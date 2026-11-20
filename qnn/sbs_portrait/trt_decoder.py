"""TensorRT FP16 engine for the masked-attention transformer decoder (inference forward of
seg._decoder_inference_forward, static shapes).

Export details:
  * the sine position embedding is baked in as constants (computed in TensorRT it is wrong,
    see trt_pixel_decoder._ConstPE);
  * the boolean attention masks become additive float masks with a finite fill (-1e4): with the
    -inf fill of nn.MultiheadAttention the FP16 engine's fused attention produces NaNs; exp(-1e4)
    is exactly 0, so the softmax is unchanged;
  * the multi-scale inputs are NHWC, matching the pixel-decoder engine's outputs (zero-copy
    hand-off), all inputs FP16; pred_masks is produced in FP16 (what the fused person kernel reads).

  build_engine(predictor, ms_shapes, mf_shape, plan)
  TRTDecoder(plan)(ms, mf) -> {"pred_logits": (1, Q, C+1) fp32, "pred_masks": (1, Q, H/4, W/4) fp16}
"""
import os
import time

import tensorrt as trt
import torch

from trt_pixel_decoder import _DT, _ConstPE

IN = ("ms0", "ms1", "ms2", "mf")
OUT = ("pred_logits", "pred_masks")
MASK_FILL = -1e4


class _Wrap(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, ms0, ms1, ms2, mf):
        ms = [x.permute(0, 3, 1, 2) for x in (ms0, ms1, ms2)]  # NHWC inputs
        o = self.pred(ms, mf, None)
        return o["pred_logits"], o["pred_masks"]


def build_engine(pred, ms_shapes, mf_shape, plan, workspace_gb=2, log=print):
    """ms_shapes: NCHW shapes of the three multi-scale features; mf_shape: mask features (NCHW)."""
    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    cache_path = os.path.join(os.path.dirname(plan), "trt_timing.cache")
    t0 = time.time()
    saved_pe, saved_fill = pred.pe_layer, getattr(pred, "_mask_fill", None)
    pred.pe_layer = _ConstPE(saved_pe, [tuple(s[-2:]) for s in ms_shapes], "cuda")
    pred._mask_fill = MASK_FILL
    try:
        nhwc = [(s[0], s[2], s[3], s[1]) for s in ms_shapes]
        args = tuple(torch.randn(*s, device="cuda") for s in nhwc + [tuple(mf_shape)])
        with torch.no_grad():
            torch.onnx.export(_Wrap(pred).eval(), args, onnx_path, opset_version=17, input_names=list(IN),
                              output_names=list(OUT), dynamo=False)
    finally:
        pred.pe_layer, pred._mask_fill = saved_pe, saved_fill
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    for i in range(net.num_inputs):
        net.get_input(i).dtype = trt.float16
    for i in range(net.num_outputs):
        o = net.get_output(i)
        o.dtype = trt.float16 if o.name == "pred_masks" else trt.float32
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cache = cfg.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(plan, "wb") as f:
        f.write(ser)
    with open(cache_path, "wb") as f:
        f.write(cfg.get_timing_cache().serialize())
    os.remove(onnx_path)
    log(f"[trt] built {plan} ({ser.nbytes / 2**20:.0f} MB) in {time.time() - t0:.0f} s")


class TRTDecoder:
    """Executes the engine on the current stream; outputs are reused buffers (consume them
    before the next call); calls from different streams are ordered."""

    def __init__(self, plan):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        e = self.engine
        self.in_spec = {n: (tuple(e.get_tensor_shape(n)), _DT[e.get_tensor_dtype(n)]) for n in IN}
        self.out = {n: torch.empty(tuple(e.get_tensor_shape(n)), dtype=_DT[e.get_tensor_dtype(n)], device="cuda")
                    for n in OUT}
        for n, t in self.out.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.last = None

    def __call__(self, ms, mf):
        stream = torch.cuda.current_stream()
        if self.last is not None and self.last != stream:
            stream.wait_stream(self.last)
        self.last = stream
        # NCHW views of NHWC memory (pixel-decoder engine) -> NHWC contiguous without a copy
        xs = dict(zip(IN, [x.permute(0, 2, 3, 1) for x in ms] + [mf]))
        keep = []
        for n, (shape, dt) in self.in_spec.items():
            x = xs[n]
            if tuple(x.shape) != shape:
                raise ValueError(f"TRT decoder built for {n}{shape}, got {tuple(x.shape)}")
            x = x.to(dt).contiguous()
            keep.append(x)
            self.ctx.set_tensor_address(n, x.data_ptr())
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        for x in keep:
            x.record_stream(stream)
        return dict(self.out)
