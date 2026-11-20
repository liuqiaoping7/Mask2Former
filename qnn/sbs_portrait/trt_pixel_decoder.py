"""TensorRT FP16 engine for the MSDeformAttn pixel decoder (static shapes).

The pixel decoder (6 MSDeformAttn encoder layers on res3-res5 + FPN to res2) runs in PyTorch
as many small kernels: the upstream FP32 MSDA im2col kernel (1.3 ms per layer) and dtype
casts around every FP32-residual LayerNorm / Linear under autocast. Here the whole
forward_features is exported to ONNX with MSDeformAttnFunction mapped to TensorRT's built-in
MultiscaleDeformableAttnPlugin_TRT (same value / sampling_locations / attention_weights
layout), and built as one FP16 engine.

  build_engine(pixel_decoder, feat_shapes, plan)
  TRTPixelDecoder(plan)(feats) -> (mask_features, [ms0, ms1, ms2])
"""
import os
import time

import onnx
import tensorrt as trt
import torch

IN = ("res2", "res3", "res4", "res5")
OUT = ("mask_features", "ms0", "ms1", "ms2")
_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}


class _MSDATRT(torch.autograd.Function):
    """MSDeformAttnFunction for export: runs the CUDA op, emits the TensorRT plugin node."""

    @staticmethod
    def forward(ctx, value, shapes, start, loc, attn, im2col_step):
        from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDA

        return MSDA.ms_deform_attn_forward(value, shapes, start, loc, attn, im2col_step)

    @staticmethod
    def symbolic(g, value, shapes, start, loc, attn, im2col_step):
        s32 = g.op("Cast", shapes, to_i=6)  # INT32
        i32 = g.op("Cast", start, to_i=6)
        out = g.op("MultiscaleDeformableAttnPlugin_TRT", value, s32, i32, loc, attn)
        # the plugin outputs (N, Lq, M, D); upstream returns (N, Lq, M*D)
        return g.op("Reshape", out, g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.int64)))


class _ConstPE(torch.nn.Module):
    """The sine position embedding depends only on the feature-map shape. Exported as a
    computed graph it comes out wrong in TensorRT (~40% relative error, both FP32 and FP16;
    the norms, convs and the MSDA plugin are exact), so it is baked in as constants."""

    def __init__(self, pe, hws, device):
        super().__init__()
        with torch.no_grad():
            for h, w in hws:
                self.register_buffer(f"pe_{h}x{w}", pe(torch.zeros(1, 1, h, w, device=device)))

    def forward(self, x, mask=None):
        return getattr(self, f"pe_{x.shape[-2]}x{x.shape[-1]}").to(x.dtype)


class _Wrap(torch.nn.Module):
    def __init__(self, pd):
        super().__init__()
        self.pd = pd

    def forward(self, res2, res3, res4, res5):
        # upstream (FP32) forward_features; precision is chosen by the TensorRT builder
        mf, _, ms = type(self.pd).forward_features(self.pd, dict(res2=res2, res3=res3, res4=res4, res5=res5))
        # multi-scale features as NHWC: upstream produces them as transpose(1, 2).view(...), i.e.
        # channels-last strides, and the compiled transformer decoder (flatten(2).permute(2, 0, 1))
        # is 2x slower on plain NCHW (19 vs 9 ms per eye)
        return (mf, *(x.permute(0, 2, 3, 1) for x in ms))


def build_engine(pd, feat_shapes, plan, workspace_gb=2, log=print, fp16=True, fp32_layers=()):
    """feat_shapes: {res2: (1, C, H, W), ...}; writes `plan` (timing cache next to it).

    fp32_layers (with fp16): layer groups pinned to FP32 -- "sampling": everything that
    produces the MSDA sampling locations / attention weights and the plugin itself (in FP16 a
    normalised location in [0, 1] is only good to ~1e-3, i.e. ~0.2 px on the 168-wide level);
    "norm": LayerNorm / GroupNorm."""
    from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn as msda_mod

    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    cache_path = os.path.join(os.path.dirname(plan), "trt_timing.cache")
    t0 = time.time()
    saved, saved_pe = msda_mod.MSDeformAttnFunction, pd.pe_layer
    msda_mod.MSDeformAttnFunction = _MSDATRT
    pd.pe_layer = _ConstPE(saved_pe, [tuple(feat_shapes[k][-2:]) for k in pd.transformer_in_features], "cuda")
    try:
        with torch.no_grad():
            args = tuple(torch.randn(*feat_shapes[k], device="cuda") for k in IN)
            torch.onnx.export(_Wrap(pd).eval(), args, onnx_path, opset_version=17, input_names=list(IN),
                              output_names=list(OUT), dynamo=False)
    finally:
        msda_mod.MSDeformAttnFunction, pd.pe_layer = saved, saved_pe
    ops = [n.op_type for n in onnx.load(onnx_path, load_external_data=False).graph.node]
    n_plugin = ops.count("MultiscaleDeformableAttnPlugin_TRT")
    if n_plugin == 0:  # upstream catches the export error and falls back to the PyTorch core
        raise RuntimeError("MSDA was not exported as the TensorRT plugin")
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
        for i in range(net.num_outputs):  # FP16 outputs: what the FP16 transformer decoder consumes
            net.get_output(i).dtype = trt.float16
        pinned = _pin_fp32(net, fp32_layers)
        if pinned:
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            log(f"[trt] {pinned} layers pinned to FP32 ({', '.join(fp32_layers)})")
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
    log(f"[trt] built {plan} ({ser.nbytes / 2**20:.0f} MB, {n_plugin} MSDA plugin nodes) in {time.time() - t0:.0f} s")


def _pin_fp32(net, groups):
    n = 0
    for i in range(net.num_layers):
        ly = net.get_layer(i)
        name = ly.name
        hit = False
        if "sampling" in groups and "/self_attn/" in name and "output_proj" not in name:
            # sampling_offsets, attention_weights + softmax, location arithmetic, plugin; value_proj
            # too: the plugin needs value / locations / weights in one precision
            hit = True
        if "sampling" in groups and ly.type in (trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3):
            hit = True
        if "norm" in groups and ly.type == trt.LayerType.NORMALIZATION:
            hit = True
        gemm = ly.type in (trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION)
        floats = all(ly.get_output(j).dtype in (trt.float32, trt.float16) for j in range(ly.num_outputs))
        if "gemm_out32" in groups and ly.type == trt.LayerType.MATRIX_MULTIPLY:
            # FP16 math, FP32 output: steers TensorRT to FP32-accumulating tactics
            ly.precision = trt.float16
            ly.set_output_type(0, trt.float32)
            n += 1
            continue
        if "nongemm" in groups and not gemm and ly.type not in (trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3):
            hit = True
        if hit and floats and ly.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.GATHER):
            if ly.type not in (trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3):
                # a plugin's layer precision would also apply to its INT32 inputs (spatial shapes,
                # level starts); its FP32 output type alone forces the FP32 kernel (out type == in type)
                ly.precision = trt.float32
            for j in range(ly.num_outputs):
                ly.set_output_type(j, trt.float32)
            n += 1
    return n


class TRTPixelDecoder:
    """Executes the engine on the current stream; outputs are reused buffers (consume them
    before the next call). Calls from different streams are ordered (the context's activation
    memory is shared)."""

    def __init__(self, plan):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
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

    def __call__(self, feats):
        stream = torch.cuda.current_stream()
        if self.last is not None and self.last != stream:
            stream.wait_stream(self.last)
        self.last = stream
        keep = []
        for n, (shape, dt) in self.in_spec.items():
            x = feats[n]
            if tuple(x.shape) != shape:
                raise ValueError(f"TRT pixel decoder built for {n}{shape}, got {tuple(x.shape)}")
            x = x.to(dt).contiguous()
            keep.append(x)
            self.ctx.set_tensor_address(n, x.data_ptr())
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        for x in keep:
            x.record_stream(stream)
        o = self.out
        return o["mask_features"], [o[k].permute(0, 3, 1, 2) for k in ("ms0", "ms1", "ms2")]  # NCHW view, NHWC memory
