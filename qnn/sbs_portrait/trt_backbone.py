"""TensorRT engine for the Swin-L backbone (static input shape, FP16).

The backbone is 60% of the model's GPU time; its ops (MatMul, LayerNorm, Softmax, GELU,
window reshapes) are all native TensorRT layers, so it is exported to ONNX and built
as a static-shape engine. The MSDeformAttn pixel decoder and the transformer decoder
stay in PyTorch.

  build_engine(backbone, plan, shape)  # export ONNX + build plan (timing cache next to it)
  TRTBackbone(plan)(x) -> {"res2": ..., "res5": ...}
"""
import os
import time

import tensorrt as trt
import torch

LEVELS = ("res2", "res3", "res4", "res5")


class _Wrap(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        o = self.backbone(x)
        return tuple(o[k] for k in LEVELS)


def build_engine(backbone, plan, shape, workspace_gb=2, log=print):
    """Export `backbone` at a fixed input `shape` (B, 3, H, W) to ONNX and build an FP16 engine."""
    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    cache_path = os.path.join(os.path.dirname(plan), "trt_timing.cache")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(_Wrap(backbone).eval(), (torch.randn(*shape, device="cuda"),), onnx_path,
                          opset_version=17, input_names=["image"], output_names=list(LEVELS), dynamo=False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
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
    os.remove(onnx_path)  # ~0.8 GB intermediate; re-export takes seconds
    log(f"[trt] built {plan} ({ser.nbytes / 2**20:.0f} MB) in {time.time() - t0:.0f} s")


class TRTBackbone:
    """Executes the engine on torch CUDA tensors on the current (non-default) stream."""

    def __init__(self, plan):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.shape = tuple(self.engine.get_tensor_shape(self.input))
        dt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.out = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)),
                                   dtype=dt[self.engine.get_tensor_dtype(n)], device="cuda") for n in LEVELS}
        for n, t in self.out.items():
            self.ctx.set_tensor_address(n, t.data_ptr())

    def __call__(self, x):
        if tuple(x.shape) != self.shape:
            raise ValueError(f"TRT backbone built for {self.shape}, got {tuple(x.shape)}")
        x = x.float().contiguous()
        self.ctx.set_tensor_address(self.input, x.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        x.record_stream(stream)
        return dict(self.out)  # outputs are reused buffers: consume before the next call
