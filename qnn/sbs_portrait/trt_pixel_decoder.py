"""TensorRT FP16 engine for the MSDeformAttn pixel decoder (dynamic shapes, optimization profiles).

The pixel decoder (6 MSDeformAttn encoder layers on res3-res5 + FPN to res2) runs in PyTorch
as many small kernels: the upstream FP32 MSDA im2col kernel (1.3 ms per layer) and dtype
casts around every FP32-residual LayerNorm / Linear under autocast. Here the whole
forward_features is exported to ONNX with MSDeformAttnFunction mapped to TensorRT's built-in
MultiscaleDeformableAttnPlugin_TRT (same value / sampling_locations / attention_weights
layout), and built as one FP16 engine. The export uses the shape-generic forward and sine
position embedding of dyn_export (spatial shapes, level starts and reference points are
computed from the input sizes), so one engine with several optimization profiles serves every
model-input size.

  build_engine(pixel_decoder, plan, profiles)
  TRTPixelDecoder(plan)(feats) -> (mask_features, [ms0, ms1, ms2])
"""
import os

import onnx
import tensorrt as trt
import torch

import trt_common as tc
from dyn_export import DynSinePE, pixel_decoder_forward

IN = ("res2", "res3", "res4", "res5")
OUT = ("mask_features", "ms0", "ms1", "ms2")
STRIDES = (4, 8, 16, 32)


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


class _MSDAUpstreamTRT(_MSDATRT):
    """Same, emitting the plugin that wraps the project's own CUDA kernel (trt_msda_plugin.py):
    INT64 shapes / level starts as the kernel takes them, output already (N, Lq, M*D)."""

    @staticmethod
    def symbolic(g, value, shapes, start, loc, attn, im2col_step):
        return g.op("MSDeformAttnUpstream_TRT", value, shapes, start, loc, attn)


MSDA_OPS = {"builtin": (_MSDATRT, "MultiscaleDeformableAttnPlugin_TRT"),
            "upstream": (_MSDAUpstreamTRT, "MSDeformAttnUpstream_TRT")}


class _Wrap(torch.nn.Module):
    def __init__(self, pd):
        super().__init__()
        self.pd = pd
        self.pe = DynSinePE(pd.pe_layer)

    def forward(self, res2, res3, res4, res5):
        # upstream (FP32) computation in its shape-generic form; precision is chosen by the builder
        mf, ms = pixel_decoder_forward(self.pd, dict(res2=res2, res3=res3, res4=res4, res5=res5), self.pe)
        # multi-scale features as NHWC: upstream produces them as transpose(1, 2).view(...), i.e.
        # channels-last strides, and the transformer decoder (flatten(2).permute(2, 0, 1)) reads
        # them that way (NCHW-contiguous made the compiled decoder 2x slower, 19 vs 9 ms per eye)
        return (mf, *(x.permute(0, 2, 3, 1) for x in ms))


def stage_shapes(chans):
    """chans: {res2: C, ...} -> f(H, W) giving the engine input shapes for a model input H x W."""
    return lambda h, w: {k: (1, chans[k], h // s, w // s) for k, s in zip(IN, STRIDES)}


def build_engine(pd, plan, profiles, chans=None, fp16=True, fp32_layers=(), trace_hw=(800, 1088),
                 workspace_gb=2, batch=1, msda="builtin", detailed=False, log=print):
    """profiles: model-input (min, opt, max) HxW list; chans: backbone channels per level.
    Writes `plan` (timing cache next to it).

    fp32_layers (with fp16): layer groups pinned to FP32 -- "sampling": everything that
    produces the MSDA sampling locations / attention weights and the plugin itself (in FP16 a
    normalised location in [0, 1] is only good to ~1e-3, i.e. ~0.2 px on the 168-wide level);
    "norm": LayerNorm / GroupNorm.
    msda: "builtin" -- TensorRT's MultiscaleDeformableAttnPlugin_TRT (FP32 / FP16); "upstream" -- the
    plugin wrapping this project's MSDeformAttn CUDA kernel (FP32 only, trt_msda_plugin.py)."""
    from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn as msda_mod

    chans = chans or {"res2": 192, "res3": 384, "res4": 768, "res5": 1536}
    shapes = stage_shapes(chans)
    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    saved = msda_mod.MSDeformAttnFunction
    fn, op_name = MSDA_OPS[msda]
    if msda == "upstream":
        import trt_msda_plugin
        trt_msda_plugin.load()  # registered before the ONNX parser looks the op up
    msda_mod.MSDeformAttnFunction = fn
    axes = {k: {2: f"h{k}", 3: f"w{k}"} for k in IN}
    axes.update({"mask_features": {2: "h4", 3: "w4"}, "ms0": {1: "h32", 2: "w32"}, "ms1": {1: "h16", 2: "w16"},
                 "ms2": {1: "h8", 2: "w8"}})
    try:
        with torch.no_grad():
            args = tuple(torch.randn(*tc.with_batch(shapes(*trace_hw), batch)[k], device="cuda") for k in IN)
            torch.onnx.export(_Wrap(pd).eval(), args, onnx_path, opset_version=17, input_names=list(IN),
                              output_names=list(OUT), dynamic_axes=axes, dynamo=False)
    finally:
        msda_mod.MSDeformAttnFunction = saved
    ops = [n.op_type for n in onnx.load(onnx_path, load_external_data=False).graph.node]
    n_plugin = ops.count(op_name)
    if n_plugin == 0:  # upstream catches the export error and falls back to the PyTorch core
        raise RuntimeError("MSDA was not exported as the TensorRT plugin")

    def configure(net, cfg):
        if fp16:
            pinned = _pin_fp32(net, fp32_layers)
            if pinned:
                cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                log(f"[trt] {pinned} layers pinned to FP32 ({', '.join(fp32_layers)})")

    # FP16 outputs: what the FP16 transformer decoder consumes
    tc.build(onnx_path, plan, profiles, shapes, fp16=fp16, plugins=True, configure=configure,
             out_dtypes={k: trt.float16 for k in OUT} if fp16 else None, workspace_gb=workspace_gb, batch=batch,
             detailed=detailed, log=log)
    os.remove(onnx_path)
    log(f"[trt] {n_plugin} MSDA plugin nodes ({op_name})")


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


class TRTPixelDecoder(tc.TRTRunner):
    """Outputs are reused buffers (consume them before the next call with the same shapes)."""

    def __init__(self, plan, msda="builtin"):
        if msda == "upstream":
            import trt_msda_plugin
            trt_msda_plugin.load()  # the plan references the plugin: register it before deserializing
        super().__init__(plan, plugins=True)

    def __call__(self, feats):
        o = super().__call__(feats)
        return o["mask_features"], [o[k].permute(0, 3, 1, 2) for k in ("ms0", "ms1", "ms2")]  # NCHW view, NHWC memory
