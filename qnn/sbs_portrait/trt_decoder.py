"""TensorRT FP16 engine for the masked-attention transformer decoder (inference forward of
seg._decoder_inference_forward, in its shape-generic form dyn_export.decoder_forward), dynamic
shapes with optimization profiles.

Export details:
  * the sine position embedding is the closed form of dyn_export.DynSinePE (the cumsum form is
    wrong in TensorRT, see trt_pixel_decoder; baked constants would fix one size only);
  * the boolean attention masks become additive float masks with a finite fill (-1e4): with the
    -inf fill of nn.MultiheadAttention the FP16 engine's fused attention produces NaNs; exp(-1e4)
    is exactly 0, so the softmax is unchanged;
  * the multi-scale inputs are NHWC, matching the pixel-decoder engine's outputs (zero-copy
    hand-off), all inputs FP16; pred_masks is produced in FP16 (what the fused person kernel reads).

  build_engine(predictor, plan, profiles)
  TRTDecoder(plan)(ms, mf) -> {"pred_logits": (1, Q, C+1) fp32, "pred_masks": (1, Q, H/4, W/4) fp16}
"""
import os

import tensorrt as trt
import torch

import trt_common as tc
from dyn_export import DynSinePE, decoder_forward

IN = ("ms0", "ms1", "ms2", "mf")
OUT = ("pred_logits", "pred_masks")
MASK_FILL = -1e4


class _Wrap(torch.nn.Module):
    def __init__(self, pred, batch=1):
        super().__init__()
        self.pred = pred
        self.pe = DynSinePE(pred.pe_layer)
        self.batch1 = batch == 1

    def forward(self, ms0, ms1, ms2, mf):
        ms = [x.permute(0, 3, 1, 2) for x in (ms0, ms1, ms2)]  # NHWC inputs
        o = decoder_forward(self.pred, ms, mf, self.pe, MASK_FILL, self.batch1)
        return o["pred_logits"], o["pred_masks"]


def stage_shapes(c=256):
    return lambda h, w: {"ms0": (1, h // 32, w // 32, c), "ms1": (1, h // 16, w // 16, c),
                         "ms2": (1, h // 8, w // 8, c), "mf": (1, c, h // 4, w // 4)}


def build_engine(pred, plan, profiles, fp16=True, trace_hw=(800, 1088), workspace_gb=2, batch=1, log=print):
    """profiles: model-input (min, opt, max) HxW list."""
    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    shapes = stage_shapes()
    axes = {"ms0": {1: "h32", 2: "w32"}, "ms1": {1: "h16", 2: "w16"}, "ms2": {1: "h8", 2: "w8"},
            "mf": {2: "h4", 3: "w4"}, "pred_masks": {2: "h4", 3: "w4"}}
    s = tc.with_batch(shapes(*trace_hw), batch)
    args = tuple(torch.randn(*s[k], device="cuda") for k in IN)
    with torch.no_grad():
        torch.onnx.export(_Wrap(pred, batch).eval(), args, onnx_path, opset_version=17, input_names=list(IN),
                          output_names=list(OUT), dynamic_axes=axes, dynamo=False)
    f16 = trt.float16 if fp16 else trt.float32
    tc.build(onnx_path, plan, profiles, shapes, fp16=fp16, in_dtypes={k: f16 for k in IN},
             out_dtypes={"pred_masks": f16, "pred_logits": trt.float32}, workspace_gb=workspace_gb, batch=batch,
             log=log)
    os.remove(onnx_path)


class TRTDecoder(tc.TRTRunner):
    """Outputs are reused buffers (consume them before the next call with the same shapes)."""

    def __call__(self, ms, mf):
        # NCHW views of NHWC memory (pixel-decoder engine) -> NHWC contiguous without a copy
        return super().__call__(dict(zip(IN, [x.permute(0, 2, 3, 1) for x in ms] + [mf])))
