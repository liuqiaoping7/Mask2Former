"""TensorRT FP16 engine for the Swin-L backbone, dynamic input size with optimization profiles.

The backbone is ~56% of the frame time; its ops (MatMul, LayerNorm, Softmax, GELU, window
reshapes) are all native TensorRT layers. It is exported with the shape-generic Swin forwards
of dyn_export.swin_dynamic() (window padding / crop and the shifted-window attention mask are
computed from the input size), so one engine serves every model-input size inside its
profiles (trt_common.PROFILES).

  build_engine(backbone, plan, profiles)   # export ONNX + build plan (timing cache next to it)
  TRTBackbone(plan)(x) -> {"res2": ..., "res5": ...}
"""
import os

import torch

import trt_common as tc
from dyn_export import swin_dynamic

LEVELS = ("res2", "res3", "res4", "res5")
STRIDES = (4, 8, 16, 32)


class _Wrap(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        o = self.backbone(x)
        return tuple(o[k] for k in LEVELS)


def build_engine(backbone, plan, profiles, fp16=True, trace_hw=(800, 1088), workspace_gb=2, batch=1, log=print):
    """profiles: model-input (min, opt, max) HxW list. trace_hw: export size (any size works with
    the shape-generic forwards; this one needs window padding at every stage)."""
    onnx_path = os.path.splitext(plan)[0] + ".onnx"
    axes = {"image": {2: "H", 3: "W"}, **{k: {2: f"h{k}", 3: f"w{k}"} for k in LEVELS}}
    with torch.no_grad(), swin_dynamic():
        torch.onnx.export(_Wrap(backbone).eval(), (torch.randn(batch, 3, *trace_hw, device="cuda"),), onnx_path,
                          opset_version=17, input_names=["image"], output_names=list(LEVELS),
                          dynamic_axes=axes, dynamo=False)
    tc.build(onnx_path, plan, profiles, lambda h, w: {"image": (1, 3, h, w)}, fp16=fp16,
             workspace_gb=workspace_gb, batch=batch, log=log)
    os.remove(onnx_path)  # ~0.8 GB intermediate; re-export takes seconds


class TRTBackbone(tc.TRTRunner):
    def __call__(self, x):
        return super().__call__({"image": x.float()})
