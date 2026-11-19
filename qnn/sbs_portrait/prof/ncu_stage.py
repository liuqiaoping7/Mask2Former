"""One-eye Mask2Former forward with NVTX ranges per stage, for ncu --nvtx filtering.

usage: ncu --nvtx --nvtx-include "S1_backbone/" ... python ncu_stage.py
Stages: S1_backbone S2_pixel_decoder S3_transformer_decoder S4_mask_upsample_to_input
        S5_upsample_crop_1080p S6_semantic_inference S7_panoptic_inference S8_person_prob
Ranges are pushed only in the final (profiled) forward; warm-up runs are unmarked.
"""
import os
import sys

import cv2
import torch
import torch.nn.functional as F

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BENCH)
import seg as segmod  # noqa: E402
import mask2former.maskformer_model as mfm  # noqa: E402

ACTIVE = [False]


def scoped(f, label):
    def g(*a, **k):
        if ACTIVE[0]:
            torch.cuda.nvtx.range_push(label)
        try:
            return f(*a, **k)
        finally:
            if ACTIVE[0]:
                torch.cuda.nvtx.range_pop()
    return g


seg = segmod.PersonSegmenter()
m = seg.model
m.backbone.forward = scoped(m.backbone.forward, "S1_backbone")
m.sem_seg_head.pixel_decoder.forward_features = scoped(m.sem_seg_head.pixel_decoder.forward_features, "S2_pixel_decoder")
m.sem_seg_head.predictor.forward = scoped(m.sem_seg_head.predictor.forward, "S3_transformer_decoder")
mfm.sem_seg_postprocess = scoped(mfm.sem_seg_postprocess, "S5_upsample_crop_1080p")


class _FP:
    def __getattr__(self, n):
        return getattr(F, n)

    interpolate = staticmethod(scoped(F.interpolate, "S4_mask_upsample_to_input"))


mfm.F = _FP()
m.semantic_inference = scoped(m.semantic_inference, "S6_semantic_inference")
m.panoptic_inference = scoped(m.panoptic_inference, "S7_panoptic_inference")

img = cv2.imread(os.path.join(os.path.expanduser(os.environ.get("SBS_DATA", "~/data/sbs")), "portrait_0121/0010.jpg"))[:, :1920]
with torch.no_grad():
    for _ in range(2):
        seg([img])
    torch.cuda.synchronize()
    ACTIVE[0] = True
    o = m([{"image": torch.as_tensor(seg.aug.get_transform(img[:, :, ::-1]).apply_image(img[:, :, ::-1]).transpose(2, 0, 1).copy()),
            "height": 1080, "width": 1920}])[0]
    torch.cuda.nvtx.range_push("S8_person_prob")
    sem = o["sem_seg"].float()
    prob = sem[0] / sem.sum(0).clamp_min(1e-6)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
print("done")
