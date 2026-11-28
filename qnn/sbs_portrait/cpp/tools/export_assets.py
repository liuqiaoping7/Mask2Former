"""Export what the C++ pipeline needs from the Python side: model constants, the engine plans built
by the Python pipeline (they must exist: run run_multires.py once) and the PiP test icon.

  python export_assets.py --out DIR
"""
import argparse
import os
import sys
from glob import glob

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SBS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SBS)
import trt_common as tc  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from pip_compose import make_icon  # noqa: E402
from seg import DEFAULT_CONFIG, PERSON_ID, REPO  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--profiles", default="exact")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(DEFAULT_CONFIG)

tag = f"{a.profiles}_{tc.plan_tag(tc.profile_list(a.profiles))}"
w = os.path.join(REPO, "weights")
plans = {"backbone": f"{w}/swinl_backbone_fp16_{tag}_f16f.plan",
         "pixel_decoder": f"{w}/pixel_decoder_fp16_{tag}_f16f.plan",
         "decoder": f"{w}/decoder_fp16_{tag}.plan"}
for p in plans.values():
    assert os.path.exists(p), f"missing {p}: build the engines with the Python pipeline first"
assert cfg.INPUT.FORMAT == "RGB"
icon = os.path.join(os.path.abspath(a.out), "icon.png")
cv2.imwrite(icon, make_icon(512))
m = cfg.MODEL.MASK_FORMER.TEST
kv = dict(plans, icon=icon,
          mean=" ".join(map(str, cfg.MODEL.PIXEL_MEAN)), std=" ".join(map(str, cfg.MODEL.PIXEL_STD)),
          object_mask_threshold=m.OBJECT_MASK_THRESHOLD, overlap_threshold=m.OVERLAP_THRESHOLD,
          min_size=cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST,
          size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY, person_id=PERSON_ID)
with open(os.path.join(a.out, "assets.txt"), "w") as f:
    f.writelines(f"{k} {v}\n" for k, v in kv.items())
print(open(os.path.join(a.out, "assets.txt")).read())
