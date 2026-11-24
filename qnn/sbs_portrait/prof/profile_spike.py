"""Optimization-profile granularity: per-eye segmentation chain time at each model-input size.

  python prof/profile_spike.py --set multires|wide|1080p|official|static1080 [--out results/mr]

For every padded model-input size of the test resolutions (both size policies) runs the three
engines (backbone -> pixel decoder -> decoder) on a real eye, 30 timed iterations after warm-up,
CUDA events per stage. "static1080" runs the former static 768x1344 plans through the same
runner (reference for the 1080p profile). Also saves pred_masks / logits of each size for the
output comparison between sets (--compare A B).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectron2.structures import ImageList  # noqa: E402

from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

SIZES = [(768, 1344), (800, 1344), (800, 1184), (800, 1088), (736, 1280), (544, 960), (512, 704), (480, 864), (480, 640)]


def load(set_name):
    if set_name != "static1080":
        return PersonSegmenter(precision="fp16", backbone="trt", fast_decoder=True, pixel_decoder="trt",
                               decoder="trt", profiles=set_name)
    from trt_backbone import TRTBackbone
    from trt_decoder import TRTDecoder
    from trt_pixel_decoder import TRTPixelDecoder
    seg = PersonSegmenter(precision="fp16")
    w = os.path.join(REPO, "weights")
    seg.trt = TRTBackbone(os.path.join(w, "swinl_backbone_fp16_768x1344.plan"))
    seg.trt_pd = TRTPixelDecoder(os.path.join(w, "pixel_decoder_fp16_768x1344.plan"))
    seg.trt_td = TRTDecoder(os.path.join(w, "decoder_fp16_768x1344.plan"))
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="multires")
    ap.add_argument("--frame", default=os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg"))
    ap.add_argument("--out", default="results/mr")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    raw = os.path.expanduser(f"~/data/sbs/mr/profile_spike")
    os.makedirs(raw, exist_ok=True)
    if a.compare:
        return compare(raw, a.compare, a.out)
    seg = load(a.set)
    m = seg.model
    L, _ = split_sbs(cv2.imread(a.frame))
    res = {}
    torch.cuda.set_stream(seg.stream)
    for hp, wp in SIZES:
        if a.set == "static1080" and (hp, wp) != (768, 1344):
            continue
        im = cv2.resize(L, (wp, hp), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).cuda().float()
        x = ImageList.from_tensors([(x - m.pixel_mean) / m.pixel_std], 32).tensor
        try:
            seg.trt.profile_for(((1, 3, hp, wp),))
        except ValueError:
            res[f"{hp}x{wp}"] = None
            continue
        ev = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(a.iters)]
        with torch.no_grad():
            for i in range(5 + a.iters):
                e = ev[i - 5] if i >= 5 else None
                if e:
                    e[0].record()
                f = seg.trt(x)
                if e:
                    e[1].record()
                mf, ms = seg.trt_pd(f)
                if e:
                    e[2].record()
                o = seg.trt_td(ms, mf)
                if e:
                    e[3].record()
        torch.cuda.synchronize()
        t = np.array([[e[k].elapsed_time(e[k + 1]) for k in range(3)] for e in ev])
        med = np.median(t, 0)
        res[f"{hp}x{wp}"] = {"backbone": float(med[0]), "pixel_decoder": float(med[1]), "decoder": float(med[2]),
                             "chain": float(np.median(t.sum(1))),
                             "profile": seg.trt.profile_for(((1, 3, hp, wp),))}
        torch.save({k: v.float().cpu() for k, v in o.items()}, os.path.join(raw, f"{a.set}_{hp}x{wp}.pt"))
        print(a.set, hp, wp, json.dumps(res[f"{hp}x{wp}"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"profile_spike_{a.set}.json"), "w"), indent=1)


def compare(raw, sets, out):
    r = {}
    for hp, wp in SIZES:
        fa, fb = (os.path.join(raw, f"{s}_{hp}x{wp}.pt") for s in sets)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = torch.load(fa), torch.load(fb)
        r[f"{hp}x{wp}"] = {
            "masks_rel": float((A["pred_masks"] - B["pred_masks"]).norm() / B["pred_masks"].norm()),
            "logits_rel": float((A["pred_logits"] - B["pred_logits"]).norm() / B["pred_logits"].norm()),
            "mask_sign_agree": float(((A["pred_masks"] >= 0) == (B["pred_masks"] >= 0)).float().mean()),
        }
        print(hp, wp, r[f"{hp}x{wp}"])
    json.dump(r, open(os.path.join(out, f"profile_compare_{sets[0]}_vs_{sets[1]}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
