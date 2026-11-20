"""Segmentation-model breakdown in the production configuration (round 2).

Per eye, CUDA-event timed (median of N): TRT backbone, pixel decoder (forward_features),
transformer decoder (predictor). Then a torch.profiler kernel table for the pixel decoder.
    python prof/seg_breakdown.py --frame ~/data/sbs/portrait_0121/0001.jpg --out results/r2/seg_breakdown
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
from seg import PersonSegmenter  # noqa: E402


def timed(fn, n):
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--compile", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L, _ = split_sbs(cv2.imread(a.frame))
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=L.shape[:2], fast_decoder=True,
                          compile_decoders=bool(a.compile))
    m = seg.model
    head = m.sem_seg_head
    x = seg.preprocess(L)
    with torch.no_grad():
        im = (x.to(m.device).float() - m.pixel_mean) / m.pixel_std
        t = ImageList.from_tensors([im], m.size_divisibility).tensor
        ac = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731

        def bb():
            with ac():
                return {k: v.clone() for k, v in seg.trt(t).items()}

        def pd(f):
            with ac():
                return head.pixel_decoder.forward_features(f)

        def td(mf, ms):
            with ac():
                return head.predictor(ms, mf, None)

        def full():
            with ac():
                return head(seg.trt(t))

        for _ in range(3):  # warm-up / compile
            full()
        torch.cuda.synchronize()
        res = {}
        res["backbone_ms"], feats = timed(bb, a.n)
        res["pixel_decoder_ms"], (mf, _, ms) = timed(lambda: pd(feats), a.n)
        res["transformer_decoder_ms"], _ = timed(lambda: td(mf, ms), a.n)
        res["seg_model_eye_ms"], _ = timed(full, a.n)
        res["shapes"] = {k: list(v.shape) for k, v in feats.items()}
        print(json.dumps(res, indent=1))

        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                pd(feats)
            torch.cuda.synchronize()
        tab = p.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=90)
        print(tab)
        rows = [(e.key, e.device_time_total / 5 / 1000, e.count / 5) for e in p.key_averages() if e.device_time_total > 0]
        res["pixel_decoder_kernels"] = sorted(rows, key=lambda r: -r[1])[:40]
    json.dump(res, open(os.path.join(a.out, "seg_breakdown.json"), "w"), indent=1)
    open(os.path.join(a.out, "pixel_decoder_kernels.txt"), "w").write(tab)


if __name__ == "__main__":
    main()
