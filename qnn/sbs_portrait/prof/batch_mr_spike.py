"""Both eyes as one batch (batch = 2) vs two batch-1 calls, at every model-input size (multi-res).

  python prof/batch_mr_spike.py [--profiles exact] [--out results/mr/batch]

For each padded model-input size of the test resolutions (official / aligned / native policies)
the three engines (backbone -> pixel decoder -> decoder) run on the real left / right eyes:
  b1  chain(L); chain(R)     -- what the pipeline does
  b2  chain(stack(L, R))     -- batch-2 engines (same profiles, built on first use)
30 timed iterations after warm-up, CUDA events per stage (sum over the two b1 calls). Also the
engines' activation memory for the profile used, and the b2 outputs vs b1 (mask sign agreement,
relative difference) -- FP16 tactics differ between the two engine sets.
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

SIZES = [(768, 1344), (800, 1344), (800, 1184), (800, 1088), (768, 1152), (736, 1280), (544, 960), (512, 704),
         (480, 864), (480, 640)]


def load(profiles, batch):
    return PersonSegmenter(precision="fp16", backbone="trt", fast_decoder=True, pixel_decoder="trt", decoder="trt",
                           profiles=profiles, batch=batch)


def chain(seg, x, ev=None):
    if ev:
        ev[0].record()
    f = seg.trt(x)
    if ev:
        ev[1].record()
    mf, ms = seg.trt_pd(f)
    if ev:
        ev[2].record()
    o = seg.trt_td(ms, mf)
    if ev:
        ev[3].record()
    return o


def act_mem(seg, b, hp, wp):
    """Activation memory (MB) of backbone / pixel decoder / decoder for the profile serving this size
    (the engines share one buffer sized for the largest)."""
    p = seg.trt.profile_for(((b, 3, hp, wp),))
    return [round(r.engine.get_device_memory_size_for_profile_v2(p) / 2**20, 1) for r in (seg.trt, seg.trt_pd, seg.trt_td)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="exact")
    ap.add_argument("--frame", default=os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="results/mr/batch")
    a = ap.parse_args()
    s1, s2 = load(a.profiles, 1), load(a.profiles, 2)
    m = s1.model
    L, R = split_sbs(cv2.imread(a.frame))
    torch.cuda.set_stream(s1.stream)
    res = {}
    for hp, wp in SIZES:
        xs = []
        for eye in (L, R):
            im = cv2.resize(eye, (wp, hp), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).cuda().float()
            xs.append((x - m.pixel_mean) / m.pixel_std)
        x1 = [ImageList.from_tensors([x], 32).tensor for x in xs]
        x2 = ImageList.from_tensors(xs, 32).tensor
        t1, t2 = [], []
        with torch.no_grad():
            for i in range(5 + a.iters):
                e1 = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(2)]
                e2 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                o1 = [{k: v.clone() for k, v in chain(s1, x, e).items()} for x, e in zip(x1, e1)]
                o2 = chain(s2, x2, e2)
                torch.cuda.synchronize()
                if i >= 5:
                    t1.append([sum(e[k].elapsed_time(e[k + 1]) for e in e1) for k in range(3)])
                    t2.append([e2[k].elapsed_time(e2[k + 1]) for k in range(3)])
        t1, t2 = np.median(np.array(t1), 0), np.median(np.array(t2), 0)
        m1 = torch.cat([o["pred_masks"] for o in o1]).float()
        m2 = o2["pred_masks"].float()
        r = {"b1_two_calls": dict(zip(("backbone", "pixel_decoder", "decoder"), map(float, t1))),
             "b2_one_call": dict(zip(("backbone", "pixel_decoder", "decoder"), map(float, t2))),
             "b1_chain": float(t1.sum()), "b2_chain": float(t2.sum()), "b2_vs_b1": float(t2.sum() / t1.sum() - 1),
             "act_mem_mb": {"b1": act_mem(s1, 1, hp, wp), "b2": act_mem(s2, 2, hp, wp)},
             "mask_sign_agree": float(((m1 >= 0) == (m2 >= 0)).float().mean()),
             "mask_rel": float((m1 - m2).norm() / m1.norm())}
        res[f"{hp}x{wp}"] = r
        print(hp, wp, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "batch_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
