"""Offline evaluation of a run_pipeline.py output (--dump).

  python eval_alpha.py --frames DIR --run OUT [--ref REF_OUT]

1) 3D quality of the final alpha (same definitions as sbs_portrait_bench.py):
   tIoU / tIoU p5 / boundary temporal flicker / sIoU / boundary L-R difference / ghost.
   Flow and disparity are recomputed from the frames with temporal_stereo (reference
   CPU implementation), independent of how the evaluated pipeline computed them.
2) Error against a reference run (e.g. the baseline): per-frame alpha MAE / max,
   IoU of alpha>0.5, person-prob MAE, and PSNR of the full-res key-frame composites.
Writes OUT/eval.json.
"""
import argparse
import json
import os
from glob import glob

import cv2
import numpy as np

import temporal_stereo as ts
from pipeline import work_size


def load(run, t):
    z = np.load(os.path.join(run, "dump", f"{t:04d}.npz"))
    return {k: z[k] for k in z.files}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default="")
    args = ap.parse_args()
    n = len(glob(os.path.join(args.run, "dump", "*.npz")))
    files = sorted(glob(os.path.join(args.frames, "*.jpg")))[:n]

    f0 = cv2.imread(files[0])
    W = f0.shape[1] // 2
    ws, _ = work_size(W, f0.shape[0])  # same work grid as the pipeline (half of a 1080p eye)
    bw = max(1, int(round(10 * ws[1] / 540.0)))  # boundary band half width: 10 px at 540 lines
    g = lambda x: cv2.cvtColor(cv2.resize(x, ws, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    min_d, num_d, _, _ = ts.estimate_disparity_range(g(f0[:, :W]), g(f0[:, W:]))
    stereo, flow = ts.StereoEstimator(min_d, num_d), ts.FlowEstimator()

    acc = ts.MetricAccumulator()
    err = {k: [] for k in ("alpha_mae", "alpha_max", "alpha_iou", "prob_mae")}
    prev, prev_g = None, None
    for t, f in enumerate(files):
        d = load(args.run, t)
        aL, aR = d["aL"].astype(np.float32), d["aR"].astype(np.float32)
        pL = d["pL"].astype(np.float32) / 255.0
        frame = cv2.imread(f)
        gL, gR = g(frame[:, :W]), g(frame[:, W:])
        dL, _ = stereo(gL, gR)
        bandL, bandR = ts.boundary_band(d["hL"], bw), ts.boundary_band(d["hR"], bw)
        if prev is not None:
            fbL, vL, _ = flow.backward_with_validity(prev_g[0], gL)
            fbR, vR, _ = flow.backward_with_validity(prev_g[1], gR)
            acc.temporal(aL, prev[0], fbL, vL, bandL)
            acc.temporal(aR, prev[1], fbR, vR, bandR)
        acc.stereo(aL, aR, dL, bandL)
        acc.ghost(aL, pL)
        prev, prev_g = (aL, aR), (gL, gR)
        if args.ref:
            r = load(args.ref, t)
            for a, ra in ((aL, r["aL"]), (aR, r["aR"])):
                diff = np.abs(a - ra.astype(np.float32))
                err["alpha_mae"].append(float(diff.mean()))
                err["alpha_max"].append(float(diff.max()))
                err["alpha_iou"].append(ts.iou(a > 0.5, ra.astype(np.float32) > 0.5))
            for k in ("pL", "pR"):
                err["prob_mae"].append(float(np.abs(d[k].astype(np.float32) - r[k].astype(np.float32)).mean() / 255.0))

    q = acc.summary()
    out = {"quality": {k: q[k] for k in ("tiou", "tiou_p5", "tflk_band", "siou", "siou_p5", "sflk_band", "ghost")}}
    if args.ref:
        out["error_vs_ref"] = {
            "alpha_mae_mean": float(np.mean(err["alpha_mae"])), "alpha_mae_worst_frame": float(np.max(err["alpha_mae"])),
            "alpha_absmax": float(np.max(err["alpha_max"])), "alpha_iou_mean": float(np.mean(err["alpha_iou"])),
            "alpha_iou_min": float(np.min(err["alpha_iou"])), "prob_mae_mean": float(np.mean(err["prob_mae"])),
        }
        ps = {}
        for kf in sorted(glob(os.path.join(args.run, "key_*.png"))):
            rf = os.path.join(args.ref, os.path.basename(kf))
            if os.path.exists(rf):
                ps[os.path.basename(kf)] = psnr(cv2.imread(kf), cv2.imread(rf))
        out["error_vs_ref"]["composite_psnr_db"] = ps
    with open(os.path.join(args.run, "eval.json"), "w") as fp:
        json.dump(out, fp, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
