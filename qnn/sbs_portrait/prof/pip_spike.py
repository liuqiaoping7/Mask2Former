"""Compositing + picture-in-picture kernels (pip_compose.py) at every test resolution.

  python prof/pip_spike.py [--dump-run ~/data/sbs/mr/opt_final/run] [--out results/mr/pip]

Per size: real frame 75, its final alphas from a pipeline dump, the pipeline background; PiP source
none / "frame" (local SBS image) / SBS 3D icon (512x512 per eye, RGBA). For each implementation:
latency per call (CUDA events around one call, median of 100: includes the
launch overhead of each implementation) and GPU kernel time per call (torch profiler, all kernels) and the error against the CPU reference (OpenCV area
resize + float blend): mean / max |diff| over the frame and PSNR of the PiP rectangles. Saves
1080p crops of the top-right corner per implementation.
"""
import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pip_compose as pc  # noqa: E402

SIZES = ["640x480", "676x504", "854x480", "960x540", "1280x720", "1252x862", "1396x842", "1440x1080", "1680x966", "1920x1080"]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=os.path.expanduser("~/data/sbs/mr/frames"))
    ap.add_argument("--bg", default=os.path.expanduser("~/data/sbs/bg_0045/0001.jpg"))
    ap.add_argument("--dump-run", default=os.path.expanduser("~/data/sbs/mr/opt_final/run"))
    ap.add_argument("--out", default="results/mr/pip")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    icon = pc.make_icon(512)
    cv2.imwrite(os.path.join(a.out, "icon_3d_sbs.png"), icon)
    bg_sbs = cv2.imread(a.bg)
    res = {}
    for s in SIZES:
        W, H = map(int, s.split("x"))
        frame = cv2.imread(sorted(glob(os.path.join(a.sets, s, "*.jpg")))[75])
        z = np.load(os.path.join(a.dump_run, s, "dump", "0075.npz"))
        aL, aR = z["aL"].astype(np.float32), z["aR"].astype(np.float32)
        h, w = aL.shape
        Wb = bg_sbs.shape[1] // 2
        bg = np.hstack([cv2.resize(bg_sbs[:, :Wb], (W, H), interpolation=cv2.INTER_LINEAR),
                        cv2.resize(bg_sbs[:, Wb:], (W, H), interpolation=cv2.INTER_LINEAR)])
        ft, bt = torch.from_numpy(frame).cuda(), torch.from_numpy(bg).cuda()
        at = [torch.from_numpy(x).cuda() for x in (aL, aR)]
        lay = pc.pip_layout(H, W)
        r = {"pip_hw": [lay["ph"], lay["pw"]]}
        for src_name, src in (("none", None), ("frame", "frame"), ("icon", icon)):
            ref = pc.reference(frame, bg, aL, aR, None if src is None else (frame if isinstance(src, str) else icon), lay)
            for kind in pc.KINDS:
                if src is None and kind == "cuda_tex1":
                    continue
                comp = pc.Composer(H, W, (w, h), bt, kind=kind, src=src)
                with torch.no_grad():
                    for _ in range(5):
                        out = comp(ft, *at)
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
                    for e0, e1 in ev:
                        e0.record()
                        out = comp(ft, *at)
                        e1.record()
                    torch.cuda.synchronize()
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(20):
                            out = comp(ft, *at)
                        torch.cuda.synchronize()
                    dev_us = sum(e.self_device_time_total for e in prof.key_averages()) / 20
                o = out.cpu().numpy()
                d = np.abs(o.astype(np.int16) - ref.astype(np.int16))
                y0, ph, pw = lay["y0"], lay["ph"], lay["pw"]
                pip_psnr = None
                if src is not None:
                    reg = [(slice(y0, y0 + ph), slice(e * W + lay["x0"][e], e * W + lay["x0"][e] + pw)) for e in range(2)]
                    pip_psnr = min(psnr(o[rg], ref[rg]) for rg in reg)
                r[f"{src_name}/{kind}"] = {"ms": float(np.median([e0.elapsed_time(e1) for e0, e1 in ev])),
                                           "kernel_ms": dev_us / 1000.0,
                                           "mean_abs": float(d.mean()), "max_abs": int(d.max()), "pip_psnr_db": pip_psnr}
                print(s, src_name, kind, json.dumps(r[f"{src_name}/{kind}"]), flush=True)
                if s == "1920x1080" and src is not None:
                    cv2.imwrite(os.path.join(a.out, f"crop_{src_name}_{kind}.png"), o[:ph + 2 * y0, W - pw - 3 * y0:W])
                del comp
        res[s] = r
    json.dump(res, open(os.path.join(a.out, "pip_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
