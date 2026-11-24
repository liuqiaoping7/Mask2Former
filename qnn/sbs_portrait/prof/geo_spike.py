"""CPU geometry parallelism at small eye sizes (native policy: GPU segmentation gets faster than the
geometry worker). Per frame: SGBM disparities (L, flipped R, L/R check) and four DIS flows (fw/bw,
both eyes), run on the work grid of the given size:

  seq   one worker: disparity, then the four flows (current pipeline)
  par2  disparity || four flows
  par3  disparity || flows L || flows R
  par5  disparity || each flow on its own thread
Each thread has its own DIS / SGBM instances (not thread safe). Outputs are checked identical.

  python prof/geo_spike.py --frames ~/data/sbs/mr/frames/960x540 [--n 40]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import temporal_stereo as ts  # noqa: E402
from pipeline import down, split_sbs, work_size  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/mr")
    a = ap.parse_args()
    files = sorted(glob(os.path.join(a.frames, "*.jpg")))[: a.n + 1]
    f0 = cv2.imread(files[0])
    L0, R0 = split_sbs(f0)
    ws, _ = work_size(L0.shape[1], L0.shape[0])
    gray = []
    for f in files:
        L, R = split_sbs(cv2.imread(f))
        gray.append([cv2.cvtColor(down(x, ws), cv2.COLOR_BGR2GRAY) for x in (L, R)])
    md, nd, _, _ = ts.estimate_disparity_range(*gray[0])
    stereo = [ts.StereoEstimator(md, nd) for _ in range(2)]
    flows = [ts.FlowEstimator() for _ in range(4)]
    pool = ThreadPoolExecutor(5)

    def jobs(t):
        (gL, gR), (pL, pR) = gray[t], gray[t - 1]
        return [lambda: stereo[0](gL, gR),
                lambda: flows[0](gL, pL), lambda: flows[1](pL, gL), lambda: flows[2](gR, pR), lambda: flows[3](pR, gR)]

    def run(mode, t):
        j = jobs(t)
        if mode == "seq":
            return [f() for f in j]
        if mode == "par2":
            d = pool.submit(j[0])
            fl = pool.submit(lambda: [f() for f in j[1:]])
            return [d.result(), *fl.result()]
        if mode == "par3":
            d = pool.submit(j[0])
            fa = pool.submit(lambda: [j[1](), j[2]()])
            fb = pool.submit(lambda: [j[3](), j[4]()])
            return [d.result(), *fa.result(), *fb.result()]
        return [x.result() for x in [pool.submit(f) for f in j]]

    res, ref = {}, None
    for mode in ("seq", "par2", "par3", "par5", "seq"):
        ts_ = []
        outs = []
        for t in range(1, len(gray)):
            t0 = time.perf_counter()
            o = run(mode, t)
            ts_.append((time.perf_counter() - t0) * 1000)
            outs.append(o)
        same = None
        if ref is None:
            ref = outs
        else:
            same = all(np.array_equal(np.nan_to_num(x, nan=-9), np.nan_to_num(y, nan=-9))
                       for o, r in zip(outs, ref) for a_, b_ in zip(o, r)
                       for x, y in zip(a_ if isinstance(a_, tuple) else (a_,), b_ if isinstance(b_, tuple) else (b_,)))
        res[mode if mode not in res else mode + "_again"] = {"median_ms": float(np.median(ts_[3:])), "identical": same}
        print(mode, res, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"work_wh": ws, **res}, open(os.path.join(a.out, f"geo_spike_{os.path.basename(a.frames.rstrip('/'))}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
