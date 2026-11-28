"""Model-input resize + normalisation: Triton (area_triton.py) vs CUDA global / texture (area_cuda.py).

  python prof/area_spike.py [--out results/mr/area]

Per test size (official size policy, real frame 75, both eyes): call latency (CUDA events around one
call, median over interleaved rounds), GPU kernel time per call by kernel (torch profiler) and the
error against cv2.resize + normalisation (PersonSegmenter.preprocess), in gray levels.
"""
import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
from detectron2.data import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from area_cuda import KINDS, AreaResize  # noqa: E402
from area_triton import area_resize_normalize  # noqa: E402
from seg import _area_taps, _linear_area_taps  # noqa: E402

SIZES = ["640x480", "676x504", "854x480", "960x540", "1280x720", "1252x862", "1396x842", "1440x1080", "1680x966", "1920x1080"]
MEAN, STD = [123.675, 116.280, 103.530], [58.395, 57.120, 57.375]  # Swin-L config, RGB order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=os.path.expanduser("~/data/sbs/mr/frames"))
    ap.add_argument("--out", default="results/mr/area")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    res = {}
    for s in SIZES:
        W, H = map(int, s.split("x"))
        frame = cv2.imread(sorted(glob(os.path.join(a.sets, s, "*.jpg")))[75])
        ft = torch.from_numpy(frame).cuda()
        nh, nw = T.ResizeShortestEdge.get_output_shape(H, W, 800, 1333)
        taps = _area_taps if nh <= H and nw <= W else _linear_area_taps
        rx, ry = taps(W, nw), taps(H, nh)
        k = max(len(t[0]) for t in rx + ry)
        pad = lambda tp: (torch.tensor([i + [i[-1]] * (k - len(i)) for i, _ in tp], dtype=torch.int32, device="cuda"),  # noqa: E731
                          torch.tensor([w + [0.0] * (k - len(w)) for _, w in tp], dtype=torch.float32, device="cuda"))
        tx, ty = pad(rx), pad(ry)
        fns = {"triton": lambda e: area_resize_normalize(e, tx, ty, mean_t, std_t, rgb=True)}
        for kind in KINDS:
            fns[kind] = AreaResize(kind, H, W, rx, ry, MEAN, STD, rgb=True)
        eyes = [ft[:, :W], ft[:, W:]]
        refs = []
        for e in range(2):
            x = frame[:, e * W:(e + 1) * W]
            if (nh, nw) != (H, W):
                x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
            refs.append(x[:, :, ::-1].transpose(2, 0, 1).astype(np.int16))
        r = {"out_hw": [nh, nw], "mode": "area" if taps is _area_taps else "linear", "taps": k}
        gray = {}
        for name, fn in fns.items():
            ds, g = [], []
            for e in range(2):
                o = (fn(eyes[e]) * std_t[:, None, None] + mean_t[:, None, None]).round().cpu().numpy().astype(np.int16)
                g.append(o)
                ds.append(np.abs(o - refs[e]))
            d = np.stack(ds)
            gray[name] = np.stack(g)
            r[name] = {"max_abs": int(d.max()), "frac_diff": float((d > 0).mean()), "mean_abs": float(d.mean()),
                       "max_abs_vs_triton": int(np.abs(gray[name] - gray["triton"]).max())}
        times = {name: [] for name in fns}
        for _ in range(a.rounds):
            for name, fn in fns.items():
                for _ in range(5):
                    fn(eyes[0])
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
                for i, (e0, e1) in enumerate(ev):
                    e0.record()
                    fn(eyes[i & 1])
                    e1.record()
                torch.cuda.synchronize()
                times[name] += [e0.elapsed_time(e1) for e0, e1 in ev]
        from torch.profiler import ProfilerActivity, profile
        for name, fn in fns.items():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(40):
                    fn(eyes[i & 1])
                torch.cuda.synchronize()
            kernels = {ev.key.split("<")[0].replace("void ", ""): ev.self_device_time_total / 40 / 1000.0
                       for ev in prof.key_averages() if ev.self_device_time_total > 0}
            r[name].update(ms=float(np.median(times[name])), kernel_ms=sum(kernels.values()), kernels=kernels)
            print(s, name, json.dumps(r[name]), flush=True)
        res[s] = r
    json.dump(res, open(os.path.join(a.out, "area_spike.json"), "w"), indent=1)
    print("\n| 尺寸 | 模型输入 | tap | " + " | ".join(fns) + " |")
    print("|" + " --- |" * (3 + len(fns)))
    for s, r in res.items():
        cells = [f"{r[n]['kernel_ms']:.3f} / {r[n]['ms']:.3f} / {r[n]['max_abs']}" for n in fns]
        print(f"| {s} | {r['out_hw'][1]}x{r['out_hw'][0]} | {r['taps']} {r['mode']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
