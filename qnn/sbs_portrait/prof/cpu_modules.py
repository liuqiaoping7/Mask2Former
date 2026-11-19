"""CPU module micro-benchmarks for perf top-down (one module per process).

usage: perf stat -M TmaL1,TmaL2 --metric-only -x ';' -D -1 --control fifo:CTL,ACK -- \
         python cpu_modules.py <module> [seconds] [ctl_fifo ack_fifo]
Counting is enabled only around the timed loop.  Prints ms/iteration.
Inputs are real frames of SVD clip 0121 (same shapes as the benchmark).
"""
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import temporal_stereo as ts  # noqa: E402

DATA = os.path.expanduser(os.environ.get("SBS_DATA", "~/data/sbs"))
FR = os.path.join(DATA, "portrait_0121/%04d.jpg")
BG = os.path.join(DATA, "bg_0045/0001.jpg")
W, H, WS = 1920, 1080, (960, 540)


def down(x, size, interp=cv2.INTER_AREA):
    return cv2.resize(x, size, interpolation=interp)


def compose(img, bg, alpha):
    a = alpha[..., None]
    return img.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)


def setup():
    f1, f2 = cv2.imread(FR % 10), cv2.imread(FR % 11)
    L1, R1, L2, R2 = f1[:, :W], f1[:, W:], f2[:, :W], f2[:, W:]
    g = lambda x: cv2.cvtColor(down(x, WS), cv2.COLOR_BGR2GRAY)
    gL1, gR1, gL2, gR2 = g(L1), g(R1), g(L2), g(R2)
    yy, xx = np.mgrid[0:540, 0:960]
    p = np.clip(1.2 - np.hypot((xx - 560) / 220.0, (yy - 300) / 260.0), 0, 1).astype(np.float32)  # person-like prob
    h = p > 0.5
    return dict(f1=f1, L1=L1, R1=R1, gL1=gL1, gR1=gR1, gL2=gL2, gR2=gR2, p=p, h=h,
                bg=cv2.imread(BG)[:, :W].copy())


def build(name, d):
    flow = ts.FlowEstimator()
    stereo = ts.StereoEstimator(-8, 32)
    if name == "imread":
        return lambda: cv2.imread(FR % 10)
    if name == "pil_resize":  # detectron2 ResizeTransform(interp=BILINEAR) on one eye
        x = np.ascontiguousarray(d["L1"][:, :, ::-1])
        return lambda: np.asarray(Image.fromarray(x).resize((1333, 750), Image.BILINEAR))
    if name == "sgbm_LR":
        return lambda: stereo(d["gL1"], d["gR1"])
    if name == "dis_flow_x4":
        return lambda: [flow(d["gL2"], d["gL1"]), flow(d["gL1"], d["gL2"]),
                        flow(d["gR2"], d["gR1"]), flow(d["gR1"], d["gR2"])]
    if name == "flow_validity_x2":  # numpy part of backward_with_validity, both eyes
        fb, ff = flow(d["gL2"], d["gL1"]), flow(d["gL1"], d["gL2"])

        def f():
            for _ in range(2):
                ff_at = ts.warp_flow(ff, fb)
                s = fb + ff_at
                mag = (fb ** 2).sum(-1) + (ff_at ** 2).sum(-1)
                ok = (s ** 2).sum(-1) < 0.01 * mag + 0.5
                pw = ts.warp_flow(d["gL1"], fb)
                photo = np.abs(pw.astype(np.float32) - d["gL2"].astype(np.float32))
                _ = ok & (photo < 20)
        return f
    if name == "disp_fill":
        dl, dr = stereo(d["gL1"], d["gR1"])
        p, h = d["p"], d["h"]

        def f():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            bL = cv2.dilate((h | (p > 0.3)).astype(np.uint8), k).astype(bool)
            bR = cv2.dilate((h | (p > 0.3)).astype(np.uint8), k).astype(bool)
            core = cv2.erode(h.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            x = dl[core]; x = x[~np.isnan(x)]; pd = np.percentile(x, 50) if x.size else 0.0
            y = dl[~np.isnan(dl)]; dd = np.percentile(y, 50)
            ts.fill_disparity(dl, bL, pd, dd); ts.fill_disparity(dr, bR, pd, dd)
        return f
    if name == "guided_x2":
        gd = d["gL1"].astype(np.float32) / 255
        return lambda: [ts.guided_filter(gd, d["p"], 4, 1e-4) for _ in range(2)]
    if name == "ema_x2":
        fb = flow(d["gL2"], d["gL1"])
        valid = np.ones((540, 960), bool); photo = np.zeros((540, 960), np.float32)
        e = ts.TemporalEMA(0.7, gate=0.5); e(d["p"], None, None, None)
        return lambda: [e(d["p"], fb, valid, photo) for _ in range(2)]
    if name == "compose_blend_x2":
        A = cv2.resize(d["p"], (W, H))
        return lambda: (compose(d["L1"], d["bg"], A), compose(d["R1"], d["bg"], A))
    if name == "compose_clip_u8":
        A = cv2.resize(d["p"], (W, H)); cl, cr = compose(d["L1"], d["bg"], A), compose(d["R1"], d["bg"], A)
        return lambda: np.hstack([cl, cr]).clip(0, 255).astype(np.uint8)
    raise SystemExit(f"unknown module {name}")


def main():
    name = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    ctl = ack = None
    if len(sys.argv) > 4:
        ctl, ack = open(sys.argv[3], "w"), open(sys.argv[4], "r")
    d = setup()
    fn = build(name, d)
    for _ in range(3):
        fn()
    t = time.perf_counter(); n = 0
    while time.perf_counter() - t < 0.5:
        fn(); n += 1
    iters = max(3, int(secs / ((time.perf_counter() - t) / n)))
    if ctl:
        ctl.write("enable\n"); ctl.flush(); ack.readline()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t
    if ctl:
        ctl.write("disable\n"); ctl.flush(); ack.readline()
    print(f"MODULE {name} iters={iters} ms_per_iter={dt / iters * 1000:.2f} threads={cv2.getNumThreads()}", flush=True)


if __name__ == "__main__":
    main()
