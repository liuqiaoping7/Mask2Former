"""SBS stereo portrait segmentation + SBS virtual background benchmark.

Input : a directory of full side-by-side frames (left | right, each eye 1080p)
        and one SBS virtual background image.
Output: SBS composites (video + key frames), per-config temporal / stereo
        consistency metrics and latency (metrics.json / metrics.md).

Configs (alpha generation, all evaluated on the same frames in one pass):
  raw        panoptic person mask per eye, per frame (hard 0/1) -- baseline
  soft       semantic person probability -> smoothstep alpha
  soft+T     + motion-compensated confidence-weighted temporal EMA (per eye)
"""
import argparse
import json
import os
import time
from glob import glob

import cv2
import numpy as np
import torch

import temporal_stereo as ts
from seg import DEFAULT_CONFIG, DEFAULT_WEIGHTS, PersonSegmenter

CONFIGS = ["raw", "soft", "soft+T"]
VIDEO_CONFIGS = ["raw"]
FLICKER_MAP_CONFIGS = ["raw"]
# timer keys that make up each config's post-processing latency
POST_KEYS = {"raw": ["post_raw"], "soft": ["post_soft"], "soft+T": ["post_soft", "post_soft+T"]}
# configs whose post-processing needs optical flow / disparity
USES_GEOMETRY = {"soft+T"}


def split_sbs(img):
    w = img.shape[1] // 2
    return img[:, :w], img[:, w:]


def down(x, size, interp=cv2.INTER_AREA):
    return cv2.resize(x, size, interpolation=interp)


def compose(img, bg, alpha):
    a = alpha[..., None]
    return img.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)


def nanpct(x, q, default):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if x.size else default


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t.setdefault(k, []).append(dt * 1000.0)

    def summary(self, skip=2):
        return {k: float(np.median(v[skip:] if len(v) > skip else v)) for k, v in self.t.items()}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir with SBS frames (*.jpg|*.png)")
    ap.add_argument("--bg", required=True, help="SBS virtual background image")
    ap.add_argument("--config-file", default=DEFAULT_CONFIG)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--work-scale", type=float, default=0.5, help="post-processing / metric resolution scale")
    ap.add_argument("--batch-eyes", action="store_true",
                    help="segment both eyes in one batch (default: one eye at a time, fits Swin-L in 8 GB)")
    ap.add_argument("--cache", default="", help="dir to cache per-frame segmentation (reuse across runs)")
    ap.add_argument("--soft", default="0.3,0.7", help="smoothstep range prob -> alpha")
    ap.add_argument("--lam", type=float, default=0.7, help="temporal EMA strength")
    ap.add_argument("--save-frames", default="0,75,149")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob(os.path.join(args.frames, "*.jpg")) + glob(os.path.join(args.frames, "*.png")))
    files = files[: args.max_frames]
    assert files, "no frames"
    f0 = cv2.imread(files[0])
    L0, R0 = split_sbs(f0)
    H, W = L0.shape[:2]
    ws = (int(W * args.work_scale), int(H * args.work_scale))
    save_idx = {int(i) for i in args.save_frames.split(",") if i}
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    # ---------------- model ----------------
    seg = PersonSegmenter(args.config_file, args.weights)
    if args.batch_eyes:
        segment = seg
    else:
        segment = lambda ims: [r for im in ims for r in seg([im])]
    for _ in range(2):  # warm-up
        segment([L0, R0])
    soft_lo, soft_hi = map(float, args.soft.split(","))

    # ---------------- stereo / flow setup ----------------
    gL0, gR0 = [cv2.cvtColor(down(x, ws), cv2.COLOR_BGR2GRAY) for x in (L0, R0)]
    min_d, num_d, dy, nmatch = ts.estimate_disparity_range(gL0, gR0)
    stereo = ts.StereoEstimator(min_d, num_d)
    flow = ts.FlowEstimator()

    bgL, bgR = split_sbs(cv2.imread(args.bg))
    bgL, bgR = down(bgL, (W, H), cv2.INTER_LINEAR), down(bgR, (W, H), cv2.INTER_LINEAR)
    bgL_w, bgR_w = down(bgL, ws), down(bgR, ws)
    stereo_info = {"pair_disp_range_work": [min_d, min_d + num_d], "vertical_offset_px_work": dy,
                   "orb_matches": nmatch}
    print("stereo:", json.dumps(stereo_info), flush=True)

    # ---------------- state ----------------
    acc = {c: ts.MetricAccumulator() for c in CONFIGS}
    ema = {k: ts.TemporalEMA(args.lam) for k in ("T_L", "T_R")}
    prev_alpha = {c: None for c in CONFIGS}
    prev_comp = {c: None for c in CONFIGS}
    flicker_map = {c: np.zeros(ws[::-1], np.float32) for c in FLICKER_MAP_CONFIGS}
    timer = Timer()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {c: cv2.VideoWriter(os.path.join(args.out, f"composite_{c}.mp4"), fourcc, 30, (W, H // 2))
               for c in VIDEO_CONFIGS}
    prev_g = None

    for t, f in enumerate(files):
        frame = f0 if t == 0 else cv2.imread(f)
        L, R = split_sbs(frame)

        # ---------------- segmentation (optionally cached) ----------------
        cf = os.path.join(args.cache, f"{t:04d}.npz") if args.cache else ""
        if cf and os.path.exists(cf):
            z = np.load(cf)
            pL, hL, pR, hR = z["pL"] / 255.0, z["hL"], z["pR"] / 255.0, z["hR"]
            pL, pR = pL.astype(np.float32), pR.astype(np.float32)
            timer.add("seg_LR", float(z["seg_ms"]) / 1000.0)
        else:
            sync(); t0 = time.perf_counter()
            (pL, hL), (pR, hR) = segment([L, R])
            sync(); dt = time.perf_counter() - t0
            timer.add("seg_LR", dt)
            if cf:
                q = lambda p: (down(p, ws) * 255 + 0.5).astype(np.uint8)
                hq = lambda h: down(h.astype(np.uint8), ws, cv2.INTER_NEAREST).astype(bool)
                np.savez_compressed(cf, pL=q(pL), hL=hq(hL), pR=q(pR), hR=hq(hR), seg_ms=dt * 1000.0)

        Ls, Rs = down(L, ws), down(R, ws)
        gL, gR = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        pL, pR = down(pL, ws), down(pR, ws)
        hL = down(hL.astype(np.uint8), ws, cv2.INTER_NEAREST).astype(bool)
        hR = down(hR.astype(np.uint8), ws, cv2.INTER_NEAREST).astype(bool)

        # ---------------- geometry: disparity + optical flow ----------------
        t0 = time.perf_counter()
        dL, dR = stereo(gL, gR)
        timer.add("disparity_LR", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if prev_g is not None:
            fbL, vL, phL = flow.backward_with_validity(prev_g[0], gL)
            fbR, vR, phR = flow.backward_with_validity(prev_g[1], gR)
        else:
            fbL = vL = phL = fbR = vR = phR = None
        timer.add("flow_LR", time.perf_counter() - t0)
        prev_g = (gL, gR)

        # ---------------- alpha per config ----------------
        alphas = {}
        t0 = time.perf_counter()
        alphas["raw"] = (hL.astype(np.float32), hR.astype(np.float32))
        timer.add("post_raw", time.perf_counter() - t0)

        t0 = time.perf_counter()
        sL, sR = ts.soft_alpha(pL, soft_lo, soft_hi), ts.soft_alpha(pR, soft_lo, soft_hi)
        alphas["soft"] = (sL, sR)
        timer.add("post_soft", time.perf_counter() - t0)

        t0 = time.perf_counter()
        tL = ema["T_L"](sL, fbL, vL, phL)
        tR = ema["T_R"](sR, fbR, vR, phR)
        alphas["soft+T"] = (tL, tR)
        timer.add("post_soft+T", time.perf_counter() - t0)

        # ---------------- metrics ----------------
        bandL_m, bandR_m = ts.boundary_band(hL, 10), ts.boundary_band(hR, 10)
        for c in CONFIGS:
            aL, aR = alphas[c]
            a = acc[c]
            cL = compose(Ls, bgL_w, aL)
            if prev_alpha[c] is not None:
                a.temporal(aL, prev_alpha[c][0], fbL, vL, bandL_m)
                a.temporal(aR, prev_alpha[c][1], fbR, vR, bandR_m)
                a.composite_warp(cL, prev_comp[c], fbL, vL)
                if c in flicker_map:
                    flicker_map[c] += np.abs(aL - ts.warp_flow(prev_alpha[c][0], fbL)) * vL
            a.stereo(aL, aR, dL, bandL_m)
            a.fidelity(aL, hL)
            a.ghost(aL, pL)
            prev_alpha[c] = (aL, aR)
            prev_comp[c] = cL

        # ---------------- outputs ----------------
        for c in VIDEO_CONFIGS:
            aL = cv2.resize(alphas[c][0], (W, H), interpolation=cv2.INTER_LINEAR)
            aR = cv2.resize(alphas[c][1], (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.hstack([compose(L, bgL, aL), compose(R, bgR, aR)]).clip(0, 255).astype(np.uint8)
            writers[c].write(cv2.resize(out, (W, H // 2), interpolation=cv2.INTER_AREA))
            if t in save_idx:
                cv2.imwrite(os.path.join(args.out, f"sbs_{c}_{t:04d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
                am = (np.hstack([alphas[c][0], alphas[c][1]]) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(args.out, f"alpha_{c}_{t:04d}.png"), am)
        if t in save_idx:
            cv2.imwrite(os.path.join(args.out, f"input_{t:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if t % 25 == 0:
            print(f"frame {t}/{len(files)}", flush=True)

    for w in writers.values():
        w.release()
    n = len(files) - 1
    for c, m in flicker_map.items():
        m = np.clip(m / max(n, 1) * 4 * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, f"flicker_heatmap_{c}.jpg"), cv2.applyColorMap(m, cv2.COLORMAP_INFERNO))

    lat = timer.summary()
    geometry_ms = lat["disparity_LR"] + lat["flow_LR"] + lat.get("disp_fill", 0.0)
    results = {
        "frames": len(files), "resolution_per_eye": [W, H], "work_resolution": list(ws),
        "model": os.path.basename(args.config_file), "batch_eyes": args.batch_eyes,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": {k: v for k, v in vars(args).items() if k not in ("frames", "bg", "out", "cache", "save_frames")},
        "stereo": stereo_info, "latency_ms": lat, "configs": {},
    }
    for c in CONFIGS:
        r = acc[c].summary()
        seg_ms = lat["seg_L"] if c == "master" else lat["seg_LR"]
        post_ms = sum(lat[k] for k in POST_KEYS[c]) + (geometry_ms if c in USES_GEOMETRY else 0.0)
        r["post_ms"] = post_ms
        r["latency_ms"] = seg_ms + post_ms
        results["configs"][c] = r
    with open(os.path.join(args.out, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=1)

    hdr = ("| config | tIoU ↑ | tIoU p5 ↑ | 边界时域抖动 ↓ | sIoU ↑ | 边界左右差 ↓ | 与 raw IoU | ghost ↓ | "
           "合成 warp 误差 ↓ | 后处理 ms | 总延迟 ms |\n|" + "---|" * 11)
    lines = [hdr]
    for c in CONFIGS:
        r = results["configs"][c]
        lines.append(f"| {c} | {r['tiou']:.4f} | {r['tiou_p5']:.4f} | {r['tflk_band']:.4f} | {r['siou']:.4f} | "
                     f"{r['sflk_band']:.4f} | {r['fid_iou']:.4f} | {r['ghost']:.4f} | {r['cwarp']:.2f} | "
                     f"{r['post_ms']:.1f} | {r['latency_ms']:.1f} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, "metrics.md"), "w") as fp:
        fp.write(md + "\n")
    print(md)
    print(json.dumps(lat, indent=1))


if __name__ == "__main__":
    main()
