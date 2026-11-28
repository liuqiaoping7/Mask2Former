"""Remaining sampling operators of the GPU path, cuda -> texture -> triton: warp and box filter
(warp_box.py, against the torch ops of gpu_post.py) and the fused person kernel (person_cuda.py,
against person_triton.py).

  python prof/ops_spike.py [--out results/mr/ops]

Real inputs per test size: frames 74 / 75 give DIS flows, SGBM disparity and grays on the work grid;
alphas from a pipeline dump; mask logits and class weights captured from the production segmenter on
both eyes of frame 75. Per implementation: GPU time per call (torch profiler: kernels + copies), call
latency (CUDA events, median over interleaved rounds) and the difference from the production
implementation (torch for warp / box, Triton for person).
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
import person_cuda  # noqa: E402
import person_triton  # noqa: E402
import temporal_stereo as ts  # noqa: E402
import warp_box as wb  # noqa: E402
from gpu_post import GPUPost  # noqa: E402
from pipeline import DEFAULT_CONFIG, DEFAULT_WEIGHTS, down, split_sbs, work_params, work_size  # noqa: E402
from seg import PersonSegmenter, _panoptic_person_rule  # noqa: E402

SIZES = ["640x480", "676x504", "854x480", "960x540", "1280x720", "1252x862", "1396x842", "1440x1080", "1680x966", "1920x1080"]


def timing(fn, rounds, iters):
    """{name: callable} -> {name: (gpu_ms, call_ms)}"""
    from torch.profiler import ProfilerActivity, profile
    times = {n: [] for n in fn}
    for _ in range(rounds):
        for n, f in fn.items():
            for _ in range(5):
                f()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for e0, e1 in ev:
                e0.record()
                f()
                e1.record()
            torch.cuda.synchronize()
            times[n] += [e0.elapsed_time(e1) for e0, e1 in ev]
    res = {}
    for n, f in fn.items():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(40):
                f()
            torch.cuda.synchronize()
        res[n] = (sum(e.self_device_time_total for e in prof.key_averages()) / 40 / 1000.0, float(np.median(times[n])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=os.path.expanduser("~/data/sbs/mr/frames"))
    ap.add_argument("--dump-run", default=os.path.expanduser("~/data/sbs/mr/opt_final/run"))
    ap.add_argument("--out", default="results/mr/ops")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seg = PersonSegmenter(DEFAULT_CONFIG, DEFAULT_WEIGHTS, precision="fp16", backbone="trt", fast_decoder=True,
                          pixel_decoder="trt", decoder="trt", profiles="exact", size_policy="official",
                          msda_plugin="builtin", trt_arena=True, fp16_features=True)
    captured = []
    orig = seg._person_fused

    def capture(masks, cls, hw, padded_hw, out_hw):
        captured.append((masks.clone(), cls.clone(), hw, padded_hw, out_hw))
        return orig(masks, cls, hw, padded_hw, out_hw)

    seg._person_fused = capture
    res = {}
    for s in SIZES:
        W, H = map(int, s.split("x"))
        files = sorted(glob(os.path.join(a.sets, s, "*.jpg")))
        ws, _ = work_size(W, H)
        r_gf = work_params(ws)["gf_r"]
        eyes = [split_sbs(cv2.imread(files[k])) for k in (74, 75)]
        g = [[cv2.cvtColor(down(eye, ws), cv2.COLOR_BGR2GRAY) for eye in e] for e in eyes]
        flow = ts.FlowEstimator()
        fb, ff = flow(g[1][0], g[0][0]), flow(g[0][0], g[1][0])
        mind, numd, _, _ = ts.estimate_disparity_range(g[1][0], g[1][1])
        d = ts.StereoEstimator(mind, numd)(g[1][0], g[1][1])[0]
        d = np.nan_to_num(d, nan=float(np.nanmedian(d))).astype(np.float32)
        z = np.load(os.path.join(a.dump_run, s, "dump", "0075.npz"))
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().cuda()  # noqa: E731
        fbt, fft, dt = t(fb), t(ff).permute(2, 0, 1).contiguous(), t(d)
        gray, alpha = t(g[0][0])[None], t(z["aL"])[None]
        assert alpha.shape[-2:] == (ws[1], ws[0]) and gray.shape[-2:] == alpha.shape[-2:]
        gp = GPUPost(ws, np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8), gf=(r_gf, 1e-4))
        guide = t(g[1][0]) / 255.0
        sa = gp.soft_alpha(alpha[0])
        x4 = torch.stack([guide, sa, guide * sa, guide * guide])
        r = {"work_wh": list(ws), "gf_r": r_gf}

        cases = {
            "warp_flow_c2": (lambda k: wb.warp(k, fft, fbt[..., 0], fbt[..., 1]), lambda: gp.warp_flow(fft, fbt)),
            "warp_flow_c1": (lambda k: wb.warp(k, gray, fbt[..., 0], fbt[..., 1]), lambda: gp.warp_flow(gray, fbt)),
            "warp_h_c1": (lambda k: wb.warp(k, alpha, dt), lambda: gp.warp_h(alpha, dt)),
            "box_n4": (lambda k: wb.box(k, x4, r_gf), lambda: gp.box(x4)),
            "box_n2": (lambda k: wb.box(k, x4[:2], r_gf), lambda: gp.box(x4[:2])),
        }
        for name, (hand, ref_fn) in cases.items():
            fns = {"torch": ref_fn, **{k: (lambda k=k, h=hand: h(k)) for k in wb.KINDS}}
            ref = ref_fn()
            err = {n: (f() - ref).abs() for n, f in fns.items()}
            tm = timing(fns, a.rounds, a.iters)
            r[name] = {n: {"gpu_ms": tm[n][0], "call_ms": tm[n][1], "max_abs": float(err[n].max()),
                           "mean_abs": float(err[n].mean()), "value_range": float(ref.max() - ref.min())} for n in fns}
            print(s, name, {n: (round(v["gpu_ms"], 4), float(f"{v['max_abs']:.2g}")) for n, v in r[name].items()}, flush=True)

        captured.clear()
        for eye in eyes[1]:
            seg.person([eye], ws, return_numpy=False)
        torch.cuda.synchronize()
        m = seg.model
        args = []
        for masks, cls, hw, padded, out_hw in captured:
            num_classes = cls.shape[-1] - 1
            scores, labels = cls.max(-1)
            keep = labels.ne(num_classes) & (scores > m.object_mask_threshold)
            h4, w4 = masks.shape[-2:]
            call = (masks.contiguous(), cls[:, 0].contiguous(), (1.0 - cls[:, -1]).contiguous(), (scores * keep).contiguous(),
                    out_hw, (hw[0] / out_hw[0]) * (h4 / padded[0]), (hw[1] / out_hw[1]) * (w4 / padded[1]))
            args.append((call, scores, keep, labels))
        assert len(args) == 2
        fns = {k: (lambda k=k: person_cuda.fused_person(k, *args[0][0])) for k in person_cuda.KINDS}
        fns["triton"] = lambda: person_triton.fused_person(*args[0][0])
        diff = {n: {"prob_max_abs": 0.0, "ids_diff": 0, "own_at_diff": 0, "own_area_max_diff": 0.0, "hard_diff": 0, "hard_px": 0}
                for n in fns}
        for call, scores, keep, labels in args:
            outs = {"triton": person_triton.fused_person(*call),
                    **{k: person_cuda.fused_person(k, *call) for k in person_cuda.KINDS}}
            hard = {n: _panoptic_person_rule(m, scores, keep, labels, o[1], o[2], o[3]) for n, o in outs.items()}
            p0, i0, o0, c0 = outs["triton"]
            for n, (p, i, o, c) in outs.items():
                v = diff[n]
                v["prob_max_abs"] = max(v["prob_max_abs"], float((p - p0).abs().max()))
                v["ids_diff"] += int((i != i0).sum())
                v["own_at_diff"] += int((o != o0).sum())
                v["own_area_max_diff"] = max(v["own_area_max_diff"], float((c - c0).abs().max()))
                v["hard_diff"] += int((hard[n] != hard["triton"]).sum())
                v["hard_px"] += int(hard[n].sum())
        tm = timing(fns, a.rounds, max(10, a.iters // 4))
        r["person"] = {n: {"gpu_ms": tm[n][0], "call_ms": tm[n][1], **diff[n]} for n in fns}
        r["person_masks"] = {"shape": list(args[0][0][0].shape), "dtype": str(args[0][0][0].dtype)}
        print(s, "person", r["person_masks"], json.dumps(r["person"]), flush=True)
        res[s] = r
    json.dump(res, open(os.path.join(a.out, "ops_spike.json"), "w"), indent=1)

    for op in ("warp_flow_c2", "warp_flow_c1", "warp_h_c1", "box_n4", "box_n2", "person"):
        cols = [c for c in ("torch", "cuda", "tex_point", "tex_linear", "triton") if c in res[SIZES[0]][op]]
        print(f"\n{op} (GPU ms)\n| size | " + " | ".join(cols) + " |")
        for s, r in res.items():
            print(f"| {s} | " + " | ".join(f"{r[op][c]['gpu_ms']:.4f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
