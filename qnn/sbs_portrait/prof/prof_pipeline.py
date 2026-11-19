"""Per-module profile of sbs_portrait_bench.py (Swin-L, per-eye, 1080p SBS).

Part A: wall time per module over N frames, in pipeline order (sync around GPU modules).
Part B: torch profiler on one eye: CUDA time / FLOPs / kernel classes per model stage.
Env: SBS_DATA (default ~/data/sbs), PROF_N (frames, default 30),
PROF_NOSYNC=1 (no extra syncs, for nsys), PROF_SKIP_B=1 (skip part B).
"""
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BENCH)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import seg as segmod  # noqa: E402
import temporal_stereo as ts  # noqa: E402
from sbs_portrait_bench import compose, down, nanpct, split_sbs  # noqa: E402
import mask2former.maskformer_model as mfm  # noqa: E402

DATA = os.path.expanduser(os.environ.get("SBS_DATA", "~/data/sbs"))
FRAMES = sorted(glob(os.path.join(DATA, "portrait_0121/*.jpg")))
BG = os.path.join(DATA, "bg_0045/0001.jpg")
N_WARM, N = 3, int(os.environ.get("PROF_N", 30))
NOSYNC = os.environ.get("PROF_NOSYNC") == "1"  # nsys mode: no extra syncs, NVTX ranges only

T = defaultdict(list)


def sync():
    torch.cuda.synchronize()


class tm:
    """Timer context; gpu=True synchronizes before and after."""

    def __init__(self, key, gpu=False):
        self.key, self.gpu = key, gpu

    def __enter__(self):
        if self.gpu and not NOSYNC:
            sync()
        torch.cuda.nvtx.range_push(self.key)
        self.t0 = time.perf_counter()

    def __exit__(self, *a):
        if self.gpu and not NOSYNC:
            sync()
        torch.cuda.nvtx.range_pop()
        T[self.key].append((time.perf_counter() - self.t0) * 1000)


def wrap(obj, name, key):
    f = getattr(obj, name)

    def g(*a, **k):
        with tm(key, gpu=True):
            return f(*a, **k)
    setattr(obj, name, g)


seg = segmod.PersonSegmenter()
model = seg.model
# instrument model stages (each eye is one call -> per-eye times are appended twice per frame)
wrap(model.backbone, "forward", "gpu.backbone")
wrap(model.sem_seg_head.pixel_decoder, "forward_features", "gpu.pixel_decoder(msda)")
wrap(model.sem_seg_head.predictor, "forward", "gpu.transformer_decoder")
_ssp = mfm.sem_seg_postprocess


class _FProxy:
    """Times F.interpolate calls made inside maskformer_model only (mask upsample to input size)."""

    def __getattr__(self, name):
        return getattr(F, name)

    def interpolate(self, *a, **k):
        with tm("gpu.post.mask_upsample_to_input", gpu=True):
            return F.interpolate(*a, **k)


mfm.F = _FProxy()


def ssp(*a, **k):
    with tm("gpu.post.upsample_crop_to_1080p", gpu=True):
        return _ssp(*a, **k)


mfm.sem_seg_postprocess = ssp
wrap(model, "semantic_inference", "gpu.post.semantic_inference")
wrap(model, "panoptic_inference", "gpu.post.panoptic_inference")


def seg_eye(img):
    """Same as PersonSegmenter.__call__ for one eye, with CPU/GPU sub-stages timed."""
    with tm("cpu.seg_preprocess(resize+chw)"):
        x = img[:, :, ::-1]
        h, w = x.shape[:2]
        x = seg.aug.get_transform(x).apply_image(x)
        inp = {"image": torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1))), "height": h, "width": w}
    with tm("gpu.model_total", gpu=True):
        with torch.no_grad():
            o = model([inp])[0]
    with tm("gpu.person_prob+isin", gpu=True):
        sem = o["sem_seg"].float()
        prob = sem[0] / sem.sum(0).clamp_min(1e-6)
        pan, info = o["panoptic_seg"]
        ids = [s["id"] for s in info if s.get("isthing") and s["category_id"] == 0]
        hard = torch.isin(pan, torch.tensor(ids, device=pan.device)) if ids else torch.zeros_like(pan, dtype=torch.bool)
    with tm("d2h.prob+mask(.cpu)", gpu=True):
        return prob.cpu().numpy().astype(np.float32), hard.cpu().numpy()


def main_part_a():
    f0 = cv2.imread(FRAMES[0])
    L0, R0 = split_sbs(f0)
    H, W = L0.shape[:2]
    ws = (W // 2, H // 2)
    for _ in range(2):
        seg_eye(L0)
    gL0, gR0 = [cv2.cvtColor(down(x, ws), cv2.COLOR_BGR2GRAY) for x in (L0, R0)]
    min_d, num_d, _, _ = ts.estimate_disparity_range(gL0, gR0)
    stereo = ts.StereoEstimator(min_d, num_d)
    flow = ts.FlowEstimator()
    bgL, bgR = split_sbs(cv2.imread(BG))
    bgL, bgR = down(bgL, (W, H), cv2.INTER_LINEAR), down(bgR, (W, H), cv2.INTER_LINEAR)
    bgR = ts.warp_h(bgR, np.full((H, W), -24.125, np.float32))
    bgL_w = down(bgL, ws)
    CONFIGS = ["raw", "soft", "soft+T", "soft+S", "soft+T+S", "final"]
    acc = {c: ts.MetricAccumulator() for c in CONFIGS}
    ema = {k: ts.TemporalEMA(0.7, gate=0.5) for k in ("T_L", "T_R", "G_L", "G_R")}
    prev_alpha = {c: None for c in CONFIGS}
    prev_comp = {c: None for c in CONFIGS}
    flicker = {c: np.zeros(ws[::-1], np.float32) for c in ("raw", "final")}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {c: cv2.VideoWriter(f"/tmp/prof_{c}.mp4", fourcc, 30, (W, H // 2)) for c in ("raw", "final")}
    prev_g = None
    lpd = 45.0
    files = FRAMES[: N_WARM + N]
    wall = []
    for t, f in enumerate(files):
        if t == N_WARM:
            T.clear()
        t_frame = time.perf_counter()
        torch.cuda.nvtx.range_push(f"frame{t}")
        with tm("cpu.imread_jpeg_decode"):
            frame = cv2.imread(f)
        L, R = split_sbs(frame)
        pL, hL = seg_eye(L)
        pR, hR = seg_eye(R)
        with tm("cpu.downscale_inputs+gray"):
            Ls, Rs = down(L, ws), down(R, ws)
            gL, gR = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
            guideL, guideR = gL.astype(np.float32) / 255.0, gR.astype(np.float32) / 255.0
        with tm("cpu.downscale_masks_1080p->540p"):
            pL, pR = down(pL, ws), down(pR, ws)
            hL = down(hL.astype(np.uint8), ws, cv2.INTER_NEAREST).astype(bool)
            hR = down(hR.astype(np.uint8), ws, cv2.INTER_NEAREST).astype(bool)
        with tm("cpu.sgbm_disparity_LR+check"):
            dL, dR = stereo(gL, gR)
        with tm("cpu.dis_flow_LR(fw+bw)+validity"):
            if prev_g is not None:
                fbL, vL, phL = flow.backward_with_validity(prev_g[0], gL)
                fbR, vR, phR = flow.backward_with_validity(prev_g[1], gR)
            else:
                fbL = vL = phL = fbR = vR = phR = None
        prev_g = (gL, gR)
        with tm("cpu.disparity_fill(morph+pct)"):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            bandL = cv2.dilate((hL | (pL > 0.3)).astype(np.uint8), k).astype(bool)
            bandR = cv2.dilate((hR | (pR > 0.3)).astype(np.uint8), k).astype(bool)
            coreL = cv2.erode(hL.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            lpd = nanpct(dL[coreL], 50, lpd)
            d_def = nanpct(dL, 50, 0.0)
            dLf = ts.fill_disparity(dL, bandL, lpd, d_def)
            dRf = ts.fill_disparity(dR, bandR, lpd, d_def)
        alphas = {}
        # ---- production path: final = soft -> G -> T -> S ----
        with tm("cpu.final.soft_alpha"):
            sL, sR = ts.soft_alpha(pL, 0.4, 0.6), ts.soft_alpha(pR, 0.4, 0.6)
        with tm("cpu.final.guided_filter_LR"):
            gfL, gfR = ts.guided_filter(guideL, sL, 4, 1e-4), ts.guided_filter(guideR, sR, 4, 1e-4)
        with tm("cpu.final.temporal_ema_LR"):
            gaL = ema["G_L"](gfL, fbL, vL, phL)
            gaR = ema["G_R"](gfR, fbR, vR, phR)
        with tm("cpu.final.stereo_fusion_LR"):
            alphas["final"] = (0.5 * (gaL + ts.warp_r2l(gaR, dLf)), 0.5 * (gaR + ts.warp_l2r(gaL, dRf)))
        with tm("cpu.final.compose_1080p_LR+encode"):
            aL = cv2.resize(alphas["final"][0], (W, H), interpolation=cv2.INTER_LINEAR)
            aR = cv2.resize(alphas["final"][1], (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.hstack([compose(L, bgL, aL), compose(R, bgR, aR)]).clip(0, 255).astype(np.uint8)
            writers["final"].write(cv2.resize(out, (W, H // 2), interpolation=cv2.INTER_AREA))
        # ---- benchmark-only: other configs, metrics, raw video ----
        with tm("bench.other_configs_alpha"):
            alphas["raw"] = (hL.astype(np.float32), hR.astype(np.float32))
            alphas["soft"] = (sL, sR)
            tL = ema["T_L"](sL, fbL, vL, phL)
            tR = ema["T_R"](sR, fbR, vR, phR)
            alphas["soft+T"] = (tL, tR)
            alphas["soft+S"] = (0.5 * (sL + ts.warp_r2l(sR, dLf)), 0.5 * (sR + ts.warp_l2r(sL, dRf)))
            alphas["soft+T+S"] = (0.5 * (tL + ts.warp_r2l(tR, dLf)), 0.5 * (tR + ts.warp_l2r(tL, dRf)))
        with tm("bench.metrics_6configs"):
            bandL_m, bandR_m = ts.boundary_band(hL, 10), ts.boundary_band(hR, 10)
            for c in CONFIGS:
                aL_, aR_ = alphas[c]
                a = acc[c]
                cL = compose(Ls, bgL_w, aL_)
                if prev_alpha[c] is not None:
                    a.temporal(aL_, prev_alpha[c][0], fbL, vL, bandL_m)
                    a.temporal(aR_, prev_alpha[c][1], fbR, vR, bandR_m)
                    a.composite_warp(cL, prev_comp[c], fbL, vL)
                    if c in flicker:
                        flicker[c] += np.abs(aL_ - ts.warp_flow(prev_alpha[c][0], fbL)) * vL
                a.stereo(aL_, aR_, dL, bandL_m)
                a.fidelity(aL_, hL)
                a.ghost(aL_, pL)
                prev_alpha[c] = (aL_, aR_)
                prev_comp[c] = cL
        with tm("bench.raw_compose+encode"):
            aL = cv2.resize(alphas["raw"][0], (W, H), interpolation=cv2.INTER_LINEAR)
            aR = cv2.resize(alphas["raw"][1], (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.hstack([compose(L, bgL, aL), compose(R, bgR, aR)]).clip(0, 255).astype(np.uint8)
            writers["raw"].write(cv2.resize(out, (W, H // 2), interpolation=cv2.INTER_AREA))
        torch.cuda.nvtx.range_pop()
        if t >= N_WARM:
            wall.append((time.perf_counter() - t_frame) * 1000)
    res = {k: {"mean_ms_per_frame": float(np.sum(v) / N), "calls": len(v)} for k, v in T.items()}
    res["_wall_ms_per_frame"] = float(np.mean(wall))
    res["_wall_ms_std"] = float(np.std(wall))
    return res


def main_part_b():
    """Kernel-level: CUDA time + FLOPs per model stage for one eye."""
    from torch.profiler import ProfilerActivity, profile, record_function
    # same model; record_function scopes stacked on top of the timing wrappers
    seg2, m = seg, model

    def scope(obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            with record_function(label):
                return f(*a, **k)
        setattr(obj, name, g)

    scope(m.backbone, "forward", "S1_backbone")
    scope(m.sem_seg_head.pixel_decoder, "forward_features", "S2_pixel_decoder")
    scope(m.sem_seg_head.predictor, "forward", "S3_transformer_decoder")
    _p = mfm.sem_seg_postprocess

    def p(*a, **k):
        with record_function("S5_upsample_crop_1080p"):
            return _p(*a, **k)
    mfm.sem_seg_postprocess = p
    _ip = mfm.F.interpolate

    class _FP2:
        def __getattr__(self, name):
            return getattr(F, name)

        def interpolate(self, *a, **k):
            with record_function("S4_mask_upsample_to_input"):
                return _ip(*a, **k)
    mfm.F = _FP2()
    scope(m, "semantic_inference", "S6_semantic_inference")
    scope(m, "panoptic_inference", "S7_panoptic_inference")
    img = cv2.imread(FRAMES[10])[:, :1920]
    for _ in range(3):
        seg2([img])
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True, record_shapes=True) as prof:
        seg2([img])
        sync()
    # assign each CUDA kernel to the innermost stage scope via the event tree
    stages = {}
    for ev in prof.events():
        if ev.name.startswith("S") and "_" in ev.name and ev.name[1].isdigit():
            stages.setdefault(ev.name, []).append(ev)
    out = {}
    for name, evs in stages.items():
        cuda_us, flops, kinds = 0.0, 0, defaultdict(float)
        for ev in evs:
            stack = [ev]
            while stack:
                e = stack.pop()
                flops += getattr(e, "flops", 0) or 0
                for k in e.kernels:
                    cuda_us += k.duration
                    kn = k.name.lower()
                    cls = ("gemm(tensor-core)" if any(s in kn for s in ("hmma", "xmma", "tensorop", "s1688", "s16816", "cutlass_80", "sm80", "sm89_xmma")) else
                           "gemm(fp32 simt)" if "gemm" in kn or "sgemm" in kn else
                           "conv(cudnn)" if "conv" in kn or "implicit" in kn or "winograd" in kn else
                           "msda(custom)" if "ms_deform" in kn or "im2col" in kn else
                           "softmax" if "softmax" in kn else
                           "reduce" if "reduce" in kn else
                           "upsample/interp" if "upsample" in kn or "interp" in kn else
                           "layernorm" if "layer_norm" in kn or "layernorm" in kn else
                           "elementwise/copy")
                    kinds[cls] += k.duration
                stack.extend(e.cpu_children)
        out[name] = {"cuda_ms": cuda_us / 1000, "gflop": flops / 1e9,
                     "kernel_ms_by_class": {k: round(v / 1000, 2) for k, v in sorted(kinds.items(), key=lambda x: -x[1])}}
    top = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70)
    return out, top


if __name__ == "__main__":
    a = main_part_a()
    print(json.dumps(a, indent=1, ensure_ascii=False))
    if os.environ.get("PROF_SKIP_B") == "1":
        sys.exit(0)
    b, top = main_part_b()
    print(json.dumps(b, indent=1, ensure_ascii=False))
    print(top)
