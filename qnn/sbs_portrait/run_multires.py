"""Several eye resolutions in one process (one segmenter / one set of dynamic-shape engines).

  python run_multires.py --sets ~/data/sbs/mr/frames --bg BG.jpg --out OUT [--sizes 640x480,...]
                         [--mode opt|ref] [--size-policy official|aligned|native] [--profiles exact]
                         [--dump] [--profile] [--interleave N]

Sequential (default): the sequences are processed one after the other, as a service receiving
videos of random resolution: per size the latency / memory stats as run_pipeline.py, plus the
switch cost (pipeline init for a new size: warm-up runs at the new shapes, ORB / SGBM setup)
and the TensorRT profile that served it. --interleave N: N sequences of different sizes are
processed round-robin frame by frame (every frame changes the engine input shapes), measuring
the per-frame cost of shape changes. Writes OUT/<size>/stats.json (+ dumps / key frames) and
OUT/summary.json.
"""
import argparse
import json
import os
import resource
import time
from glob import glob

import cv2
import numpy as np
import torch

from pipeline import SBSPortraitPipeline, Timer, split_sbs
from run_pipeline import MemMonitor, git_rev, pip_source, run_sequence
from seg import DEFAULT_CONFIG, DEFAULT_WEIGHTS, PersonSegmenter


def make_seg(mode, size_policy, profiles, trt_arena=True, fp16_features=True, person_kernel="cuda"):
    if mode == "opt":
        return PersonSegmenter(DEFAULT_CONFIG, DEFAULT_WEIGHTS, precision="fp16", backbone="trt", fast_decoder=True,
                               pixel_decoder="trt", decoder="trt", profiles=profiles, size_policy=size_policy,
                               trt_arena=trt_arena, fp16_features=fp16_features, person_kernel=person_kernel)
    return PersonSegmenter(DEFAULT_CONFIG, DEFAULT_WEIGHTS, precision="fp32", fused=False, size_policy=size_policy)


def engine_info(seg, W, H):
    nh, nw = seg.input_hw(H, W)
    d = seg.model.size_divisibility
    hp, wp = -(-nh // d) * d, -(-nw // d) * d
    info = {"model_input_hw": [nh, nw], "padded_hw": [hp, wp]}
    if seg.trt is not None:
        p = seg.trt.profile_for(((1, 3, hp, wp),))
        info["profile"] = p
        info["profile_range"] = [list(map(list, (r[0][2:], r[1][2:], r[2][2:]))) for r in [seg.trt.ranges[p]["image"]]][0]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="directory with one <W>x<H> frame directory per size")
    ap.add_argument("--sizes", default="", help="comma list (default: sizes.json order)")
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--mode", default="opt", choices=("opt", "ref"))
    ap.add_argument("--size-policy", default="official", choices=("official", "native", "aligned"))
    ap.add_argument("--pip", default="", help="picture-in-picture: frame | icon | <SBS RGBA png>")
    ap.add_argument("--compose-kernel", default="triton", help="pip_compose.KINDS")
    ap.add_argument("--trt-arena", type=int, default=1, help="1: TRTChain arena, 0: per-engine buffers + pool")
    ap.add_argument("--fp16-features", type=int, default=1, help="FP16 backbone -> pixel decoder tensors")
    ap.add_argument("--profiles", default="exact")
    ap.add_argument("--profile", action="store_true", help="sync per GPU module (module breakdown)")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--key-frames", default="0,75,149")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--interleave", type=int, default=0)
    ap.add_argument("--person-kernel", default="cuda", choices=("cuda", "triton"))
    ap.add_argument("--post-kernels", default="triton", choices=("triton", "torch"))
    a = ap.parse_args()
    sizes = a.sizes.split(",") if a.sizes else json.load(open(os.path.join(a.sets, "sizes.json")))
    keys = {int(k) for k in a.key_frames.split(",") if k}
    bg = cv2.imread(a.bg)
    os.makedirs(a.out, exist_ok=True)

    mon = MemMonitor()
    mon.start()
    t0 = time.perf_counter()
    seg = make_seg(a.mode, a.size_policy, a.profiles, bool(a.trt_arena), bool(a.fp16_features), a.person_kernel)
    seg_init_s = time.perf_counter() - t0
    gpu_peak = rss_peak = 0
    summary = {"commit": git_rev(), "mode": a.mode, "size_policy": a.size_policy, "profiles": a.profiles,
               "person_kernel": a.person_kernel, "post_kernels": a.post_kernels,
               "seg_init_s": seg_init_s, "sizes": {}}
    files = {s: sorted(glob(os.path.join(a.sets, s, "*.jpg")))[: a.max_frames] for s in sizes}

    if a.interleave:
        summary["interleave"] = interleave(seg, sizes[: a.interleave], files, bg, a, mon)
    else:
        for s in sizes:
            timer = Timer(sync=a.profile)
            mon.reset()
            t1 = time.perf_counter()
            pipe = SBSPortraitPipeline(cv2.imread(files[s][0]), bg, mode=a.mode, seg=seg, timer=timer,
                                       pip=pip_source(a.pip), compose_kernel=a.compose_kernel,
                                       post_kernels=a.post_kernels)
            init_s = time.perf_counter() - t1
            st = run_sequence(pipe, files[s], os.path.join(a.out, s), timer, mon, a.dump, keys, a.skip)
            pipe.close()
            del pipe
            torch.cuda.empty_cache()  # per-size GPU peaks without the previous size's cached blocks
            st.update({"init_s": init_s, **engine_info(seg, *st["eye_wh"])})
            json.dump(st, open(os.path.join(a.out, s, "stats.json"), "w"), indent=1)
            gpu_peak, rss_peak = max(gpu_peak, st["gpu_mem_peak_mb_nvml"]), max(rss_peak, st["rss_peak_mb"])
            summary["sizes"][s] = {k: st[k] for k in ("e2e_ms", "fps", "init_s", "work_wh", "model_input_hw", "padded_hw",
                                                      "gpu_mem_peak_mb_nvml", "rss_peak_mb") if k in st}
            summary["sizes"][s]["profile"] = st.get("profile")
            summary["sizes"][s]["modules_ms"] = {k: round(v, 2) for k, v in list(st["modules_ms"].items())[:8]}
            print(s, json.dumps(summary["sizes"][s]), flush=True)
    mon.stop()
    summary.update({"gpu_mem_peak_mb_nvml": max(gpu_peak, mon.gpu_peak / 2**20),
                    "rss_peak_mb": max(rss_peak, mon.rss_peak / 2**20),
                    "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    if seg.trt is not None:
        summary["shape_switches"] = {"backbone": seg.trt.switches, "pixel_decoder": seg.trt_pd.switches,
                                     "decoder": seg.trt_td.switches}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "sizes"}, indent=1))


def interleave(seg, sizes, files, bg, a, mon):
    """Round-robin over the sequences, one frame each (each frame changes the input shapes)."""
    timers = {s: Timer(sync=a.profile) for s in sizes}
    pipes = {s: SBSPortraitPipeline(cv2.imread(files[s][0]), bg, mode=a.mode, seg=seg, timer=timers[s],
                                    post_kernels=a.post_kernels) for s in sizes}
    n = min(len(files[s]) for s in sizes)
    ms = {s: [] for s in sizes}
    mon.reset()
    for t in range(n):
        for s in sizes:
            frame = cv2.imread(files[s][t])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipes[s].process(frame)
            torch.cuda.synchronize()
            ms[s].append((time.perf_counter() - t0) * 1000.0)
    for p in pipes.values():
        p.close()
    return {"sizes": sizes, "frames": n, "gpu_mem_peak_mb_nvml": mon.gpu_peak / 2**20,
            "e2e_ms_median": {s: float(np.median(v[a.skip:])) for s, v in ms.items()}}


if __name__ == "__main__":
    main()
