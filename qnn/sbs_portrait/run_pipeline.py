"""Run the production pipeline on an SBS frame sequence and record latency / memory.

  python run_pipeline.py --frames DIR --bg BG.jpg --out OUT [--dump] [--profile]

--profile : synchronize CUDA around GPU modules for a per-module breakdown
            (end-to-end numbers of a --profile run include those syncs).
--dump    : save per-frame alphas / probs / masks (OUT/dump/*.npz) for eval_alpha.py.
Writes OUT/stats.json, OUT/key_XXXX.png (full-res SBS) and OUT/preview.mp4 (half-res).
"""
import argparse
import json
import os
import resource
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
import psutil
import torch

from pipeline import SBSPortraitPipeline, Timer


class MemMonitor(threading.Thread):
    """Samples this process's GPU memory (NVML) and RSS every `period` seconds."""

    def __init__(self, period=0.01):
        super().__init__(daemon=True)
        import pynvml
        pynvml.nvmlInit()
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.pid, self.proc = os.getpid(), psutil.Process()
        self.period, self.stop_ = period, threading.Event()
        self.gpu_peak = self.rss_peak = 0

    def gpu_used(self):
        for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
            if p.pid == self.pid:
                return p.usedGpuMemory or 0
        return 0

    def run(self):
        while not self.stop_.is_set():
            self.gpu_peak = max(self.gpu_peak, self.gpu_used())
            self.rss_peak = max(self.rss_peak, self.proc.memory_info().rss)
            time.sleep(self.period)

    def reset(self):
        self.gpu_peak = self.gpu_used()
        self.rss_peak = self.proc.memory_info().rss

    def stop(self):
        self.stop_.set()
        self.join()


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--key-frames", default="0,75,149")
    ap.add_argument("--skip", type=int, default=5, help="warm-up frames excluded from latency stats")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.dump:
        os.makedirs(os.path.join(args.out, "dump"), exist_ok=True)
    files = sorted(glob(os.path.join(args.frames, "*.jpg")))[: args.max_frames]
    keys = {int(k) for k in args.key_frames.split(",") if k}

    mon = MemMonitor()
    mon.start()
    t_init = time.perf_counter()
    timer = Timer(sync=args.profile)
    pipe = SBSPortraitPipeline(cv2.imread(files[0]), cv2.imread(args.bg), timer=timer)
    init_s = time.perf_counter() - t_init
    timer.clear()
    torch.cuda.reset_peak_memory_stats()
    mon.reset()
    W, H = pipe.W, pipe.H
    writer = cv2.VideoWriter(os.path.join(args.out, "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H // 2))

    # I/O off the critical path: decode of the next frame is prefetched, encoding is asynchronous
    io = ThreadPoolExecutor(1, thread_name_prefix="decode")
    enc = ThreadPoolExecutor(1, thread_name_prefix="encode")
    nxt = io.submit(cv2.imread, files[0])
    pending = None
    frame_ms = []
    for t, f in enumerate(files):
        if t:
            timer.new_frame()
        torch.cuda.nvtx.range_push(f"frame{t}")  # per-frame range for nsys (prof/nsys_analyze.py)
        t0 = time.perf_counter()
        with timer("read"):  # wait for the prefetched frame
            frame = nxt.result()
            if t + 1 < len(files):
                nxt = io.submit(cv2.imread, files[t + 1])
        r = pipe.process(frame)
        with timer("encode"):  # bounded queue: at most one encode in flight
            if pending is not None:
                pending.result()
            out = r["out"].copy()  # r["out"] is a reused pinned buffer
            pending = enc.submit(lambda o: writer.write(cv2.resize(o, (W, H // 2), interpolation=cv2.INTER_AREA)), out)
        torch.cuda.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.nvtx.range_pop()
        # outputs for correctness checks (not timed)
        if args.dump:
            r = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in r.items()}
            np.savez_compressed(os.path.join(args.out, "dump", f"{t:04d}.npz"),
                                aL=r["aL"].astype(np.float16), aR=r["aR"].astype(np.float16),
                                pL=(r["pL"] * 255 + 0.5).astype(np.uint8), pR=(r["pR"] * 255 + 0.5).astype(np.uint8),
                                hL=r["hL"], hR=r["hR"])
        if t in keys:
            cv2.imwrite(os.path.join(args.out, f"key_{t:04d}.png"), r["out"])
    if pending is not None:
        pending.result()
    writer.release()
    mon.stop()

    fm = np.array(frame_ms[args.skip:])
    mods = timer.summary(skip=args.skip)
    stats = {
        "commit": git_rev(), "frames": len(files), "profile_sync": args.profile,
        "e2e_ms": {"median": float(np.median(fm)), "mean": float(fm.mean()), "p95": float(np.percentile(fm, 95)),
                   "std": float(fm.std())},
        "fps": float(1000.0 / np.median(fm)),
        "modules_ms": dict(sorted(mods.items(), key=lambda x: -x[1])),
        "gpu_mem_peak_mb_nvml": mon.gpu_peak / 2**20,
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "rss_peak_mb": mon.rss_peak / 2**20,
        "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "init_s": init_s,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fp:
        json.dump(stats, fp, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
