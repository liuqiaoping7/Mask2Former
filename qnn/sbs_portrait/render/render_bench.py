"""The render path alone, Python half of cpp/render/render_bench.cpp: a producer thread writes
pre-made device frames as fast as the queue allows, the main thread renders them.

  python -m render.render_bench --api gl|vulkan [--present zerocopy|copy] [--check N] [--validate]
"""
import argparse
import json
import time

import torch

from .ring import Presenter, run_decoupled


def patterns(w, h, n=8):
    """B: left->right, G: top->bottom, R: checkerboard that moves with the frame index."""
    y, x = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
    out = []
    for f in range(n):
        r = torch.where((((x + f * 8) // 64 + y // 64) % 2).bool(), 200, (x * 7 + y * 3 + f) & 63)
        out.append(torch.stack([x * 255 // (w - 1), y * 255 // (h - 1), r], -1).to(torch.uint8).contiguous())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="gl", choices=("gl", "vulkan"))
    ap.add_argument("--present", default="zerocopy", choices=("zerocopy", "copy"))
    ap.add_argument("--size", default="1920x1080", help="eye size; the frame is twice as wide")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--vsync", type=int, default=0)
    ap.add_argument("--queue", default="block", choices=("block", "drop"))
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--window", default="")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--hold", type=float, default=0.0, help="producer pause before the last frame (screenshots)")
    a = ap.parse_args()
    ew, h = map(int, a.size.split("x"))
    w = 2 * ew
    pres = Presenter(a.api, w, h, slots=a.slots, zerocopy=a.present == "zerocopy", vsync=bool(a.vsync),
                     drop=a.queue == "drop", window=tuple(map(int, a.window.split("x"))) if a.window else None,
                     check=a.check, validate=a.validate)
    print(pres.description, flush=True)
    pats = patterns(w, h)
    own = torch.empty_like(pats[0])

    def produce():
        warm = min(30, a.frames // 4)
        t0 = time.perf_counter()
        for f in range(a.frames):
            if f == warm:
                t0 = time.perf_counter()
            if a.hold and f == a.frames - 1:
                time.sleep(a.hold)
            dst = pres.begin()
            if pres.closed:
                break
            dst = own if dst is None else dst
            dst.copy_(pats[f % len(pats)], non_blocking=True)  # stands in for compose
            pres.end(dst)
            torch.cuda.current_stream().synchronize()
        return (time.perf_counter() - t0) * 1000 / (a.frames - warm)

    ms = run_decoupled(pres, produce)
    st = pres.stats()
    pres.close()
    print(json.dumps({"size": f"{w}x{h}", "frames": a.frames, "producer_ms_per_frame": ms, "render": st}))


if __name__ == "__main__":
    main()
