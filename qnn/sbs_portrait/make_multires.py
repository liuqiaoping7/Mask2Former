"""Multi-resolution test sequences: the 1080p SBS clip re-rendered at other eye sizes (VGA .. 1080p).

  python make_multires.py --src ~/data/sbs/portrait_0121 --out ~/data/sbs/mr/frames [--random 4 --seed 0]

Each eye is centre-cropped to the target aspect ratio (same crop window on both eyes, so the
pair stays rectified and disparities keep their sign / ordering) and resized with INTER_AREA,
then saved as JPEG quality 95. Sizes: fixed anchors (VGA 4:3 and 16:9, qHD, 720p, 1080p 4:3,
1080p) plus --random sizes with height uniform in [480, 1080] and aspect uniform in
[4:3, 16:9] (even dimensions). Writes <out>/<W>x<H>/NNNN.jpg and <out>/sizes.json.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np

ANCHORS = [(640, 480), (854, 480), (960, 540), (1280, 720), (1440, 1080), (1920, 1080)]


def random_sizes(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        h = int(rng.uniform(480, 1080)) // 2 * 2
        w = int(round(h * rng.uniform(4 / 3, 16 / 9) / 2)) * 2
        w = min(max(w, 640), 1920)
        if (w, h) not in ANCHORS + out:
            out.append((w, h))
    return out


def render(frame, size):
    W = frame.shape[1] // 2
    H = frame.shape[0]
    w, h = size
    cw = min(W, int(round(H * w / h)))  # crop width at the target aspect (full height)
    ch = min(H, int(round(cw * h / w)))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    eyes = [frame[y0:y0 + ch, o + x0:o + x0 + cw] for o in (0, W)]
    return np.hstack([cv2.resize(e, (w, h), interpolation=cv2.INTER_AREA) for e in eyes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=150)
    a = ap.parse_args()
    files = sorted(glob(os.path.join(a.src, "*.jpg")))[: a.frames]
    sizes = ANCHORS + random_sizes(a.random, a.seed)
    for w, h in sizes:
        d = os.path.join(a.out, f"{w}x{h}")
        os.makedirs(d, exist_ok=True)
        f0 = cv2.imread(files[0])
        if (w, h) == (f0.shape[1] // 2, f0.shape[0]):  # source size: link, no JPEG re-encode
            for f in files:
                dst = os.path.join(d, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(f), dst)
            print(f"{w}x{h}: {len(files)} frames (links)")
            continue

        def one(f, d=d, w=w, h=h):
            cv2.imwrite(os.path.join(d, os.path.basename(f)), render(cv2.imread(f), (w, h)),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, files))
        print(f"{w}x{h}: {len(files)} frames")
    json.dump([f"{w}x{h}" for w, h in sizes], open(os.path.join(a.out, "sizes.json"), "w"))


if __name__ == "__main__":
    main()
