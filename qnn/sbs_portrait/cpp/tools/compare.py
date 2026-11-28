"""Compare the stage dumps of the C++ pipeline with the Python reference, stage by stage in pipeline
order (worst case over the frames and eyes). NaN positions must agree.

  python compare.py REF_DIR CPP_DIR
"""
import os
import sys
from glob import glob

import numpy as np

DT = {"f4": np.float32, "f2": np.float16, "u1": np.uint8}
ORDER = ("gray", "disp", "fb", "ff", "pre", "logits", "masks", "prob", "hard", "valid", "photo", "dfill", "tema",
         "alpha", "out")


def load(path):
    with open(path, "rb") as f:
        head = f.readline().split()
        shape = [int(v) for v in head[1:]]
        return np.frombuffer(f.read(), DT[head[0].decode()]).reshape(shape)


def main():
    ref, cpp = sys.argv[1:3]
    print(f"{'stage':8s} {'files':>5s} {'max |diff|':>11s} {'mean |diff|':>11s} {'differing':>10s}  notes")
    for stage in ORDER:
        files = sorted(glob(os.path.join(ref, stage + "_*.bin")))
        if not files:
            continue
        mx, total, n, diff, notes = 0.0, 0.0, 0, 0, set()
        for f in files:
            g = os.path.join(cpp, os.path.basename(f))
            if not os.path.exists(g):
                notes.add("missing in C++")
                continue
            a, b = load(f), load(g)
            if stage == "pre":  # C++ keeps the zero padding of the model input
                pad = b.copy()
                pad[:, :a.shape[1], :a.shape[2]] = 0
                if pad.any():
                    notes.add("non-zero padding")
                b = b[:, :a.shape[1], :a.shape[2]]
            if a.shape != b.shape:
                notes.add(f"shape {a.shape} vs {b.shape}")
                continue
            a, b = a.astype(np.float64), b.astype(np.float64)
            na, nb = np.isnan(a), np.isnan(b)
            if (na != nb).any():
                notes.add(f"NaN positions differ ({int((na != nb).sum())})")
            d = np.abs(np.where(na | nb, 0.0, a - b))
            mx, total, n, diff = max(mx, d.max()), total + d.sum(), n + d.size, diff + int((d > 0).sum())
        print(f"{stage:8s} {len(files):5d} {mx:11.3g} {total / max(n, 1):11.3g} {diff / max(n, 1):10.2e}  {'; '.join(sorted(notes))}")


if __name__ == "__main__":
    main()
