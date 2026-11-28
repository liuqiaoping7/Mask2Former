"""Convert the C++ --dump output (raw .bin) to the npz files of run_pipeline.py --dump, in place, so
that eval_alpha.py can read the run.

  python to_npz.py RUN_DIR [RUN_DIR ...]
"""
import os
import sys
from glob import glob

import numpy as np

from compare import load

for run in sys.argv[1:]:
    d = os.path.join(run, "dump")
    for first in sorted(glob(os.path.join(d, "aL_*.bin"))):
        t = first[-8:-4]
        r = {k: load(os.path.join(d, f"{k}_{t}.bin")) for k in ("aL", "aR", "pL", "pR", "hL", "hR")}
        np.savez_compressed(os.path.join(d, f"{t}.npz"), aL=r["aL"].astype(np.float16), aR=r["aR"].astype(np.float16),
                            pL=(r["pL"] * 255 + 0.5).astype(np.uint8), pR=(r["pR"] * 255 + 0.5).astype(np.uint8),
                            hL=r["hL"].astype(bool), hR=r["hR"].astype(bool))
        for k in r:
            os.remove(os.path.join(d, f"{k}_{t}.bin"))
