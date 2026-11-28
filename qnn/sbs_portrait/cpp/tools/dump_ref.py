"""Dump the intermediate tensors of the Python production pipeline for the first N frames, with the
same names as `sbs_portrait --stage-dir`, for tools/compare.py.

  python dump_ref.py --frames DIR --bg BG.jpg --out DIR [--n 10] [--pip frame|icon]
"""
import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from pipeline import SBSPortraitPipeline  # noqa: E402
from run_pipeline import pip_source  # noqa: E402

DTYPE = {"float32": "f4", "float16": "f2", "uint8": "u1", "bool": "u1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pip", default="")
    ap.add_argument("--compose-kernel", default="triton")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob(os.path.join(a.frames, "*.jpg")))[: a.n]
    pipe = SBSPortraitPipeline(cv2.imread(files[0]), cv2.imread(a.bg), pip=pip_source(a.pip),
                               compose_kernel=a.compose_kernel)
    frame = [0]
    count = {}

    def save(name, x, eye=None):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.ascontiguousarray(x)
        tag = f"{name}_{frame[0]}" + ("" if eye is None else f"_{eye}")
        with open(os.path.join(a.out, tag + ".bin"), "wb") as f:
            f.write((" ".join([DTYPE[x.dtype.name], *map(str, x.shape)]) + "\n").encode())
            f.write(x.tobytes())

    def eye_of(name):  # per-frame call counter: first call = left eye
        e = count.get(name, 0)
        count[name] = e + 1
        return e

    def wrap(obj, attr, after):
        fn = getattr(obj, attr)

        def run(*args, **kw):
            r = fn(*args, **kw)
            after(r, args)
            return r

        setattr(obj, attr, run)

    seg, gp = pipe.seg, pipe.gp
    wrap(seg, "preprocess_gpu", lambda r, args: save("pre", r, eye_of("pre")))
    chain = seg.chain

    def run_chain(x):
        out = chain(x)
        e = eye_of("chain")
        save("logits", out["pred_logits"][0].float(), e)
        save("masks", out["pred_masks"][0], e)
        return out

    seg.chain = run_chain
    wrap(seg, "_person_fused", lambda r, args: [save(n, v, e) for e in [eye_of("person")]
                                                for n, v in (("prob", r[0]), ("hard", r[1]))])

    def after_geo(r, args):
        gL, gR, dL, dR, flows = r[:5]
        for e, (g, d) in enumerate(((gL, dL), (gR, dR))):
            save("gray", g, e)
            save("disp", d, e)
            if flows is not None:
                save("fb", flows[2 * e], e)
                save("ff", flows[2 * e + 1], e)

    wrap(pipe, "cpu_geometry", after_geo)

    def after_validity(r, args):
        e = eye_of("validity")
        save("valid", r[0], e)
        save("photo", r[1], e)

    wrap(gp, "flow_validity", after_validity)
    wrap(gp, "fill_disparity", lambda r, args: [save("dfill", v, e) for e, v in enumerate(r)])
    wrap(gp, "ema", lambda r, args: save("tema", r, "LR".index(args[0])))
    wrap(gp, "fusion", lambda r, args: [save("alpha", v, e) for e, v in enumerate(r)])
    for t, f in enumerate(files):
        frame[0] = t
        count.clear()
        save("out", pipe.process(cv2.imread(f))["out"])
    with open(os.path.join(a.out, "init.txt"), "w") as f:
        f.write(f"bg_shift {pipe.bg_shift}\nperson_disp {pipe.person_disp}\n")
    print(f"bg_shift {pipe.bg_shift} init person_disp {pipe.person_disp}")


if __name__ == "__main__":
    main()
