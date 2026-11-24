"""Builder workspace limit vs engine scratch (activation) memory and speed, 1080p input (768x1344,
one exact profile), each of the three engines. The scratch TensorRT asks for at run time includes the
tactic workspace, so a smaller limit can shrink the arena (trt_common.TRTChain) at some speed cost.

  python prof/workspace_spike.py [--limits 2048,512,256,128,64] [--out results/mr/arena]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trt_common as tc  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

PROF = [((768, 1344),) * 3]


def timed(fn, n=30):
    for _ in range(5):
        fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return float(np.median([a.elapsed_time(b) for a, b in ev]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limits", default="2048,512,256,128,64")
    ap.add_argument("--fp16-features", action="store_true", help="also: backbone -> pixel decoder tensors in FP16")
    ap.add_argument("--out", default="results/mr/arena")
    a = ap.parse_args()
    import trt_backbone as tb
    import trt_decoder as tt
    import trt_pixel_decoder as tp
    seg = PersonSegmenter(precision="fp16", fast_decoder=True)
    m = seg.model
    chans = {k: v.channels for k, v in m.backbone.output_shape().items()}
    orig = tc.build
    res = {}
    x = torch.randn(1, 3, 768, 1344, device="cuda")
    stream = torch.cuda.Stream()
    runs = [(lim, False) for lim in map(int, a.limits.split(","))] + ([(2048, True)] if a.fp16_features else [])
    for lim, f16 in runs:
        tc.build = lambda *ar, _l=lim, **kw: orig(*ar, **{**kw, "workspace_mb": _l})  # noqa: E731
        plans = {k: os.path.join(REPO, "weights", f"ws_{k}_{lim}{'_f16' if f16 else ''}.plan") for k in ("bb", "pd", "td")}
        tb.build_engine(m.backbone, plans["bb"], PROF, out_fp16=f16)
        tp.build_engine(m.sem_seg_head.pixel_decoder, plans["pd"], PROF, chans=chans, in_fp16=f16)
        tt.build_engine(m.sem_seg_head.predictor, plans["td"], PROF)
        tc.build = orig
        bb, pd, td = tb.TRTBackbone(plans["bb"]), tp.TRTPixelDecoder(plans["pd"]), tt.TRTDecoder(plans["td"])
        chain = tc.TRTChain([bb, pd, td], [None, {k: k for k in ("res2", "res3", "res4", "res5")},
                                          {"ms0": "ms0", "ms1": "ms1", "ms2": "ms2", "mf": "mask_features"}])
        with torch.cuda.stream(stream), torch.no_grad():
            out = chain(x)
            masks = out["pred_masks"].float().clone()
            t = timed(lambda: chain(x))
        st = chain.layouts[(1, 3, 768, 1344)]["stages"]
        r = {"scratch_mb": [s["scratch"] / 2**20 for s in st], "arena_mb": chain.arena.numel() / 2**20, "chain_ms": t,
             "plan_mb": {k: os.path.getsize(p) / 2**20 for k, p in plans.items()}}
        if "ref" not in res:
            res["ref"] = masks
        r["masks_rel_vs_2048"] = float((masks - res["ref"]).norm() / res["ref"].norm())
        name = f"{lim}{'_fp16_features' if f16 else ''}"
        res[name] = r
        print(name, json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in r.items()}), flush=True)
        del chain, bb, pd, td, out
        tc.release()
        for p in plans.values():
            os.remove(p)
    res.pop("ref")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "workspace_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
