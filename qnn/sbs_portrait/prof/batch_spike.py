"""Re-evaluate batching both eyes (batch=2) after round 2: all three stages are TensorRT FP16.

Item 6 (round 1) rejected batch=2 when the decoders were memory-bound torch code (+49 ms,
+1.9 GB). Builds batch-2 engines with the existing builders and compares, per stage and for
the whole chain, two batch-1 runs (current production engines) against one batch-2 run.

    python prof/batch_spike.py --build backbone     # separate process, ~10 GB host RAM
    python prof/batch_spike.py --build heads        # pixel decoder + decoder (batch 2)
    python prof/batch_spike.py                      # compare (no building in this process)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectron2.structures import ImageList  # noqa: E402

from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

W = os.path.join(REPO, "weights")
BB2, PD2, TD2 = (os.path.join(W, f"{n}_b2_768x1344.plan") for n in ("swinl_backbone_fp16", "pixel_decoder_fp16", "decoder_fp16"))
FRAME = os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg")


def eyes_input(seg):
    import cv2
    fr = torch.from_numpy(cv2.imread(FRAME)).cuda()
    w = fr.shape[1] // 2
    xs = [seg.preprocess_gpu(fr[:, :w]), seg.preprocess_gpu(fr[:, w:])]
    return ImageList.from_tensors(xs, seg.model.size_divisibility).tensor.contiguous()  # 2, 3, 768, 1344


def timed(fn, n=30):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(float(np.median(ts)), 2)


def act_mb(runner):
    e = runner.engine
    v = getattr(e, "device_memory_size_v2", None)
    return round((v if v is not None else e.device_memory_size) / 2**20, 1)


def build(what):
    if what == "backbone":
        from trt_backbone import build_engine
        seg = PersonSegmenter(precision="fp32")
        seg.model.eval()  # Swin's train() returns None: do not chain .eval()
        build_engine(seg.model.backbone, BB2, (2, 3, 768, 1344))
        return
    from trt_decoder import build_engine as build_td
    from trt_pixel_decoder import build_engine as build_pd
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=(1080, 1920), fast_decoder=True)
    shp = {k: (2, *v.shape[1:]) for k, v in seg.trt.out.items()}
    build_pd(seg.model.sem_seg_head.pixel_decoder, shp, PD2)
    from trt_pixel_decoder import TRTPixelDecoder
    o = TRTPixelDecoder(PD2).out
    nchw = lambda s: (s[0], s[3], s[1], s[2])  # noqa: E731
    build_td(seg.model.sem_seg_head.predictor, [nchw(tuple(o[k].shape)) for k in ("ms0", "ms1", "ms2")],
             tuple(o["mask_features"].shape), TD2)


def compare(out_dir):
    from trt_backbone import TRTBackbone
    from trt_decoder import TRTDecoder
    from trt_pixel_decoder import TRTPixelDecoder
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=(1080, 1920), fast_decoder=True,
                          pixel_decoder="trt", decoder="trt")
    bb2, pd2, td2 = TRTBackbone(BB2), TRTPixelDecoder(PD2), TRTDecoder(TD2)
    bb1, pd1, td1 = seg.trt, seg.trt_pd, seg.trt_td
    res = {}
    with torch.no_grad(), torch.cuda.stream(seg.stream):
        x2 = eyes_input(seg)
        x1 = [x2[i:i + 1].contiguous() for i in range(2)]
        clone = lambda d: {k: v.clone() for k, v in d.items()}  # noqa: E731
        f1 = [clone(bb1(x)) for x in x1]
        f2 = clone(bb2(x2))
        h1 = [(lambda mf, ms: (mf.clone(), [m.clone() for m in ms]))(*pd1(f)) for f in f1]
        mf2, ms2 = pd2(f2)
        h2 = (mf2.clone(), [m.clone() for m in ms2])
        # per stage: two batch-1 calls vs one batch-2 call
        res["backbone_ms"] = {"b1x2": timed(lambda: [bb1(x) for x in x1]), "b2": timed(lambda: bb2(x2))}
        res["pixel_decoder_ms"] = {"b1x2": timed(lambda: [pd1(f) for f in f1]), "b2": timed(lambda: pd2(f2))}
        res["decoder_ms"] = {"b1x2": timed(lambda: [td1(ms, mf) for mf, ms in h1]), "b2": timed(lambda: td2(h2[1], h2[0]))}

        def chain1():
            return [td1(*reversed(pd1(bb1(x)))) for x in x1]

        def chain2():
            return td2(*reversed(pd2(bb2(x2))))

        res["chain_ms"] = {"b1x2": timed(chain1), "b2": timed(chain2)}
        # correctness: each eye of the batch-2 chain vs its batch-1 chain
        o1 = []
        for x in x1:
            o = td1(*reversed(pd1(bb1(x))))
            o1.append({k: v.clone() for k, v in o.items()})
        o2 = chain2()
        torch.cuda.synchronize()
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
        res["per_eye_vs_b1"] = [{"logits_rel": rel(o2["pred_logits"][i], o1[i]["pred_logits"][0]),
                                 "masks_rel": rel(o2["pred_masks"][i], o1[i]["pred_masks"][0]),
                                 "mask_sign_agree": float(((o2["pred_masks"][i] >= 0) == (o1[i]["pred_masks"][0] >= 0)).float().mean())}
                                for i in range(2)]
    res["activation_mb"] = {"b1": [act_mb(r) for r in (bb1, pd1, td1)], "b2": [act_mb(r) for r in (bb2, pd2, td2)]}
    res["plan_mb"] = {os.path.basename(p): round(os.path.getsize(p) / 2**20) for p in (BB2, PD2, TD2)}
    print(json.dumps(res, indent=1))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "batch_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", choices=["backbone", "heads"])
    ap.add_argument("--out", default="results/r2/batch_spike")
    a = ap.parse_args()
    build(a.build) if a.build else compare(a.out)
