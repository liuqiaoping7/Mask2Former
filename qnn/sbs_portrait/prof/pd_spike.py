"""Round 2 upper-bound spike: TensorRT pixel decoder vs the current torch path (one eye).

Builds the engine in a separate step (--build), then in a fresh process (--bench) compares
outputs (vs the FP32 upstream decoder and the current FP16 torch path) and times both.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectron2.structures import ImageList  # noqa: E402

from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

PLAN = os.environ.get("PD_PLAN", os.path.join(REPO, "weights", "pixel_decoder_fp16_768x1344.plan"))


def timed(fn, n=20):
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default="results/r2/pd_spike")
    a = ap.parse_args()
    L, _ = split_sbs(cv2.imread(a.frame))
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=L.shape[:2], fast_decoder=True)
    m = seg.model
    pd = m.sem_seg_head.pixel_decoder
    with torch.no_grad():
        im = (seg.preprocess(L).to(m.device).float() - m.pixel_mean) / m.pixel_std
        t = ImageList.from_tensors([im], m.size_divisibility).tensor
        feats = {k: v.clone() for k, v in seg.trt(t).items()}
        if a.build:
            from trt_pixel_decoder import build_engine
            build_engine(pd, {k: tuple(v.shape) for k, v in feats.items()}, PLAN, fp16=os.environ.get("PD_FP32") != "1",
                         fp32_layers=tuple(x for x in os.environ.get("PD_PIN", "").split(",") if x))
            return
        from trt_pixel_decoder import TRTPixelDecoder
        eng = TRTPixelDecoder(PLAN)
        ref_mf, _, ref_ms = type(pd).forward_features(pd, feats)            # FP32 upstream
        with torch.autocast("cuda", dtype=torch.float16):
            cur_mf, _, cur_ms = pd.forward_features(feats)                   # current FP16 torch path
        trt_mf, trt_ms = eng(feats)
        torch.cuda.synchronize()
        res = {"err_vs_fp32": {"torch_fp16": [rel(cur_mf, ref_mf)] + [rel(x, y) for x, y in zip(cur_ms, ref_ms)],
                               "trt_fp16": [rel(trt_mf, ref_mf)] + [rel(x, y) for x, y in zip(trt_ms, ref_ms)]}}

        def cur():
            with torch.autocast("cuda", dtype=torch.float16):
                pd.forward_features(feats)

        res["torch_fp16_ms"] = timed(cur)
        res["trt_ms"] = timed(lambda: eng(feats))
        # end-to-end head: masks / class probabilities from the predictor on each decoder output
        pred = m.sem_seg_head.predictor
        with torch.autocast("cuda", dtype=torch.float16):
            o_ref = pred(ref_ms, ref_mf, None)
            o_trt = pred(trt_ms, trt_mf, None)
            o_cur = pred(cur_ms, cur_mf, None)
        res["torch_fp16_pred_masks_rel_err"] = rel(o_cur["pred_masks"], o_ref["pred_masks"])
        res["torch_fp16_mask_sign_agree"] = float(((o_cur["pred_masks"] >= 0) == (o_ref["pred_masks"] >= 0)).float().mean())
        res["pred_masks_rel_err"] = rel(o_trt["pred_masks"], o_ref["pred_masks"])
        res["pred_logits_rel_err"] = rel(o_trt["pred_logits"], o_ref["pred_logits"])
        res["mask_sign_agree"] = float(((o_trt["pred_masks"] >= 0) == (o_ref["pred_masks"] >= 0)).float().mean())
        res["out_dtypes"] = [str(x.dtype) for x in [trt_mf, *trt_ms]]
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "pd_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
