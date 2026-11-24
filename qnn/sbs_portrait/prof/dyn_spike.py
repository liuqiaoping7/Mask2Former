"""Multi-resolution spike: shape-generic export forwards (dyn_export.py) and dynamic TensorRT engines.

  --eager           upstream vs shape-generic forwards in PyTorch FP32 at several input sizes
                    (backbone, pixel decoder, transformer decoder; expected equal up to fp rounding)
  --onnx STAGE      export STAGE (backbone / pd / td) as a dynamic ONNX, build an FP32 engine with
                    one wide profile and compare it with PyTorch FP32 at several sizes: checks that
                    the graph is really shape generic before any FP16 / profile question
Writes results/mr/dyn_spike_<mode>.json.
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

import dyn_export as dx  # noqa: E402
from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter, _decoder_inference_forward  # noqa: E402

SIZES = [(480, 640), (576, 1024), (800, 1088), (768, 1344)]  # padded model-input H x W


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def inputs(seg, eye, sizes):
    """Model input for each size: the eye resized to the unpadded size, padded like ImageList."""
    m = seg.model
    out = {}
    for hp, wp in sizes:
        h, w = hp - 8, wp - 16  # exercise the padding path too
        im = cv2.resize(eye, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).cuda().float()
        x = (x - m.pixel_mean) / m.pixel_std
        out[(hp, wp)] = ImageList.from_tensors([x], m.size_divisibility).tensor
        assert tuple(out[(hp, wp)].shape[-2:]) == (hp, wp)
    return out


@torch.no_grad()
def eager(seg, xs):
    import types

    m = seg.model
    pd, pred = m.sem_seg_head.pixel_decoder, m.sem_seg_head.predictor
    pred.forward = types.MethodType(_decoder_inference_forward, pred)
    pe_pd, pe_td = dx.DynSinePE(pd.pe_layer).cuda(), dx.DynSinePE(pred.pe_layer).cuda()
    res = {}
    for (h, w), x in xs.items():
        r = {}
        f_ref = m.backbone(x)
        with dx.swin_dynamic():
            f_dyn = m.backbone(x)
        r["backbone"] = {k: rel(f_dyn[k], f_ref[k]) for k in f_ref}
        mf_ref, _, ms_ref = type(pd).forward_features(pd, f_ref)
        mf_dyn, ms_dyn = dx.pixel_decoder_forward(pd, f_ref, pe_pd)
        r["pe_pd"] = max(rel(pe_pd(f_ref[k]), pd.pe_layer(f_ref[k])) for k in ("res3", "res4", "res5"))
        r["pixel_decoder"] = {"mask_features": rel(mf_dyn, mf_ref), **{f"ms{i}": rel(a, b) for i, (a, b) in enumerate(zip(ms_dyn, ms_ref))}}
        o_ref = pred(ms_ref, mf_ref, None)
        o_dyn = dx.decoder_forward(pred, ms_ref, mf_ref, pe_td)
        r["decoder"] = {k: rel(o_dyn[k], o_ref[k]) for k in o_ref}
        res[f"{h}x{w}"] = r
        print(h, w, json.dumps(r))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg"))
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--onnx", choices=("backbone", "pd", "td"))
    ap.add_argument("--out", default="results/mr")
    a = ap.parse_args()
    L, _ = split_sbs(cv2.imread(a.frame))
    seg = PersonSegmenter(precision="fp32")
    xs = inputs(seg, L, SIZES)
    os.makedirs(a.out, exist_ok=True)
    if a.eager:
        res = eager(seg, xs)
        json.dump(res, open(os.path.join(a.out, "dyn_spike_eager.json"), "w"), indent=1)
    if a.onnx:
        res = onnx_check(seg, xs, a.onnx)
        json.dump(res, open(os.path.join(a.out, f"dyn_spike_fp32_{a.onnx}.json"), "w"), indent=1)


@torch.no_grad()
def onnx_check(seg, xs, stage):
    import types

    import trt_common as tc

    m = seg.model
    pd, pred = m.sem_seg_head.pixel_decoder, m.sem_seg_head.predictor
    pred.forward = types.MethodType(_decoder_inference_forward, pred)
    prof = [((480, 640), (640, 1088), (800, 1344))]
    plan = os.path.join(REPO, "weights", f"spike_fp32_{stage}.plan")
    if stage == "backbone":
        import trt_backbone as tb
        tb.build_engine(m.backbone, plan, prof, fp16=False)
        run = tb.TRTBackbone(plan)
    elif stage == "pd":
        import trt_pixel_decoder as tp
        tp.build_engine(pd, plan, prof, fp16=False)
        run = tp.TRTPixelDecoder(plan)
    else:
        import trt_decoder as tt
        tt.build_engine(pred, plan, prof, fp16=False)
        run = tt.TRTDecoder(plan)
    res = {}
    for (h, w), x in xs.items():
        f = m.backbone(x)
        if stage == "backbone":
            o = run(x)
            r = {k: rel(o[k], f[k]) for k in f}
        else:
            mf, _, ms = type(pd).forward_features(pd, f)
            if stage == "pd":
                omf, oms = run(f)
                r = {"mask_features": rel(omf, mf), **{f"ms{i}": rel(a, b) for i, (a, b) in enumerate(zip(oms, ms))}}
            else:
                ref = pred(ms, mf, None)
                o = run([t.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2) for t in ms], mf)
                r = {k: rel(o[k], ref[k]) for k in ref}
        res[f"{h}x{w}"] = r
        print(stage, h, w, json.dumps(r))
    tc.release()
    os.remove(plan)
    return res


if __name__ == "__main__":
    main()
