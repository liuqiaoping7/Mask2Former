"""TensorRT's built-in MultiscaleDeformableAttnPlugin_TRT vs a plugin wrapping this project's own
MSDeformAttn CUDA kernel (trt_msda_plugin.py): accuracy first, speed only as a reference.

  python prof/msda_plugin_spike.py [--out results/mr/msda_plugin]

A  single MSDA op, real inputs (first encoder layer of the pixel decoder on a 1080p eye): engines with
   only that op, plugin builtin / upstream x builder FP32 (strict, no TF32) / FP16, against the
   PyTorch op (the same upstream CUDA kernel, FP32).
B  the whole pixel decoder engine (shape-generic export, one exact profile per size) for 768x1344,
   800x1088 and 480x640 model inputs: builtin / upstream x FP32 / FP16 against PyTorch FP32
   forward_features; per-call time.
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

import trt_common as tc  # noqa: E402
from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

SIZES = [(768, 1344), (800, 1088), (480, 640)]
W = os.path.join(REPO, "weights")


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


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


class _OneOp(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, value, shapes, start, loc, attn):
        o = self.fn.apply(value, shapes, start, loc, attn, 64)
        return o.reshape(o.shape[0], o.shape[1], -1)  # builtin: (N, Lq, M, D) -> (N, Lq, M*D); upstream: no-op


def op_level(m, feats, out):
    import tensorrt as trt

    import trt_msda_plugin
    from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDA
    from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn as msda_mod
    from trt_pixel_decoder import MSDA_OPS

    import dyn_export as dx
    trt_msda_plugin.load()
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
    cap = {}
    orig = msda_mod.MSDeformAttnFunction

    class _Grab:
        @staticmethod
        def apply(value, shapes, start, loc, attn, step):
            if not cap:
                cap.update(value=value.clone(), shapes=shapes.clone(), start=start.clone(), loc=loc.clone(), attn=attn.clone())
            return orig.apply(value, shapes, start, loc, attn, step)

    msda_mod.MSDeformAttnFunction = _Grab
    try:
        with torch.no_grad():
            dx.pixel_decoder_forward(m.sem_seg_head.pixel_decoder, feats)
    finally:
        msda_mod.MSDeformAttnFunction = orig
    x = [cap[k] for k in ("value", "shapes", "start", "loc", "attn")]
    ref = MSDA.ms_deform_attn_forward(*x, 64)
    ref = ref.reshape(ref.shape[0], ref.shape[1], -1)
    names = ["value", "shapes", "start", "loc", "attn"]
    shp = {n: tuple(t.shape) for n, t in zip(names, x)}
    res = {"shapes": {k: list(v) for k, v in shp.items()}}
    for msda in ("builtin", "upstream"):
        fn, _ = MSDA_OPS[msda]
        onnx_path = os.path.join(W, f"msda_op_{msda}.onnx")
        with torch.no_grad():
            torch.onnx.export(_OneOp(fn), tuple(x), onnx_path, opset_version=17, input_names=names, output_names=["out"],
                              dynamo=False)
        for prec in ("fp32", "fp16", "fp16io"):
            # fp16io: value / locations / weights as FP16 network inputs, so TensorRT can run the plugin
            # in FP16 (with FP32 inputs it keeps an FP32 plugin even in an FP16 build)
            plan = os.path.join(W, f"msda_op_{msda}_{prec}.plan")
            io = {k: trt.float16 for k in ("value", "loc", "attn")} if prec == "fp16io" else None
            tc.build(onnx_path, plan, [((1, 1),) * 3], lambda h, w: shp, fp16=prec != "fp32", plugins=True,
                     in_dtypes=io, detailed=True)
            run = tc.TRTRunner(plan, plugins=True)
            feeds = dict(zip(names, x))
            o = run(feeds)["out"].float()
            d = (o - ref).abs()
            res[f"{msda}/{prec}"] = {"rel": rel(o, ref), "max_abs": float(d.max()), "bit_exact": bool(d.max() == 0),
                                     "ms": timed(lambda: run(feeds)), "plugin_precision": _plugin_io(plan)}
            print("op", msda, prec, json.dumps(res[f"{msda}/{prec}"]), flush=True)
            del run
            tc.release()
            os.remove(plan)
        os.remove(onnx_path)
    res["torch_ms"] = timed(lambda: MSDA.ms_deform_attn_forward(*x, 64))
    json.dump(res, open(os.path.join(out, "op_level.json"), "w"), indent=1)


def _plugin_io(plan):
    """Data types at the MSDA plugin layers (engine inspector, needs a DETAILED build): value input and
    output of each plugin layer, counted over the engine."""
    import collections

    import tensorrt as trt

    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(plan, "rb").read())
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    cnt = collections.Counter()
    for ly in info.get("Layers", []):
        if isinstance(ly, dict) and ly.get("LayerType") in ("PluginV2", "PluginV3"):
            fin = ly["Inputs"][0].get("Format/Datatype", "?")
            fout = ly["Outputs"][0].get("Format/Datatype", "?")
            cnt[f"{fin} -> {fout}"] += 1
    return dict(cnt)


def pd_level(m, eye, out):
    from trt_pixel_decoder import TRTPixelDecoder, build_engine

    pd = m.sem_seg_head.pixel_decoder
    chans = {k: v.channels for k, v in m.backbone.output_shape().items()}
    prof = [((h, w),) * 3 for h, w in SIZES]
    feats = {}
    with torch.no_grad():
        for h, w in SIZES:
            im = cv2.resize(eye, (w, h), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).cuda().float()
            x = ImageList.from_tensors([(x - m.pixel_mean) / m.pixel_std], 32).tensor
            f = m.backbone(x)
            mf, _, ms = type(pd).forward_features(pd, f)
            feats[(h, w)] = (f, mf, ms, timed(lambda: type(pd).forward_features(pd, f), 10))
    res = {}
    for msda in ("builtin", "upstream"):
        for prec in ("fp32", "fp16"):
            plan = os.path.join(W, f"msda_pd_{msda}_{prec}.plan")
            build_engine(pd, plan, prof, chans=chans, fp16=prec == "fp16", msda=msda, detailed=True)
            res[f"{msda}/{prec}/plugin_io"] = _plugin_io(plan)
            print("pd", msda, prec, "plugin io", res[f"{msda}/{prec}/plugin_io"], flush=True)
            run = TRTPixelDecoder(plan, msda=msda)
            for (h, w), (f, mf, ms, t_torch) in feats.items():
                omf, oms = run(f)
                r = {"mask_features": rel(omf, mf), **{f"ms{i}": rel(a, b) for i, (a, b) in enumerate(zip(oms, ms))},
                     "ms": timed(lambda: run(f)), "torch_fp32_ms": t_torch}
                res[f"{msda}/{prec}/{h}x{w}"] = r
                print("pd", msda, prec, h, w, json.dumps({k: round(v, 6) for k, v in r.items()}), flush=True)
            res[f"{msda}/{prec}/plugin_io"] = res[f"{msda}/{prec}/plugin_io"]
            del run
            tc.release()
            os.remove(plan)
    json.dump(res, open(os.path.join(out, "pd_level.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg"))
    ap.add_argument("--out", default="results/mr/msda_plugin")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seg = PersonSegmenter(precision="fp32")
    m = seg.model
    L, _ = split_sbs(cv2.imread(a.frame))
    with torch.no_grad():
        im = cv2.resize(L, (1333, 750), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).cuda().float()
        feats = m.backbone(ImageList.from_tensors([(x - m.pixel_mean) / m.pixel_std], 32).tensor)
    op_level(m, feats, a.out)
    pd_level(m, L, a.out)


if __name__ == "__main__":
    main()
