"""Round 2 spike (user suggestion): the whole model (Swin-L backbone + MSDeformAttn pixel
decoder + transformer decoder) as ONE TensorRT engine vs the current chain of three engines.

    python prof/merge_spike.py --build     # separate process (exports Swin-L: ~10 GB host RAM)
    python prof/merge_spike.py             # compare outputs / time against the 3-engine chain
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectron2.structures import ImageList  # noqa: E402

from seg import REPO, PersonSegmenter  # noqa: E402

PLAN = os.path.join(REPO, "weights", "model_fp16_768x1344.plan")


class Whole(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        head = self.m.sem_seg_head
        f = self.m.backbone(x)
        pd = head.pixel_decoder
        mf, _, ms = type(pd).forward_features(pd, f)
        o = head.predictor(ms, mf, None)
        return o["pred_logits"], o["pred_masks"]


def frame_input(seg):
    import cv2

    from pipeline import split_sbs
    L, _ = split_sbs(cv2.imread(os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg")))
    x = seg.preprocess_gpu(torch.from_numpy(np.ascontiguousarray(L)).cuda())
    return ImageList.from_tensors([x], seg.model.size_divisibility).tensor.contiguous()


def build():
    from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn as msda_mod

    import trt_decoder
    from trt_pixel_decoder import _MSDATRT, _ConstPE
    seg = PersonSegmenter(precision="fp32", fast_decoder=True)  # torch backbone, FP32 weights
    m = seg.model.cuda().eval()
    x = frame_input(seg)
    with torch.no_grad():
        f = m.backbone(x)
        pd, pred = m.sem_seg_head.pixel_decoder, m.sem_seg_head.predictor
        _, _, ms = type(pd).forward_features(pd, f)
    pd.pe_layer = _ConstPE(pd.pe_layer, [tuple(f[k].shape[-2:]) for k in pd.transformer_in_features], "cuda")
    pred.pe_layer = _ConstPE(pred.pe_layer, [tuple(z.shape[-2:]) for z in ms], "cuda")
    pred._mask_fill = trt_decoder.MASK_FILL
    msda_mod.MSDeformAttnFunction = _MSDATRT
    onnx_path = PLAN.replace(".plan", ".onnx")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(Whole(m).eval(), (x,), onnx_path, opset_version=17, input_names=["image"],
                          output_names=["pred_logits", "pred_masks"], dynamo=False)
    lg = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(lg, "")
    b = trt.Builder(lg)
    net = b.create_network(0)
    ps = trt.OnnxParser(net, lg)
    if not ps.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
    for i in range(net.num_outputs):
        o = net.get_output(i)
        o.dtype = trt.float16 if o.name == "pred_masks" else trt.float32
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cache_path = os.path.join(os.path.dirname(PLAN), "trt_timing.cache")
    cache = cfg.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("build failed")
    open(PLAN, "wb").write(ser)
    open(cache_path, "wb").write(cfg.get_timing_cache().serialize())
    for fn in os.listdir(os.path.dirname(onnx_path)):  # ONNX + external weight files
        if fn.startswith(os.path.basename(onnx_path)) or fn.endswith(".weight") or fn.endswith(".bias"):
            pass
    os.remove(onnx_path)
    print(f"[trt] built {PLAN} ({ser.nbytes / 2**20:.0f} MB) in {time.time() - t0:.0f} s")


def timed(fn, n=20):
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def compare(out_dir):
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=(1080, 1920), fast_decoder=True,
                          pixel_decoder="trt", decoder="trt")
    x = frame_input(seg)
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(PLAN, "rb").read())
    ctx = eng.create_execution_context()
    dt = {trt.float32: torch.float32, trt.float16: torch.float16}
    outs = {n: torch.empty(tuple(eng.get_tensor_shape(n)), dtype=dt[eng.get_tensor_dtype(n)], device="cuda")
            for n in ("pred_logits", "pred_masks")}
    for n, t in [("image", x)] + list(outs.items()):
        ctx.set_tensor_address(n, t.data_ptr())
    s = torch.cuda.current_stream()

    def merged():
        ctx.execute_async_v3(s.cuda_stream)

    def chain():
        mf, ms = seg.trt_pd(seg.trt(x))
        return seg.trt_td(ms, mf)

    with torch.no_grad():
        merged()
        c = {k: v.clone() for k, v in chain().items()}
        torch.cuda.synchronize()
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
        res = {"merged_ms": timed(merged), "chain_ms": timed(chain),
               "logits_rel_vs_chain": rel(outs["pred_logits"], c["pred_logits"]),
               "masks_rel_vs_chain": rel(outs["pred_masks"], c["pred_masks"]),
               "mask_sign_agree_vs_chain": float(((outs["pred_masks"] >= 0) == (c["pred_masks"] >= 0)).float().mean()),
               "plan_mb": os.path.getsize(PLAN) / 2**20}
    print(json.dumps(res, indent=1))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "merge_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default="results/r2/merge_spike")
    a = ap.parse_args()
    build() if a.build else compare(a.out)
