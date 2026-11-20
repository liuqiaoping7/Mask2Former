"""Round 2 upper-bound spike: transformer decoder (inference forward) as a TensorRT engine.

    python prof/td_spike.py --build [--fp32]     # separate process: export + build
    python prof/td_spike.py                      # compare with eager FP32 / compiled FP16 torch, time
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectron2.structures import ImageList  # noqa: E402

from pipeline import split_sbs  # noqa: E402
from seg import REPO, PersonSegmenter  # noqa: E402

DT = {trt.float32: torch.float32, trt.float16: torch.float16}


class Wrap(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, ms0, ms1, ms2, mf):
        o = self.pred([ms0, ms1, ms2], mf, None)
        return o["pred_logits"], o["pred_masks"]


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


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default="results/r2/td_spike")
    a = ap.parse_args()
    plan = os.path.join(REPO, "weights", f"td_{'fp32' if a.fp32 else 'fp16'}{os.environ.get('TD_PIN', '')}{os.environ.get('TD_FILL', '')}.plan")
    L, _ = split_sbs(cv2.imread(os.path.expanduser("~/data/sbs/portrait_0121/0001.jpg")))
    seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=L.shape[:2], fast_decoder=True,
                          compile_decoders=not a.build, pixel_decoder="trt")
    m = seg.model
    pred = m.sem_seg_head.predictor
    with torch.no_grad():
        x = seg.preprocess_gpu(torch.from_numpy(np.ascontiguousarray(L)).cuda())
        t = ImageList.from_tensors([x], m.size_divisibility).tensor
        with torch.autocast("cuda", dtype=torch.float16):
            mf, ms = seg.trt_pd(seg.trt(t))
        ins = [z.float().contiguous() for z in ms] + [mf.float().contiguous()]
        names = ["ms0", "ms1", "ms2", "mf"]
        if a.build:
            from trt_pixel_decoder import _ConstPE
            pred.pe_layer = _ConstPE(pred.pe_layer, [tuple(z.shape[-2:]) for z in ms], "cuda")
            if os.environ.get("TD_FILL"):
                pred._mask_fill = float(os.environ["TD_FILL"])
            onnx_path = plan.replace(".plan", ".onnx")
            torch.onnx.export(Wrap(pred).eval(), tuple(ins), onnx_path, opset_version=17, input_names=names,
                              output_names=["logits", "masks"], dynamo=False)
            lg = trt.Logger(trt.Logger.WARNING)
            b = trt.Builder(lg)
            net = b.create_network(0)
            ps = trt.OnnxParser(net, lg)
            if not ps.parse_from_file(onnx_path):
                raise RuntimeError("\n".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
            cfg = b.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            if not a.fp32:
                cfg.set_flag(trt.BuilderFlag.FP16)
                pin = tuple(x for x in os.environ.get("TD_PIN", "").split(",") if x)
                if pin:
                    from trt_pixel_decoder import _pin_fp32
                    print("[trt] pinned", _pin_fp32(net, pin), "layers to FP32", pin)
                    cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            t0 = time.time()
            ser = b.build_serialized_network(net, cfg)
            open(plan, "wb").write(ser)
            os.remove(onnx_path)
            print(f"[trt] built {plan} ({ser.nbytes / 2**20:.0f} MB) in {time.time() - t0:.0f} s")
            return
        eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(plan, "rb").read())
        ctx = eng.create_execution_context()
        outs = {n: torch.empty(tuple(eng.get_tensor_shape(n)), dtype=DT[eng.get_tensor_dtype(n)], device="cuda")
                for n in ("logits", "masks")}
        for n, z in list(zip(names, ins)) + list(outs.items()):
            ctx.set_tensor_address(n, z.data_ptr())
        s = torch.cuda.current_stream()

        def run_trt():
            ctx.execute_async_v3(s.cuda_stream)

        run_trt()
        # references: compiled FP16 production path, and an eager FP32 run of the same forward
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3):
                cur = pred(ms, mf, None)
        torch.cuda.synchronize()
        res = {"trt_ms": timed(run_trt)}

        def run_cur():
            with torch.autocast("cuda", dtype=torch.float16):
                pred(ms, mf, None)

        res["compiled_ms"] = timed(run_cur)
        ref = pred._orig_mod if hasattr(pred, "_orig_mod") else None
        fwd = pred.forward
        if hasattr(fwd, "_torchdynamo_orig_callable"):
            ref = fwd._torchdynamo_orig_callable
        r32 = (ref or fwd)([z.float() for z in ms], mf.float(), None)
        for name, o in (("trt", {"pred_logits": outs["logits"], "pred_masks": outs["masks"]}), ("compiled_fp16", cur)):
            res[name] = {"logits_rel": rel(o["pred_logits"], r32["pred_logits"]),
                         "masks_rel": rel(o["pred_masks"], r32["pred_masks"]),
                         "mask_sign_agree": float(((o["pred_masks"] >= 0) == (r32["pred_masks"] >= 0)).float().mean())}
    print(json.dumps(res, indent=1))
    os.makedirs(a.out + ("_fp32" if a.fp32 else ""), exist_ok=True)
    json.dump(res, open(os.path.join(a.out + ("_fp32" if a.fp32 else ""), "td_spike.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
