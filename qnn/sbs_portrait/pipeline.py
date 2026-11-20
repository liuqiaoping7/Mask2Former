"""Production SBS portrait pipeline: only the `final` alpha path of sbs_portrait_bench.py.

frame -> per-eye person segmentation -> disparity + optical flow -> soft alpha ->
guided filter -> temporal EMA -> stereo fusion -> 1080p SBS composite.

Evaluation (multi-config comparison, consistency metrics) lives in eval_alpha.py and
runs offline on dumped alphas, so it is not on the latency path.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np
import torch

import temporal_stereo as ts
from gpu_post import GPUPost
from seg import DEFAULT_CONFIG, DEFAULT_WEIGHTS, PersonSegmenter


def split_sbs(img):
    w = img.shape[1] // 2
    return img[:, :w], img[:, w:]


def down(x, size, interp=cv2.INTER_AREA):
    return cv2.resize(x, size, interpolation=interp)


def nanpct(x, q, default):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if x.size else default


class Timer:
    """Per-module wall time, accumulated per frame (a module may run several times per frame,
    e.g. once per eye). sync=True synchronizes CUDA around GPU modules (profiling mode)."""

    def __init__(self, sync=False):
        self.sync, self.t, self.frames = sync, {}, [{}]

    def new_frame(self):
        self.frames.append({})

    def clear(self):
        self.t, self.frames = {}, [{}]

    @contextmanager
    def __call__(self, key, gpu=False):
        if gpu and self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if gpu and self.sync:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        self.t.setdefault(key, []).append(dt)
        cur = self.frames[-1]
        cur[key] = cur.get(key, 0.0) + dt

    def summary(self, skip=3):
        """Median over frames of the per-frame total of each module (ms/frame)."""
        fr = [f for f in self.frames if f][skip:] or [f for f in self.frames if f]
        keys = sorted({k for f in fr for k in f})
        return {k: float(np.median([f.get(k, 0.0) for f in fr])) for k in keys}


class SBSPortraitPipeline:
    def __init__(self, first_frame, bg_sbs, config_file=DEFAULT_CONFIG, weights=DEFAULT_WEIGHTS,
                 work_scale=0.5, soft=(0.4, 0.6), gf=(4, 1e-4), lam=0.7, gate=0.5, bg_margin=4.0,
                 precision="fp16", backbone="trt", post="gpu", overlap=True, timer=None):
        """post: "gpu" (torch post-processing + compositing, see gpu_post.py) or "cpu" (reference).
        overlap: run the CPU geometry (downscale, SGBM, DIS) in a worker thread while the GPU
        segments the same frame (OpenCV and CUDA waits release the GIL)."""
        self.timer = timer or Timer()
        self.soft, self.gf = soft, gf
        L0, R0 = split_sbs(first_frame)
        self.H, self.W = L0.shape[:2]
        H, W = self.H, self.W
        self.ws = ws = (int(W * work_scale), int(H * work_scale))
        s = work_scale

        self.seg = PersonSegmenter(config_file, weights, precision=precision, backbone=backbone, eye_hw=(H, W),
                                   fast_decoder=True, compile_decoders=True, pixel_decoder="trt")
        for _ in range(2):  # warm-up
            self.segment_eyes(L0, R0, record=False)

        gL0, gR0 = [cv2.cvtColor(down(x, ws), cv2.COLOR_BGR2GRAY) for x in (L0, R0)]
        min_d, num_d, _, _ = ts.estimate_disparity_range(gL0, gR0)
        self.stereo = ts.StereoEstimator(min_d, num_d)
        self.flow = ts.FlowEstimator()
        self.ema = {"L": ts.TemporalEMA(lam, gate=gate), "R": ts.TemporalEMA(lam, gate=gate)}

        # virtual background + depth placement (same as the benchmark)
        bgL, bgR = split_sbs(bg_sbs)
        bgL, bgR = down(bgL, (W, H), cv2.INTER_LINEAR), down(bgR, (W, H), cv2.INTER_LINEAR)
        gbL, gbR = [cv2.cvtColor(down(x, ws), cv2.COLOR_BGR2GRAY) for x in (bgL, bgR)]
        bmin, bnum, _, _ = ts.estimate_disparity_range(gbL, gbR)
        bd_l, _ = ts.StereoEstimator(bmin, bnum)(gbL, gbR)
        _, h0, _, _ = self.segment_eyes(L0, R0, record=False)
        dL0, _ = self.stereo(gL0, gR0)
        core = cv2.erode(h0.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        person_p5 = nanpct(dL0[core], 5, 0.0) / s
        self.person_disp = nanpct(dL0[core], 50, 0.0)
        bg_p95 = nanpct(bd_l, 95, 0.0) / s
        self.bg_shift = max(0.0, bg_p95 - person_p5 + bg_margin)
        if self.bg_shift > 0:
            bgR = ts.warp_h(bgR, np.full((H, W), -self.bg_shift, np.float32))
        self.bgL, self.bgR = bgL, bgR
        self.prev_g = None
        self.post = post
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="geometry") if overlap else None
        if post == "gpu":
            self.gp = GPUPost(ws, bgL, bgR, soft=soft, gf=gf, lam=lam, gate=gate, person_disp=self.person_disp)
            self.up_stream = torch.cuda.Stream()  # geometry results are uploaded from the worker thread

    # ------------------------------------------------------------------ stages
    def segment_eyes(self, L, R, record=True, gpu=False, pre=None):
        """Person prob / hard mask for both eyes at work resolution (numpy).

        Person-only inference evaluated directly on the work grid (seg.person), instead of
        the official full-res semantic + panoptic post-processing followed by a downscale.
        """
        tm = self.timer if record else Timer()
        out = []
        for eye in (L, R):
            p = [pre[len(out) // 2]] if pre is not None else None
            prob, hard = self.seg.person([eye], self.ws, timer=tm, return_numpy=not gpu, pre=p)[0]
            out += [prob, hard]
        return out  # pL, hL, pR, hR

    def geometry(self, L, R):
        tm = self.timer
        with tm("geo.downscale"):
            Ls, Rs = down(L, self.ws), down(R, self.ws)
            gL, gR = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        with tm("geo.disparity"):
            dL, dR = self.stereo(gL, gR)
        with tm("geo.flow"):
            if self.prev_g is not None:
                fl = (self.flow.backward_with_validity(self.prev_g[0], gL),
                      self.flow.backward_with_validity(self.prev_g[1], gR))
            else:
                fl = ((None, None, None), (None, None, None))
        self.prev_g = (gL, gR)
        return gL, gR, dL, dR, fl

    def fill_disparity(self, dL, dR, pL, hL, pR, hR):
        with self.timer("alpha.disp_fill"):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            bandL = cv2.dilate((hL | (pL > 0.3)).astype(np.uint8), k).astype(bool)
            bandR = cv2.dilate((hR | (pR > 0.3)).astype(np.uint8), k).astype(bool)
            coreL = cv2.erode(hL.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            self.person_disp = nanpct(dL[coreL], 50, self.person_disp)
            d_def = nanpct(dL, 50, 0.0)
            return (ts.fill_disparity(dL, bandL, self.person_disp, d_def),
                    ts.fill_disparity(dR, bandR, self.person_disp, d_def))

    def alpha(self, pL, pR, gL, gR, dLf, dRf, fl):
        tm = self.timer
        (fbL, vL, phL), (fbR, vR, phR) = fl
        with tm("alpha.soft"):
            sL, sR = ts.soft_alpha(pL, *self.soft), ts.soft_alpha(pR, *self.soft)
        with tm("alpha.guided"):
            guL, guR = ts.guided_filter(gL.astype(np.float32) / 255.0, sL, *self.gf), \
                ts.guided_filter(gR.astype(np.float32) / 255.0, sR, *self.gf)
        with tm("alpha.ema"):
            tL = self.ema["L"](guL, fbL, vL, phL)
            tR = self.ema["R"](guR, fbR, vR, phR)
        with tm("alpha.fusion"):
            aL = 0.5 * (tL + ts.warp_r2l(tR, dLf))
            aR = 0.5 * (tR + ts.warp_l2r(tL, dRf))
        return aL, aR

    def compose(self, L, R, aL, aR):
        with self.timer("compose"):
            W, H = self.W, self.H
            AL = cv2.resize(aL, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            AR = cv2.resize(aR, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            cl = L.astype(np.float32) * AL + self.bgL.astype(np.float32) * (1.0 - AL)
            cr = R.astype(np.float32) * AR + self.bgR.astype(np.float32) * (1.0 - AR)
            return np.hstack([cl, cr]).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ frame
    def process(self, frame):
        return self.process_gpu(frame) if self.post == "gpu" else self.process_cpu(frame)

    def process_gpu(self, frame):
        """CPU: decode-side downscale, SGBM disparity, raw DIS flows. GPU: segmentation, flow
        validity, disparity fill, soft / guided / EMA / fusion, 1080p compositing."""
        tm, gp = self.timer, self.gp
        L, R = split_sbs(frame)
        geo = self.pool.submit(self.cpu_geometry, L, R, True) if self.pool else None
        with tm("gpu.upload_frame"):
            frame_t = gp.pin.up("frame", frame)
        # model-input resize on the GPU from the uploaded frame (OpenCV INTER_AREA weights);
        # takes 7 ms of CPU work off the critical path and out of the OpenCV thread pool
        with tm("seg.preprocess"):
            pre = [self.seg.preprocess_gpu(frame_t[:, :self.W]), self.seg.preprocess_gpu(frame_t[:, self.W:])]
        pL, hL, pR, hR = self.segment_eyes(L, R, gpu=True, pre=pre)
        with tm("geo.wait"):  # time the GPU path waits for the CPU geometry (0 when fully hidden)
            gL, gR, dL, dR, flows, (dev, done) = geo.result() if geo else self.cpu_geometry(L, R, True)
        with tm("gpu.post", gpu=True):
            cur = torch.cuda.current_stream()
            cur.wait_event(done)  # uploads ran on up_stream while the GPU was segmenting
            for x in dev:
                if x is not None:
                    x.record_stream(cur)
            gLt, gRt, dLt, dRt, fbL, ffL, fbR, ffR = dev
            gLt, gRt = gLt.float(), gRt.float()
            if flows is not None:
                vL, phL = gp.flow_validity(self.prev_gt[0], gLt, fbL, ffL)
                vR, phR = gp.flow_validity(self.prev_gt[1], gRt, fbR, ffR)
            else:
                fbL = fbR = vL = vR = phL = phR = None
            self.prev_gt = (gLt, gRt)
            dLf, dRf = gp.fill_disparity(dLt, dRt, pL, hL, pR, hR)
            sL, sR = gp.soft_alpha(pL), gp.soft_alpha(pR)
            tL = gp.ema("L", gp.guided(gLt / 255.0, sL), fbL, vL, phL)
            tR = gp.ema("R", gp.guided(gRt / 255.0, sR), fbR, vR, phR)
            aL, aR = gp.fusion(tL, tR, dLf, dRf)
        with tm("gpu.compose+d2h", gpu=True):
            out = gp.pin.down("out", gp.compose(frame_t, aL, aR))
        return {"out": out, "aL": aL, "aR": aR, "pL": pL, "pR": pR, "hL": hL, "hR": hR}

    def cpu_geometry(self, L, R, upload=False):
        """Gray work-res images, SGBM disparities and raw DIS flows (fw + bw) for both eyes.
        upload: also copy them to the GPU on up_stream (pinned, async) right away, so the copies
        overlap the segmentation instead of following it; returns (tensors, done event) extra."""
        tm = self.timer
        with tm("geo.downscale"):
            gL = cv2.cvtColor(down(L, self.ws), cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(down(R, self.ws), cv2.COLOR_BGR2GRAY)
        with tm("geo.disparity"):
            dL, dR = self.stereo(gL, gR)
        with tm("geo.flow"):
            flows = None
            if self.prev_g is not None:
                pgL, pgR = self.prev_g
                flows = (self.flow(gL, pgL), self.flow(pgL, gL), self.flow(gR, pgR), self.flow(pgR, gR))
        self.prev_g = (gL, gR)
        if not upload:
            return gL, gR, dL, dR, flows
        with tm("geo.upload"), torch.cuda.stream(self.up_stream):
            up = self.gp.pin.up
            fl = [up(k, f) for k, f in zip(("fbL", "ffL", "fbR", "ffR"), flows)] if flows is not None else [None] * 4
            dev = [up("gL", gL), up("gR", gR), up("dL", dL), up("dR", dR), *fl]
            done = torch.cuda.Event()
            done.record(self.up_stream)
        return gL, gR, dL, dR, flows, (dev, done)

    def process_cpu(self, frame):
        L, R = split_sbs(frame)
        pL, hL, pR, hR = self.segment_eyes(L, R)
        gL, gR, dL, dR, fl = self.geometry(L, R)
        dLf, dRf = self.fill_disparity(dL, dR, pL, hL, pR, hR)
        aL, aR = self.alpha(pL, pR, gL, gR, dLf, dRf, fl)
        out = self.compose(L, R, aL, aR)
        return {"out": out, "aL": aL, "aR": aR, "pL": pL, "pR": pR, "hL": hL, "hR": hR}
