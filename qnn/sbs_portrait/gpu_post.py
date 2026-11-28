"""GPU (torch) version of the pipeline's alpha post-processing and SBS compositing.

Mirrors the CPU reference (temporal_stereo.py / pipeline.py, OpenCV + numpy) operation by
operation, at the same work resolution:
  flow validity  fw-bw consistency + photometric check   (cv2.remap -> grid_sample, border)
  disparity fill elliptic dilation (31x31 at 540 lines) (exact: per-row distance to the nearest foreground
                 pixel vs the ellipse's half width at each row offset), 9x9 erosion
                 (min-pool), nanquantile
  soft alpha     smoothstep
  guided filter  9x9 box filter = reflect pad + separable avg_pool (cv2 BORDER_REFLECT_101),
                 the four first-pass and two second-pass box filters batched
  temporal EMA   motion-compensated, confidence + motion gate
  stereo fusion  horizontal warps through the filled disparities
  compose        bilinear alpha upsample to the eye size (cv2 INTER_LINEAR), blend with the
                 GPU-resident background, truncate to uint8 like numpy astype
Everything stays on the GPU; only the per-frame inputs (gray images, flows, disparities,
the full-res frame) are uploaded through reusable pinned buffers and the composite is
downloaded once.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import warp_box as wb


class Pinned:
    """Reusable pinned host buffers + async H2D/D2H (avoids pageable copies and re-pinning)."""

    def __init__(self, device):
        self.dev, self.buf = device, {}

    def _get(self, key, arr):
        b = self.buf.get(key)
        if b is None or b.shape != arr.shape or b.dtype != torch.from_numpy(arr[:0]).dtype:
            b = self.buf[key] = torch.empty(arr.shape, dtype=torch.from_numpy(arr[:0]).dtype, pin_memory=True)
        return b

    def up(self, key, arr):
        b = self._get(key, np.ascontiguousarray(arr))
        b.numpy()[...] = arr
        return b.to(self.dev, non_blocking=True)

    def down(self, key, t):
        b = self.buf.get(key)
        if b is None or tuple(b.shape) != tuple(t.shape) or b.dtype != t.dtype:
            b = self.buf[key] = torch.empty(tuple(t.shape), dtype=t.dtype, pin_memory=True)
        b.copy_(t, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return b.numpy()


class GPUPost:
    def __init__(self, ws, bgL, bgR, soft=(0.4, 0.6), gf=(4, 1e-4), lam=0.7, gate=0.5, sigma=10.0,
                 person_disp=0.0, ellipse=31, core=9, device="cuda", kernels="triton"):
        """ellipse / core: person band dilation (OpenCV ellipse) and core erosion sizes (odd).
        kernels: warps and box filters as "triton" kernels (warp_box.py) or "torch" ops."""
        assert kernels in ("triton", "torch")
        self.kernels = kernels
        self.dev = torch.device(device)
        self.ws = ws
        w, h = ws
        self.soft, self.gf, self.lam, self.gate, self.sigma = soft, gf, lam, gate, sigma
        gy, gx = torch.meshgrid(torch.arange(h, device=self.dev, dtype=torch.float32),
                                torch.arange(w, device=self.dev, dtype=torch.float32), indexing="ij")
        self.gx, self.gy = gx, gy
        self.core = core
        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ellipse, ellipse))
        c = ellipse // 2
        hw = []  # half width of each row (rows are runs centred on the middle column)
        for row in ell:
            on = np.flatnonzero(row)
            h = len(on) // 2
            assert len(on) == 0 or list(on) == list(range(c - h, c + 1 + h)), "ellipse rows must be centred runs"
            hw.append(h if len(on) else -1)
        self.ell_hw = torch.tensor(hw, dtype=torch.float32, device=self.dev)
        # background as CHW uint8 on the GPU (uploaded once)
        self.bg = torch.from_numpy(np.ascontiguousarray(np.stack([bgL, bgR]).transpose(0, 3, 1, 2))).to(self.dev)
        self.prev = {"L": None, "R": None}
        self.person_disp = torch.tensor(float(person_disp), device=self.dev)
        self.pin = Pinned(self.dev)

    # ------------------------------------------------------------ warps
    def _sample(self, img, x, y):
        """img (C,H,W) float sampled at pixel coords (x,y) (H,W); bilinear, border replicate."""
        h, w = img.shape[-2:]
        grid = torch.stack([x * (2.0 / (w - 1)) - 1.0, y * (2.0 / (h - 1)) - 1.0], -1)[None]
        return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]

    def warp_flow(self, img, flow):
        """out(x) = img(x + flow(x)); img (C,H,W), flow (H,W,2)."""
        if self.kernels == "triton":
            return wb.warp("triton", img.contiguous(), flow[..., 0], flow[..., 1])
        return self._sample(img, self.gx + flow[..., 0], self.gy + flow[..., 1])

    def warp_h(self, img, dx):
        if self.kernels == "triton":
            return wb.warp("triton", img.contiguous(), dx.contiguous())
        return self._sample(img, self.gx + dx, self.gy)

    # ------------------------------------------------------------ stages
    def flow_validity(self, g_prev, g_cur, fb, ff):
        """Same rule as temporal_stereo.FlowEstimator.backward_with_validity (fb: cur->prev)."""
        ff_at = self.warp_flow(ff.permute(2, 0, 1), fb).permute(1, 2, 0)
        s = fb + ff_at
        mag = (fb ** 2).sum(-1) + (ff_at ** 2).sum(-1)
        fwbw_ok = (s ** 2).sum(-1) < 0.01 * mag + 0.5
        prev_w = self.warp_flow(g_prev[None], fb)[0].round()  # cv2.remap on uint8 rounds
        photo = (prev_w - g_cur).abs()
        return fwbw_ok & (photo < 20), photo

    def fill_disparity(self, dL, dR, pL, hL, pR, hR):
        def band(h, p):
            return self.dilate_ellipse(h | (p > 0.3))

        bandL, bandR = band(hL, pL), band(hR, pR)
        core = -F.max_pool2d(-hL.float()[None, None], self.core, stride=1, padding=self.core // 2)[0, 0] > 0.5
        nan = torch.tensor(float("nan"), device=self.dev)
        pd = torch.nanquantile(torch.where(core, dL, nan).flatten(), 0.5)
        self.person_disp = torch.where(torch.isnan(pd), self.person_disp, pd)
        dd = torch.nanquantile(dL.flatten(), 0.5)
        dd = torch.where(torch.isnan(dd), torch.zeros_like(dd), dd)

        def fill(d, b):
            inv = torch.isnan(d)
            return torch.where(inv & b, self.person_disp, torch.where(inv, dd, d))

        return fill(dL, bandL), fill(dR, bandR)

    def dilate_ellipse(self, m):
        """Binary dilation of m (H, W bool) by the 31x31 OpenCV ellipse, = conv2d(m, ellipse) > 0
        exactly, without the 961-tap convolution: dh = distance from each pixel to the nearest
        foreground pixel of its row (two cumulative scans); a pixel is in the dilation iff for
        some row offset dy, dh(y + dy, x) <= half_width(dy)."""
        H, W = m.shape
        idx = torch.arange(W, device=m.device).expand(H, W)
        left = torch.where(m, idx, -4 * W).cummax(1).values                    # last fg <= x
        right = torch.where(m, idx, 5 * W).flip(1).cummin(1).values.flip(1)    # first fg >= x
        dh = torch.minimum(idx - left, right - idx).float()
        r = (len(self.ell_hw) - 1) // 2
        win = F.pad(dh[None, None], (0, 0, r, r), value=float("inf"))[0, 0].unfold(0, 2 * r + 1, 1)  # H, W, 31
        return (win <= self.ell_hw).any(-1)

    def soft_alpha(self, p):
        lo, hi = self.soft
        t = ((p - lo) / (hi - lo)).clamp(0.0, 1.0)
        return t * t * (3 - 2 * t)

    def box(self, x):
        """9x9 mean of each (H, W) plane of x (N, H, W): reflect-101 pad, then a vertical and a
        horizontal 1-D average (separable: 2 x 9 instead of 81 taps)."""
        r = self.gf[0]
        if self.kernels == "triton":
            return wb.box("triton", x.contiguous(), r)
        y = F.pad(x[:, None], (r, r, r, r), mode="reflect")
        y = F.avg_pool2d(y, (2 * r + 1, 1), stride=1)
        return F.avg_pool2d(y, (1, 2 * r + 1), stride=1)[:, 0]

    def guided(self, guide, p):
        eps = self.gf[1]
        mi, mp, mgp, mgg = self.box(torch.stack([guide, p, guide * p, guide * guide]))
        cov = mgp - mi * mp
        var = mgg - mi * mi
        a = cov / (var + eps)
        b = mp - a * mi
        ma, mb = self.box(torch.stack([a, b]))
        return (ma * guide + mb).clamp(0.0, 1.0)

    def ema(self, eye, p, fb, valid, photo):
        prev = self.prev[eye]
        if prev is None or fb is None:
            self.prev[eye] = p
            return p
        prev_w = self.warp_flow(prev[None], fb)[0]
        w = self.lam * valid.float() * torch.exp(-photo / self.sigma)
        if self.gate:
            w = w * (1.0 - (p - prev_w).abs() / self.gate).clamp(0.0, 1.0)
        a = w * prev_w + (1.0 - w) * p
        self.prev[eye] = a
        return a

    def fusion(self, tL, tR, dLf, dRf):
        aL = 0.5 * (tL + self.warp_h(tR[None], -dLf)[0])
        aR = 0.5 * (tR + self.warp_h(tL[None], dRf)[0])
        return aL, aR

    def compose(self, frame_t, aL, aR):
        """frame_t: (H, 2W, 3) uint8 on GPU; returns (H, 2W, 3) uint8 on GPU."""
        H, W2 = frame_t.shape[:2]
        W = W2 // 2
        img = frame_t.permute(2, 0, 1).reshape(3, H, 2, W).permute(2, 0, 1, 3).float()  # 2,3,H,W
        a = F.interpolate(torch.stack([aL, aR])[:, None], size=(H, W), mode="bilinear", align_corners=False)
        out = img * a + self.bg.float() * (1.0 - a)  # 2,3,H,W
        out = out.clamp(0, 255).to(torch.uint8)  # truncation, like numpy astype
        return out.permute(1, 2, 0, 3).reshape(3, H, W2).permute(1, 2, 0)
