"""Optical flow / stereo disparity utilities and temporal-stereo consistency metrics.

All post-processing and metrics run at "work" resolution (half of 1080p by default).
Disparity convention: x_left = x_right + d  (d > 0 for objects in front of the
zero-parallax plane).
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
class Grid:
    _cache = {}

    @classmethod
    def get(cls, h, w):
        key = (h, w)
        if key not in cls._cache:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            cls._cache[key] = (gx, gy)
        return cls._cache[key]


def warp_flow(img, flow, interp=cv2.INTER_LINEAR):
    """Backward warp: out(x) = img(x + flow(x))."""
    h, w = flow.shape[:2]
    gx, gy = Grid.get(h, w)
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], interp, borderMode=cv2.BORDER_REPLICATE)


def warp_h(img, dx, interp=cv2.INTER_LINEAR):
    """Horizontal backward warp: out(x) = img(x + dx(x))."""
    h, w = dx.shape[:2]
    gx, gy = Grid.get(h, w)
    return cv2.remap(img, gx + dx.astype(np.float32), gy, interp, borderMode=cv2.BORDER_REPLICATE)


def warp_r2l(a_r, d_l):
    """Right-eye map sampled into left-eye coordinates with left disparity."""
    return warp_h(a_r, -d_l)


# ----------------------------------------------------------------------------
# optical flow
# ----------------------------------------------------------------------------
class FlowEstimator:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, gray_a, gray_b):
        """Flow a->b such that a(x) ~ b(x + flow(x))."""
        return self.dis.calc(gray_a, gray_b, None)

    def backward_with_validity(self, gray_prev, gray_cur):
        """Flow cur->prev (to warp prev into cur) and a validity mask.

        Validity = forward-backward consistency AND photometric consistency.
        """
        fb = self(gray_cur, gray_prev)  # cur -> prev
        ff = self(gray_prev, gray_cur)  # prev -> cur
        ff_at = warp_flow(ff, fb)
        s = fb + ff_at
        mag = (fb ** 2).sum(-1) + (ff_at ** 2).sum(-1)
        fwbw_ok = (s ** 2).sum(-1) < 0.01 * mag + 0.5
        prev_w = warp_flow(gray_prev, fb)
        photo = np.abs(prev_w.astype(np.float32) - gray_cur.astype(np.float32))
        valid = fwbw_ok & (photo < 20)
        return fb, valid, photo


# ----------------------------------------------------------------------------
# stereo disparity
# ----------------------------------------------------------------------------
def estimate_disparity_range(gray_l, gray_r, margin=8):
    """Estimate (min_disp, num_disp, vertical_offset, n_matches) from ORB matches on a rectified pair."""
    orb = cv2.ORB_create(4000)
    kl, dl = orb.detectAndCompute(gray_l, None)
    kr, dr = orb.detectAndCompute(gray_r, None)
    if dl is None or dr is None:
        return -16, 64, 0.0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    dx, dy = [], []
    for x in m:
        pl, pr = kl[x.queryIdx].pt, kr[x.trainIdx].pt
        dx.append(pl[0] - pr[0])
        dy.append(pl[1] - pr[1])
    dx, dy = np.array(dx), np.array(dy)
    ok = np.abs(dy - np.median(dy)) < 2
    dx = dx[ok]
    lo, hi = np.percentile(dx, 1) - margin, np.percentile(dx, 99) + margin
    min_d = int(np.floor(lo))
    num_d = int(np.ceil((hi - min_d) / 16.0) * 16)
    return min_d, max(num_d, 16), float(np.median(dy)), int(ok.sum())


class StereoEstimator:
    def __init__(self, min_disp, num_disp, block=5):
        self.min_disp = min_disp
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def _disp(self, a, b):
        d = self.sgbm.compute(a, b).astype(np.float32) / 16.0
        d[d < self.min_disp] = np.nan
        return d

    def __call__(self, gray_l, gray_r):
        """Return left/right disparity maps (NaN = invalid) after L/R check."""
        dl = self._disp(gray_l, gray_r)
        dr = np.fliplr(self._disp(np.fliplr(gray_r).copy(), np.fliplr(gray_l).copy())).copy()
        dr_at_l = warp_h(np.nan_to_num(dr, nan=-1e4), -np.nan_to_num(dl), cv2.INTER_NEAREST)
        dl_at_r = warp_h(np.nan_to_num(dl, nan=-1e4), np.nan_to_num(dr), cv2.INTER_NEAREST)
        dl[~(np.abs(dl - dr_at_l) <= 1.5)] = np.nan
        dr[~(np.abs(dr - dl_at_r) <= 1.5)] = np.nan
        return dl, dr


# ----------------------------------------------------------------------------
# alpha post-processing
# ----------------------------------------------------------------------------
def soft_alpha(prob, lo=0.3, hi=0.7):
    """Map class probability to alpha with a smoothstep between lo and hi."""
    t = np.clip((prob - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def iou(a, b, region=None):
    if region is not None:
        a, b = a & region, b & region
    u = (a | b).sum()
    return float((a & b).sum() / u) if u > 0 else 1.0


def boundary_band(mask, width):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


class MetricAccumulator:
    """Per-config accumulator of temporal / stereo consistency and fidelity metrics.

    tiou       motion-compensated temporal IoU (flow-valid pixels)
    tflk_band  boundary temporal flicker: mean |a_t - warp(a_{t-1})| in the boundary band
    siou       L/R IoU after warping right alpha to the left eye (L/R-checked disparity)
    sflk_band  boundary L/R difference: mean |a_L - warp_R->L(a_R)| in the boundary band
    fid_iou    IoU with the per-frame raw panoptic mask
    ghost      lag / trailing / halo: sum |a - target| over confident pixels / person area
    cwarp      motion-compensated RGB difference of consecutive composites (0-255)
    """

    def __init__(self):
        self.v = {k: [] for k in ("tiou", "tflk_band", "siou", "sflk_band", "fid_iou", "ghost", "cwarp")}

    def temporal(self, a_cur, a_prev, flow_b, valid, band):
        a_prev_w = warp_flow(a_prev, flow_b)
        bc, bp = a_cur > 0.5, a_prev_w > 0.5
        self.v["tiou"].append(iou(bc, bp, valid))
        r = valid & band
        if r.any():
            self.v["tflk_band"].append(float(np.abs(a_cur - a_prev_w)[r].mean()))

    def stereo(self, a_l, a_r, d_l_strict, band_l):
        valid = ~np.isnan(d_l_strict)
        a_r2l = warp_r2l(a_r, np.nan_to_num(d_l_strict))
        self.v["siou"].append(iou(a_l > 0.5, a_r2l > 0.5, valid))
        r = valid & band_l
        if r.any():
            self.v["sflk_band"].append(float(np.abs(a_l - a_r2l)[r].mean()))

    def fidelity(self, a, ref_hard):
        self.v["fid_iou"].append(iou(a > 0.5, ref_hard))

    def ghost(self, a, prob):
        """Current frame confident (p<0.1 -> 0, p>0.9 -> 1) but alpha disagrees.

        Over-smoothing in time lowers flicker but raises this, so it must be
        read together with the stability metrics.
        """
        conf = (prob < 0.1) | (prob > 0.9)
        err = np.abs(a - (prob > 0.5).astype(np.float32))[conf].sum()
        self.v["ghost"].append(float(err / max((prob > 0.5).sum(), 1)))

    def composite_warp(self, c_cur, c_prev, flow_b, valid):
        cw = warp_flow(c_prev, flow_b)
        self.v["cwarp"].append(float(np.abs(c_cur - cw).mean(-1)[valid].mean()))

    def summary(self):
        out = {}
        for k, v in self.v.items():
            if v:
                a = np.array(v)
                out[k] = float(a.mean())
                if k in ("tiou", "siou"):
                    out[k + "_p5"] = float(np.percentile(a, 5))
        return out
