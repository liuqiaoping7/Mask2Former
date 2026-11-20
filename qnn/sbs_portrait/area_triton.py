"""Fused model-input preprocessing on the GPU (Triton): OpenCV INTER_AREA downscale of a uint8
BGR HWC image (any row stride, e.g. one eye of the SBS frame) + cvRound + BGR->RGB + mean/std
normalisation, written as float32 CHW. One pass over the output; each output pixel reads its
<= K x K source taps directly (no K-times-larger intermediates as with gather + sum).

Weights / indices come from seg._area_taps (cv::computeResizeAreaTab); the horizontal weighted
sum is formed per source row first and then weighted vertically, the order cv::resizeArea uses.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _area_norm_kernel(SRC, ROW, IX, WX, IY, WY, MEAN, STD, OUT, NH, NW,
                      K: tl.constexpr, FLIP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    inb = offs < NH * NW
    oy = offs // NW
    ox = offs % NW
    for c in tl.static_range(3):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for ky in tl.static_range(K):
            sy = tl.load(IY + oy * K + ky, mask=inb, other=0)
            wy = tl.load(WY + oy * K + ky, mask=inb, other=0.0)
            row = tl.zeros([BLOCK], dtype=tl.float32)
            for kx in tl.static_range(K):
                sx = tl.load(IX + ox * K + kx, mask=inb, other=0)
                wx = tl.load(WX + ox * K + kx, mask=inb, other=0.0)
                v = tl.load(SRC + sy * ROW + sx * 3 + c, mask=inb, other=0).to(tl.float32)
                row += wx * v
            acc += wy * row
        acc = tl.minimum(tl.maximum(libdevice.rint(acc), 0.0), 255.0)  # saturate_cast<uchar>
        oc = 2 - c if FLIP else c
        acc = (acc - tl.load(MEAN + oc)) / tl.load(STD + oc)
        tl.store(OUT + oc * NH * NW + offs, acc, mask=inb)


def area_resize_normalize(eye, taps_x, taps_y, mean, std, rgb, block=256):
    """eye: (H, W, 3) uint8 CUDA, unit channel / pixel strides (row stride free); taps_*: (idx
    (n, K) long, w (n, K) float32) from _area_taps; mean / std: (3,) float32 in model channel
    order. Returns (3, nh, nw) float32."""
    assert eye.dtype == torch.uint8 and eye.stride(2) == 1 and eye.stride(1) == 3
    (ix, wx), (iy, wy) = taps_x, taps_y
    nh, nw, k = iy.shape[0], ix.shape[0], max(ix.shape[1], iy.shape[1])
    if ix.shape[1] != k or iy.shape[1] != k:
        raise ValueError("pad both tap tables to the same K")
    out = torch.empty(3, nh, nw, device=eye.device, dtype=torch.float32)
    grid = (triton.cdiv(nh * nw, block),)
    _area_norm_kernel[grid](eye, eye.stride(0), ix.to(torch.int32), wx, iy.to(torch.int32), wy,
                            mean, std, out, nh, nw, K=k, FLIP=rgb, BLOCK=block)
    return out
