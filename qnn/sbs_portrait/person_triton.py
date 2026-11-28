"""Fused person post-processing kernel (Triton).

Replaces, for one image, the chain
    grid_sample(Q mask logits -> out grid) -> sigmoid -> (2,Q)x(Q,HW) matmul
    -> argmax_q(score_q * keep_q * mask_q) -> (mask >= 0.5) -> per-query own-area counts
which materialises and re-reads a Q x H x W tensor (200 x 960 x 540) several times
(memory bound), by one pass per output pixel that reads only the small 1/4-res logits
(Q x h4 x w4, L2-resident) and keeps everything else in registers. Computes in FP32.

Exactly the same sampling as F.grid_sample(mode="bilinear", padding_mode="zeros",
align_corners=False) at pixel coordinate (i + 0.5) * scale - 0.5; own = (logit >= 0),
identical to sigmoid(logit) >= 0.5; argmax ties resolve to the first query like torch.argmax.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _person_kernel(M, WP, WD, SK, PROB, IDS, OWNAT, OWNCNT,
                   Q, H4, W4, OH, OW, SY, SX, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    inb = offs < OH * OW
    i = offs // OW
    j = offs % OW
    iy = (i.to(tl.float32) + 0.5) * SY - 0.5
    ix = (j.to(tl.float32) + 0.5) * SX - 0.5
    y0f = tl.floor(iy)
    x0f = tl.floor(ix)
    wy1 = iy - y0f
    wx1 = ix - x0f
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    y0 = y0f.to(tl.int32)
    x0 = x0f.to(tl.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    vy0 = (y0 >= 0) & (y0 < H4)
    vy1 = (y1 >= 0) & (y1 < H4)
    vx0 = (x0 >= 0) & (x0 < W4)
    vx1 = (x1 >= 0) & (x1 < W4)
    m00 = inb & vy0 & vx0
    m01 = inb & vy0 & vx1
    m10 = inb & vy1 & vx0
    m11 = inb & vy1 & vx1
    o00 = y0 * W4 + x0
    o01 = y0 * W4 + x1
    o10 = y1 * W4 + x0
    o11 = y1 * W4 + x1

    num = tl.zeros([BLOCK], dtype=tl.float32)
    den = tl.zeros([BLOCK], dtype=tl.float32)
    best = tl.full([BLOCK], -1.0, dtype=tl.float32)
    bid = tl.zeros([BLOCK], dtype=tl.int32)
    bown = tl.zeros([BLOCK], dtype=tl.int32)
    plane = H4 * W4
    for q in range(Q):
        base = M + q * plane
        v00 = tl.load(base + o00, mask=m00, other=0.0).to(tl.float32)
        v01 = tl.load(base + o01, mask=m01, other=0.0).to(tl.float32)
        v10 = tl.load(base + o10, mask=m10, other=0.0).to(tl.float32)
        v11 = tl.load(base + o11, mask=m11, other=0.0).to(tl.float32)
        s = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11)
        sig = 1.0 / (1.0 + tl.exp(-s))
        num += tl.load(WP + q) * sig
        den += tl.load(WD + q) * sig
        own = s >= 0.0
        tl.store(OWNCNT + pid * Q + q, tl.sum((own & inb).to(tl.int32), axis=0))
        prod = tl.load(SK + q) * sig
        upd = prod > best
        best = tl.where(upd, prod, best)
        bid = tl.where(upd, q, bid)
        bown = tl.where(upd, own.to(tl.int32), bown)
    tl.store(PROB + offs, num / tl.maximum(den, 1e-6), mask=inb)
    tl.store(IDS + offs, bid, mask=inb)
    tl.store(OWNAT + offs, bown.to(tl.int8), mask=inb)


def fused_person(masks, w_person, w_den, score_keep, out_hw, scale_y, scale_x, block=256):
    """masks: (Q, h4, w4) logits (fp16/fp32, contiguous); w_person / w_den / score_keep: (Q,) fp32.

    Returns prob (OH, OW) fp32, ids (OH, OW) int64, own_at (OH, OW) bool, own_area (Q,) fp32.
    """
    q, h4, w4 = masks.shape
    oh, ow = out_hw
    dev = masks.device
    prob = torch.empty(oh, ow, device=dev, dtype=torch.float32)
    ids = torch.empty(oh, ow, device=dev, dtype=torch.int32)
    own_at = torch.empty(oh, ow, device=dev, dtype=torch.int8)
    grid = (triton.cdiv(oh * ow, block),)
    # per-block partial counts: a global atomic per query and block costs ~1/3 of the kernel (contention)
    part = torch.empty(grid[0], q, device=dev, dtype=torch.int32)
    _person_kernel[grid](masks.contiguous(), w_person.float().contiguous(), w_den.float().contiguous(),
                         score_keep.float().contiguous(), prob, ids, own_at, part,
                         q, h4, w4, oh, ow, float(scale_y), float(scale_x), BLOCK=block)
    return prob, ids.long(), own_at.bool(), part.sum(0).float()
