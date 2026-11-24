"""Shape-generic forwards for exporting the three model stages as dynamic-shape TensorRT engines.

The static engines (one input size, 1080p eyes) were exported with every shape-dependent value
frozen into the graph. For eyes of any size (VGA .. 1080p) the model input varies, so the ONNX
graphs must compute those values from the input shape instead. The upstream code freezes them
in five places; each is replaced here by a numerically identical, traceable form (in tracing,
`x.shape[i]` is a 0-dim tensor, Python int arithmetic on it is recorded, `int()` / numpy is not):

  Swin BasicLayer      Hp = int(np.ceil(H / ws)) * ws and the shifted-window attention mask
                       built by slice assignment  ->  integer ceil and a region-id arithmetic
                       mask ((y >= Hp-ws) + (y >= Hp-s)) * 3 + (x >= Wp-ws) + (x >= Wp-s)
  Swin block           `if pad_r > 0 or pad_b > 0: crop` is decided at trace time (1080p input
                       needs no padding at stage 1, other sizes do)  ->  always pad / crop
                       (zero-size pad and full crop are no-ops)
  sine position        cumsum over a ones mask (wrong in TensorRT, see trt_pixel_decoder) and,
  embedding            when baked as constants, valid for one size only  ->  arange-based
                       closed form ((i + 1) / (n + eps) * scale), even channels sin, odd cos
  MSDA encoder         spatial shapes via torch.as_tensor(python list), reference points via
                       linspace(0.5, H - 0.5, H)  ->  Shape ops and (arange + 0.5) / H
  decoder              per-size cache keyed by the traced size tuple  ->  keyed by level

All replacements are checked for equality with the upstream forward in eager mode
(check_* functions, used by prof/dyn_spike.py).
"""
import contextlib
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- position embedding
class DynSinePE(torch.nn.Module):
    """PositionEmbeddingSine (normalize=True, no padding mask) computed from the input size only."""

    def __init__(self, pe):
        super().__init__()
        f = pe.num_pos_feats
        dim_t = torch.arange(f, dtype=torch.float32)
        self.register_buffer("dim_t", pe.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / f), persistent=False)
        self.register_buffer("even", (torch.arange(f) % 2) == 0, persistent=False)
        assert pe.normalize, "only the normalized variant is used by Mask2Former"
        self.scale = pe.scale

    def _axis(self, n, device):
        # cumsum of a ones mask is 1..n; upstream divides by the last value (+ eps)
        e = (torch.arange(n, device=device, dtype=torch.float32) + 1.0) / (n + 1e-6) * self.scale
        p = e[:, None] / self.dim_t.to(device)
        return torch.where(self.even.to(device), p.sin(), p.cos()).t()  # F, n

    def forward(self, x, mask=None):
        assert mask is None
        h, w = x.shape[-2:]
        py = self._axis(h, x.device)[:, :, None].expand(-1, -1, w)
        px = self._axis(w, x.device)[:, None, :].expand(-1, h, -1)
        return torch.cat([py, px], 0)[None].expand(x.shape[0], -1, -1, -1)


# ---------------------------------------------------------------- Swin
def _basic_layer_forward(self, x, H, W):
    ws, s = self.window_size, self.shift_size
    Hp = (H + ws - 1) // ws * ws
    Wp = (W + ws - 1) // ws * ws
    ry = torch.arange(Hp, device=x.device)
    rx = torch.arange(Wp, device=x.device)
    hy = (ry >= Hp - ws).long() + (ry >= Hp - s).long()
    hx = (rx >= Wp - ws).long() + (rx >= Wp - s).long()
    img_mask = (hy[:, None] * 3 + hx[None, :]).float()[None, :, :, None]  # 1 Hp Wp 1, region ids 0..8
    from mask2former.modeling.backbone.swin import window_partition

    mw = window_partition(img_mask, ws).view(-1, ws * ws)
    attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    for blk in self.blocks:
        blk.H, blk.W = H, W
        x = blk(x, attn_mask)
    if self.downsample is not None:
        return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
    return x, H, W, x, H, W


def _swin_block_forward(self, x, mask_matrix):
    B, L, C = x.shape
    H, W = self.H, self.W
    ws = self.window_size
    shortcut = x
    x = self.norm1(x).view(B, H, W, C)
    pad_r = (ws - W % ws) % ws
    pad_b = (ws - H % ws) % ws
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    Hp, Wp = x.shape[1], x.shape[2]
    if self.shift_size > 0:
        x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        attn_mask = mask_matrix
    else:
        attn_mask = None
    xw = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
    aw = self.attn(xw, mask=attn_mask)
    x = aw.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
    if self.shift_size > 0:
        x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    x = x[:, :H, :W, :].reshape(B, H * W, C)  # always crop (no-op without padding)
    x = shortcut + self.drop_path(x)
    return x + self.drop_path(self.mlp(self.norm2(x)))


@contextlib.contextmanager
def swin_dynamic():
    """Temporarily replace the Swin BasicLayer / block forwards by the shape-generic versions."""
    from mask2former.modeling.backbone import swin

    saved = swin.BasicLayer.forward, swin.SwinTransformerBlock.forward
    swin.BasicLayer.forward, swin.SwinTransformerBlock.forward = _basic_layer_forward, _swin_block_forward
    try:
        yield
    finally:
        swin.BasicLayer.forward, swin.SwinTransformerBlock.forward = saved


# ---------------------------------------------------------------- pixel decoder
def pixel_decoder_forward(pd, feats, pe=None):
    """MSDeformAttnPixelDecoder.forward_features with shape-generic encoder inputs (no padding mask:
    batch-1 inputs have none; valid ratios are 1). pe: position-embedding module (default upstream)."""
    pe = pe or pd.pe_layer
    srcs, pos = [], []
    for idx, f in enumerate(pd.transformer_in_features[::-1]):
        x = feats[f]
        srcs.append(pd.input_proj[idx](x))
        pos.append(pe(x))
    tr = pd.transformer
    nl = len(srcs)
    hw = [s.shape[-2:] for s in srcs]
    src_flat = torch.cat([s.flatten(2).transpose(1, 2) for s in srcs], 1)
    pos_flat = torch.cat([p.flatten(2).transpose(1, 2) + tr.level_embed[i].view(1, 1, -1) for i, p in enumerate(pos)], 1)
    shapes = torch.stack([torch._shape_as_tensor(s)[2:] for s in srcs]).to(src_flat.device)  # (L, 2) int64
    sizes = shapes.prod(1)
    start = torch.cat((sizes.new_zeros((1,)), sizes.cumsum(0)[:-1]))
    refs = []
    for h, w in hw:
        ry, rx = torch.meshgrid((torch.arange(h, device=src_flat.device, dtype=torch.float32) + 0.5) / h,
                                (torch.arange(w, device=src_flat.device, dtype=torch.float32) + 0.5) / w, indexing="ij")
        refs.append(torch.stack((rx.reshape(-1), ry.reshape(-1)), -1))
    ref = torch.cat(refs, 0)[None, :, None, :].expand(src_flat.shape[0], -1, nl, -1)
    y = src_flat
    for layer in tr.encoder.layers:
        y = layer(y, pos_flat, ref, shapes, start, None)
    bs = y.shape[0]
    out, off = [], 0
    for h, w in hw:
        out.append(y[:, off:off + h * w].transpose(1, 2).reshape(bs, -1, h, w))
        off = off + h * w
    for idx, f in enumerate(pd.in_features[:pd.num_fpn_levels][::-1]):
        x = feats[f]
        cur = pd.lateral_convs[idx](x)
        y = cur + F.interpolate(out[-1], size=cur.shape[-2:], mode="bilinear", align_corners=False)
        out.append(pd.output_convs[idx](y))
    return pd.mask_features(out[-1]), out[:pd.maskformer_num_feature_levels]


# ---------------------------------------------------------------- transformer decoder
def decoder_forward(self, x, mask_features, pe=None, mask_fill=None, batch1=True):
    """seg._decoder_inference_forward with the per-size cache keyed by level (the traced size
    tuples are new objects at every call) and an optional position-embedding module / finite
    additive mask fill (TensorRT export); batch1=False: per-head masks for batch > 1."""
    pe = pe or self.pe_layer
    src, pos, sizes = [], [], []
    for i in range(self.num_feature_levels):
        sizes.append(x[i].shape[-2:])
        pos.append(pe(x[i], None).flatten(2).permute(2, 0, 1))
        src.append((self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))
    bs = src[0].shape[1]
    query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
    mf_at = {}

    def attn_mask_for(out, level):
        if level not in mf_at:
            mf_at[level] = F.interpolate(mask_features, size=sizes[level], mode="bilinear", align_corners=False)
        emb = self.mask_embed(self.decoder_norm(out).transpose(0, 1))
        m = (torch.einsum("bqc,bchw->bqhw", emb, mf_at[level]) < 0).flatten(2)
        m = m & ~m.all(-1, keepdim=True)
        if mask_fill is not None:
            m = torch.zeros(m.shape, dtype=emb.dtype, device=m.device).masked_fill(m, mask_fill)
        if batch1:
            return m[0]  # batch 1: a 2-D mask broadcasts over the heads
        return m.unsqueeze(1).expand(-1, self.num_heads, -1, -1).flatten(0, 1)

    attn_mask = attn_mask_for(output, 0)
    for i in range(self.num_layers):
        level = i % self.num_feature_levels
        output = self.transformer_cross_attention_layers[i](
            output, src[level], memory_mask=attn_mask, memory_key_padding_mask=None,
            pos=pos[level], query_pos=query_embed)
        output = self.transformer_self_attention_layers[i](
            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
        output = self.transformer_ffn_layers[i](output)
        if i < self.num_layers - 1:
            attn_mask = attn_mask_for(output, (i + 1) % self.num_feature_levels)
    dec = self.decoder_norm(output).transpose(0, 1)
    return {"pred_logits": self.class_embed(dec),
            "pred_masks": torch.einsum("bqc,bchw->bqhw", self.mask_embed(dec), mask_features)}


def ceil_to(x, d):
    return int(math.ceil(x / d) * d)
