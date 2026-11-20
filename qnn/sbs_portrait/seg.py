"""Mask2Former person segmenter for SBS eyes (official panoptic + semantic inference)."""
import contextlib
import os
import sys

import cv2
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

import detectron2.data.transforms as T  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

from mask2former import add_maskformer2_config  # noqa: E402

PERSON_ID = 0  # COCO "person" (contiguous id 0 in COCO panoptic metadata)
DEFAULT_CONFIG = os.path.join(
    REPO, "configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
DEFAULT_WEIGHTS = os.path.join(REPO, "weights/model_final_f07440.pkl")


class PersonSegmenter:
    def __init__(self, config_file=DEFAULT_CONFIG, weights=DEFAULT_WEIGHTS, opts=(), precision="fp32",
                 backbone="torch", eye_hw=(1080, 1920), trt_plan=None, fused=True, fast_decoder=False,
                 compile_decoders=False, pixel_decoder="torch"):
        """precision: "fp32" (reference) or "fp16" -- FP16 autocast for the whole model (the pixel
        decoder's FP32 forcing is lifted, only the MSDeformAttn kernel stays FP32; TF32 matmul is
        enabled for the remaining FP32 GEMMs) and FP16 person post-processing in person().
        backbone: "torch" or "trt" -- static-shape TensorRT FP16 engine for the backbone, used by
        person() (built on first use for eyes of size eye_hw; the torch backbone weights are
        then released). With "trt" only person() is available.
        fused: person() post-processing with the fused Triton kernel (person_triton.py) instead
        of the grid_sample / matmul / argmax torch ops.
        fast_decoder: inference-only transformer-decoder forward without redundant work
        (see _decoder_inference_forward); outputs only pred_logits / pred_masks.
        compile_decoders: torch.compile (Inductor, max-autotune without CUDA graphs, static
        shapes) the pixel decoder and the transformer decoder; first use compiles for ~2 min,
        the Inductor cache under weights/inductor_cache makes later starts fast.
        pixel_decoder: "torch" or "trt" (needs backbone="trt") -- the whole MSDeformAttn pixel
        decoder as one static-shape TensorRT FP16 engine with the built-in MSDA plugin
        (trt_pixel_decoder.py); only the transformer decoder is then compiled."""
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(list(opts) + [
            "MODEL.WEIGHTS", weights,
            "MODEL.MASK_FORMER.TEST.SEMANTIC_ON", "True",
            "MODEL.MASK_FORMER.TEST.PANOPTIC_ON", "True",
            "MODEL.MASK_FORMER.TEST.INSTANCE_ON", "False",
        ])
        cfg.freeze()
        self.cfg = cfg
        self.model = build_model(cfg).eval()
        DetectionCheckpointer(self.model).load(weights)
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST] * 2, cfg.INPUT.MAX_SIZE_TEST)
        self.fmt = cfg.INPUT.FORMAT
        assert precision in ("fp32", "fp16")
        self.fp16 = precision == "fp16"
        if self.fp16:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _enable_pixel_decoder_fp16(self.model)
        self.stream = torch.cuda.Stream()
        self.fused = fused
        if fast_decoder:
            import types

            pred = self.model.sem_seg_head.predictor
            pred.forward = types.MethodType(_decoder_inference_forward, pred)
        if compile_decoders:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(REPO, "weights", "inductor_cache"))
            head = self.model.sem_seg_head
            opts = dict(mode="max-autotune-no-cudagraphs", dynamic=False)
            if pixel_decoder != "trt":
                head.pixel_decoder.forward_features = torch.compile(head.pixel_decoder.forward_features, **opts)
            head.predictor.forward = torch.compile(head.predictor.forward, **opts)
        self.trt = self.trt_pd = None
        if backbone == "trt":
            from trt_backbone import TRTBackbone, build_engine
            nh, nw = T.ResizeShortestEdge.get_output_shape(eye_hw[0], eye_hw[1], cfg.INPUT.MIN_SIZE_TEST,
                                                           cfg.INPUT.MAX_SIZE_TEST)
            d = self.model.size_divisibility
            shape = (1, 3, -(-nh // d) * d, -(-nw // d) * d)
            plan = trt_plan or os.path.join(REPO, "weights", f"swinl_backbone_fp16_{shape[2]}x{shape[3]}.plan")
            if not os.path.exists(plan):
                build_engine(self.model.backbone, plan, shape)
            self.trt = TRTBackbone(plan)
            self.model.backbone.to("meta")  # release the torch backbone weights
            if pixel_decoder == "trt":
                from trt_pixel_decoder import TRTPixelDecoder
                from trt_pixel_decoder import build_engine as build_pd
                pd_plan = os.path.join(REPO, "weights", f"pixel_decoder_fp16_{shape[2]}x{shape[3]}.plan")
                if not os.path.exists(pd_plan):
                    build_pd(self.model.sem_seg_head.pixel_decoder,
                             {k: tuple(v.shape) for k, v in self.trt.out.items()}, pd_plan)
                self.trt_pd = TRTPixelDecoder(pd_plan)
                self.model.sem_seg_head.pixel_decoder.to("meta")
            torch.cuda.empty_cache()

    def preprocess(self, im):
        """BGR HxWx3 uint8 -> model input CHW uint8 tensor (shortest edge = MIN_SIZE_TEST).

        Same output size as detectron2's ResizeShortestEdge, but resized with OpenCV
        INTER_AREA instead of PIL BILINEAR (closest match to PIL's antialiased downscale:
        mean |diff| ~0.28 gray level, ~7x faster).
        """
        h, w = im.shape[:2]
        nh, nw = T.ResizeShortestEdge.get_output_shape(h, w, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        x = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        if self.fmt == "RGB":
            x = x[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def _person_fused(self, masks, cls, hw, padded_hw, out_hw):
        from person_triton import fused_person

        m = self.model
        (h, w), (Hp, Wp), (oh, ow) = hw, padded_hw, out_hw
        h4, w4 = masks.shape[-2:]
        num_classes = cls.shape[-1] - 1
        scores, labels = cls.max(-1)
        keep = labels.ne(num_classes) & (scores > m.object_mask_threshold)
        prob, ids, own_at, own_area = fused_person(
            masks, cls[:, PERSON_ID], 1.0 - cls[:, -1], scores * keep, (oh, ow),
            (h / oh) * (h4 / Hp), (w / ow) * (w4 / Wp))
        return prob, _panoptic_person_rule(m, scores, keep, labels, ids, own_at, own_area)

    @torch.no_grad()
    def __call__(self, imgs_bgr, timer=None):
        """imgs_bgr: list of HxWx3 uint8 BGR (processed as one batch).

        Returns list of (prob, hard): prob = person share of the semantic score
        (float32 HxW in [0,1]); hard = union of panoptic person segments (bool).
        timer: optional pipeline.Timer, records seg.preprocess / seg.model / seg.person / seg.d2h.
        """
        tm = timer or _NullTimer()
        with tm("seg.preprocess"):
            inputs = []
            for im in imgs_bgr:
                inputs.append({"image": self.preprocess(im), "height": im.shape[0], "width": im.shape[1]})
        with tm("seg.model", gpu=True):
            outs = self.model(inputs)
        res = []
        for o in outs:
            with tm("seg.person", gpu=True):
                sem = o["sem_seg"].float()
                prob = sem[PERSON_ID] / sem.sum(0).clamp_min(1e-6)
                pan, info = o["panoptic_seg"]
                ids = [s["id"] for s in info if s.get("isthing") and s["category_id"] == PERSON_ID]
                if ids:
                    hard = torch.isin(pan, torch.tensor(ids, device=pan.device))
                else:
                    hard = torch.zeros_like(pan, dtype=torch.bool)
            with tm("seg.d2h", gpu=True):
                res.append((prob.cpu().numpy().astype(np.float32), hard.cpu().numpy()))
        return res


    @torch.no_grad()
    def person(self, imgs_bgr, out_size, timer=None, return_numpy=True, pre=None):
        """Person-only inference evaluated directly at out_size=(w, h) (e.g. 960x540).

        Same model forward as __call__, but replaces the official post-processing
        (upsample 200 masks to the padded input, crop, upsample to full res, 133-class
        semantic inference, per-segment panoptic loop with .item() syncs) by:
          * one grid_sample of the 1/4-res mask logits straight onto the output grid
            (sampling positions follow the official crop + resize geometry; for an
            output pixel the position is the centre of its full-res footprint),
          * one sigmoid, and the person share of the semantic score as a (2, Q) x (Q, HW)
            matmul: sum_q p(person|q) m_q / sum_q (1 - p(void|q)) m_q,
          * a vectorised panoptic rule for the person mask with the same thresholds
            (keep: not void and score > OBJECT_MASK_THRESHOLD; segment = argmax of
            score * mask; valid if area / own_area >= OVERLAP_THRESHOLD), no host syncs.
        Returns list of (prob float32 HxW, hard bool HxW) at out_size: numpy arrays, or CUDA
        tensors ready on the caller's current stream when return_numpy=False.
        pre: already preprocessed inputs (list of self.preprocess(img)); imgs_bgr is then unused.
        """
        tm = timer or _NullTimer()
        if pre is None:
            with tm("seg.preprocess"):
                pre = [self.preprocess(im) for im in imgs_bgr]
        xs = pre
        sizes = [tuple(x.shape[-2:]) for x in xs]
        pdt = torch.float16 if self.fp16 else torch.float32
        caller = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            res = self._person(xs, sizes, out_size, tm, pdt, return_numpy)
        if not return_numpy:  # hand the GPU results over to the caller's stream
            caller.wait_stream(self.stream)
            for t in (t for r in res for t in r):
                t.record_stream(caller)
        return res

    def _person(self, xs, sizes, out_size, tm, pdt, return_numpy):
        import torch.nn.functional as F
        from detectron2.structures import ImageList

        m = self.model
        with tm("seg.model", gpu=True):
            ims = [(x.to(m.device, non_blocking=True).float() - m.pixel_mean) / m.pixel_std for x in xs]
            il = ImageList.from_tensors(ims, m.size_divisibility)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                feats = self.trt(il.tensor) if self.trt is not None else m.backbone(il.tensor)
                if self.trt_pd is not None:
                    mf, ms = self.trt_pd(feats)
                    out = m.sem_seg_head.predictor(ms, mf, None)
                else:
                    out = m.sem_seg_head(feats)
        cls_all = out["pred_logits"].float().softmax(-1)  # B, Q, C+1 (tiny, keep FP32)
        masks = out["pred_masks"].to(pdt)                   # B, Q, Hp/4, Wp/4
        num_classes = cls_all.shape[-1] - 1
        Hp, Wp = il.tensor.shape[-2:]
        ow, oh = out_size
        res = []
        for b, (h, w) in enumerate(sizes):
            if self.fused:
                with tm("seg.person", gpu=True):
                    prob, hard = self._person_fused(masks[b], cls_all[b], (h, w), (Hp, Wp), (oh, ow))
                if return_numpy:
                    with tm("seg.d2h", gpu=True):
                        res.append((prob.cpu().numpy(), hard.cpu().numpy()))
                else:
                    res.append((prob, hard))
                continue
            with tm("seg.person", gpu=True):
                h4, w4 = masks.shape[-2:]
                # continuous mask-grid coordinate of each output pixel centre (official geometry:
                # mask grid -> padded input (x4) -> crop [0:h, 0:w] -> resize to the eye size)
                ys = (torch.arange(oh, device=masks.device, dtype=torch.float32) + 0.5) * (h / oh) * (h4 / Hp)
                xs_ = (torch.arange(ow, device=masks.device, dtype=torch.float32) + 0.5) * (w / ow) * (w4 / Wp)
                gy, gx = torch.meshgrid(ys / h4 * 2 - 1, xs_ / w4 * 2 - 1, indexing="ij")
                grid = torch.stack([gx, gy], -1)[None].to(pdt)
                mk = F.grid_sample(masks[b:b + 1], grid, mode="bilinear", align_corners=False)[0].sigmoid_()  # Q,oh,ow
                cls = cls_all[b]
                wts = torch.stack([cls[:, PERSON_ID], 1.0 - cls[:, -1]]).to(pdt)  # 2, Q
                num, den = (wts @ mk.flatten(1)).float().view(2, oh, ow)
                prob = num / den.clamp_min(1e-6)
                # panoptic person mask, vectorised
                scores, labels = cls.max(-1)
                keep = labels.ne(num_classes) & (scores > m.object_mask_threshold)
                ids = ((scores * keep).to(pdt)[:, None, None] * mk).argmax(0)       # oh, ow
                own = mk >= 0.5
                own_at = own.gather(0, ids[None])[0]
                flat = ids.flatten()
                q = mk.shape[0]
                area = torch.zeros(q, device=mk.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
                seg_area = torch.zeros(q, device=mk.device).scatter_add_(0, flat, own_at.flatten().float())
                own_area = own.flatten(1).sum(1).float()
                valid = keep & (area > 0) & (own_area > 0) & (seg_area > 0) & \
                    (area >= m.overlap_threshold * own_area) & labels.eq(PERSON_ID)
                hard = valid[ids] & own_at
            if return_numpy:
                with tm("seg.d2h", gpu=True):
                    res.append((prob.cpu().numpy(), hard.cpu().numpy()))
            else:
                res.append((prob, hard))
        return res


def _pixel_decoder_fp16_forward_features(self, features):
    """MSDeformAttnPixelDecoder.forward_features without the FP32 forcing (the upstream method
    is decorated with autocast(enabled=False) and casts its inputs with .float()). Under FP16
    autocast the projections, encoder FFN/linears and FPN convs run in FP16; the MSDeformAttn
    CUDA kernel itself stays FP32 through _FP32MSDeformAttnFunction."""
    srcs, pos = [], []
    for idx, f in enumerate(self.transformer_in_features[::-1]):
        x = features[f]
        srcs.append(self.input_proj[idx](x))
        pos.append(self.pe_layer(x))
    y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
    bs = y.shape[0]
    split = [None] * self.transformer_num_feature_levels
    for i in range(self.transformer_num_feature_levels):
        if i < self.transformer_num_feature_levels - 1:
            split[i] = level_start_index[i + 1] - level_start_index[i]
        else:
            split[i] = y.shape[1] - level_start_index[i]
    y = torch.split(y, split, dim=1)
    out, multi_scale_features, num_cur_levels = [], [], 0
    for i, z in enumerate(y):
        out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
    for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        x = features[f]
        cur_fpn = self.lateral_convs[idx](x)
        y = cur_fpn + torch.nn.functional.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear",
                                                      align_corners=False)
        out.append(self.output_convs[idx](y))
    for o in out:
        if num_cur_levels < self.maskformer_num_feature_levels:
            multi_scale_features.append(o)
            num_cur_levels += 1
    return self.mask_features(out[-1]), out[0], multi_scale_features


def _enable_pixel_decoder_fp16(model):
    import types

    from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn as msda_mod

    if not getattr(msda_mod, "_fp32_wrapped", False):
        orig = msda_mod.MSDeformAttnFunction

        class _FP32MSDeformAttnFunction:
            """The MSDA CUDA kernel only dispatches float/double; a half input would raise and the
            module would silently fall back to the slow pure-PyTorch path. Run it in FP32."""

            @staticmethod
            def apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                out = orig.apply(value.float(), shapes, level_start_index, sampling_locations.float(),
                                 attention_weights.float(), im2col_step)
                return out.to(value.dtype)

        msda_mod.MSDeformAttnFunction = _FP32MSDeformAttnFunction
        msda_mod._fp32_wrapped = True
    pd = model.sem_seg_head.pixel_decoder
    pd.forward_features = types.MethodType(_pixel_decoder_fp16_forward_features, pd)


def _decoder_inference_forward(self, x, mask_features, mask=None):
    """MultiScaleMaskedTransformerDecoder.forward for inference, mathematically equal outputs.

    Upstream computes, after every layer, full-resolution mask logits (einsum over the 1/4-res
    mask features), bilinearly downsamples them to the next attention size, applies sigmoid,
    repeats the boolean mask for every head and fixes fully-masked rows with a data-dependent
    index (host sync); it also builds class logits / masks for every layer (aux outputs).
    Here: bilinear resampling commutes with the per-query linear mask head, so mask_features
    is resampled once per attention size and the logits are computed there directly;
    sigmoid(x) < 0.5 <=> x < 0; with batch 1 a 2-D mask broadcasts over heads; the row fix is
    a mask expression; class logits / full-res masks are produced for the last layer only.
    """
    import torch.nn.functional as F

    src, pos, size_list = [], [], []
    for i in range(self.num_feature_levels):
        size_list.append(tuple(x[i].shape[-2:]))
        pos.append(self.pe_layer(x[i], None).flatten(2).permute(2, 0, 1))
        src.append((self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))
    bs = src[0].shape[1]
    query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
    mf_at = {}

    def attn_mask_for(out, size):
        if size not in mf_at:
            mf_at[size] = F.interpolate(mask_features, size=size, mode="bilinear", align_corners=False)
        emb = self.mask_embed(self.decoder_norm(out).transpose(0, 1))
        m = (torch.einsum("bqc,bchw->bqhw", emb, mf_at[size]) < 0).flatten(2)  # True = not attended
        m = m & ~m.all(-1, keepdim=True)  # fully masked rows attend everywhere (upstream rule)
        if bs == 1:
            return m[0]
        return m.unsqueeze(1).expand(-1, self.num_heads, -1, -1).flatten(0, 1)

    attn_mask = attn_mask_for(output, size_list[0])
    for i in range(self.num_layers):
        level = i % self.num_feature_levels
        output = self.transformer_cross_attention_layers[i](
            output, src[level], memory_mask=attn_mask, memory_key_padding_mask=None,
            pos=pos[level], query_pos=query_embed)
        output = self.transformer_self_attention_layers[i](
            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
        output = self.transformer_ffn_layers[i](output)
        if i < self.num_layers - 1:
            attn_mask = attn_mask_for(output, size_list[(i + 1) % self.num_feature_levels])
    dec = self.decoder_norm(output).transpose(0, 1)
    return {"pred_logits": self.class_embed(dec),
            "pred_masks": torch.einsum("bqc,bchw->bqhw", self.mask_embed(dec), mask_features)}


def _panoptic_person_rule(model, scores, keep, labels, ids, own_at, own_area):
    """Vectorised Mask2Former panoptic rule restricted to the person class (no host syncs)."""
    q = scores.shape[0]
    flat = ids.flatten()
    area = torch.zeros(q, device=ids.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    seg_area = torch.zeros(q, device=ids.device).scatter_add_(0, flat, own_at.flatten().float())
    valid = keep & (area > 0) & (own_area > 0) & (seg_area > 0) & \
        (area >= model.overlap_threshold * own_area) & labels.eq(PERSON_ID)
    return valid[ids] & own_at


class _NullTimer:
    def __call__(self, key, gpu=False):
        return contextlib.nullcontext()
