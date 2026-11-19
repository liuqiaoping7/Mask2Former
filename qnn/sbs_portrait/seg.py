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
    def __init__(self, config_file=DEFAULT_CONFIG, weights=DEFAULT_WEIGHTS, opts=()):
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
    def person(self, imgs_bgr, out_size, timer=None):
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
        Returns list of (prob float32 HxW, hard bool HxW) numpy arrays at out_size.
        """
        import torch.nn.functional as F
        from detectron2.structures import ImageList

        tm = timer or _NullTimer()
        m = self.model
        with tm("seg.preprocess"):
            xs = [self.preprocess(im) for im in imgs_bgr]
            sizes = [tuple(x.shape[-2:]) for x in xs]
        with tm("seg.model", gpu=True):
            ims = [(x.to(m.device, non_blocking=True).float() - m.pixel_mean) / m.pixel_std for x in xs]
            il = ImageList.from_tensors(ims, m.size_divisibility)
            out = m.sem_seg_head(m.backbone(il.tensor))
        cls_all = out["pred_logits"].float().softmax(-1)  # B, Q, C+1
        masks = out["pred_masks"].float()                   # B, Q, Hp/4, Wp/4
        num_classes = cls_all.shape[-1] - 1
        Hp, Wp = il.tensor.shape[-2:]
        ow, oh = out_size
        res = []
        for b, (h, w) in enumerate(sizes):
            with tm("seg.person", gpu=True):
                h4, w4 = masks.shape[-2:]
                # continuous mask-grid coordinate of each output pixel centre (official geometry:
                # mask grid -> padded input (x4) -> crop [0:h, 0:w] -> resize to the eye size)
                ys = (torch.arange(oh, device=masks.device, dtype=torch.float32) + 0.5) * (h / oh) * (h4 / Hp)
                xs_ = (torch.arange(ow, device=masks.device, dtype=torch.float32) + 0.5) * (w / ow) * (w4 / Wp)
                gy, gx = torch.meshgrid(ys / h4 * 2 - 1, xs_ / w4 * 2 - 1, indexing="ij")
                grid = torch.stack([gx, gy], -1)[None]
                mk = F.grid_sample(masks[b:b + 1], grid, mode="bilinear", align_corners=False)[0].sigmoid()  # Q,oh,ow
                cls = cls_all[b]
                wts = torch.stack([cls[:, PERSON_ID], 1.0 - cls[:, -1]])          # 2, Q
                num, den = (wts @ mk.flatten(1)).view(2, oh, ow)
                prob = num / den.clamp_min(1e-6)
                # panoptic person mask, vectorised
                scores, labels = cls.max(-1)
                keep = labels.ne(num_classes) & (scores > m.object_mask_threshold)
                ids = ((scores * keep)[:, None, None] * mk).argmax(0)               # oh, ow
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
            with tm("seg.d2h", gpu=True):
                res.append((prob.cpu().numpy(), hard.cpu().numpy()))
        return res


class _NullTimer:
    def __call__(self, key, gpu=False):
        return contextlib.nullcontext()
