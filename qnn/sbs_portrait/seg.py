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


class _NullTimer:
    def __call__(self, key, gpu=False):
        return contextlib.nullcontext()
