"""Ultralytics YOLO backbone wrapper for person ReID.

Loads an Ultralytics YOLO model (detection / segmentation / classification),
drops its task head, and exposes a clean feature extractor that returns a single
global feature vector ``(B, in_planes)``. This lets a lightweight YOLO act as a
drop-in backbone for the ReID pipeline.

The ultralytics import is intentionally done lazily inside ``__init__`` so that
existing (ViT / ResNet) experiments do not require ultralytics to be installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOBackbone(nn.Module):
    """Wraps an Ultralytics YOLO model and exposes ``forward(x) -> (B, in_planes)``.

    Two feature-extraction modes, auto-detected from the model task (overridable
    via ``cfg.MODEL.YOLO.TASK``):

    * ``classify``: run the backbone up to the final ``Classify`` head, then apply
      only its ``conv``+``pool`` to obtain the pre-logit feature vector. This
      deliberately bypasses the head's ``linear`` and the eval-mode ``softmax``.
    * ``detect`` / ``segment`` / ``pose``: run the routing-aware backbone+neck,
      collect the multi-scale maps the head would consume (P3/P4/P5), and fuse
      them into one vector according to ``cfg.MODEL.YOLO.FUSION``.
    """

    def __init__(self, cfg):
        super(YOLOBackbone, self).__init__()
        ycfg = cfg.MODEL.YOLO
        weights = ycfg.WEIGHTS
        if not weights:
            raise ValueError(
                "cfg.MODEL.YOLO.WEIGHTS is empty. Set it to an ultralytics model "
                "(e.g. 'yolo11n.pt', 'yolo11n-seg.pt', 'yolo11n-cls.pt')."
            )

        # Lazy import: keeps ultralytics out of the import path for non-YOLO runs.
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for MODEL.NAME='yolo'. "
                "Install it with `pip install ultralytics`."
            ) from e

        yolo = YOLO(weights)
        self.net = yolo.model  # inner nn.Module (BaseModel subclass)

        # Resolve task ('auto' -> inferred from the model / head type).
        task = ycfg.TASK
        if task == 'auto':
            task = getattr(self.net, 'task', None)
            if task is None:
                head_name = type(self.net.model[-1]).__name__.lower()
                task = 'classify' if 'classify' in head_name else 'detect'
        self.task = task

        self.fusion = ycfg.FUSION
        self.feature_layers = list(ycfg.FEATURE_LAYERS) if len(ycfg.FEATURE_LAYERS) else None

        head = self.net.model[-1]
        if self.task == 'classify':
            # Keep the Classify head: we reuse its conv+pool to form the feature.
            self.head = head
            self._classify_from = head.f
        else:
            # Detection-style head: record which feature maps it consumes, then
            # drop the head's parameters (replace with Identity, preserving the
            # Sequential length / layer indices used by the routing logic).
            head_from = head.f if isinstance(head.f, (list, tuple)) else [head.f]
            self._feature_source = self.feature_layers if self.feature_layers is not None else list(head_from)
            self.net.model[-1] = nn.Identity()

        if ycfg.FREEZE_BACKBONE:
            for p in self.net.parameters():
                p.requires_grad_(False)

        # Infer the output feature dimension with a one-time dry forward.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, int(cfg.INPUT.SIZE_TRAIN[0]), int(cfg.INPUT.SIZE_TRAIN[1]))
            self.in_planes = self.forward(dummy).shape[1]
        if was_training:
            self.train()

    def _run_backbone(self, x):
        """Routing-aware forward over every layer except the head.

        Mirrors ultralytics ``BaseModel._predict_once`` but saves *all* layer
        outputs so any layer index can be gathered afterwards. Returns
        ``(last_output, all_outputs)``.
        """
        y = []
        for m in self.net.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return x, y

    def _fuse(self, maps):
        if self.fusion == 'deepest':
            return F.adaptive_avg_pool2d(maps[-1], 1).flatten(1)
        if self.fusion == 'upsample_concat':
            size = maps[0].shape[-2:]  # highest-resolution map comes first
            ups = [F.interpolate(m, size=size, mode='bilinear', align_corners=False) for m in maps]
            return F.adaptive_avg_pool2d(torch.cat(ups, dim=1), 1).flatten(1)
        # default: 'gap_concat'
        pooled = [F.adaptive_avg_pool2d(m, 1).flatten(1) for m in maps]
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        last, y = self._run_backbone(x)
        if self.task == 'classify':
            xin = last if self._classify_from == -1 else y[self._classify_from]
            return self.head.pool(self.head.conv(xin)).flatten(1)
        maps = [last if j == -1 else y[j] for j in self._feature_source]
        return self._fuse(maps)
