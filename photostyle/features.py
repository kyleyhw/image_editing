"""Frozen image descriptors: ResNet-18 global features + per-channel CDFs.

Fixes the Phase 6 confounds (PROJECT_PLAN Phase 7):

* ImageNet normalisation before the backbone (previously missing);
* the backbone is frozen and in eval mode, so BatchNorm statistics never
  drift with tiny batches;
* aspect ratio is preserved (the backbone ends in adaptive average pooling,
  so any input size works);
* features are computed once per image and cached, so the head trains on a
  CPU in seconds.

Descriptor layout (1280 floats): 3 x 256 per-channel CDF values, then 512
ResNet-18 pooled features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FEATURE_DIM = 3 * 256 + 512


def channel_cdfs(img: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Hard per-channel CDF of a (3, H, W) image in [0, 1] -> (3 * bins,)."""
    out = []
    for c in range(3):
        h = torch.histc(img[c].flatten(), bins=bins, min=0.0, max=1.0)
        out.append(torch.cumsum(h / h.sum().clamp_min(1.0), 0))
    return torch.cat(out)


class FeatureExtractor:
    """Frozen ResNet-18 + CDF descriptor with an optional on-disk cache."""

    def __init__(self, cache_dir: Path | None = None, short_side: int = 224):
        weights = tvm.ResNet18_Weights.DEFAULT
        net = tvm.resnet18(weights=weights)
        self.backbone = torch.nn.Sequential(*list(net.children())[:-1]).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.short_side = short_side
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (3, H, W) in [0, 1] -> (FEATURE_DIM,)."""
        _, h, w = img.shape
        s = self.short_side / min(h, w)
        x = F.interpolate(img[None], size=(max(1, round(h * s)), max(1, round(w * s))),
                          mode="bilinear", align_corners=False, antialias=True)
        spatial = self.backbone((x - IMAGENET_MEAN) / IMAGENET_STD).flatten()
        return torch.cat([channel_cdfs(img), spatial])

    def cached(self, img: torch.Tensor, key: str) -> torch.Tensor:
        """Like __call__, but memoised on disk under a hash of ``key``."""
        if self.cache_dir is None:
            return self(img)
        path = self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".npy")
        if path.exists():
            return torch.from_numpy(np.load(path))
        f = self(img)
        np.save(path, f.numpy())
        return f
