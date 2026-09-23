"""Learned sky segmentation (owner decision A4.5 #4).

UperNet + ConvNeXt-tiny trained on ADE20K (``openmmlab/upernet-convnext-tiny``,
MIT licence; ADE20K's images are research-use, see the commercial checklist in
PROJECT_PLAN A4.5). The model runs on an aspect-preserving proxy (long side
``edge``, padded to a multiple of 32); the "sky" class probability is
upsampled and its edges re-snapped to the full-resolution image with a
guided filter, as for the heuristic mask.

``sky_mask(img)`` falls back to the heuristic ``regional.sky_probability`` if
the model cannot be loaded (offline, no cache), so callers never fail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from photostyle.regional import _luma, guided_filter, sky_probability

MODEL_ID = "openmmlab/upernet-convnext-tiny"
SKY_CLASS = 2  # ADE20K "sky"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_model = None
_failed = False


def _load(cache_dir: str = "data/cache/hf_models"):
    global _model, _failed
    if _model is None and not _failed:
        try:
            from transformers import UperNetForSemanticSegmentation

            _model = UperNetForSemanticSegmentation.from_pretrained(MODEL_ID, cache_dir=cache_dir).eval()
        except Exception:  # offline / not cached: use the heuristic
            _failed = True
    return _model


@torch.no_grad()
def sky_mask(img: torch.Tensor, edge: int = 384, use_model: bool = True) -> torch.Tensor:
    """(B, 3, H, W) sRGB in [0, 1] -> (B, 1, H, W) sky probability in [0, 1]."""
    m = _load() if use_model else None
    if m is None:
        return sky_probability(img)
    B, _, H, W = img.shape
    s = min(1.0, edge / max(H, W))
    h, w = max(32, round(H * s)), max(32, round(W * s))
    x = F.interpolate(img, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    ph, pw = (-h) % 32, (-w) % 32
    x = F.pad((x - _MEAN) / _STD, (0, pw, 0, ph), mode="replicate")
    p = m(pixel_values=x).logits.softmax(1)[:, SKY_CLASS:SKY_CLASS + 1, :h, :w]
    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
    return guided_filter(_luma(img), p, r=max(2, min(H, W) // 128), eps=1e-3).clamp(0, 1)
