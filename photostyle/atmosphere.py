"""Depth-aware atmosphere and scene light (PROJECT_PLAN Phase 18, re-scoped by A2.4).

Parametric, editable operations that only re-weight existing pixels:

  haze        blend toward an atmospheric colour A with distance:
                  t = far ** falloff,  x' = x (1 - h t) + A h t        (h > 0 adds haze)
              and the inverse for h < 0 (dehaze): x' = (x - A |h| t) / (1 - |h| t)
  clarity     depth-weighted local contrast (unsharp mask), separately
              for near and far:  x' = x + (k_near n + k_far f) (x - blur(x))
  sky light   exposure / warmth / saturation on the sky mask (Phase 16)

Depth: Depth Anything V2 **Small** (Apache-2.0; the Base/Large/Giant
checkpoints are CC BY-NC 4.0 and excluded under A1.1), run on a <= 518 px
proxy and upsampled with a guided filter so edges follow the image. If the
model is unavailable, a transparent fallback is used: vertical position plus
the dark-channel haze prior (He et al. 2009).

All strengths default to 0 (identity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from photostyle.regional import RegionalRenderer, box_filter, guided_filter, sky_probability

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_model = None


def _load_model(cache_dir: str = "data/cache/hf_models"):
    global _model
    if _model is None:
        from transformers import AutoModelForDepthEstimation

        _model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, cache_dir=cache_dir).eval()
    return _model


@torch.no_grad()
def depth_map(img: torch.Tensor, use_model: bool = True) -> torch.Tensor:
    """(1, 3, H, W) in [0, 1] -> (1, 1, H, W) 'nearness' in [0, 1] (1 = near)."""
    _, _, H, W = img.shape
    y = (0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2])[:, None]
    near = None
    if use_model:
        try:
            m = _load_model()
            s = 518 / max(H, W)
            h, w = max(14, round(H * s / 14) * 14), max(14, round(W * s / 14) * 14)
            x = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            d = m(pixel_values=(x - mean) / std).predicted_depth[:, None]      # relative inverse depth
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            lo, hi = torch.quantile(d.flatten(), 0.02), torch.quantile(d.flatten(), 0.98)
            near = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        except Exception:  # model unavailable: fall back
            near = None
    if near is None:
        dark = box_filter(img.min(1, keepdim=True).values, 7)                  # dark channel ~ haze ~ distance
        pos = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)       # lower in frame ~ nearer
        near = (0.6 * pos + 0.4 * (1 - dark)).clamp(0, 1)
    return guided_filter(y, near, r=max(2, min(H, W) // 64), eps=1e-3).clamp(0, 1)


@dataclass
class SceneParams:
    haze: float = 0.0          # -1 (remove) .. +1 (add)
    haze_warmth: float = -0.3  # haze colour: negative = cool (Clean Cool default), positive = warm
    falloff: float = 1.5       # how quickly haze grows with distance
    clarity_near: float = 0.0  # -1 .. +1
    clarity_far: float = 0.0
    sky_exposure: float = 0.0  # stops
    sky_warmth: float = 0.0
    sky_saturation: float = 0.0

    def is_identity(self) -> bool:
        d = asdict(self)
        return all(abs(d[k]) < 1e-6 for k in ("haze", "clarity_near", "clarity_far", "sky_exposure",
                                              "sky_warmth", "sky_saturation"))


@torch.no_grad()
def apply_scene(img: torch.Tensor, p: SceneParams, near: torch.Tensor | None = None,
                sky: torch.Tensor | None = None) -> torch.Tensor:
    """img (1, 3, H, W) in [0, 1] -> edited image. ``near``/``sky`` may be precomputed."""
    if p.is_identity():
        return img
    if near is None:
        near = depth_map(img)
    far = 1 - near
    x = img
    if abs(p.haze) > 1e-6:
        # Atmospheric light: a bright (95th percentile) grey from the farthest pixels,
        # tinted by haze_warmth. Sky is left mostly alone: dehazing it only clips it
        # to white, and hazing it just greys it.
        w = far > torch.quantile(far.flatten(), 0.9)
        y = x.mean(1, keepdim=True)
        a = torch.quantile(y[w], 0.95) if w.any() else y.max()
        A = a.view(1, 1, 1, 1).expand(1, 3, 1, 1).clone()
        A[:, 0] += 0.04 * p.haze_warmth
        A[:, 2] -= 0.04 * p.haze_warmth
        s_ = sky if sky is not None else sky_probability(img)
        t = far ** p.falloff * (1 - (0.7 if p.haze > 0 else 1.0) * s_)
        if p.haze > 0:
            x = x * (1 - p.haze * t) + A * p.haze * t
        else:
            # Remove only the haze that is there: the dark channel over A estimates
            # the haze fraction (He et al. 2009), so haze-free dark terrain is kept
            # and the darkest channel cannot be pushed below zero.
            dark = box_filter(x.min(1, keepdim=True).values, 3)
            present = (dark / A.mean().clamp_min(1e-3)).clamp(0, 1)
            k = (-p.haze * t * present).clamp(max=0.85)
            x = (x - A * k) / (1 - k)
    if abs(p.clarity_near) > 1e-6 or abs(p.clarity_far) > 1e-6:
        r_ = max(2, min(x.shape[2], x.shape[3]) // 100)
        detail = x - box_filter(x, r_)
        x = x + (p.clarity_near * near + p.clarity_far * far) * 1.5 * detail
    if abs(p.sky_exposure) + abs(p.sky_warmth) + abs(p.sky_saturation) > 1e-6:
        s = sky if sky is not None else sky_probability(img)
        one = torch.ones(1)
        x = RegionalRenderer._adjust(x, s, one * p.sky_exposure, one * p.sky_warmth, one * p.sky_saturation)
    return x.clamp(0, 1)
