"""Regional edits for landscapes (PROJECT_PLAN Phase 16, amendment A2.3).

``RegionalRenderer`` = ``GlobalRenderer`` followed by three editable,
identity-at-zero regional stages, all predicted jointly by the head:

1. **Graduated filter** (7 params): a soft half-plane mask
       m(x, y) = sigmoid((n . p - offset) / feather),  n = (sin a, cos a)
   (a = 0 darkens/cools the top of the frame). Exposure (stops), warmth
   (red-blue balance), saturation and a green-magenta tint are applied
   inside the mask. This is the single most common landscape adjustment.
2. **Luminance range masks** (9 params): smooth shadow / midtone /
   highlight weights of luminance, each with an RGB offset: explicit, slider-
   editable split toning ("colour wheels").
3. **Sky mask** (3 params): per-image sky probability (``sky_probability``)
   with exposure, warmth and saturation offsets.

The sky probability is a transparent heuristic: blue or bright low-texture
pixels, weighted toward the top of the frame, refined with a guided filter
so the mask follows skylines and ridges without halos. It is a placeholder
for a pretrained segmenter. Candidate models must have a permissive licence
(many ADE20K checkpoints are non-commercial), a decision left to the owner.
Any callable (B, 3, H, W) -> (B, 1, H, W) in [0, 1] can be passed as
``sky_fn``.

Only the global stage can live in a ``.cube``; the regional stages are
exported as parameters (EditParams) plus masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from photostyle.render import GlobalRenderer


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2])[:, None]


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), k, stride=1)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int = 8, eps: float = 1e-3) -> torch.Tensor:
    """He et al. guided filter (grey guide). guide, src: (B, 1, H, W)."""
    mean_i, mean_p = box_filter(guide, r), box_filter(src, r)
    cov_ip = box_filter(guide * src, r) - mean_i * mean_p
    var_i = box_filter(guide * guide, r) - mean_i ** 2
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_filter(a, r) * guide + box_filter(b, r)


@torch.no_grad()
def sky_probability(img: torch.Tensor) -> torch.Tensor:
    """Heuristic sky probability (B, 1, H, W) in [0, 1]. Not differentiable (not needed)."""
    B, _, H, W = img.shape
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = _luma(img)
    blue = torch.sigmoid((b - torch.maximum(r, g) + 0.02) / 0.03) * torch.sigmoid((y - 0.25) / 0.05)
    mx, mn = img.max(1, keepdim=True).values, img.min(1, keepdim=True).values
    grey_bright = torch.sigmoid((y - 0.6) / 0.05) * torch.sigmoid((0.12 - (mx - mn)) / 0.03)
    gx = (y[..., :, 1:] - y[..., :, :-1]).abs()
    gy = (y[..., 1:, :] - y[..., :-1, :]).abs()
    tex = F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))
    smooth = torch.sigmoid((0.03 - box_filter(tex, 3)) / 0.01)
    pos = torch.linspace(1.0, 0.0, H).view(1, 1, H, 1).expand(B, 1, H, W) ** 0.7
    p = torch.clamp(torch.maximum(blue, grey_bright) * (0.5 + 0.5 * smooth) * (0.3 + 0.7 * pos), 0, 1)
    r_ = max(2, min(H, W) // 32)
    return guided_filter(y, p, r=r_, eps=1e-3).clamp(0, 1)


REGIONAL_LAYOUT = {"grad": 7, "tone": 9, "sky": 3}


class RegionalRenderer(nn.Module):
    def __init__(self, kind: str = "per_channel", knots: int = 9, sky_fn=sky_probability):
        super().__init__()
        self.base = GlobalRenderer(kind, knots)
        self.K = knots
        self.sky_fn = sky_fn
        self.layout = {**self.base.layout, **REGIONAL_LAYOUT}
        self.num_params = sum(self.layout.values())

    def unpack(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for k, n in self.layout.items():
            out[k] = p[:, i:i + n]
            i += n
        return out

    def split(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(global params, regional params)."""
        return p[:, : self.base.num_params], p[:, self.base.num_params:]

    @staticmethod
    def _adjust(x, mask, expo, warm, sat, tint=None):
        """Exposure (stops), warmth, saturation (and tint) applied inside ``mask``."""
        B = x.shape[0]
        x = x * torch.pow(2.0, expo.view(B, 1, 1, 1) * mask)
        shift = torch.stack([warm, torch.zeros_like(warm), -warm], 1).view(B, 3, 1, 1) * 0.1
        if tint is not None:
            shift = shift + torch.stack([tint, -tint, tint], 1).view(B, 3, 1, 1) * 0.05
        x = x + shift * mask
        y = _luma(x)
        return y + (x - y) * (1 + sat.view(B, 1, 1, 1) * mask)

    def grad_mask(self, q: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = q.shape[0]
        ang = 0.6 * torch.tanh(q[:, 0])                      # about +-34 degrees
        off = 0.9 * torch.tanh(q[:, 1])
        feather = 0.08 + 0.5 * F.softplus(q[:, 2] + 0.5)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        # distance "above" the line: positive toward the top of the frame at angle 0
        d = -(torch.sin(ang).view(B, 1, 1) * xx + torch.cos(ang).view(B, 1, 1) * yy) - off.view(B, 1, 1)
        return torch.sigmoid(d / feather.view(B, 1, 1))[:, None]

    def tone_masks(self, x: torch.Tensor) -> torch.Tensor:
        y = _luma(x)
        s = torch.sigmoid((0.3 - y) / 0.07)
        h = torch.sigmoid((y - 0.7) / 0.07)
        return torch.cat([s, (1 - s) * (1 - h), h], 1)      # (B, 3, H, W)

    def forward(self, img: torch.Tensor, params: torch.Tensor, sky: torch.Tensor | None = None) -> torch.Tensor:
        pg, _ = self.split(params)
        q = self.unpack(params)
        x = self.base(img, pg)
        B, _, H, W = x.shape
        g = q["grad"]
        x = self._adjust(x, self.grad_mask(g, H, W), g[:, 3], g[:, 4], g[:, 5], g[:, 6])
        w = self.tone_masks(x)
        off = q["tone"].view(B, 3, 3) * 0.1                   # band x RGB
        x = x + torch.einsum("bkhw,bkc->bchw", w, off)
        if sky is None:
            sky = self.sky_fn(img)
        s = q["sky"]
        x = self._adjust(x, sky, s[:, 0], s[:, 1], s[:, 2])
        return x.clamp(0, 1)
