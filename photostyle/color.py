"""Differentiable colour conversions (sRGB <-> CIELAB, D65)."""

from __future__ import annotations

import torch

_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) sRGB in [0, 1] -> (B, 3, H, W) CIELAB (L in [0, 100])."""
    lin = srgb_to_linear(img)
    xyz = torch.einsum("ij,bjhw->bihw", _M_RGB2XYZ, lin) / _WHITE.view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz > d**3, xyz.clamp_min(1e-8) ** (1 / 3), xyz / (3 * d**2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


def hsv_saturation(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, H, W) HSV saturation, differentiable."""
    mx = img.max(1).values
    mn = img.min(1).values
    return (mx - mn) / mx.clamp_min(1e-4)
