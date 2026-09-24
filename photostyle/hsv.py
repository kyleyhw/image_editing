"""RGB <-> HSV on (3, H, W) tensors (hue in 0..1)."""

from __future__ import annotations

import torch


def rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(0)
    mx, _ = x.max(0)
    mn, _ = x.min(0)
    d = mx - mn
    m = d > 1e-6
    safe = d + 1e-12
    rc, gc, bc = (mx - r) / safe, (mx - g) / safe, (mx - b) / safe
    h = torch.where(mx == r, bc - gc, torch.where(mx == g, 2 + rc - bc, 4 + gc - rc))
    h = torch.where(m, (h / 6) % 1, torch.zeros_like(h))
    s = torch.where(mx > 1e-6, d / (mx + 1e-12), torch.zeros_like(mx))
    return torch.stack([h, s, mx])


def hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    h, s, v = x.unbind(0)

    def f(n):
        k = (n + h * 6) % 6
        return v - v * s * torch.clamp(torch.minimum(k, 4 - k), 0, 1)

    return torch.stack([f(5), f(3), f(1)])
