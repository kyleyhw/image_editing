"""Tutorial recipes as teacher grades.

Some looks are defined by how people make them rather than by a set of photos.
A recipe is a fixed, hand-written grade (per-hue shifts, split toning, tone
curve) transcribed from public tutorials. It is applied to openly licensed
input photos to make before/after pairs, and the content-adaptive head is then
trained on those pairs (``photostyle style train NAME --recipe teacher``). The
head re-derives the grade per photo within the editable renderer (curves +
colour matrix + vignette), so the result stays a normal, editable style.

cyberpunk, from the consensus of:
  - TourBox, "Creating Cyberpunk Color Grading in Photoshop": temperature to blue,
    tint to magenta; red -> magenta, green -> cyan (saturation -100), blue -> cyan,
    purple -> magenta; boost blue / purple saturation; darken red and orange,
    brighten blue; blue + magenta in shadows, midtones and highlights.
  - Spoon Graphics, "Cyberpunk style colour grading & neon effects": pink/blue via
    temperature and tint; highlights down, shadows up; split toning bright pink
    highlights / rich blue shadows; dehaze up; darkened edges.
  - Denny's Tips, "Cyberpunk Lightroom tutorial": S-curve (darker shadows, brighter
    highlights); blue channel lifted in the shadows.
"""

from __future__ import annotations

import math

import torch


def rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(0)
    mx, _ = x.max(0)
    mn, _ = x.min(0)
    d = mx - mn
    h = torch.zeros_like(mx)
    m = d > 1e-6
    rc = torch.where(m, (mx - r) / (d + 1e-12), torch.zeros_like(d))
    gc = torch.where(m, (mx - g) / (d + 1e-12), torch.zeros_like(d))
    bc = torch.where(m, (mx - b) / (d + 1e-12), torch.zeros_like(d))
    h = torch.where(mx == r, bc - gc, torch.where(mx == g, 2 + rc - bc, 4 + gc - rc))
    h = torch.where(m, (h / 6) % 1, torch.zeros_like(h))
    s = torch.where(mx > 1e-6, d / (mx + 1e-12), torch.zeros_like(mx))
    return torch.stack([h, s, mx])


def hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    h, s, v = x.unbind(0)
    k = lambda n: (n + h * 6) % 6  # noqa: E731
    f = lambda n: v - v * s * torch.clamp(torch.minimum(k(n), 4 - k(n)), 0, 1)  # noqa: E731
    return torch.stack([f(5), f(3), f(1)])


def _band(h: torch.Tensor, centre: float, width: float) -> torch.Tensor:
    """Smooth weight of hue h (0..1) around ``centre`` degrees."""
    d = (h * 360 - centre + 180) % 360 - 180
    return torch.exp(-0.5 * (d / width) ** 2)


# hue band: (centre deg, width deg, hue shift deg, saturation x, value x)
CYBERPUNK_HSL = [
    (0, 18, -22, 1.05, 0.88),     # red -> magenta, darker
    (30, 18, -28, 0.80, 0.85),    # orange -> red/pink, darker, less saturated
    (60, 16, -12, 1.10, 0.95),    # yellow slightly toward orange, keep punch
    (120, 35, 55, 0.30, 0.90),    # green -> cyan, heavily desaturated
    (180, 20, 12, 1.20, 1.00),    # aqua -> blue-cyan
    (225, 25, -12, 1.25, 1.10),   # blue -> cyan-ish, more saturated, brighter
    (275, 20, 25, 1.25, 1.00),    # purple -> magenta
    (315, 20, 5, 1.20, 1.00),     # magenta, more saturated
]


def cyberpunk(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) sRGB in [0, 1] -> graded, following the tutorials above."""
    x = img.clamp(0, 1)
    # white balance: temperature to blue, tint to magenta
    x = x * torch.tensor([0.95, 0.93, 1.08]).view(3, 1, 1)
    # per-hue shifts
    hsv = rgb_to_hsv(x.clamp(0, 1))
    h, s, v = hsv.unbind(0)
    dh = torch.zeros_like(h)
    sm = torch.ones_like(h)
    vm = torch.ones_like(h)
    wsum = torch.full_like(h, 1e-6)
    for c, w, sh, sx, vx in CYBERPUNK_HSL:
        wt = _band(h, c, w)
        dh += wt * sh
        sm += wt * (sx - 1)
        vm += wt * (vx - 1)
        wsum += wt
    chroma = s.clamp(0, 1)                      # greys are left alone by hue tools
    h = (h + chroma * dh / 360) % 1
    s = (s * (1 + chroma * (sm - 1))).clamp(0, 1)
    v = (v * (1 + chroma * (vm - 1))).clamp(0, 1)
    x = hsv_to_rgb(torch.stack([h, s, v]))
    # tone: S-curve with deep blacks and rolled-off highlights
    y = x.clamp(0, 1)
    sm_ = y * y * (3 - 2 * y)
    y = y + 0.45 * (sm_ - y)
    y = ((y - 0.03) / 0.97).clamp(0, 1)
    # split toning: rich blue shadows, pink highlights
    L = (0.2126 * y[0] + 0.7152 * y[1] + 0.0722 * y[2]).clamp(0, 1)
    ws = (1 - L) ** 2
    wh = L ** 2
    shadow = torch.tensor([-0.05, 0.02, 0.13]).view(3, 1, 1)
    high = torch.tensor([0.09, -0.05, 0.06]).view(3, 1, 1)
    y = y + ws * shadow + wh * high
    # darkened edges
    H, W = y.shape[1:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    y = y * (1 - 0.22 * ((xx ** 2 + yy ** 2) / 2).clamp(0, 1) ** 1.2)
    return y.clamp(0, 1)


RECIPES = {"cyberpunk": cyberpunk}


def _selfcheck() -> None:                        # hsv round trip
    x = torch.rand(3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5), "hsv round trip"
    assert math.isclose(float(_band(torch.tensor(0.0), 0, 10)), 1.0)
