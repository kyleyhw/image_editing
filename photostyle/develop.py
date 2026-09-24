"""Lightroom-style develop tools, so that grading tutorials can be followed literally.

Tutorials describe looks in Lightroom / Camera Raw terms ("Temperature -25",
"Green saturation -100", "shadows: rich blue"). ``develop(img, settings)``
applies such settings, using the same slider names, ranges and sign
conventions, so a recipe (photostyle/recipes.py) is just a list of slider
values with their sources.

Adobe does not publish its exact maths. Each tool below implements its
documented behaviour, with the mapping from slider value to effect stated in
its docstring. Sign conventions follow Lightroom:
- Temperature + is warmer, Tint + is magenta.
- HSL hue + moves a colour to the next band by hue angle (Red + goes toward
  Orange, Green + toward Aqua, Blue - toward Aqua, Purple + toward Magenta).
- Calibration: Red primary - goes toward magenta, Green primary + toward cyan,
  Blue primary - toward cyan.

All tools take and return (3, H, W) sRGB tensors in [0, 1].
"""

from __future__ import annotations

import torch

from photostyle.hsv import hsv_to_rgb, rgb_to_hsv

# Lightroom's eight colour-mixer bands (hue centre, degrees)
BANDS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 270, "magenta": 315}


def _lum(x: torch.Tensor) -> torch.Tensor:
    return (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).clamp(0, 1)


def _scale_lum(x: torch.Tensor, new_l: torch.Tensor) -> torch.Tensor:
    """Change luminance while keeping hue and saturation (ratio on RGB)."""
    old = _lum(x).clamp_min(1e-4)
    return x * (new_l / old)


def white_balance(x, temperature=0.0, tint=0.0):
    """Temperature +-100: red/blue gains of up to +-15 %. Tint +-100: green gain of up to -+12 %
    (tint + is magenta). Luminance is kept."""
    u, m = temperature / 100, tint / 100
    g = torch.tensor([1 + 0.15 * u, 1 - 0.12 * m, 1 - 0.15 * u]).view(3, 1, 1)
    y = x * g
    return _scale_lum(y, _lum(x)).clamp(0, 1)


def exposure(x, stops=0.0):
    return (x * 2 ** stops).clamp(0, 1)


def contrast(x, amount=0.0):
    """+-100: blend toward / away from a smoothstep S-curve around mid-grey (strength 0.5 at 100)."""
    a = amount / 100 * 0.5
    L = _lum(x)
    s = L * L * (3 - 2 * L)
    return _scale_lum(x, (L + a * (s - L)).clamp(0, 1)).clamp(0, 1)


def tone_regions(x, highlights=0.0, shadows=0.0, whites=0.0, blacks=0.0):
    """Luminance changes within soft tone ranges (+-100 moves the range by up to +-0.2 / 0.12):
    blacks ~L<0.2, shadows ~0.25, highlights ~0.75, whites ~L>0.8."""
    L = _lum(x)
    d = (0.20 * shadows / 100 * torch.exp(-((L - 0.25) / 0.16) ** 2)
         + 0.20 * highlights / 100 * torch.exp(-((L - 0.75) / 0.16) ** 2)
         + 0.12 * blacks / 100 * (1 - L) ** 6
         + 0.12 * whites / 100 * L ** 6)
    return _scale_lum(x, (L + d).clamp(0, 1)).clamp(0, 1)


def fade(x, amount=0.0):
    """Matte / faded blacks: the black point is lifted to ``amount``/100 * 0.25."""
    f = amount / 100 * 0.25
    return f + (1 - f) * x


def saturation(x, sat=0.0, vibrance=0.0):
    """Saturation +-100 scales chroma by 0..2. Vibrance does the same, weighted toward muted colours."""
    L = _lum(x).unsqueeze(0)
    c = x - L
    mx, mn = x.max(0).values, x.min(0).values
    cur = (mx - mn).clamp(0, 1)
    k = 1 + sat / 100 + (vibrance / 100) * (1 - cur)
    return (L + c * k).clamp(0, 1)


def _band(h, centre, width=22.0):
    d = (h * 360 - centre + 180) % 360 - 180
    return torch.exp(-0.5 * (d / width) ** 2)


def hsl(x, hue=None, sat=None, lum=None):
    """Colour mixer. ``hue``/``sat``/``lum`` map band name -> slider (+-100).
    Hue +-100 rotates that band by +-30 degrees; Saturation -100 removes its colour, +100 doubles it;
    Luminance +-100 changes its brightness by +-30 %. Greys are unaffected (weights scale with chroma)."""
    hue, sat, lum = hue or {}, sat or {}, lum or {}
    hsv = rgb_to_hsv(x)
    h, s, v = hsv.unbind(0)
    dh, ds, dv = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    for name, c in BANDS.items():
        w = _band(h, c)
        dh += w * hue.get(name, 0) / 100 * 30
        ds += w * sat.get(name, 0) / 100
        dv += w * lum.get(name, 0) / 100 * 0.3
    chroma = s.clamp(0, 1)
    h = (h + chroma * dh / 360) % 1
    s = (s * (1 + ds)).clamp(0, 1)
    v = (v * (1 + chroma * dv)).clamp(0, 1)
    return hsv_to_rgb(torch.stack([h, s, v]))


def _tint(hue_deg):
    t = hsv_to_rgb(torch.tensor([[[hue_deg / 360]], [[1.0]], [[1.0]]]))[:, 0, 0]
    return (t - t.mean()).view(3, 1, 1)


def color_grading(x, shadows=(0, 0), midtones=(0, 0), highlights=(0, 0), balance=0.0):
    """Colour wheels: (hue degrees, saturation 0-100) per range. Saturation 100 adds up to 0.25 of the
    hue's colour where the range applies; balance +-100 shifts the shadow/highlight split."""
    L = _lum(x)
    p = (L + balance / 200).clamp(0, 1)
    w = {"s": (1 - p) ** 2, "m": 4 * p * (1 - p), "h": p ** 2}
    y = x.clone()
    for key, (hue, s) in (("s", shadows), ("m", midtones), ("h", highlights)):
        if s:
            y = y + w[key] * (s / 100) * 0.25 * _tint(hue)
    return y.clamp(0, 1)


def calibration(x, red_hue=0.0, green_hue=0.0, blue_hue=0.0, red_sat=0.0, green_sat=0.0, blue_sat=0.0):
    """Camera-calibration primaries, applied to chroma only so neutrals stay neutral.
    Red hue - pushes red toward magenta (+ toward orange); Green hue + toward cyan (- toward yellow);
    Blue hue - toward cyan (+ toward purple). +-100 moves 35 % of a primary into its neighbour.
    Saturation +-100 scales that primary's chroma by 0..2."""
    k = 0.35
    M = torch.eye(3)
    r, g, b = red_hue / 100, green_hue / 100, blue_hue / 100
    M[1, 0] += k * max(r, 0); M[2, 0] += k * max(-r, 0)       # noqa: E702
    M[2, 1] += k * max(g, 0); M[0, 1] += k * max(-g, 0)       # noqa: E702
    M[0, 2] += k * max(b, 0); M[1, 2] += k * max(-b, 0)       # noqa: E702
    M = M * torch.tensor([1 + red_sat / 100, 1 + green_sat / 100, 1 + blue_sat / 100]).view(1, 3)
    L = _lum(x).unsqueeze(0)
    c = x - L
    return (L + torch.einsum("ij,jhw->ihw", M, c)).clamp(0, 1)


def channel_curve(x, channel, lift=0.0, pull=0.0):
    """Point-curve endpoints of one channel (Lightroom's "drag the bottom-left point up" / "top-right
    point down"): output = lift + (1 - lift - pull) * input, values in 0..1 of full scale."""
    y = x.clone()
    i = "rgb".index(channel)
    y[i] = lift + (1 - lift - pull) * x[i]
    return y.clamp(0, 1)


def dehaze(x, amount=0.0):
    """Global approximation: contrast + 0.4 x amount and saturation + 0.2 x amount (Dehaze adds
    contrast and colour; its local, depth-like part is not modelled)."""
    return saturation(contrast(x, 0.4 * amount), sat=0.2 * amount)


def vignette(x, amount=0.0):
    """Amount -100 darkens the corners by 60 %, rising with radius squared."""
    H, W = x.shape[1:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    r2 = ((xx ** 2 + yy ** 2) / 2).clamp(0, 1)
    return (x * (1 + amount / 100 * 0.6 * r2)).clamp(0, 1)


ORDER = ["white_balance", "exposure", "contrast", "tone_regions", "channel_curves", "fade", "hsl",
         "saturation", "calibration", "color_grading", "dehaze", "vignette"]
TOOLS = {"white_balance": white_balance, "exposure": exposure, "contrast": contrast, "tone_regions": tone_regions,
         "fade": fade, "hsl": hsl, "saturation": saturation, "calibration": calibration,
         "color_grading": color_grading, "dehaze": dehaze, "vignette": vignette}


def develop(img: torch.Tensor, settings: dict) -> torch.Tensor:
    """Apply ``settings`` ({tool: {param: value}}) in Lightroom's processing order."""
    x = img.clamp(0, 1)
    for tool in ORDER:
        if tool not in settings:
            continue
        if tool == "channel_curves":
            for ch, (lift, pull) in settings[tool].items():
                x = channel_curve(x, ch, lift, pull)
        else:
            x = TOOLS[tool](x, **settings[tool])
    return x
