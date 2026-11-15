"""Irodori's image pipeline: the reference implementation (A6.3, docs/pipeline.md).

One pipeline serves the editor, the looks and training. This module is the
reference: differentiable PyTorch, batched, and the source of every constant.
The WebGL2 implementation (studio/web/src/lib/pipeline.js) reads the same
constants from ``pipeline_spec.json`` (written by ``write_spec``) and is tested
against this module (tools/check_pipeline.py).

Order of operations for a pixel with display-encoded sRGB input ``x``:

    linearise -> sRGB to linear Rec. 2020 (working space)
    1 white balance     channel gains (von Kries) from temperature / tint
    2 exposure          multiply by 2^E
    3 tone              luminance tone curve from the six tone sliders;
                        hue-preserving (colour scaled by a luminance ratio)
    4 point curves      Lightroom-style RGB and per-channel curves (PCHIP)
    5 colour            in Oklab: calibration, vibrance / saturation,
                        8-band HSL, colour grading, black-and-white mix
    6 vignette          exposure-like, by radius
    output              Rec. 2020 to linear sRGB, luminance-preserving gamut
                        clip, sRGB encoding

Images are (B, 3, H, W) or (3, H, W) float tensors in [0, 1]. Parameters are a
flat dict of slider values (floats or (B,) tensors, so gradients flow) plus an
optional ``curves`` dict of control points. Missing sliders take their default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray

Value = Union[float, torch.Tensor]
Params = Mapping[str, Any]

# --------------------------------------------------------------------------- colour science
# sRGB and Rec. 2020 primaries to CIE XYZ (D65), from IEC 61966-2-1 and ITU-R BT.2020.
SRGB_TO_XYZ = np.array([[0.4123907993, 0.3575843394, 0.1804807884],
                        [0.2126390059, 0.7151686788, 0.0721923054],
                        [0.0193308187, 0.1191947798, 0.9505321522]])
REC2020_TO_XYZ = np.array([[0.6369580483, 0.1446169036, 0.1688809752],
                           [0.2627002120, 0.6779980715, 0.0593017165],
                           [0.0000000000, 0.0280726930, 1.0609850577]])
# Oklab (Ottosson 2020): XYZ (D65) -> LMS, then cube root, then LMS' -> Lab.
XYZ_TO_LMS = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                       [0.0329845436, 0.9293118715, 0.0361456387],
                       [0.0482003018, 0.2643662691, 0.6338517070]])
LMS_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])

# Display P3 (D65, the gamut of current phone and laptop screens), SMPTE EG 432-1.
P3_TO_XYZ = np.array([[0.4865709486, 0.2656676932, 0.1982172852],
                      [0.2289745641, 0.6917385218, 0.0792869141],
                      [0.0000000000, 0.0451133819, 1.0439443689]])

SRGB_TO_W = np.linalg.inv(REC2020_TO_XYZ) @ SRGB_TO_XYZ        # working space = linear Rec. 2020
W_TO_SRGB = np.linalg.inv(SRGB_TO_W)
W_TO_P3 = np.linalg.inv(P3_TO_XYZ) @ REC2020_TO_XYZ
W_TO_LMS = XYZ_TO_LMS @ REC2020_TO_XYZ
LMS_TO_W = np.linalg.inv(W_TO_LMS)
LAB_TO_LMS = np.linalg.inv(LMS_TO_LAB)
LUMA = REC2020_TO_XYZ[1]                                         # Y weights of the working space

MID_GREY = 0.18                    # scene-referred mid grey (18 % reflectance), the contrast pivot
EPS_Y = 1e-5                       # linear-light floor of the luminance ratio (sRGB code value 0.03/255)


def srgb_decode(x: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF (IEC 61966-2-1): display code values to linear light. Above 1 the power law
    continues, so over-range values survive a round trip."""
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)


def srgb_encode(y: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF. Defined for y >= 0 (values above 1 extend the power law)."""
    y = y.clamp_min(0)
    return torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)


def _mat(m: NDArray[np.float64], x: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 matrix to the channel axis of (B, 3, H, W)."""
    return torch.einsum("ij,bjhw->bihw", torch.as_tensor(m, dtype=x.dtype, device=x.device), x)


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().clamp_min(1e-12) ** (1 / 3)


def to_oklab(c: torch.Tensor) -> torch.Tensor:
    """Linear working RGB -> Oklab (L, a, b)."""
    return _mat(LMS_TO_LAB, _cbrt(_mat(W_TO_LMS, c)))


def from_oklab(lab: torch.Tensor) -> torch.Tensor:
    return _mat(LMS_TO_W, _mat(LAB_TO_LMS, lab) ** 3)


def _np_oklab_hue(rgb_display: tuple[float, float, float]) -> float:
    """Oklab hue angle (degrees) of an sRGB display colour."""
    lin = srgb_decode(torch.tensor(rgb_display, dtype=torch.float64)).numpy()
    lab = LMS_TO_LAB @ np.cbrt(XYZ_TO_LMS @ (SRGB_TO_XYZ @ lin))
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def _hsv_display(h_deg: float) -> tuple[float, float, float]:
    """The fully saturated sRGB colour at an HSV hue angle."""
    h = (h_deg % 360) / 60
    x = 1 - abs(h % 2 - 1)
    return [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][int(h) % 6]


# --------------------------------------------------------------------------- slider schema
# Lightroom's eight colour-mixer bands at their HSV hue angles (as in photostyle/develop.py). Their
# Oklab hue angles are computed, so a band means the same colours in both modules.
BAND_NAMES = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
BAND_HSV = [0, 30, 60, 120, 180, 225, 270, 315]
BAND_OKLAB = [_np_oklab_hue(_hsv_display(h)) for h in BAND_HSV]
GRADE_RANGES = ["shadows", "midtones", "highlights", "global"]


def _s(name: str, label: str, group: str, lo: float, hi: float, default: float = 0.0, step: float = 1.0) -> dict:
    return {"name": name, "label": label, "group": group, "min": lo, "max": hi, "default": default, "step": step}


SLIDERS: list[dict] = [
    _s("temp", "Temperature", "basic", -100, 100), _s("tint", "Tint", "basic", -100, 100),
    _s("exposure", "Exposure", "basic", -5, 5, 0, 0.01), _s("contrast", "Contrast", "basic", -100, 100),
    _s("highlights", "Highlights", "basic", -100, 100), _s("shadows", "Shadows", "basic", -100, 100),
    _s("whites", "Whites", "basic", -100, 100), _s("blacks", "Blacks", "basic", -100, 100),
    _s("vibrance", "Vibrance", "basic", -100, 100), _s("saturation", "Saturation", "basic", -100, 100),
    *[_s(f"hsl_{k}_{b}", b.capitalize(), f"hsl_{k}", -100, 100) for k in ("hue", "sat", "lum") for b in BAND_NAMES],
    *[s for r in GRADE_RANGES for s in (_s(f"grade_{r}_hue", "Hue", f"grade_{r}", 0, 360),
                                        _s(f"grade_{r}_sat", "Saturation", f"grade_{r}", 0, 100),
                                        _s(f"grade_{r}_lum", "Luminance", f"grade_{r}", -100, 100))],
    _s("grade_balance", "Balance", "grade", -100, 100), _s("grade_blending", "Blending", "grade", 0, 100, 50),
    _s("bw", "Black & white", "bw", 0, 1), *[_s(f"bw_{b}", b.capitalize(), "bw", -100, 100) for b in BAND_NAMES],
    *[_s(f"cal_{p}_{k}", f"{p.capitalize()} {'hue' if k == 'hue' else 'saturation'}", "calibration", -100, 100)
      for p in ("red", "green", "blue") for k in ("hue", "sat")],
    _s("cal_shadow_tint", "Shadows tint", "calibration", -100, 100),
    _s("vig_amount", "Amount", "vignette", -100, 100), _s("vig_midpoint", "Midpoint", "vignette", 0, 100, 50),
    _s("vig_feather", "Feather", "vignette", 0, 100, 50), _s("vig_roundness", "Roundness", "vignette", -100, 100),
    _s("texture", "Texture", "detail", -100, 100), _s("clarity", "Clarity", "detail", -100, 100),
    _s("dehaze", "Dehaze", "detail", -100, 100),
    _s("sharpen", "Sharpening", "detail", 0, 150), _s("sharpen_masking", "Masking", "detail", 0, 100),
    _s("nr_luma", "Noise reduction", "detail", 0, 100), _s("nr_colour", "Colour noise", "detail", 0, 100),
    _s("grain", "Amount", "effects", 0, 100), _s("grain_size", "Size", "effects", 0, 100, 25),
    _s("grain_roughness", "Roughness", "effects", 0, 100, 50),
    _s("halation", "Halation", "effects", 0, 100),
    _s("lens_amount", "Blur amount", "lens", 0, 100), _s("lens_focus", "Focus depth", "lens", 0, 1, 0.5, 0.01),
    _s("lens_range", "Focal range", "lens", 0, 100, 20),
]
DEFAULTS: dict[str, float] = {s["name"]: s["default"] for s in SLIDERS}
NAMES = [s["name"] for s in SLIDERS]

# --------------------------------------------------------------------------- 1 white balance
# Temperature moves the assumed illuminant along the Planckian locus by up to 80 mired either side of
# 6504 K (4278 K to 13 559 K): 80 mired is about the gap between tungsten-halogen and daylight, the
# span a JPEG temperature slider is expected to correct. + is warmer (the illuminant is taken to be
# bluer, so blue is divided down). Tint + is magenta: green gain 2^(-0.35) ~ -22 % at +100.
WB_MIRED_PER_UNIT = 0.8
WB_REF_K = 6504.0
TINT_STOPS = 0.35


def planck_xy(kelvin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CIE 1931 xy of a blackbody (Kim et al. 2002 cubic fit, 1667-25000 K)."""
    t = kelvin.clamp(1667, 25000)
    i = 1e3 / t
    x = torch.where(t <= 4000, -0.2661239 * i**3 - 0.2343589 * i**2 + 0.8776956 * i + 0.179910,
                    -3.0258469 * i**3 + 2.1070379 * i**2 + 0.2226347 * i + 0.240390)
    y = torch.where(t <= 2222, -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683,
                    torch.where(t <= 4000, -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867,
                                3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483))
    return x, y


def _white_w(kelvin: torch.Tensor) -> torch.Tensor:
    """Working-space RGB of a blackbody white with Y = 1. Shape (..., 3)."""
    x, y = planck_xy(kelvin)
    xyz = torch.stack([x / y, torch.ones_like(x), (1 - x - y) / y], -1)
    return xyz @ torch.as_tensor(np.linalg.inv(REC2020_TO_XYZ).T, dtype=xyz.dtype)


def wb_gains(temp: torch.Tensor, tint: torch.Tensor) -> torch.Tensor:
    """(B, 3) gains; luminance of a neutral is unchanged, and temp = tint = 0 is the identity."""
    mired = 1e6 / WB_REF_K - WB_MIRED_PER_UNIT * temp
    g = _white_w(torch.full_like(temp, WB_REF_K)) / _white_w(1e6 / mired)
    g = g * torch.stack([torch.ones_like(tint), 2 ** (-TINT_STOPS * tint / 100), torch.ones_like(tint)], -1)
    return g / (g @ torch.as_tensor(LUMA, dtype=g.dtype)).unsqueeze(-1)


# --------------------------------------------------------------------------- 2a lens blur (Phase 24)
# Depth of field from a depth map (relative inverse depth in [0, 1], 1 = nearest; in the browser from
# Depth Anything V2 Small). Circle of confusion, in pixels:
#   CoC = R * clamp((|d - d_focus| - range) / 0.5, 0, 1),   R = 2 % of the long edge at amount 100,
# so blur grows with distance in depth from the focus plane, beyond an in-focus range. The blur is a
# disc gathered from LENS_TAPS points (a Vogel spiral, evenly covering the disc), averaged in linear
# light so bright points bloom into bokeh discs. A sample counts if its own disc reaches the pixel;
# a sample behind the pixel counts only within the pixel's own disc, so a blurred background does
# not spill over a sharp foreground (scatter-as-gather with occlusion).
LENS_TAPS = 48
LENS_R = 0.02
LENS_SPAN = 0.5


def vogel(n: int = LENS_TAPS) -> torch.Tensor:
    """(n, 2) points evenly covering the unit disc (golden-angle spiral)."""
    k = torch.arange(n, dtype=torch.float64)
    r = torch.sqrt((k + 0.5) / n)
    a = k * math.pi * (3 - math.sqrt(5))
    return torch.stack([r * torch.cos(a), r * torch.sin(a)], 1)


def coc(depth: torch.Tensor, p: Params, rmax: float) -> torch.Tensor:
    B = depth.shape[0]
    f = _get(p, "lens_focus", B).to(depth).view(B, 1, 1, 1)
    rng = _get(p, "lens_range", B).to(depth).view(B, 1, 1, 1) / 100 * 0.3
    return rmax * ((((depth - f).abs() - rng) / LENS_SPAN).clamp(0, 1))


def lens_blur(c: torch.Tensor, depth: torch.Tensor, p: Params, src_edge: int) -> torch.Tensor:
    """Depth-of-field blur of linear colour c (B, 3, H, W) with depth (B, 1, H, W). The disc radius is
    relative to the source photo's long edge ``src_edge`` (unchanged by cropping)."""
    B, _, H, W = c.shape
    amt = float(_get(p, "lens_amount", 1)[0]) / 100
    if amt <= 0:
        return c
    rmax = LENS_R * amt * src_edge
    cp = coc(depth, p, rmax)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64) + 0.5, torch.arange(W, dtype=torch.float64) + 0.5, indexing="ij")
    acc = c.clone()
    wsum = torch.ones_like(cp)
    for o in vogel():
        dx, dy = float(o[0]) * rmax, float(o[1]) * rmax
        dist = math.hypot(dx, dy)
        gx = ((xx + dx) / W * 2 - 1).float()
        gy = ((yy + dy) / H * 2 - 1).float()
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1).to(c)
        cs = torch.nn.functional.grid_sample(c, grid, mode="bilinear", padding_mode="border", align_corners=False)
        ds = torch.nn.functional.grid_sample(depth, grid, mode="bilinear", padding_mode="border", align_corners=False)
        rs = coc(ds, p, rmax)
        behind = (ds < depth).to(c)                 # farther than the pixel (smaller inverse depth)
        reach = behind * torch.minimum(rs, cp) + (1 - behind) * rs
        w = (reach - dist + 0.5).clamp(0, 1)
        acc = acc + w * cs
        wsum = wsum + w
    return acc / wsum


# --------------------------------------------------------------------------- 3 tone
# The six tone sliders displace control points of a monotone curve on perceptual luminance
# L = sRGB-encode(Y). Abscissae: black, deep shadow, shadow, mid grey (18 % -> 0.4614), highlight,
# bright highlight, white, and 1.35 = sRGB-encode(2.0), one stop over white, so that highlights - and
# whites - can bring back values pushed over white by exposure. Each row is the displacement of every
# point at slider +100. Amounts are chosen so +-100 is a strong but usable edit, in line with
# photostyle/develop.py (e.g. shadows +100 lifts the shadow point by 0.10).
TONE_X = [0.0, 0.06, 0.18, 0.4614, 0.75, 0.92, 1.0, 1.35]
TONE_ROWS: dict[str, list[float]] = {
    "blacks":     [0.08, 0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    "shadows":    [0.0, 0.04, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0],
    "contrast":   [0.0, -0.03, -0.06, 0.0, 0.06, 0.03, 0.0, 0.0],
    "highlights": [0.0, 0.0, 0.0, 0.02, 0.10, 0.06, 0.03, 0.0],
    "whites":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.08, 0.08],
}
# highlights - additionally pulls the over-white point down to white (full recovery of +1 stop at -100)
TONE_RECOVERY = 0.35
MONO_EPS = 1e-4


def tone_points(p: Params, batch: int) -> torch.Tensor:
    """(B, 8) curve ordinates for the tone sliders, made strictly increasing."""
    y = torch.tensor(TONE_X, dtype=torch.float32).expand(batch, -1).clone()
    for k, row in TONE_ROWS.items():
        v = _get(p, k, batch) / 100
        y = y + v.unsqueeze(-1) * torch.tensor(row)
    h = _get(p, "highlights", batch) / 100
    y[:, -1] = y[:, -1] - TONE_RECOVERY * torch.relu(-h)
    return _monotone(y)


def _monotone(y: torch.Tensor) -> torch.Tensor:
    """Running maximum plus a tiny step, so PCHIP never inverts tones."""
    out = [y[:, 0]]
    for i in range(1, y.shape[1]):
        out.append(torch.maximum(y[:, i], out[-1] + MONO_EPS))
    return torch.stack(out, 1)


def pchip(xs: torch.Tensor, ys: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Monotone cubic interpolation (Fritsch & Carlson 1980), batched and differentiable in ``ys``.
    xs (n,), ys (B, n), t (B, N). Outside [xs0, xs-1] the end segments extend linearly."""
    h = xs[1:] - xs[:-1]
    d = (ys[:, 1:] - ys[:, :-1]) / h
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    dl, dr = d[:, :-1], d[:, 1:]
    same = (dl * dr) > 0
    hm = (w1 + w2) / (w1 / torch.where(same, dl, torch.ones_like(dl)) + w2 / torch.where(same, dr, torch.ones_like(dr)))
    m_in = torch.where(same, hm, torch.zeros_like(hm))
    m = torch.cat([d[:, :1], m_in, d[:, -1:]], 1)
    i = (torch.bucketize(t.detach().contiguous(), xs) - 1).clamp(0, len(xs) - 2)
    x0 = xs[i]
    hi = h[i]
    u = (t - x0) / hi
    y0, y1 = ys.gather(1, i), ys.gather(1, i + 1)
    m0, m1 = m.gather(1, i), m.gather(1, i + 1)
    inside = (u >= 0) & (u <= 1)
    uc = u.clamp(0, 1)
    h00, h10 = 2 * uc**3 - 3 * uc**2 + 1, uc**3 - 2 * uc**2 + uc
    h01, h11 = -2 * uc**3 + 3 * uc**2, uc**3 - uc**2
    cubic = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
    lin = torch.where(u < 0, y0 + (t - x0) * m0, y1 + (t - x0 - hi) * m1)
    return torch.where(inside, cubic, lin)


TONE_KEYS = ("contrast", "highlights", "shadows", "whites", "blacks")


def apply_tone(c: torch.Tensor, p: Params, detail: torch.Tensor | None = None,
               maps: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
    """Hue-preserving tone: the curve acts on perceptual luminance, colour is scaled by Y'/Y.
    ``detail`` is an optional change of perceptual luminance (texture, clarity, sharpening, NR)
    added before the curve."""
    B = c.shape[0]
    Y = _mat(LUMA[None, :], c)[:, :1]                                     # (B, 1, H, W)
    L = srgb_encode(Y)
    if detail is not None:
        L = (L + detail).clamp_min(0)
    xs = torch.tensor(TONE_X, dtype=c.dtype, device=c.device)
    local = [(mk, m) for mk, m in zip(p.get("masks") or [], maps or [])
             if any((mk.get("sliders") or {}).get(k) for k in TONE_KEYS)]
    if local:
        # per-pixel ordinates: the slider fields enter linearly, then monotone, then PCHIP
        y = torch.tensor(TONE_X, dtype=c.dtype).view(1, -1, 1, 1).expand(B, -1, *L.shape[2:]).clone()
        for k, row in TONE_ROWS.items():
            y = y + local_field(p, k, B, maps, c) / 100 * torch.tensor(row, dtype=c.dtype).view(1, -1, 1, 1)
        h = local_field(p, "highlights", B, maps, c) / 100
        y[:, -1:] = y[:, -1:] - TONE_RECOVERY * torch.relu(-h)
        ym = [y[:, :1]]
        for i in range(1, y.shape[1]):
            ym.append(torch.maximum(y[:, i:i + 1], ym[-1] + MONO_EPS))
        Lp = pchip_px(xs, torch.cat(ym, 1), L)
    else:
        Lp = pchip(xs, tone_points(p, B).to(c), L.flatten(1)).view_as(L)
    Yp = srgb_decode(Lp.clamp_min(0))
    # c' = k c + (Y' - k Y): a pure luminance ratio k = Y'/Y wherever Y >= EPS_Y; below it the ratio is
    # capped and the rest of the new luminance is added as grey, so a lifted black becomes grey.
    k = Yp / Y.clamp_min(EPS_Y)
    return k * c + (Yp - k * Y)


# --------------------------------------------------------------------------- 3a detail (spatial)
# Local means at three scales of the photo after white balance and exposure (A6, Phase 24):
#   PX   1 px at full resolution: sharpening and luminance noise (pixel-level detail)
#   MID  0.4 % of the long edge (6 px at 1600 px): texture and colour noise (fine structure)
#   BIG  1.5 % of the long edge (24 px at 1600 px): clarity, dehaze and halation (local contrast,
#        the scale at which haze and glow vary)
# ``px_scale`` = this image's long edge / the full-resolution long edge, so pixel-sized operators
# (sharpening, noise, grain) keep their size between the preview and the export.
SIGMA_PX = 1.0
SIGMA_MID = 0.004
SIGMA_BIG = 0.015
CLARITY_K = 0.6         # clarity +100 adds 0.6 x (L - local mean), weighted to the mid-tones
TEXTURE_K = 0.8         # texture +100 adds 0.8 x (L - fine mean)
SHARPEN_K = 1.0         # sharpening 100 adds 1.0 x (L - 1 px mean) (Lightroom's range is 0-150)
SHARPEN_TAU = 0.04      # masking 100 restricts sharpening to edges with local contrast above 0.04
NR_SIGMA = 0.03         # luminance NR smooths differences below ~0.03 of perceptual L (noise, not edges)
DEHAZE_W = 0.9          # dehaze +100 removes 90 % of the estimated haze (dark-channel prior)
DEHAZE_T0 = 0.15        # transmission floor: prevents noise blow-up in dense haze
DEHAZE_VEIL = 0.35      # dehaze -100 blends 35 % toward the atmospheric light
VEIL_MAX = 0.35         # ...capped at linear 0.35 (perceptual 0.63), so a night scene's lamps do not set a white veil
HALATION_THR = 0.75     # glow comes from perceptual luminance above 0.75 (highlights)
HALATION_K = 1.5        # halation 100 adds 1.5 x the blurred highlight excess (linear light)
HALATION_TINT = (1.0, 0.35, 0.12)   # red-orange, the colour of film halation (red layer exposed by backscatter)
GRAIN_K = 0.10          # grain 100: +-0.10 of perceptual L in the mid-tones


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur (reflect padding), kernel radius 3 sigma."""
    if sigma < 0.3:
        return x
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    pad_x = min(r, x.shape[3] - 1)
    pad_y = min(r, x.shape[2] - 1)
    y = torch.nn.functional.pad(x, (pad_x, pad_x, 0, 0), mode="reflect")
    y = torch.nn.functional.conv2d(y, k[r - pad_x: r + pad_x + 1].view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    y = torch.nn.functional.pad(y, (0, 0, pad_y, pad_y), mode="reflect")
    return torch.nn.functional.conv2d(y, k[r - pad_y: r + pad_y + 1].view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


SPATIAL = ("texture", "clarity", "dehaze", "sharpen", "nr_luma", "nr_colour", "halation")


def needs_spatial(p: Params) -> bool:
    return any(float(torch.as_tensor(p.get(k, 0)).abs().max()) > 0 for k in SPATIAL) or any(
        (mk.get("sliders") or {}).get(k) for mk in p.get("masks") or [] for k in ("clarity", "texture", "dehaze"))


def local_means(c: torch.Tensor, px_scale: float) -> dict[str, torch.Tensor]:
    """Local means. px: perceptual L at 1 px; mid: [linear RGB, L] at 0.4 %; big: [L, dark
    channel min_c(c), highlight excess] at 1.5 %. The dark channel and the highlight excess are
    taken per pixel before blurring (the dark-channel prior; glow spreads from each light)."""
    L = srgb_encode(_mat(LUMA[None, :], c))
    edge = max(c.shape[2], c.shape[3])
    dark = c.amin(1, keepdim=True)
    hi = ((L - HALATION_THR) / (1 - HALATION_THR)).clamp(0, 1) ** 2
    return {"px": gaussian_blur(L, SIGMA_PX * px_scale), "mid": gaussian_blur(torch.cat([c, L], 1), SIGMA_MID * edge),
            "big": gaussian_blur(torch.cat([L, dark, hi], 1), SIGMA_BIG * edge)}


def haze_light(c: torch.Tensor) -> torch.Tensor:
    """Atmospheric light: the 99th percentile of linear luminance, per image (B,)."""
    Y = _mat(LUMA[None, :], c).flatten(1)
    return torch.quantile(Y.float(), 0.99, dim=1).to(c).clamp_min(1e-3)


def apply_dehaze(c: torch.Tensor, p: Params, m: dict[str, torch.Tensor],
                 maps: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
    """Dark-channel dehaze (He, Sun & Tang 2011), with the dark channel averaged at the BIG scale:
    t = 1 - w * dark / A,  c' = (c - A (1 - t)) / max(t, t0). Negative amounts add a uniform veil
    toward A instead (there is no depth, so the haze is not graded with distance)."""
    B = c.shape[0]
    amt = local_field(p, "dehaze", B, maps, c) / 100
    A = haze_light(c).view(B, 1, 1, 1)
    dark = (m["big"][:, 1:2] / A).clamp(0, 1)
    t = 1 - DEHAZE_W * amt.clamp_min(0) * dark
    out = (c - A * (1 - t)) / t.clamp_min(DEHAZE_T0)
    veil = (-amt).clamp_min(0) * DEHAZE_VEIL
    return torch.where(amt >= 0, out, c * (1 - veil) + A.clamp_max(VEIL_MAX) * veil)


def detail_delta(L0: torch.Tensor, p: Params, m: dict[str, torch.Tensor],
                 maps: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
    """Change of perceptual luminance from texture, clarity, sharpening and luminance NR."""
    B = L0.shape[0]
    g = lambda k: local_field(p, k, B, maps, L0) / 100      # noqa: E731
    Lpx, Lmid, Lbig = m["px"], m["mid"][:, 3:], m["big"][:, :1]
    d = CLARITY_K * g("clarity") * (L0 - Lbig) * (4 * L0.clamp(0, 1) * (1 - L0.clamp(0, 1)))
    d = d + TEXTURE_K * g("texture") * (L0 - Lmid)
    tau = SHARPEN_TAU * g("sharpen_masking")
    e = (L0 - Lmid).abs()
    mask = ((e - tau + 0.01) / 0.02).clamp(0, 1)
    d = d + SHARPEN_K * g("sharpen") * (L0 - Lpx) * mask
    w = torch.exp(-(((L0 - Lpx) / NR_SIGMA) ** 2))
    return d - g("nr_luma") * w * (L0 - Lpx)


def apply_colour_nr(c: torch.Tensor, p: Params, m: dict[str, torch.Tensor]) -> torch.Tensor:
    """Colour noise: pull each pixel's Oklab (a, b) toward the MID-scale local mean."""
    B = c.shape[0]
    k = (_get(p, "nr_colour", B).to(c) / 100).view(B, 1, 1)
    lab, lab_m = to_oklab(c), to_oklab(m["mid"][:, :3])
    ab = lab[:, 1:] * (1 - k.unsqueeze(1)) + lab_m[:, 1:] * k.unsqueeze(1)
    return from_oklab(torch.cat([lab[:, :1], ab], 1))


def apply_halation(c: torch.Tensor, p: Params, m: dict[str, torch.Tensor]) -> torch.Tensor:
    B = c.shape[0]
    k = (_get(p, "halation", B).to(c) / 100).view(B, 1, 1, 1)
    h = m["big"][:, 2:3]
    return c + HALATION_K * k * h * torch.tensor(HALATION_TINT, dtype=c.dtype).view(1, 3, 1, 1)


def _pcg(v: torch.Tensor) -> torch.Tensor:
    """PCG hash on uint32 (O'Neill 2014), in int64 arithmetic; the shader computes the same."""
    M = 0xFFFFFFFF
    s = (v * 747796405 + 2891336453) & M
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M
    return ((w >> 22) ^ w) & M


def value_noise(h: int, w: int, cell: float, seed: int, offset: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """Smooth value noise in [-1, 1] on a grid of ``cell`` pixels (pixel centres, as the shader)."""
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64) + 0.5 + offset[1],
                            torch.arange(w, dtype=torch.float64) + 0.5 + offset[0], indexing="ij")
    u, v = xx / cell, yy / cell
    iu, iv = torch.floor(u), torch.floor(v)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)

    def hv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        k = _pcg(((a.long() & 0xFFFF) | ((b.long() & 0xFFFF) << 16)) ^ seed)
        return k.double() / 0xFFFFFFFF * 2 - 1

    n = (hv(iu, iv) * (1 - fu) * (1 - fv) + hv(iu + 1, iv) * fu * (1 - fv)
         + hv(iu, iv + 1) * (1 - fu) * fv + hv(iu + 1, iv + 1) * fu * fv)
    return n.float()


def apply_grain(out: torch.Tensor, p: Params, px_scale: float) -> torch.Tensor:
    """Monochrome film grain on the display-encoded output: value noise of 1-4 px cells (size),
    with a finer octave mixed in by roughness, strongest in the mid-tones."""
    B, _, H, W = out.shape
    amt = _get(p, "grain", B).to(out)
    if not bool((amt > 0).any()):
        return out
    size = float(_get(p, "grain_size", 1)[0]) / 100
    rough = float(_get(p, "grain_roughness", 1)[0]) / 100
    cell = (1 + 3 * size) * px_scale
    n = (1 - 0.5 * rough) * value_noise(H, W, cell, 17) + 0.5 * rough * value_noise(H, W, cell / 2, 91)
    Y = _mat(SRGB_TO_XYZ[1:2], srgb_decode(out))
    L = srgb_encode(Y)
    wmid = (1 - (2 * L - 1) ** 2).clamp(0, 1)
    dL = GRAIN_K * (amt / 100).view(B, 1, 1, 1) * wmid * n.to(out)
    Yn = srgb_decode((L + dL).clamp(0, 1))
    k = Yn / Y.clamp_min(EPS_Y)
    return srgb_encode((k * srgb_decode(out) + (Yn - k * Y)).clamp(0, 1))


# --------------------------------------------------------------------------- local adjustments (Phase 23)
# A mask is {"type": "linear" | "radial" | "brush" | "all", "invert": bool, shape parameters in source
# coordinates (fractions of the frame, before crop), optional "range": {"lum": [lo, hi], "feather": f,
# "hue": deg, "hue_width": deg}, "sliders": {name: value}}. Its sliders are added where the mask is
# on: s(p) = s + sum_j m_j(p) s_j (A6.3). Up to MAX_MASKS masks.
MAX_MASKS = 4
LOCAL_SLIDERS = ["temp", "tint", "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
                 "saturation", "clarity", "texture", "dehaze"]


def _smooth(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0, 1)
    return t * t * (3 - 2 * t)


def mask_shape(mk: Mapping[str, Any], h: int, w: int) -> torch.Tensor:
    """(H, W) shape mask in source space, before range and invert."""
    yy, xx = torch.meshgrid((torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij")
    kind = mk.get("type", "all")
    if kind == "linear":
        # full effect on the start side, fading to none at the end (Lightroom's linear gradient)
        x0, y0, x1, y1 = (float(mk[k]) for k in ("x0", "y0", "x1", "y1"))
        dx, dy = (x1 - x0) * w, (y1 - y0) * h
        t = ((xx - x0) * w * dx + (yy - y0) * h * dy) / max(dx * dx + dy * dy, 1e-9)
        return 1 - _smooth(t)
    if kind == "radial":
        cx, cy, rx, ry = (float(mk[k]) for k in ("cx", "cy", "rx", "ry"))
        a = math.radians(float(mk.get("angle", 0)))
        px, py = (xx - cx) * w, (yy - cy) * h
        u = (px * math.cos(a) + py * math.sin(a)) / max(rx * w, 1e-6)
        v = (-px * math.sin(a) + py * math.cos(a)) / max(ry * h, 1e-6)
        r = torch.sqrt(u * u + v * v)
        f = max(float(mk.get("feather", 50)) / 100, 0.01)
        return 1 - _smooth((r - (1 - f)) / f)
    if kind == "brush":
        import base64
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(mk["data"]))).convert("L")
        t = torch.from_numpy(np.asarray(im).copy()).float()[None, None] / 255
        return torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return torch.ones(h, w)


def mask_range(mk: Mapping[str, Any], L: torch.Tensor, hue: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
    """Range refinement from the pixel's own perceptual luminance and Oklab hue (1 where no range)."""
    rg = mk.get("range") or {}
    out = torch.ones_like(L)
    if "lum" in rg:
        lo, hi = (float(v) for v in rg["lum"])
        fe = max(float(rg.get("feather", 10)) / 100, 1e-3)
        out = out * _smooth((L - lo + fe) / fe) * _smooth((hi + fe - L) / fe)
    if "hue" in rg:
        c = math.radians(float(rg["hue"]))
        wdt = math.radians(float(rg.get("hue_width", 30)))
        d = (torch.remainder(hue - c + math.pi, 2 * math.pi) - math.pi).abs()
        out = out * (1 - _smooth((d - wdt) / (0.5 * wdt))) * _smooth(chroma / 0.03)
    return out


def mask_maps(masks: Sequence[Mapping[str, Any]], src_hw: tuple[int, int], crop: Mapping[str, float] | None,
              c0: torch.Tensor) -> list[torch.Tensor]:
    """(B, 1, H, W) mask for each mask, in output space (after crop), from the global-edit colour c0."""
    B, _, H, W = c0.shape
    L = srgb_encode(_mat(LUMA[None, :], c0))
    lab = to_oklab(c0)
    hue, chroma = torch.atan2(lab[:, 2:3], lab[:, 1:2]), torch.sqrt(lab[:, 1:2] ** 2 + lab[:, 2:3] ** 2)
    out = []
    for mk in list(masks)[:MAX_MASKS]:
        shp = mask_shape(mk, *src_hw)[None, None].expand(B, 1, -1, -1).to(c0)
        shp = apply_geometry(shp.expand(B, 3, -1, -1), crop)[:, :1] if crop else shp
        m = shp * mask_range(mk, L, hue, chroma)
        out.append(1 - m if mk.get("invert") else m)
    return out


def local_field(p: Params, k: str, B: int, maps: Sequence[torch.Tensor] | None, like: torch.Tensor) -> torch.Tensor:
    """A slider as a field: the global value plus each mask's value where that mask is on.
    (B, 1, 1, 1) without masks, else (B, 1, H, W)."""
    v = _get(p, k, B).to(like).view(B, 1, 1, 1)
    for mk, m in zip(p.get("masks") or [], maps or []):
        dv = float((mk.get("sliders") or {}).get(k, 0))
        if dv:
            v = v + m * dv
    return v


def pchip_px(xs: torch.Tensor, ys: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """PCHIP with per-pixel ordinates: ys (B, n, H, W), t (B, 1, H, W)."""
    B, n, H, W = ys.shape
    flat_y = ys.permute(0, 2, 3, 1).reshape(B * H * W, n)
    return pchip(xs, flat_y, t.reshape(B * H * W, 1)).view(B, 1, H, W)


# --------------------------------------------------------------------------- 4 point curves
CURVE_KEYS = ("rgb", "r", "g", "b")


def curve_points(curves: Mapping[str, Any] | None, key: str) -> list[tuple[float, float]] | None:
    """Control points of one curve on [0, 1], or None for the identity. Points are (input, output)."""
    pts = (curves or {}).get(key)
    if not pts or len(pts) < 2:
        return None
    pts = sorted((float(a), float(b)) for a, b in pts)
    if pts == [(0.0, 0.0), (1.0, 1.0)]:
        return None
    return pts


def eval_curve(pts: list[tuple[float, float]], t: torch.Tensor) -> torch.Tensor:
    """A point curve on [0, 1]: PCHIP through the points, flat beyond the first and last point
    (as in Lightroom: a raised black point lifts everything below it), identity slope above 1 so
    over-range values stay ordered."""
    xs = torch.tensor([a for a, _ in pts], dtype=t.dtype, device=t.device)
    ys = torch.tensor([[b for _, b in pts]], dtype=t.dtype, device=t.device)
    flat = t.flatten().unsqueeze(0)
    inner = pchip(xs, ys, flat.clamp(float(xs[0]), float(xs[-1]))).view_as(t)
    return torch.where(t > 1, inner + (t - 1), inner)


def apply_curves(c: torch.Tensor, curves: Mapping[str, Any] | None) -> torch.Tensor:
    """Lightroom's point curve: RGB applies the same curve to each channel, then R, G, B.
    Applied to display-encoded values, as the curve panel shows them."""
    if not any(curve_points(curves, k) for k in CURVE_KEYS):
        return c
    e = srgb_encode(c)
    rgb = curve_points(curves, "rgb")
    if rgb:
        e = eval_curve(rgb, e)
    chans = []
    for i, k in enumerate("rgb"):
        pts = curve_points(curves, k)
        chans.append(eval_curve(pts, e[:, i]) if pts else e[:, i])
    return srgb_decode(torch.stack(chans, 1))


# --------------------------------------------------------------------------- 5 colour (Oklab)
CAL_SHIFT = 0.35          # +-100 moves 35 % of a primary into its neighbour (photostyle/develop.py)
CAL_SHADOW = 0.08         # shadows tint: Oklab a offset at +100, weighted (1 - L)^2
SAT_GAIN = 1.0            # saturation +100 doubles chroma, -100 removes it
VIB_C0 = 0.25             # vibrance acts fully below Oklab chroma 0 and not at all from 0.25 (~ sRGB primaries)
HSL_HUE_DEG = 30.0        # HSL hue +-100 rotates its band by +-30 degrees
HSL_LUM = 0.30            # HSL luminance +-100 changes Oklab L by +-30 % (weighted by chroma)
CHROMA_FULL = 0.08        # chroma at which a colour counts as fully coloured (HSL lum, B&W mix); skin ~0.05-0.1
GRADE_C = 0.08            # colour grading saturation 100 adds 0.08 of Oklab chroma (~ a strong tint)
GRADE_L = 0.10            # colour grading luminance +-100 changes L by +-0.10
BW_MIX = 0.40             # B&W mix +-100 brightens / darkens that band's grey by up to 40 %


def _get(p: Params, k: str, batch: int) -> torch.Tensor:
    v = p.get(k, DEFAULTS.get(k, 0.0))
    t = torch.as_tensor(v, dtype=torch.float32)
    return t.expand(batch) if t.dim() == 0 else t


def band_weights(h: torch.Tensor) -> torch.Tensor:
    """(8, ...) weights of the eight hue bands at Oklab hue ``h`` (radians). Cosine-squared tents
    between neighbouring band centres: smooth, and they sum to 1 at every hue."""
    centres = [math.radians(c) for c in BAND_OKLAB]
    ws = []
    for k in range(8):
        lo, c, hi = centres[k - 1], centres[k], centres[(k + 1) % 8]
        span_l = (c - lo) % (2 * math.pi)
        span_r = (hi - c) % (2 * math.pi)
        d = torch.remainder(h - c + math.pi, 2 * math.pi) - math.pi          # signed distance to centre
        t = torch.where(d < 0, -d / span_l, d / span_r)
        ws.append(torch.where(t < 1, torch.cos(t.clamp(0, 1) * math.pi / 2) ** 2, torch.zeros_like(t)))
    return torch.stack(ws)


def _grade_dir(hue_deg: torch.Tensor) -> torch.Tensor:
    """Unit Oklab (a, b) direction of a colour-wheel hue given in conventional (HSV) degrees: the
    wheel's hue maps through the band anchors, so 'orange 30' is the Oklab hue of sRGB orange."""
    anch = torch.tensor(BAND_HSV + [360.0])
    ok = np.unwrap(np.radians(BAND_OKLAB + [BAND_OKLAB[0] + 360.0]))
    okt = torch.tensor(ok, dtype=torch.float32)
    hd = torch.remainder(hue_deg, 360.0)
    i = (torch.bucketize(hd, anch) - 1).clamp(0, 7)
    u = (hd - anch[i]) / (anch[i + 1] - anch[i])
    ang = okt[i] + u * (okt[i + 1] - okt[i])
    return torch.stack([torch.cos(ang), torch.sin(ang)], -1)


def apply_colour(c: torch.Tensor, p: Params, maps: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
    B = c.shape[0]
    g = lambda k: _get(p, k, B).to(c).view(B, 1, 1)          # noqa: E731

    # calibration: a neutral-preserving matrix on chroma in the working space
    r, gg, b = g("cal_red_hue") / 100, g("cal_green_hue") / 100, g("cal_blue_hue") / 100
    if any(bool((v != 0).any()) for v in (r, gg, b, g("cal_red_sat"), g("cal_green_sat"), g("cal_blue_sat"))):
        z = torch.zeros_like(r)
        M = torch.stack([
            torch.stack([1 + z, CAL_SHIFT * torch.relu(-gg), CAL_SHIFT * torch.relu(b)], -1),
            torch.stack([CAL_SHIFT * torch.relu(r), 1 + z, CAL_SHIFT * torch.relu(-b)], -1),
            torch.stack([CAL_SHIFT * torch.relu(-r), CAL_SHIFT * torch.relu(gg), 1 + z], -1)], -2)   # (B,1,1,3,3)
        M = M * torch.stack([1 + g("cal_red_sat") / 100, 1 + g("cal_green_sat") / 100,
                             1 + g("cal_blue_sat") / 100], -1).unsqueeze(-2)
        Y = _mat(LUMA[None, :], c)
        ch = c - Y
        ch = torch.einsum("bij,bjhw->bihw", M[:, 0, 0], ch)
        c = Y + ch

    lab = to_oklab(c)
    L, a, bb = lab[:, 0], lab[:, 1], lab[:, 2]
    C = torch.sqrt(a * a + bb * bb + 1e-12)
    h = torch.atan2(bb, a)
    cw = (C / CHROMA_FULL).clamp(0, 1)
    cw = cw * cw * (3 - 2 * cw)                                     # smoothstep: greys are not affected

    W = band_weights(h)                                             # (8, B, H, W)
    hue = sum(W[k] * g(f"hsl_hue_{n}") for k, n in enumerate(BAND_NAMES)) / 100
    sat = sum(W[k] * g(f"hsl_sat_{n}") for k, n in enumerate(BAND_NAMES)) / 100
    lum = sum(W[k] * g(f"hsl_lum_{n}") for k, n in enumerate(BAND_NAMES)) / 100

    vib = g("vibrance") / 100
    vw = 1 - (C / VIB_C0).clamp(0, 1)
    C = C * (1 + SAT_GAIN * g("saturation") / 100).clamp_min(0) * (1 + vib * vw * vw).clamp_min(0)
    C = C * (1 + sat).clamp_min(0)
    for mk, m in zip(p.get("masks") or [], maps or []):
        ds = float((mk.get("sliders") or {}).get("saturation", 0))
        if ds:
            C = C * (1 + SAT_GAIN * m[:, 0] * ds / 100).clamp_min(0)
    h = h + torch.deg2rad(hue * HSL_HUE_DEG) * cw
    L = L * (1 + lum * HSL_LUM * cw)

    if bool((g("bw") > 0.5).any()):
        mix = sum(W[k] * g(f"bw_{n}") for k, n in enumerate(BAND_NAMES)) / 100
        Lbw = L * (1 + mix * BW_MIX * cw)
        on = (g("bw") > 0.5).to(c)
        L = on * Lbw + (1 - on) * L
        C = (1 - on) * C

    a, bb = C * torch.cos(h), C * torch.sin(h)

    # colour grading: tints by tonal range (masks on Oklab L, shifted by balance)
    pL = (L + g("grade_balance") / 200).clamp(0, 1)
    blend = 0.5 + g("grade_blending") / 100                         # 50 -> masks (1-p)^2, 2p(1-p), p^2
    ws = (1 - pL) ** (2 / blend)
    wh = pL ** (2 / blend)
    wm = (1 - ws - wh).clamp_min(0)
    for rng, w in (("shadows", ws), ("midtones", wm), ("highlights", wh), ("global", torch.ones_like(pL))):
        s = g(f"grade_{rng}_sat") / 100
        dirn = _grade_dir(_get(p, f"grade_{rng}_hue", B).float()).to(c).view(B, 1, 1, 2)
        a = a + w * s * GRADE_C * dirn[..., 0]
        bb = bb + w * s * GRADE_C * dirn[..., 1]
        L = L + w * g(f"grade_{rng}_lum") / 100 * GRADE_L
    a = a + (1 - L).clamp(0, 1) ** 2 * g("cal_shadow_tint") / 100 * CAL_SHADOW
    return from_oklab(torch.stack([L, a, bb], 1))


# --------------------------------------------------------------------------- 6 vignette
VIG_STOPS = 2.0           # amount -100 darkens the corners by two stops (x 0.25), +100 brightens by two


def vignette_mask(h: int, w: int, mid: torch.Tensor, feather: torch.Tensor, roundness: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) in [0, 1]: 0 at the centre, 1 in the corners. Radius 1 is the frame's corner;
    roundness 0 follows the frame's aspect, +100 is a circle, -100 a squarer shape."""
    yy, xx = torch.meshgrid((torch.arange(h) + 0.5) / h * 2 - 1, (torch.arange(w) + 0.5) / w * 2 - 1, indexing="ij")
    asp = w / h
    B = mid.shape[0]
    rnd = roundness.view(B, 1, 1) / 100
    sx = 1 + rnd.clamp_min(0) * (asp - 1)                            # circle: equal scale in x and y
    xs = xx * sx
    pw = 2 * (2 ** (-rnd.clamp_max(0) * 1.5))                       # squarer: higher p-norm
    r = ((xs.abs() ** pw + yy.abs() ** pw) / (sx ** pw + 1)) ** (1 / pw)
    m0 = (mid.view(B, 1, 1) / 100).clamp(0.01, 1)
    f = (feather.view(B, 1, 1) / 100).clamp(0.01, 1)
    lo, hi = m0 * (1 - f), m0 + (1 - m0) * f
    t = ((r - lo) / (hi - lo)).clamp(0, 1)
    return (t * t * (3 - 2 * t)).unsqueeze(1)


def apply_vignette(c: torch.Tensor, p: Params) -> torch.Tensor:
    B, _, H, W = c.shape
    amt = _get(p, "vig_amount", B).to(c)
    if not bool((amt != 0).any()):
        return c
    m = vignette_mask(H, W, _get(p, "vig_midpoint", B), _get(p, "vig_feather", B),
                      _get(p, "vig_roundness", B)).to(c)
    return c * 2 ** (amt.view(B, 1, 1, 1) / 100 * VIG_STOPS * m)



# --------------------------------------------------------------------------- 0 geometry
GEO_DEFAULTS = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "angle": 0.0, "vertical": 0.0, "horizontal": 0.0,
                "distortion": 0.0, "scale": 100.0}
PERSPECTIVE_K = 0.5       # vertical / horizontal +-100: projective term +-0.5 per half-frame (about +-25 deg of tilt)
DISTORTION_K = 0.2        # distortion +-100: radial term +-0.2 at the frame corner (strong barrel / pincushion)


def geometry_affine(width: int, height: int, crop: Mapping[str, float] | None) -> NDArray[np.float64]:
    """3x3 matrix taking output uv (0..1, v down) to centred source coordinates q, in units of half
    the long edge, after the crop rectangle, the straighten rotation (+ clockwise) and the scale."""
    c = {**GEO_DEFAULTS, **(crop or {})}
    W, H = float(width), float(height)
    n = max(W, H) / 2
    th = math.radians(c["angle"])
    to_px = np.array([[c["w"] * W, 0, c["x"] * W], [0, c["h"] * H, c["y"] * H], [0, 0, 1]])
    centre = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    s = 100.0 / max(float(c["scale"]), 1.0)
    norm = np.array([[s / n, 0, 0], [0, s / n, 0], [0, 0, 1]])
    return norm @ rot @ centre @ to_px


def geometry_map(width: int, height: int, crop: Mapping[str, float] | None, u: torch.Tensor, v: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Source uv of output uv. After the affine part: perspective correction as a homography
    (x, y) -> (x, y) / (1 + kh x + kv y), then radial lens correction q (1 - kd r^2) with r = 1 at the
    frame corner. Positive distortion corrects barrel distortion (edges sample nearer the centre)."""
    c = {**GEO_DEFAULTS, **(crop or {})}
    A = torch.as_tensor(geometry_affine(width, height, c), dtype=u.dtype)
    x = A[0, 0] * u + A[0, 1] * v + A[0, 2]
    y = A[1, 0] * u + A[1, 1] * v + A[1, 2]
    kh, kv = PERSPECTIVE_K * c["horizontal"] / 100, PERSPECTIVE_K * c["vertical"] / 100
    w = 1 + kh * x + kv * y
    x, y = x / w, y / w
    n = max(width, height) / 2
    rmax2 = (width * width + height * height) / (4 * n * n)
    k = 1 - DISTORTION_K * c["distortion"] / 100 * (x * x + y * y) / rmax2
    x, y = x * k, y * k
    return 0.5 + x * n / width, 0.5 + y * n / height


def geometry_matrix(width: int, height: int, crop: Mapping[str, float] | None) -> NDArray[np.float64]:
    """The affine part as a map to source uv (exact when there is no perspective or distortion)."""
    n = max(width, height) / 2
    back = np.array([[n / width, 0, 0.5], [0, n / height, 0.5], [0, 0, 1]])
    return back @ geometry_affine(width, height, crop)


def output_size(width: int, height: int, crop: Mapping[str, float] | None) -> tuple[int, int]:
    c = {"w": 1.0, "h": 1.0, **(crop or {})}
    return max(1, round(c["w"] * width)), max(1, round(c["h"] * height))


def apply_geometry(x: torch.Tensor, crop: Mapping[str, float] | None) -> torch.Tensor:
    """Crop, straighten, scale, perspective and lens correction of (B, C, H, W) by bilinear
    resampling; points outside the source are black."""
    if not crop or all(float(crop.get(k, d)) == d for k, d in GEO_DEFAULTS.items()):
        return x
    B, _, H, W = x.shape
    ow, oh = output_size(W, H, crop)
    v, u = torch.meshgrid((torch.arange(oh, dtype=torch.float64) + 0.5) / oh, (torch.arange(ow, dtype=torch.float64) + 0.5) / ow, indexing="ij")
    su, sv = geometry_map(W, H, crop, u, v)
    grid = torch.stack([su * 2 - 1, sv * 2 - 1], -1).unsqueeze(0).expand(B, -1, -1, -1).to(x)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# --------------------------------------------------------------------------- output
def gamut_clip(c: torch.Tensor, y_row: NDArray[np.float64] = SRGB_TO_XYZ[1]) -> torch.Tensor:
    """Bring linear output RGB into [0, 1] by moving each out-of-range colour toward its own
    luminance (hue and luminance kept, chroma reduced) rather than clipping channels, which shifts
    hue. ``y_row`` is the luminance row of the output primaries."""
    Y = _mat(y_row[None, :], c).clamp(0, 1)
    d = c - Y
    up = torch.where(d > 1e-9, (1 - Y) / d.clamp_min(1e-9), torch.full_like(d, 1e9))
    dn = torch.where(d < -1e-9, -Y / d.clamp_max(-1e-9), torch.full_like(d, 1e9))
    t = torch.minimum(up, dn).amin(1, keepdim=True).clamp(0, 1)
    return (Y + t * d).clamp(0, 1)


def render(img: torch.Tensor, params: Params | None = None, legacy: Any = None, output: str = "srgb",
           px_scale: float = 1.0, depth: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the pipeline. ``img`` is display-encoded sRGB in [0, 1], (3, H, W) or (B, 3, H, W).
    ``legacy`` is an optional callable applied first to the display-encoded image (the Phase-A5
    looks, until Phase 21 replaces them). ``output`` is "srgb" or "p3" (Display P3: same transfer
    curve, wider primaries). ``px_scale`` is this image's long edge over the full-resolution long
    edge (pixel-sized detail and grain keep their size in previews). ``depth`` (B, 1, H, W) or
    (1, H, W) in the source frame enables lens blur."""
    p = dict(params or {})
    single = img.dim() == 3
    x = img.unsqueeze(0) if single else img
    if legacy is not None:
        x = legacy(x)
    src_hw = (x.shape[2], x.shape[3])
    x = apply_geometry(x, p.get("crop"))
    B = x.shape[0]
    c = _mat(SRGB_TO_W, srgb_decode(x.clamp(0, 1)))
    c = c * wb_gains(_get(p, "temp", B), _get(p, "tint", B)).to(c).view(B, 3, 1, 1)
    c = c * (2 ** _get(p, "exposure", B).to(c)).view(B, 1, 1, 1)
    if depth is not None and float(_get(p, "lens_amount", 1)[0]) > 0:
        d = depth if depth.dim() == 4 else depth.unsqueeze(0)
        d = torch.nn.functional.interpolate(d.float(), size=src_hw, mode="bilinear", align_corners=False)
        d = apply_geometry(d.expand(B, 1, -1, -1).repeat(1, 3, 1, 1), p.get("crop"))[:, :1] if p.get("crop") else d.expand(B, 1, -1, -1)
        c = lens_blur(c, d.to(c), p, max(src_hw))
    maps = mask_maps(p["masks"], src_hw, p.get("crop"), c) if p.get("masks") else None
    for mk, mm in zip(p.get("masks") or [], maps or []):
        sl = mk.get("sliders") or {}
        if sl.get("temp") or sl.get("tint"):
            g = wb_gains(torch.tensor([float(sl.get("temp", 0))]), torch.tensor([float(sl.get("tint", 0))]))[0].to(c)
            c = c * torch.exp(mm * torch.log(g).view(1, 3, 1, 1))
        if sl.get("exposure"):
            c = c * 2 ** (mm * float(sl["exposure"]))
    detail, m = None, None
    if needs_spatial(p):
        m = local_means(c, px_scale)
        c = apply_dehaze(c, p, m, maps)
        c = apply_colour_nr(c, p, m)
        detail = detail_delta(srgb_encode(_mat(LUMA[None, :], c)), p, m, maps)
    c = apply_tone(c, p, detail, maps)
    c = apply_curves(c, p.get("curves"))
    c = apply_colour(c, p, maps)
    if m is not None:
        c = apply_halation(c, p, m)
    c = apply_vignette(c, p)
    if output == "p3":
        out = srgb_encode(gamut_clip(_mat(W_TO_P3, c), P3_TO_XYZ[1]))
    else:
        out = srgb_encode(gamut_clip(_mat(W_TO_SRGB, c)))
    out = apply_grain(out, p, px_scale)
    return out[0] if single else out


# --------------------------------------------------------------------------- spec for the browser
SPEC_PATH = Path(__file__).resolve().parent.parent / "studio" / "web" / "src" / "lib" / "pipeline_spec.json"


def spec() -> dict:
    """Everything the WebGL implementation needs, as plain numbers."""
    m = lambda a: [float(v) for v in np.asarray(a).flatten()]     # noqa: E731
    dirs = _grade_dir(torch.arange(0, 361, dtype=torch.float32)).tolist()
    return {
        "version": 1,
        "sliders": SLIDERS, "bands": BAND_NAMES, "band_hsv": BAND_HSV, "band_oklab": BAND_OKLAB,
        "grade_ranges": GRADE_RANGES, "grade_dirs": dirs,
        "m": {"srgb_to_w": m(SRGB_TO_W), "w_to_srgb": m(W_TO_SRGB), "w_to_lms": m(W_TO_LMS),
              "lms_to_w": m(LMS_TO_W), "lms_to_lab": m(LMS_TO_LAB), "lab_to_lms": m(LAB_TO_LMS),
              "luma": m(LUMA), "srgb_y": m(SRGB_TO_XYZ[1]), "w_to_xyz": m(REC2020_TO_XYZ),
              "w_to_p3": m(W_TO_P3), "p3_y": m(P3_TO_XYZ[1])},
        "wb": {"mired_per_unit": WB_MIRED_PER_UNIT, "ref_k": WB_REF_K, "tint_stops": TINT_STOPS},
        "tone": {"x": TONE_X, "rows": TONE_ROWS, "recovery": TONE_RECOVERY, "mono_eps": MONO_EPS},
        "k": {"eps_y": EPS_Y, "cal_shift": CAL_SHIFT, "cal_shadow": CAL_SHADOW, "sat_gain": SAT_GAIN,
              "vib_c0": VIB_C0, "hsl_hue_deg": HSL_HUE_DEG, "hsl_lum": HSL_LUM, "chroma_full": CHROMA_FULL,
              "grade_c": GRADE_C, "grade_l": GRADE_L, "bw_mix": BW_MIX, "vig_stops": VIG_STOPS},
        "masks": {"max": MAX_MASKS, "local": LOCAL_SLIDERS},
        "geometry": {"perspective_k": PERSPECTIVE_K, "distortion_k": DISTORTION_K},
        "lens": {"taps": vogel().tolist(), "r": LENS_R, "span": LENS_SPAN},
        "detail": {"sigma_px": SIGMA_PX, "sigma_mid": SIGMA_MID, "sigma_big": SIGMA_BIG, "clarity": CLARITY_K,
                   "texture": TEXTURE_K, "sharpen": SHARPEN_K, "sharpen_tau": SHARPEN_TAU, "nr_sigma": NR_SIGMA,
                   "dehaze_w": DEHAZE_W, "dehaze_t0": DEHAZE_T0, "dehaze_veil": DEHAZE_VEIL, "veil_max": VEIL_MAX, "halation_thr": HALATION_THR,
                   "halation_k": HALATION_K, "halation_tint": list(HALATION_TINT), "grain": GRAIN_K},
    }


def write_spec(path: Path = SPEC_PATH) -> None:
    path.write_text(json.dumps(spec(), indent=1))


if __name__ == "__main__":
    write_spec()
    print(f"wrote {SPEC_PATH}")
