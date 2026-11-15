"""Tutorial looks as slider settings of the image pipeline (A6.4, Phase 21 task 19).

A tutorial recipe (photostyle/recipe_library.py, photostyle/recipes.py) is a list of Lightroom
slider values with sources. ``to_pipeline`` maps the develop-tool settings onto the sliders of
``photostyle/pipeline.py``, so a look becomes an editable preset: applying it moves the same
sliders the user edits.

Mapping (develop tool -> pipeline slider):
  white_balance, exposure, contrast, tone_regions, hsl, saturation/vibrance, calibration,
  black_white, color_grading      one to one: both follow Lightroom's slider names and signs
  point_curve, channel_curves,
  parametric_curve, fade          composed into the point curves (RGB, R, G, B)
  dehaze                          contrast + 0.4 x amount, saturation + 0.2 x amount (the global
                                  approximation develop.py uses; local dehaze is not a slider)
  vignette                        converted so the corner darkening matches: develop's amount a
                                  scales the corner by (1 + 0.6 a/100); the pipeline's v scales it
                                  by 2^(2 v/100), so v = 100 log2(1 + 0.6 a/100) / 2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from photostyle.pipeline import BAND_NAMES, DEFAULTS
from photostyle.develop import _pchip

CURVE_SAMPLES = 16           # composed curves are sampled at 16 points (the shader's per-curve budget)


def _clip(name: str, v: float) -> float:
    from photostyle.pipeline import SLIDERS
    s = next(s for s in SLIDERS if s["name"] == name)
    return float(np.clip(v, s["min"], s["max"]))


def _curve_fn(points_255: list) -> Any:
    xs, ys = zip(*sorted(points_255))
    xs, ys = list(xs), list(ys)
    if xs[0] > 0:
        xs, ys = [0] + xs, [ys[0]] + ys
    if xs[-1] < 255:
        xs, ys = xs + [255], ys + [ys[-1]]
    return lambda t: np.clip(_pchip(xs, ys, np.asarray(t) * 255), 0, 255) / 255


def to_pipeline(settings: dict) -> dict:
    """develop() settings -> pipeline params (sliders + curves)."""
    p: dict[str, Any] = {}

    def add(k: str, v: float) -> None:
        p[k] = _clip(k, p.get(k, DEFAULTS.get(k, 0.0)) + v)

    s = settings
    if "white_balance" in s:
        add("temp", s["white_balance"].get("temperature", 0))
        add("tint", s["white_balance"].get("tint", 0))
    if "exposure" in s:
        add("exposure", s["exposure"].get("stops", 0))
    if "contrast" in s:
        add("contrast", s["contrast"].get("amount", 0))
    for k, v in s.get("tone_regions", {}).items():
        add(k, v)
    if "saturation" in s:
        add("saturation", s["saturation"].get("sat", 0))
        add("vibrance", s["saturation"].get("vibrance", 0))
    for kind in ("hue", "sat", "lum"):
        for band, v in s.get("hsl", {}).get(kind, {}).items():
            add(f"hsl_{kind}_{band}", v)
    for k, v in s.get("calibration", {}).items():
        add({"shadow_tint": "cal_shadow_tint"}.get(k, f"cal_{k}"), v)
    if "black_white" in s:
        p["bw"] = 1.0
        for band, v in (s["black_white"].get("mix") or {}).items():
            add(f"bw_{band}", v)
    cg = s.get("color_grading", {})
    for rng in ("shadows", "midtones", "highlights"):
        if rng in cg:
            hue, sat = cg[rng]
            p[f"grade_{rng}_hue"] = float(hue % 360)
            add(f"grade_{rng}_sat", sat)
    if "balance" in cg:
        add("grade_balance", cg["balance"])
    if "dehaze" in s:
        a = s["dehaze"].get("amount", 0)
        add("contrast", 0.4 * a)
        add("saturation", 0.2 * a)
    if "vignette" in s:
        a = s["vignette"].get("amount", 0)
        add("vig_amount", 100 * math.log2(max(1e-3, 1 + 0.6 * a / 100)) / 2)

    curves = _curves(s)
    if curves:
        p["curves"] = curves
    return {k: v for k, v in p.items() if k == "curves" or v != DEFAULTS.get(k, 0.0)}


def _curves(s: dict) -> dict:
    """Compose develop's curve-like tools in their order (parametric, point, channel, fade)."""
    out: dict[str, list[list[float]]] = {}
    pc = s.get("parametric_curve", {})
    pt = s.get("point_curve", {})
    fade = s.get("fade", {}).get("amount", 0) / 100 * 0.25
    stages_rgb = []
    if any(pc.values()):
        # develop.parametric_curve: +-0.1 bumps centred on the four quarters of the tonal range
        def param(t: np.ndarray, pc: dict = pc) -> np.ndarray:
            d = (0.10 * pc.get("shadows", 0) / 100 * np.exp(-((t - 0.125) / 0.1) ** 2)
                 + 0.10 * pc.get("darks", 0) / 100 * np.exp(-((t - 0.375) / 0.12) ** 2)
                 + 0.10 * pc.get("lights", 0) / 100 * np.exp(-((t - 0.625) / 0.12) ** 2)
                 + 0.10 * pc.get("highlights", 0) / 100 * np.exp(-((t - 0.875) / 0.1) ** 2))
            return np.clip(t + d, 0, 1)
        stages_rgb.append(param)
    if pt.get("rgb"):
        stages_rgb.append(_curve_fn(pt["rgb"]))
    if fade:
        stages_rgb.append(lambda t, f=fade: f + (1 - f) * t)
    if stages_rgb:
        if len(stages_rgb) == 1 and pt.get("rgb") and not pc and not fade:
            out["rgb"] = [[a / 255, b / 255] for a, b in sorted(pt["rgb"])]
        else:
            t = np.linspace(0, 1, CURVE_SAMPLES)
            y = t
            for f in stages_rgb:
                y = f(y)
            y = np.maximum.accumulate(y)
            out["rgb"] = [[float(a), float(b)] for a, b in zip(t, y)]
    for ch in "rgb":
        fns = []
        if pt.get(ch):
            fns.append(_curve_fn(pt[ch]))
        if ch in s.get("channel_curves", {}):
            lift, pull = s["channel_curves"][ch]
            fns.append(lambda t, lo=lift, hi=pull: lo + (1 - lo - hi) * t)
        if not fns:
            continue
        if len(fns) == 1 and not pt.get(ch):
            lift, pull = s["channel_curves"][ch]
            out[ch] = [[0.0, float(lift)], [1.0, float(1 - pull)]]
            continue
        t = np.linspace(0, 1, CURVE_SAMPLES)
        y = t
        for f in fns:
            y = f(y)
        out[ch] = [[float(a), float(b)] for a, b in zip(t, np.maximum.accumulate(y))]
    return out


def look_sliders(name: str) -> dict:
    """The pipeline settings of a tutorial look (calibration factors applied, as the trained
    looks used)."""
    from photostyle import recipes as R
    return to_pipeline(R.settings(R.calibrated(name)))


def scale(p: dict, alpha: float) -> dict:
    """A look at strength ``alpha``: sliders scale toward their defaults; curve points move toward
    the diagonal (y' = x + alpha (y - x)); colour-wheel hues and the B&W switch are not scaled."""
    out: dict[str, Any] = {}
    for k, v in p.items():
        if k == "curves":
            out[k] = {c: [[x, x + alpha * (y - x)] for x, y in pts] for c, pts in v.items()}
        elif k == "bw" or k.endswith("_hue") and k.startswith("grade_"):
            out[k] = v
        else:
            d = DEFAULTS.get(k, 0.0)
            out[k] = d + alpha * (v - d)
    return out


__all__ = ["to_pipeline", "look_sliders", "scale", "BAND_NAMES"]


def compose_curve(inner: list | None, outer: list | None) -> list | None:
    """Points of outer(inner(x)), sampled at CURVE_SAMPLES points (one curve per channel in the
    shader, so a look's curve and the user's curve are composed on the CPU)."""
    if not inner:
        return outer
    if not outer:
        return inner
    def ev(pts: list, t: np.ndarray) -> np.ndarray:
        xs, ys = zip(*sorted(pts))
        return np.clip(_pchip(list(xs), list(ys), np.clip(t, xs[0], xs[-1])), 0, 1) if len(xs) > 2 else \
            np.interp(t, xs, ys)
    t = np.linspace(0, 1, CURVE_SAMPLES)
    y = np.maximum.accumulate(ev(outer, ev(inner, t)))
    return [[float(a), float(b)] for a, b in zip(t, y)]


def combine(*layers: dict) -> dict:
    """Sum edits in order (auto correction, look at its strength, the user's adjustments): numeric
    sliders add their offsets from the defaults; a colour-wheel hue is taken from the last layer
    that gives that wheel any saturation; black & white is on if any layer turns it on; curves
    compose (later layers act on the result of earlier ones); the crop is the last layer's."""
    out: dict[str, Any] = {}
    for layer in layers:
        for k, v in layer.items():
            if k == "curves":
                cur = out.setdefault("curves", {})
                for ch, pts in v.items():
                    cur[ch] = compose_curve(cur.get(ch), pts)
            elif k == "crop":
                out["crop"] = v
            elif k == "bw":
                out["bw"] = max(out.get("bw", 0.0), v)
            elif k.startswith("grade_") and k.endswith("_hue"):
                if layer.get(k.replace("_hue", "_sat"), 0):
                    out[k] = v
            else:
                d = DEFAULTS.get(k, 0.0)
                out[k] = _clip(k, out.get(k, d) + (v - d))
    return out
