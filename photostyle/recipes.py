"""Looks transcribed from public grading tutorials, as Lightroom-style slider settings.

A recipe is a set of slider values for photostyle.develop.develop(). Every value
records where it comes from:
- "numeric": the tutorial states the number.
- "direction": the tutorial states the direction only, and the amount is
  chosen here.
- "conflict": the tutorials disagree; the note says which one was followed.
The tables hold the tutorials' own amounts. A look may also have a
calibration (``photostyle/calibration/<name>.json``, written by
tools/calibrate_recipe.py): one non-negative factor per direction-only row,
fitted to a target (reference photos, or a look the owner picked). The
tutorials' numbers and every sign stay as stated; a factor of 0 means the
target does not use that step.

``photostyle style train NAME --recipe teacher`` grades openly licensed input
photos with the recipe and trains the content-adaptive head on those pairs, so
the look stays an editable style (curves + colour matrix + vignette) fitted per
photo. Tools a global renderer cannot express (clarity, grain) are listed as
not modelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from photostyle.develop import develop
from photostyle.hsv import hsv_to_rgb, rgb_to_hsv  # noqa: F401  (re-exported for older callers)

SOURCES = {
    # cyberpunk
    "TB": ("TourBox, Creating Cyberpunk Color Grading in Photoshop: A Step-by-Step Guide",
           "https://www.tourboxtech.com/en/news/cyberpunk-colors.html"),
    "SG": ("Spoon Graphics, How to Apply Cyberpunk Style Color Grading & Neon Effects to Your Photos",
           "https://blog.spoongraphics.co.uk/tutorials/how-to-apply-cyberpunk-style-color-grading-neon-effects-to-your-photos"),
    "DT": ("Denny's Tips, Cyberpunk Lightroom Tutorial", "https://www.dennystips.com/cyberpunk-lightroom-tutorial/"),
    "DD": ("Drew Deltz, Editing Like a Pro: Transforming Photos into Cyberpunk Masterpieces with Photoshop",
           "https://drewdeltz.systeme.io/blog/editing-like-a-pro-transforming-photos-into-cyberpunk-masterpieces-with-photoshop"),
    # fujifilm (Superia 400 / Classic Negative family)
    "LP": ("Legendary Presets, Iconic Fuji Film Looks: Stock-by-Stock Lightroom Guide (Superia 100/400/800)",
           "https://legendarypresets.com/guide-to-iconic-fuji-film-looks/"),
    "PL": ("PresetLove, Superia 400 Film Preset", "https://presetlove.com/presets/superia-400/"),
    "ST": ("Scott Tucker, My Fujifilm Superia X-Tra 400 Film Simulation", "https://www.scotttuckerphoto.com/blog/superia400"),
    "JP": ("J.M. Peltier, Fujifilm Classic Chrome vs. Classic Neg", "https://www.jmpeltier.com/fujifilm-classic-chrome-vs-classic-neg/"),
    "ES": ("The Editing Studio, Fujifilm Look Lightroom Preset: Complete Guide (Classic Chrome)",
           "https://theeditingstudio.co/blog/fujifilm-look-lightroom-preset"),
}

# (tool, param) -> (value, [sources], kind, note)
CYBERPUNK = [
    ("white_balance", "temperature", -25, ["TB", "SG"], "direction", "temperature toward blue"),
    ("white_balance", "tint", 20, ["TB", "SG"], "direction", "tint toward magenta"),
    ("contrast", "amount", 25, ["DT", "DD"], "direction", "S-curve: darker shadows, brighter highlights"),
    ("tone_regions", "highlights", -40, ["SG", "DD"], "direction", "decrease highlights"),
    ("tone_regions", "shadows", 30, ["SG", "DD"], "direction", "increase shadows (HDR-style detail)"),
    ("tone_regions", "whites", 10, ["DD"], "direction", "boost whites slightly"),
    ("channel_curves", "b", (0.08, 0.06), ["DT"], "direction",
     "blue channel: bottom-left point up (blue shadows), top-right point down (yellow highlights)"),
    ("hsl", "hue.red", -40, ["TB"], "direction", "red -> magenta"),
    ("hsl", "hue.green", 80, ["TB"], "direction", "light green -> cyan"),
    ("hsl", "hue.blue", -40, ["TB"], "direction", "blue -> cyan"),
    ("hsl", "hue.purple", 40, ["TB"], "direction", "purple -> magenta"),
    ("hsl", "hue.magenta", -20, ["TB"], "direction", "magenta -> purple"),
    ("hsl", "sat.green", -100, ["TB"], "numeric", "green saturation -100"),
    ("hsl", "sat.yellow", 20, ["TB"], "direction", "boost yellow saturation"),
    ("hsl", "sat.blue", 30, ["TB"], "direction", "boost blue saturation"),
    ("hsl", "sat.purple", 30, ["TB"], "direction", "boost purple saturation"),
    ("hsl", "lum.red", -20, ["TB"], "direction", "decrease red brightness"),
    ("hsl", "lum.orange", -20, ["TB"], "direction", "decrease orange brightness"),
    ("hsl", "lum.blue", 20, ["TB"], "direction", "enhance blue brightness"),
    ("calibration", "red_hue", -100, ["DT", "TB"], "numeric", "red primary all the way left (toward magenta/pink)"),
    ("calibration", "green_hue", 50, ["DT", "TB"], "conflict",
     "DT: green primary to the right (purple and cyan); TB: toward yellow. DT is followed (Lightroom-specific)"),
    ("calibration", "blue_hue", -40, ["TB"], "direction", "blue primary toward cyan"),
    ("color_grading", "shadows", (220, 35), ["SG", "TB", "DT"], "direction", "rich blue shadows"),
    ("color_grading", "midtones", (285, 15), ["TB"], "direction", "magenta and blue in the midtones"),
    ("color_grading", "highlights", (320, 30), ["SG", "TB"], "direction", "bright pink highlights"),
    ("dehaze", "amount", 20, ["SG", "DT"], "direction", "boost dehaze to restore contrast"),
    ("vignette", "amount", -25, ["SG"], "direction", "darkened edges"),
]
CYBERPUNK_NOT_MODELLED = ["clarity - (softer glow; SG, DT): a local operation", "neon glow layers (SG, DD)"]

FUJIFILM = [
    ("white_balance", "temperature", -8, ["LP"], "numeric",
     "white balance 5000-5200 K instead of 5500 K daylight, i.e. slightly cooler (converted to this scale)"),
    ("contrast", "amount", 10, ["PL", "JP"], "direction", "a bit more contrast than neutral (Classic Negative)"),
    ("tone_regions", "shadows", 15, ["PL"], "direction", "softens shadows"),
    ("tone_regions", "whites", 10, ["PL"], "direction", "increases whites"),
    ("tone_regions", "blacks", -10, ["PL"], "direction", "deepens blacks"),
    ("fade", "amount", 20, ["ST"], "direction", "the 'full film look' version adds shadow fade"),
    ("hsl", "hue.green", 10, ["LP"], "numeric", "green hue +10 toward aqua"),
    ("hsl", "sat.yellow", -12, ["LP"], "numeric", "yellow saturation -10 to -15"),
    ("hsl", "sat.blue", -10, ["JP"], "direction", "cooler colours desaturated more than reds and oranges"),
    ("hsl", "sat.aqua", -10, ["JP"], "direction", "cooler colours desaturated more than reds and oranges"),
    ("hsl", "sat.red", 5, ["ST"], "direction", "vivid reds"),
    ("hsl", "sat.green", 5, ["ST"], "direction", "green greens"),
    ("saturation", "vibrance", 10, ["PL"], "direction", "amplifies vibrance"),
    ("color_grading", "shadows", (95, 14), ["PL", "ST"], "direction", "lime green in the shadows / slight greens in shadows"),
    ("color_grading", "highlights", (35, 12), ["PL", "ST"], "direction", "warm brown highlights / warm but not green whites"),
]
FUJIFILM_NOT_MODELLED = ["grain 25-30 (LP)"]

# Classic Chrome: The Editing Studio gives ranges; the midpoint of each range is used.
CLASSIC_CHROME = [
    ("tone_regions", "highlights", -25, ["ES"], "numeric", "highlights -20 to -30"),
    ("tone_regions", "shadows", 15, ["ES"], "numeric", "shadows +10 to +20"),
    ("tone_regions", "blacks", 15, ["ES"], "numeric", "blacks +10 to +20"),
    ("contrast", "amount", -15, ["ES"], "numeric", "contrast -10 to -20"),
    ("exposure", "stops", 0.1, ["ES"], "numeric", "exposure 0 to +0.2"),
    ("saturation", "vibrance", -18, ["ES"], "numeric", "vibrance -15 to -20"),
    ("saturation", "sat", -10, ["ES"], "numeric", "saturation -10"),
    ("hsl", "sat.green", -22, ["ES"], "numeric", "green saturation -20 to -25"),
    ("hsl", "sat.blue", -18, ["ES"], "numeric", "blue saturation -15 to -20"),
    ("hsl", "sat.red", -8, ["ES"], "numeric", "red saturation -5 to -10"),
    ("hsl", "hue.blue", -15, ["JP"], "direction", "blues keep some vibrance and shift toward cyan"),
    ("color_grading", "shadows", (210, 8), ["ES"], "direction", "very slight cool grey in the shadows"),
]
CLASSIC_CHROME_NOT_MODELLED = ["grain 15-20 amount, 20-25 size, 40-50 roughness (ES)"]

RECIPE_TABLES = {"cyberpunk": CYBERPUNK, "fujifilm": FUJIFILM, "classic_chrome": CLASSIC_CHROME}

# the twenty-look library (photostyle/recipe_library.py)
from photostyle.recipe_library import LOOKS, S as _LIB_SOURCES  # noqa: E402

SOURCES.update(_LIB_SOURCES)
RECIPE_TABLES.update({name: look["rows"] for name, look in LOOKS.items()})


CALIBRATION_DIR = Path(__file__).parent / "calibration"


def calibration(name: str) -> dict:
    f = CALIBRATION_DIR / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def calibrated(name: str) -> list:
    """The recipe table with its calibration factors applied (direction-only rows)."""
    fac = calibration(name).get("factors", {})
    out = []
    for tool, param, value, srcs, kind, note in RECIPE_TABLES[name]:
        f = fac.get(f"{tool}.{param}", 1.0) if kind == "direction" else 1.0
        if f == 1.0 or isinstance(value, list):     # lists are curve points: never scaled
            pass
        elif isinstance(value, tuple) and tool == "color_grading":
            value = (value[0], value[1] * f)
        elif isinstance(value, tuple):
            value = tuple(v * f for v in value)
        else:
            value = value * f
        out.append((tool, param, value, srcs, kind, note))
    return out


def settings(table) -> dict:
    """Recipe table -> develop() settings. "a.b" params nest ({"hsl": {"hue": {"red": ...}}})."""
    out: dict = {}
    for tool, param, value, *_ in table:
        t = out.setdefault(tool, {})
        if "." in param:
            kind, band = param.split(".")
            t.setdefault(kind, {})[band] = value
        else:
            t[param] = value
    return out


def _make(name):
    s = settings(calibrated(name))
    return lambda img: develop(img, s)


RECIPES = {name: _make(name) for name in RECIPE_TABLES}
cyberpunk = RECIPES["cyberpunk"]
fujifilm = RECIPES["fujifilm"]


def describe(name: str) -> str:
    """Markdown table of a recipe with its sources (for docs and style cards)."""
    cal = calibration(name)
    fac = cal.get("factors", {})
    rows = ["| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |",
            "|---|---|---|---|---|---|---|"]
    for (tool, param, value, srcs, kind, note), (*_, v2, _s, _k, _n) in zip(RECIPE_TABLES[name], calibrated(name)):
        f = fac.get(f"{tool}.{param}") if kind == "direction" else None
        c = "" if f is None else (f"x{f:g} = {v2}" if f else "x0 (unused)")
        rows.append(f"| {tool} | {param} | {value} | {c} | {', '.join(srcs)} | {kind} | {note} |")
    if cal:
        rows = [f"Calibration target: {cal.get('target_note', cal.get('target', ''))}", ""] + rows
    refs = sorted({s for row in RECIPE_TABLES[name] for s in row[3]})
    rows += ["", *[f"- **{k}**: [{SOURCES[k][0]}]({SOURCES[k][1]})" for k in refs]]
    return "\n".join(rows)


def _selfcheck() -> None:
    x = torch.rand(3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5), "hsv round trip"
    grey = torch.full((3, 4, 4), 0.5)
    assert torch.allclose(develop(grey, {}), grey)
