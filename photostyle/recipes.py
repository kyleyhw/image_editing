"""Looks transcribed from public grading tutorials, as Lightroom-style slider settings.

A recipe is a set of slider values for photostyle.develop.develop(). Every value
records where it comes from:
- "numeric": the tutorial states the number.
- "direction": the tutorial states the direction only, and the amount is
  chosen here.
- "conflict": the tutorials disagree; the note says which one was followed.
- "direction, calibrated": the amount was fitted to reference photos of the
  look (tools/calibrate_recipe.py). The sign stays the tutorial's; 0 means
  the references do not support it.

``photostyle style train NAME --recipe teacher`` grades openly licensed input
photos with the recipe and trains the content-adaptive head on those pairs, so
the look stays an editable style (curves + colour matrix + vignette) fitted per
photo. Tools a global renderer cannot express (clarity, grain) are listed as
not modelled.
"""

from __future__ import annotations

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
    ('white_balance', 'temperature', -8, ['LP'], 'numeric', 'white balance 5000-5200 K instead of 5500 K daylight, i.e. slightly cooler (converted to this scale)'),
    ('contrast', 'amount', 40.0, ['PL', 'JP'], 'direction, calibrated', 'a bit more contrast than neutral (Classic Negative); amount calibrated to the Fuji scans (x4 of 10)'),
    ('tone_regions', 'shadows', 0.0, ['PL'], 'direction, calibrated', 'softens shadows; calibrated to 0: the Fuji scans do not support it'),
    ('tone_regions', 'whites', 40.0, ['PL'], 'direction, calibrated', 'increases whites; amount calibrated to the Fuji scans (x4 of 10)'),
    ('tone_regions', 'blacks', -40.0, ['PL'], 'direction, calibrated', 'deepens blacks; amount calibrated to the Fuji scans (x4 of -10)'),
    ('fade', 'amount', 20.0, ['ST'], 'direction, calibrated', "the 'full film look' version adds shadow fade; amount confirmed by the Fuji scans"),
    ('hsl', 'hue.green', 10, ['LP'], 'numeric', 'green hue +10 toward aqua'),
    ('hsl', 'sat.yellow', -12, ['LP'], 'numeric', 'yellow saturation -10 to -15'),
    ('hsl', 'sat.blue', -40.0, ['JP'], 'direction, calibrated', 'cooler colours desaturated more than reds and oranges; amount calibrated to the Fuji scans (x4 of -10)'),
    ('hsl', 'sat.aqua', -40.0, ['JP'], 'direction, calibrated', 'cooler colours desaturated more than reds and oranges; amount calibrated to the Fuji scans (x4 of -10)'),
    ('hsl', 'sat.red', 0.0, ['ST'], 'direction, calibrated', 'vivid reds; calibrated to 0: the Fuji scans do not support it'),
    ('hsl', 'sat.green', 0.0, ['ST'], 'direction, calibrated', 'green greens; calibrated to 0: the Fuji scans do not support it'),
    ('saturation', 'vibrance', 0.0, ['PL'], 'direction, calibrated', 'amplifies vibrance; calibrated to 0: the Fuji scans do not support it'),
    ('color_grading', 'shadows', (95, 7.0), ['PL', 'ST'], 'direction, calibrated', 'lime green in the shadows / slight greens in shadows; amount calibrated to the Fuji scans (x0.5 of (95, 14))'),
    ('color_grading', 'highlights', (35, 0.0), ['PL', 'ST'], 'direction, calibrated', 'warm brown highlights / warm but not green whites; calibrated to 0: the Fuji scans do not support it'),
]
FUJIFILM_NOT_MODELLED = ["grain 25-30 (LP)"]

RECIPE_TABLES = {"cyberpunk": CYBERPUNK, "fujifilm": FUJIFILM}


def settings(table) -> dict:
    """Recipe table -> develop() settings."""
    out: dict = {}
    for tool, param, value, *_ in table:
        t = out.setdefault(tool, {})
        if tool == "hsl":
            kind, band = param.split(".")
            t.setdefault(kind, {})[band] = value
        else:
            t[param] = value
    return out


def _make(name):
    s = settings(RECIPE_TABLES[name])
    return lambda img: develop(img, s)


RECIPES = {name: _make(name) for name in RECIPE_TABLES}
cyberpunk = RECIPES["cyberpunk"]
fujifilm = RECIPES["fujifilm"]


def describe(name: str) -> str:
    """Markdown table of a recipe with its sources (for docs and style cards)."""
    rows = ["| tool | setting | value | source | kind | what the tutorial says |", "|---|---|---|---|---|---|"]
    for tool, param, value, srcs, kind, note in RECIPE_TABLES[name]:
        rows.append(f"| {tool} | {param} | {value} | {', '.join(srcs)} | {kind} | {note} |")
    refs = sorted({s for row in RECIPE_TABLES[name] for s in row[3]})
    rows += ["", *[f"- **{k}**: [{SOURCES[k][0]}]({SOURCES[k][1]})" for k in refs]]
    return "\n".join(rows)


def _selfcheck() -> None:
    x = torch.rand(3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5), "hsv round trip"
    grey = torch.full((3, 4, 4), 0.5)
    assert torch.allclose(develop(grey, {}), grey)
