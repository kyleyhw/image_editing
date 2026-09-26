"""How faithfully does each trained look reproduce its tutorial recipe?

For each look, on up to 12 of its own subject photos: mean and 99th-percentile absolute
difference (0-255) between the recipe's output and the trained look, and the share of pixels
whose hue moves by more than 30 degrees (the "green sky" failure: per-channel curves
cannot express a hue-preserving highlight compression on saturated colours).

    uv run python tools/check_library.py [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle import newstyle as ns  # noqa: E402
from photostyle.engine import StylePack  # noqa: E402
from photostyle.features import FeatureExtractor  # noqa: E402
from photostyle.hsv import rgb_to_hsv  # noqa: E402
from photostyle.recipe_library import LOOKS  # noqa: E402
from photostyle.recipes import RECIPES  # noqa: E402


def hue_jump(a: torch.Tensor, b: torch.Tensor) -> float:
    ha, hb = rgb_to_hsv(a), rgb_to_hsv(b)
    sat = (ha[1] > 0.25) & (hb[1] > 0.25) & (ha[2] > 0.2)
    d = ((ha[0] - hb[0] + 0.5) % 1 - 0.5).abs() * 360
    return float((d[sat] > 30).float().mean()) if sat.any() else 0.0


def check(name: str, fx) -> dict:
    p = ns.Project.load(name)
    pk = StylePack.load(Path("stylepacks") / name)
    files = [c["file"] for i, c in enumerate(p.candidates) if not ns.flagged(p, i)][:12]
    diffs, jumps = [], []
    with torch.no_grad():
        for f in files:
            x = ns._tensor(f, 384)
            t = pk.renderer(x[None], pk.head(fx(x)[None]))[0]
            r = RECIPES[name](x)
            diffs.append(((t - r).abs() * 255).flatten())
            jumps.append(hue_jump(r, t))
    d = torch.cat(diffs).numpy()
    return {"mean": round(float(d.mean()), 1), "p99": round(float(np.percentile(d, 99)), 1),
            "hue_jump_pct": round(100 * float(np.mean(jumps)), 2), "worst_hue_jump_pct": round(100 * float(np.max(jumps)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    fx = FeatureExtractor()
    res = {n: check(n, fx) for n in LOOKS if not args.only or n in args.only}
    for n, r in res.items():
        print(f"{n:16} mean {r['mean']:5}  p99 {r['p99']:6}  hue jumps {r['hue_jump_pct']:5}%  worst photo {r['worst_hue_jump_pct']:5}%")
    Path("tests/reports").mkdir(exist_ok=True, parents=True)
    Path("tests/reports/library_fidelity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
