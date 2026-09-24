"""Calibrate the direction-only amounts of a tutorial recipe against reference photos.

Tutorials often give a direction without an amount ("lime green in the shadows",
"deepens blacks"). This scales only those rows (kind == "direction") by a
non-negative factor. Numeric rows stay as the tutorial states them, and no
direction can flip. The factors are chosen so that the recipe applied to
ordinary open input photos matches the reference photos' colour statistics
(the same CIELAB statistics used to pick references).

    uv run python tools/calibrate_recipe.py fujifilm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle import newstyle as ns  # noqa: E402
from photostyle import recipes  # noqa: E402
from photostyle.develop import develop  # noqa: E402
from photostyle.stats import colour_stats  # noqa: E402
from photostyle.io import to_pil  # noqa: E402

FACTORS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def scaled(table, mult):
    out = []
    for i, (tool, param, value, srcs, kind, note) in enumerate(table):
        m = mult.get(i, 1.0)
        if isinstance(value, tuple) and tool == "color_grading":
            value = (value[0], value[1] * m)
        elif isinstance(value, tuple):
            value = tuple(v * m for v in value)
        else:
            value = value * m
        out.append((tool, param, value, srcs, kind, note))
    return out


def stats_vec(imgs):
    return np.array([[colour_stats(to_pil(x))[k] for k in ns.STAT_KEYS] for x in imgs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--inputs", type=int, default=60)
    ap.add_argument("--edge", type=int, default=256)
    args = ap.parse_args()
    table = recipes.RECIPE_TABLES[args.name]
    p = ns.Project.load(args.name)
    refs = [ns._tensor(p.candidates[i]["file"], args.edge) for i in p.refs]
    pool = ns.input_pool(exclude=args.name, include_owner=False)
    ins = [ns._tensor(f, args.edge) for f in random.Random(0).sample(pool, min(args.inputs, len(pool)))]
    R = stats_vec(refs)
    mu, sd = R.mean(0), stats_vec(ins).std(0) + 1e-6

    def loss(mult):
        s = recipes.settings(scaled(table, mult))
        with torch.no_grad():
            out = stats_vec([develop(x, s) for x in ins])
        return float((((out.mean(0) - mu) / sd) ** 2).mean())

    free = [i for i, row in enumerate(table) if row[4] == "direction"]
    mult = {i: 1.0 for i in free}
    best = loss(mult)
    print(f"start loss {best:.3f} (identity {loss({i: 0.0 for i in free}):.3f})", flush=True)
    for sweep in range(2):
        for i in free:
            for f in FACTORS:
                if f == mult[i]:
                    continue
                trial = {**mult, i: f}
                v = loss(trial)
                if v < best - 1e-4:
                    best, mult = v, trial
            print(f"sweep {sweep} row {i} {table[i][0]}.{table[i][1]}: x{mult[i]} loss {best:.3f}", flush=True)
    res = {"name": args.name, "loss": best, "n_refs": len(refs), "n_inputs": len(ins),
           "factors": {f"{table[i][0]}.{table[i][1]}": mult[i] for i in free},
           "values": {f"{t}.{pa}": v for t, pa, v, *_ in scaled(table, mult)}}
    out = Path("data/styles") / args.name / "recipe_calibration.json"
    out.write_text(json.dumps(res, indent=1, default=list))
    print(json.dumps(res["factors"], indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
