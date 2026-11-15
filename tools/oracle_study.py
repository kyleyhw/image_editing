"""Phase 21 task 17: how close can the slider pipeline get to the FiveK experts?

For each photo, the best slider values are found by gradient descent on the Oklab distance to the
expert's edit (an oracle: it sees the target). The resulting error is a floor for any model that
predicts those sliders. Three slider sets are compared:
  correction   10 sliders (white balance, exposure, tone, vibrance, saturation)
  + colour     plus the 24 HSL sliders
  + curve      plus a free RGB curve (8 interior points)
Baselines: the unedited original, and the mean expert edit... (identity only here).

    uv run python tools/oracle_study.py --n 128 --edge 128 --expert c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import auto as A  # noqa: E402
from photostyle.colour_diff import ciede2000, srgb_to_lab  # noqa: E402


def de00(a: torch.Tensor, b: torch.Tensor) -> float:
    A_ = a.clamp(0, 1).permute(0, 2, 3, 1).double().numpy()
    B_ = b.clamp(0, 1).permute(0, 2, 3, 1).double().numpy()
    return float(ciede2000(srgb_to_lab(A_), srgb_to_lab(B_)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--edge", type=int, default=128)
    ap.add_argument("--expert", default="c")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "tests/reports/phase21_oracle.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    xs, ys, names = A.load_pairs(ROOT / "data/fivek", args.split, args.expert, args.edge, args.n)
    print(f"{len(xs)} pairs at {xs.shape[2]}x{xs.shape[3]}", flush=True)
    res: dict = {"n": len(xs), "edge": args.edge, "expert": args.expert}
    res["identity"] = {"dok": float(A.oklab_distance(xs, ys).mean() * 100), "psnr": float(np.mean([A.psnr(x, y) for x, y in zip(xs, ys)])),
                       "de00": de00(xs, ys)}
    print("identity", res["identity"], flush=True)
    for name, sliders, curves in [("correction", A.CORRECTION, False), ("correction+colour", A.CORRECTION + A.COLOUR, False),
                                  ("correction+colour+curve", A.CORRECTION + A.COLOUR, True)]:
        outs, vals = [], []
        for i in range(0, len(xs), 64):
            o, v = A.oracle(xs[i:i + 64], ys[i:i + 64], sliders, steps=args.steps, curves=curves)
            outs.append(o)
            vals.append(v)
        out, val = torch.cat(outs), torch.cat(vals)
        r = A.ranges(sliders)
        sat = (val.abs() >= 0.98 * r).float().mean(0)
        res[name] = {"dok": float(A.oklab_distance(out, ys).mean() * 100),
                     "psnr": float(np.mean([A.psnr(o, y) for o, y in zip(out, ys)])), "de00": de00(out, ys),
                     "at_limit": {n: round(float(s), 3) for n, s in zip(sliders, sat) if s > 0},
                     "median_abs": {n: round(float(v), 2) for n, v in zip(sliders, val.abs().median(0).values) if n in A.CORRECTION}}
        print(name, {k: v for k, v in res[name].items() if k != "median_abs"}, flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
