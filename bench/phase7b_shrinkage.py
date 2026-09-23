"""Phase 7 follow-up: does shrinkage calibration stabilise small-data heads?

For n in {10, 25, 50, 100} x 3 seeds: train the Phase 7 head (per-channel
renderer), then evaluate it raw (alpha = 1) and with alpha calibrated on the
validation set (photostyle.train.calibrate_shrinkage). Same data, splits and
test subset as the Phase 7 curve.

Usage:
    uv run python -m bench.phase7b_shrinkage --out bench/results/phase7b
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from bench.learning_curve import Pairs, evaluate, hwc, l1_loss, render_items, seed_all, summarise, train
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.render import GlobalRenderer
from photostyle.train import calibrate_shrinkage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase7b"))
    ap.add_argument("--sizes", default="10,25,50,100")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(Path("data/fivek_landscape_c"), "c", fx)
    pool = data.split("train") + data.split("validation")
    random.Random(1234).shuffle(pool)
    val, pool = pool[:30], pool[30:]
    test = random.Random(99).sample(data.split("test"), 200)
    r = GlobalRenderer("per_channel")
    res = {"runs": []}
    for n in map(int, args.sizes.split(",")):
        for seed in (0, 1, 2):
            seed_all(seed)
            items = random.Random(seed).sample(pool, n)
            m = Head(FEATURE_DIM, r.num_params)
            m.set_norm(torch.stack([it["feat"] for it in pool]))
            train(m, r, items, val)
            raw = summarise(evaluate([hwc(o) for o in render_items(m, r, test)[0]], test))
            a = calibrate_shrinkage(m, r, items, val, loss_fn=l1_loss)
            cal = summarise(evaluate([hwc(o) for o in render_items(m, r, test)[0]], test))
            res["runs"].append({"n": n, "seed": seed, "alpha": a, "raw": raw, "calibrated": cal})
            print(f"n={n} seed={seed} raw={raw['psnr']:.2f} alpha={a:.2f} cal={cal['psnr']:.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            (args.out / "results.json").write_text(json.dumps(res, indent=1))
    summ = {}
    for n in sorted({x["n"] for x in res["runs"]}):
        rr = [x for x in res["runs"] if x["n"] == n]
        summ[n] = {k: {"mean": float(np.mean([x[k]["psnr"] for x in rr])), "min": float(min(x[k]["psnr"] for x in rr))}
                   for k in ("raw", "calibrated")}
    res["summary"] = summ
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
