"""Phase 9 gate: renderer ablation on FiveK landscapes (expert C).

Same head, same Phase 7 protocol; only the renderer changes:

  shared            one shared tone curve + matrix (Phase 3 renderer)
  per_channel       per-channel monotone curves + matrix
  lut4 / lut8       per_channel + residual 9^3 basis LUTs (N = 4 / 8)
  zeng33            shared curve + three 33^3 basis LUTs (Zeng et al.-like)
  csr               CSRNet-style FiLM-modulated per-pixel MLP (not editable)

Gate (PROJECT_PLAN Phase 9): keep the LUT stage only if it beats
curves + matrix by > 0.3 dB at <= 250 pairs.

Usage:
    uv run python -m bench.phase9 --out bench/results/phase9
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from bench.learning_curve import Pairs, evaluate, hwc, render_items, seed_all, summarise, train
from photostyle.csr import CSRLite
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.lut import LUTRenderer
from photostyle.render import GlobalRenderer


def make(name: str):
    return {
        "shared": lambda: GlobalRenderer("shared"),
        "per_channel": lambda: GlobalRenderer("per_channel"),
        "lut4": lambda: LUTRenderer("per_channel", n_basis=4, size=9),
        "lut8": lambda: LUTRenderer("per_channel", n_basis=8, size=9),
        "zeng33": lambda: LUTRenderer("shared", n_basis=3, size=33),
        "csr": lambda: CSRLite(),
    }[name]()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase9"))
    ap.add_argument("--sizes", default="100,250,all")
    ap.add_argument("--renderers", default="shared,per_channel,lut4,lut8,zeng33,csr")
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, "c", fx)
    pool = data.split("train") + data.split("validation")
    rng = random.Random(1234)
    rng.shuffle(pool)
    val, pool = pool[:30], pool[30:]
    test = random.Random(99).sample(data.split("test"), args.test)   # same subset as Phase 7

    out_path = args.out / "results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {"runs": []}
    done = {(r["n"], r["renderer"]) for r in results["runs"]}
    for s in args.sizes.split(","):
        n = len(pool) if s == "all" else int(s)
        items = random.Random(0).sample(pool, n)
        for name in args.renderers.split(","):
            if (n, name) in done:
                continue
            seed_all(0)
            r = make(name)
            m = Head(FEATURE_DIM, r.num_params)
            m.set_norm(torch.stack([it["feat"] for it in pool]))
            info = train(m, r, items, val)
            outs, _ = render_items(m, r, test)
            summ = summarise(evaluate([hwc(o) for o in outs], test))
            results["runs"].append({"n": n, "renderer": name, "params": r.num_params, **info, **summ})
            print(f"n={n} {name:12s} psnr={summ['psnr']:.2f} dE={summ['delta_e']:.2f} "
                  f"steps={info['steps']} [{time.time() - t0:.0f}s]", flush=True)
            out_path.write_text(json.dumps(results, indent=1))

    by = {(r["n"], r["renderer"]): r["psnr"] for r in results["runs"]}
    gate = {}
    for n in sorted({r["n"] for r in results["runs"]}):
        if (n, "per_channel") in by:
            best_lut = max((by.get((n, k), -1e9) for k in ("lut4", "lut8")), default=-1e9)
            gate[n] = {"lut_gain_db": best_lut - by[(n, "per_channel")],
                       "keep_lut": n <= 250 and best_lut - by[(n, "per_channel")] > 0.3}
    results["gate"] = gate
    out_path.write_text(json.dumps(results, indent=1))
    print("gate:", gate)


if __name__ == "__main__":
    main()
