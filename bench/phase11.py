"""Phase 11 gate: learning a look from example photos only (no pairs), measured.

Zeng et al.'s disjoint protocol on FiveK expert C landscapes: the pool is
split in half. The *inputs* are unedited originals from half A; the *style
examples* are expert-C retouches of half B (different photos). The model
never sees a pair. PSNR on the paired test set then measures how close
unpaired learning gets to expert C.

Loss ablation (all include the parameter L2 regulariser):
  swd            colour-distribution matching to the examples (+ fidelity)
  pseudo         distort-and-recover pseudo-pairs from the examples only
  swd+pseudo     both (the Pilot A recipe without a look profile)
at 25 / 100 / 500 examples. References: identity and the paired model at
the same n (from the Phase 7 curve). Gate: adopt the simplest loss within
0.5 dB of the best.

Usage:
    uv run python -m bench.phase11 --out bench/results/phase11
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from bench.learning_curve import Pairs, evaluate, hwc, render_items, summarise
from photostyle.features import FeatureExtractor
from photostyle.train import learn_unpaired

VARIANTS = {"swd": dict(use_pseudo=False, use_swd=True, use_fidelity=True),
            "pseudo": dict(use_pseudo=True, use_swd=False, use_fidelity=False),
            "swd+pseudo": dict(use_pseudo=True, use_swd=True, use_fidelity=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase11"))
    ap.add_argument("--sizes", default="25,100,500")
    ap.add_argument("--inputs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, "c", fx)
    pool = data.split("train") + data.split("validation")
    random.Random(1234).shuffle(pool)
    half = len(pool) // 2
    inputs = [it["src"] if it["src"] is not None else it["src_s8"].float() / 255 for it in pool[:half][: args.inputs]]
    styles = pool[half:]
    test = random.Random(99).sample(data.split("test"), args.test)     # same subset as Phase 7
    res = {"identity": summarise(evaluate([hwc(it["src"]) for it in test], test)), "runs": []}
    print(f"identity {res['identity']['psnr']:.2f} [{time.time() - t0:.0f}s]", flush=True)

    for n in map(int, args.sizes.split(",")):
        ex = [it["tgt_s8"].float() / 255 for it in random.Random(0).sample(styles, min(n, len(styles)))]
        for name, kw in VARIANTS.items():
            head, r, _, info = learn_unpaired(ex, inputs, fx, steps=args.steps, **kw)
            outs, theta = render_items(head, r, test)
            s = summarise(evaluate([hwc(o) for o in outs], test))
            res["runs"].append({"n": n, "variant": name, **s, "theta_spread": float(theta.std(0).mean())})
            print(f"n={n} {name:11s} psnr={s['psnr']:.2f} dE={s['delta_e']:.2f} [{time.time() - t0:.0f}s]", flush=True)
            (args.out / "results.json").write_text(json.dumps(res, indent=1))

    gate = {}
    order = ["swd", "pseudo", "swd+pseudo"]          # simplest first
    for n in sorted({r["n"] for r in res["runs"]}):
        by = {r["variant"]: r["psnr"] for r in res["runs"] if r["n"] == n}
        best = max(by.values())
        gate[n] = {"best": best, "adopt": next(v for v in order if by.get(v, -1e9) >= best - 0.5)}
    res["gate"] = gate
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    print("gate:", gate, f"done in {time.time() - t0:.0f}s")



if __name__ == "__main__":
    main()
