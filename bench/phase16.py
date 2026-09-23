"""Phase 16 gate: do regional edits beat global-only edits on FiveK landscapes?

Compares GlobalRenderer("per_channel") with RegionalRenderer (global +
graduated filter + range masks + heuristic sky mask), trained with the same
Phase 7 protocol at a few training sizes. Reported for the whole image and
for the top third (~sky) and bottom two thirds (~ground). Also computes both
oracle ceilings and writes a sky-mask overlay sheet for the halo audit.

Usage:
    uv run python -m bench.phase16 --out bench/results/phase16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench.learning_curve import (
    Pairs,
    hwc,
    oracle_params,
    render_items,
    seed_all,
    small,
    train,
)
from bench.metrics import psnr
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.regional import RegionalRenderer, sky_probability
from photostyle.render import GlobalRenderer


def region_psnr(outs, items) -> dict[str, float]:
    res = {"all": [], "top": [], "bottom": []}
    for o, it in zip(outs, items):
        p, t = np.clip(hwc(o), 0, 1), hwc(it["tgt"])
        h = p.shape[0] // 3
        res["all"].append(psnr(p, t))
        res["top"].append(psnr(p[:h], t[:h]))
        res["bottom"].append(psnr(p[h:], t[h:]))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase16"))
    ap.add_argument("--sizes", default="250,all")
    ap.add_argument("--test", type=int, default=120)
    ap.add_argument("--oracle", type=int, default=40)
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
    test = random.Random(99).sample(data.split("test"), args.test)
    renderers = {"global": GlobalRenderer("per_channel"), "regional": RegionalRenderer("per_channel")}
    results = {"oracle": {}, "runs": []}

    for name, r in renderers.items():
        ot = test[: args.oracle]
        with torch.no_grad():
            outs = [r(it["src"][None], oracle_params(r, it, steps=300))[0] for it in ot]
        results["oracle"][name] = region_psnr(outs, ot)
        print(f"oracle/{name}: {results['oracle'][name]} [{time.time() - t0:.0f}s]", flush=True)

    for s in args.sizes.split(","):
        n = len(pool) if s == "all" else int(s)
        train_items = random.Random(0).sample(pool, n)
        for name, r in renderers.items():
            seed_all(0)
            m = Head(FEATURE_DIM, r.num_params)
            m.set_norm(torch.stack([it["feat"] for it in pool]))
            info = train(m, r, train_items, val)
            outs, _ = render_items(m, r, test)
            res = region_psnr(outs, test)
            results["runs"].append({"n": n, "renderer": name, **info, **res})
            print(f"n={n} {name}: {res} steps={info['steps']} [{time.time() - t0:.0f}s]", flush=True)
            (args.out / "results.json").write_text(json.dumps(results, indent=1))

    # Halo-audit sheet: sky-mask overlays on 12 AdobeMIT test images.
    pick = [it for it in test if it["meta"]["license"] == "AdobeMIT"][:12]
    tiles = []
    for it in pick:
        img = small(it, "src")
        p = sky_probability(img[None])[0, 0]
        over = img.clone()
        over[0] = torch.clamp(over[0] * (1 - 0.6 * p) + 0.9 * 0.6 * p, 0, 1)
        over[2] = torch.clamp(over[2] * (1 - 0.6 * p), 0, 1)
        both = torch.cat([img, over], 2)
        tiles.append(Image.fromarray((both.permute(1, 2, 0).numpy() * 255).round().astype("uint8")))
    w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (w * 2, sum(t.height for t in tiles[::2]) + 10), "white")
    y = 0
    for i in range(0, len(tiles), 2):
        sheet.paste(tiles[i], (0, y))
        if i + 1 < len(tiles):
            sheet.paste(tiles[i + 1], (w, y))
        y += max(t.height for t in tiles[i:i + 2])
    sheet.save(args.out / "sky_mask_audit.jpg", quality=85)

    g = {r_["n"]: r_ for r_ in results["runs"] if r_["renderer"] == "global"}
    rg = {r_["n"]: r_ for r_ in results["runs"] if r_["renderer"] == "regional"}
    results["gain_db"] = {n: {k: rg[n][k] - g[n][k] for k in ("all", "top", "bottom")} for n in g if n in rg}
    results["gate"] = {n: v["all"] >= 0.5 for n, v in results["gain_db"].items()}
    print("gain:", results["gain_db"], "gate:", results["gate"], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
