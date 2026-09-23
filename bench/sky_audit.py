"""Sky mask audit: heuristic (Phase 16) vs learned segmenter (photostyle.sky).

Same 12 AdobeMIT FiveK test images as the Phase 16 halo audit, plus the
owner's photos (private output). No ground truth exists here, so the figure
is the main evidence; as a false-positive proxy we report the mean sky
probability in the bottom half of each frame (for these landscapes that is
water, ground, snow, foliage: it should be near 0), and per-image runtime.

Usage:
    uv run python -m bench.sky_audit --out bench/results/sky_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from bench.phase18 import to_t
from bench.pilot_a import grid
from photostyle.regional import sky_probability
from photostyle.sky import sky_mask
from photostyle.train import shrink


def overlay(img: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    o = img.clone()
    o[0] = (o[0] * (1 - 0.6 * p) + 0.9 * 0.6 * p).clamp(0, 1)
    o[2] = (o[2] * (1 - 0.6 * p)).clamp(0, 1)
    return o


def run(paths: list[Path], labels: list[str]):
    rows, recs = [], []
    for path, lab in zip(paths, labels):
        img = shrink(to_t(path), 1024)[None]
        t = time.time()
        pm = sky_mask(img)[0, 0]
        dt = time.time() - t
        ph = sky_probability(img)[0, 0]
        h = pm.shape[0] // 2
        recs.append({"image": lab, "model_s": dt,
                     "bottom_half_heuristic": float(ph[h:].mean()), "bottom_half_model": float(pm[h:].mean()),
                     "top_half_heuristic": float(ph[:h].mean()), "top_half_model": float(pm[:h].mean())})
        rows.append([img[0], overlay(img[0], ph), overlay(img[0], pm)])
        print(f"{lab}: model {dt:.1f}s  bottom-half sky heuristic {recs[-1]['bottom_half_heuristic']:.3f} "
              f"model {recs[-1]['bottom_half_model']:.3f}", flush=True)
    return rows, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/sky_audit"))
    ap.add_argument("--owner", type=Path, default=Path("data/owner"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.data / "meta.csv").open() as f:
        meta = [r for r in csv.DictReader(f) if r["split"] == "test" and r["license"].strip() == "AdobeMIT"]
    # Phase 16's audit used the first 12 AdobeMIT images of its test sample; use the first 12 here.
    meta = meta[:12]
    rows, recs = run([args.data / "original" / m["file"] for m in meta], [m["file"] for m in meta])
    summary = {k: float(np.mean([r[k] for r in recs])) for k in recs[0] if k != "image"}
    (args.out / "results.json").write_text(json.dumps({"images": recs, "mean": summary}, indent=1))
    grid(rows, ["input (FiveK, AdobeMIT)", "heuristic sky (orange)", "learned sky (orange)"],
         args.out / "sky_audit.jpg", tw=300)
    print("mean:", summary, flush=True)

    man = args.owner / "manifest.csv"
    if man.exists():
        with man.open() as f:
            own = list(csv.DictReader(f))
        priv = args.owner / "outputs" / "sky_audit"
        priv.mkdir(parents=True, exist_ok=True)
        prow, _ = run([args.owner / "srgb" / f"{r['id']}.jpg" for r in own],
                      [f"photo {i}" for i in range(1, len(own) + 1)])
        grid(prow, ["your photo", "heuristic sky", "learned sky"], priv / "sky_audit.jpg", tw=300)


if __name__ == "__main__":
    main()
