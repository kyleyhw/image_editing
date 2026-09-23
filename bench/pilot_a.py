"""Pilot A: the Clean Cool look on landscapes, learned without pairs (A2.2).

Training signals (no expert targets are used):

  1. profile loss      rendered FiveK landscape *originals* are pulled toward
                       the Clean Cool look profile (per regime: night/day);
  2. distribution loss sliced Wasserstein between rendered Lab pixels and the
                       openly licensed seed set of the same regime;
  3. distort & recover each seed image is randomly re-graded with the
                       renderer, and the model must undo it (pseudo-pairs);
  4. regulariser       small L2 on parameters.

Models: ``head`` (content-adaptive) and ``preset`` (one learned parameter
vector, same losses). References: identity, a hand-built static Clean Cool
preset, and per-channel histogram matching to the seed set.

Evaluation on held-out FiveK *test* originals, reported for all four cells
(night/day x urban/nature) and for two regions (top third ~ sky, bottom two
thirds ~ ground):

  * profile distance: mean |stat - target| / tolerance (lower = closer to look);
  * parameter spread: effect size |mean_A - mean_B| / pooled sd of predicted
    parameters between night and day, and between urban and nature.

Usage:
    uv run python -m bench.pilot_a --out bench/results/pilot_a
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from bench.learning_curve import Pairs, hist_match, seed_all, shrink, small, to_tensor
from photostyle.export import bake_lut, write_cube
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head, Preset
from photostyle.looks import (
    CLEAN_COOL,
    chroma_keep_loss,
    chroma_retention,
    clip_fraction,
    detail_similarity,
    fidelity_loss,
    lab_pixels,
    profile_distance,
    profile_loss,
    sliced_wasserstein,
)
from photostyle.render import GlobalRenderer
from photostyle.stats import colour_stats, regime_of

EDGE = 256


def regime_t(img: torch.Tensor) -> str:
    arr = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    return regime_of(colour_stats(arr))


def handmade_preset(r: GlobalRenderer) -> torch.Tensor:
    """A static Clean Cool grade built by hand: crushed blacks, blue-lifted shadows,
    slightly cooler, gently desaturated. The baseline the adaptive model must beat."""
    th = torch.zeros(1, r.num_params)
    K = r.K
    th[0, 0], th[0, K], th[0, 2 * K] = -0.03, -0.025, 0.0     # black points R, G, B
    th[0, 1:3] = -0.15                                         # R: deeper low-mids
    th[0, 2 * K + 1:2 * K + 3] = 0.10                          # B: lifted low-mids
    q = r.unpack(th)
    desat = -0.08 * (torch.eye(3) - torch.full((3, 3), 1 / 3))
    q["dM"].copy_(desat.reshape(1, 9))
    q["bias"].copy_(torch.tensor([[-0.005, 0.0, 0.01]]))
    return th


def load_seed(root: Path) -> list[dict]:
    items = []
    with Path("data/manifests/clean_cool_landscape.csv").open() as f:
        for row in csv.DictReader(f):
            p = root / row["file"]
            if p.exists():
                items.append({"img": shrink(to_tensor(Image.open(p)), EDGE), "regime": row["regime"]})
    return items


def pseudo_pairs(seed: list[dict], r: GlobalRenderer, fx: FeatureExtractor, k: int, rng: torch.Generator):
    """Distort-and-recover: random global re-grades of seed images."""
    pairs = []
    for it in seed:
        for _ in range(k):
            th = torch.zeros(1, r.num_params)
            q = r.unpack(th)
            q["curve"].copy_(torch.randn(q["curve"].shape, generator=rng) * 0.12)
            q["curve"].view(1, 3, r.K)[:, :, 0] = torch.randn(1, 3, generator=rng) * 0.03
            q["dM"].copy_(torch.randn(q["dM"].shape, generator=rng) * 0.04)
            q["bias"].copy_(torch.randn(q["bias"].shape, generator=rng) * 0.02)
            with torch.no_grad():
                d = r(it["img"][None], th)[0]
            pairs.append({"src": d, "tgt": it["img"], "feat": fx(d), "regime": it["regime"]})
    return pairs


def no_vignette(theta: torch.Tensor) -> torch.Tensor:
    """Clean Cool has no vignette: the first run used it to darken corners and game the stats."""
    return torch.cat([theta[:, :-1], torch.zeros_like(theta[:, -1:])], 1)


def train(model, r, inputs, seed, pseudo, steps, lr=3e-3, w=(1.0, 0.5, 2.0, 1.0), keep_chroma=0.0):
    """Items may carry a precomputed training-size sky mask in "sky_s" (``--learned-sky``)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    seed_pix = {g: torch.cat([lab_pixels(s["img"][None], 4096) for s in seed if s["regime"] == g])
                for g in ("night", "day")}
    log = []
    for step in range(1, steps + 1):
        model.train()
        batch = random.sample(inputs, 8)
        theta = no_vignette(model(torch.stack([b["feat"] for b in batch])))
        srcs = [small(b, "src")[None] for b in batch]
        outs = [r(x, th[None]) for x, th in zip(srcs, theta)]
        lp = sum(profile_loss(o, [b["regime"]], CLEAN_COOL, ref=x)
                 for o, b, x in zip(outs, batch, srcs)) / len(batch)
        lf = sum(fidelity_loss(o, x) for o, x in zip(outs, srcs)) / len(batch)
        if keep_chroma:
            lf = lf + keep_chroma * sum(chroma_keep_loss(o, x, sky=b.get("sky_s"))
                                        for o, x, b in zip(outs, srcs, batch)) / len(batch)
        ls = 0.0
        for g in ("night", "day"):
            og = [o for o, b in zip(outs, batch) if b["regime"] == g]
            if og and len(seed_pix[g]):
                ls = ls + sliced_wasserstein(torch.cat([lab_pixels(o, 1024) for o in og]), seed_pix[g])
        pb = random.sample(pseudo, 8)
        pt = no_vignette(model(torch.stack([p["feat"] for p in pb])))
        lr_ = sum((r(p["src"][None], th[None])[0] - p["tgt"]).abs().mean() for p, th in zip(pb, pt)) / 8
        loss = (w[0] * lp + w[1] * ls + w[2] * lr_ + w[3] * lf
                + 1e-3 * (theta.pow(2).mean() + pt.pow(2).mean()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            log.append({"step": step, "profile": float(lp), "swd": float(ls), "recover": float(lr_),
                        "fidelity": float(lf)})
            print(f"  step {step}: profile={float(lp):.3f} swd={float(ls):.4f} recover={float(lr_):.4f} "
                  f"fidelity={float(lf):.4f}", flush=True)
    model.eval()
    return log


def regions(img: torch.Tensor) -> dict[str, torch.Tensor]:
    h = img.shape[1]
    return {"all": img, "top": img[:, : h // 3], "bottom": img[:, h // 3:]}


@torch.no_grad()
def evaluate(name, theta, r, test, results):
    rows = []
    for it, th in zip(test, theta):
        out = r(it["src"][None], th[None])[0] if th is not None else it["_out"]
        src = it["src"][None]
        row = {"file": it["file"], "regime": it["regime"], "subject": it["meta"]["subject"],
               "new_clip": float(clip_fraction(out[None]) - clip_fraction(src)),
               "detail": float(detail_similarity(out[None], src)),
               "chroma_kept": chroma_retention(out[None], src)}
        if "sky_full" in it:
            row["sky_chroma_kept"] = chroma_retention(out[None], src, sky=it["sky_full"], sky_only=True)
        ref_regions = regions(it["src"])
        for rg, im in regions(out).items():
            d = profile_distance(im, it["regime"], CLEAN_COOL, ref=ref_regions[rg])
            row[f"dist_{rg}"] = float(np.mean(list(d.values())))
            if rg == "all":
                row.update({f"d_{k}": v for k, v in d.items()})
        rows.append(row)
    results["methods"][name] = rows
    cells = {}
    for reg in ("night", "day"):
        for sub in ("man_made", "nature"):
            sel = [x for x in rows if x["regime"] == reg and x["subject"] == sub]
            if sel:
                cells[f"{reg}/{sub}"] = {
                    "n": len(sel), **{f"dist_{rg}": float(np.mean([x[f"dist_{rg}"] for x in sel]))
                                      for rg in ("all", "top", "bottom")},
                    "new_clip": float(np.mean([x["new_clip"] for x in sel])),
                    "detail": float(np.mean([x["detail"] for x in sel])),
                    "chroma_kept": float(np.mean([x["chroma_kept"] for x in sel]))}
                if "sky_chroma_kept" in sel[0]:
                    cells[f"{reg}/{sub}"]["sky_chroma_kept"] = float(np.nanmean([x["sky_chroma_kept"] for x in sel]))
    results["cells"][name] = cells
    print(f"{name:10s} " + "  ".join(f"{c}: {v['dist_all']:.2f} (clip {v['new_clip']:+.3f}, "
                                     f"detail {v['detail']:.2f}, chroma {v['chroma_kept']:.2f})" for c, v in cells.items()), flush=True)


def effect_sizes(theta: torch.Tensor, groups: list[str], a: str, b: str) -> list[float]:
    ta = theta[[g == a for g in groups]]
    tb = theta[[g == b for g in groups]]
    if len(ta) < 2 or len(tb) < 2:
        return []
    pooled = torch.sqrt((ta.var(0) + tb.var(0)) / 2).clamp_min(1e-6)
    return ((ta.mean(0) - tb.mean(0)).abs() / pooled).tolist()


def grid(rows_imgs, labels, path, tw=300):
    rows = []
    for imgs in rows_imgs:
        ims = [Image.fromarray((i.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8"))
               for i in imgs]
        for im in ims:
            im.thumbnail((tw, tw))
        row = Image.new("RGB", (len(ims) * (tw + 8) + 8, max(i.height for i in ims) + 8), "white")
        for j, im in enumerate(ims):
            row.paste(im, (8 + j * (tw + 8), 4))
        rows.append(row)
    head = Image.new("RGB", (rows[0].width, 22), "white")
    d = ImageDraw.Draw(head)
    for j, lab in enumerate(labels):
        d.text((8 + j * (tw + 8), 5), lab, fill="black")
    out = Image.new("RGB", (rows[0].width, 22 + sum(r.height for r in rows)), "white")
    out.paste(head, (0, 0))
    y = 22
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--seed-dir", type=Path, default=Path("data/styles/clean_cool_landscape/images"))
    ap.add_argument("--owner", type=Path, default=Path("data/owner"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/pilot_a"))
    ap.add_argument("--private-out", type=Path, default=Path("data/owner/outputs/pilot_a"))
    ap.add_argument("--inputs", type=int, default=800)
    ap.add_argument("--test", type=int, default=160)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-chroma", type=float, default=0.0,
                    help="weight of the sky / saturated-subject chroma floor (v2); 0 = the original pilot")
    ap.add_argument("--tag", default="", help="suffix for the checkpoint name")
    ap.add_argument("--learned-sky", action="store_true",
                    help="use the learned sky segmenter (photostyle.sky) for the chroma floor and sky metrics (v3)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    args.private_out.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    t0 = time.time()

    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, "c", fx)
    pool = random.Random(7).sample(data.split("train") + data.split("validation"), args.inputs)
    test = random.Random(8).sample(data.split("test"), args.test)
    for it in pool + test:
        it["regime"] = regime_t(small(it, "src"))
    if args.learned_sky:
        from photostyle.sky import sky_mask

        for it in pool:
            it["sky_s"] = sky_mask(small(it, "src")[None])
        for it in test:
            it["sky_full"] = sky_mask(it["src"][None])
        print(f"sky masks for {len(pool) + len(test)} images [{time.time() - t0:.0f}s]", flush=True)
    seed = load_seed(args.seed_dir)
    r = GlobalRenderer("per_channel")
    pseudo = pseudo_pairs(seed, r, fx, k=16, rng=torch.Generator().manual_seed(0))
    counts = {g: sum(it["regime"] == g for it in test) for g in ("night", "day")}
    print(f"inputs={len(pool)} test={len(test)} {counts} seed={len(seed)} pseudo={len(pseudo)} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    results = {"profile": CLEAN_COOL.name, "counts": {"inputs": len(pool), "test": len(test), **counts,
                                                      "seed": len(seed), "pseudo_pairs": len(pseudo)},
               "methods": {}, "cells": {}, "spread": {}, "train_log": {}}
    feats_test = torch.stack([it["feat"] for it in test])

    # references
    evaluate("identity", torch.zeros(len(test), r.num_params), r, test, results)
    hm = handmade_preset(r)
    evaluate("handmade", hm.expand(len(test), -1), r, test, results)
    ref_cdf = np.zeros((3, 256))
    for s in seed:
        for c in range(3):
            h, _ = np.histogram(s["img"][c].numpy(), bins=256, range=(0, 1))
            ref_cdf[c] += np.cumsum(h) / h.sum()
    ref_cdf /= len(seed)
    for it in test:
        it["_out"] = torch.from_numpy(hist_match(it["src"], ref_cdf)).permute(2, 0, 1).float()
    evaluate("histmatch", [None] * len(test), r, test, results)

    models = {}
    for kind in ("preset", "head"):
        seed_all(0)
        m = Preset(r.num_params) if kind == "preset" else Head(FEATURE_DIM, r.num_params)
        if kind == "head":
            m.set_norm(torch.stack([it["feat"] for it in pool]))
        print(f"training {kind} ...", flush=True)
        results["train_log"][kind] = train(m, r, pool, seed, pseudo, args.steps, keep_chroma=args.keep_chroma)
        with torch.no_grad():
            theta = no_vignette(m(feats_test))
        evaluate(kind, theta, r, test, results)
        models[kind] = (m, theta)

    theta = models["head"][1]
    results["spread"] = {
        "night_vs_day": effect_sizes(theta, [it["regime"] for it in test], "night", "day"),
        "urban_vs_nature": effect_sizes(theta, [it["meta"]["subject"] for it in test], "man_made", "nature"),
        "theta_std_mean": float(theta.std(0).mean()),
    }
    for k in ("night_vs_day", "urban_vs_nature"):
        es = results["spread"][k]
        results["spread"][k + "_n_params_d>0.8"] = int(sum(e > 0.8 for e in es))
    print("spread:", {k: v for k, v in results["spread"].items() if not isinstance(v, list)}, flush=True)

    # Gate: in every cell the head must beat the hand-made static preset on
    # profile distance (whole image, top and bottom regions) WITHOUT paying for
    # it in fidelity: new clipping <= 1 % and detail similarity >= 0.95 and
    # no worse than the hand-made preset's by more than 0.02.
    gate = {}
    for cell, v in results["cells"]["head"].items():
        h = results["cells"]["handmade"][cell]
        gate[cell] = (all(v[f"dist_{rg}"] < h[f"dist_{rg}"] for rg in ("all", "top", "bottom"))
                      and v["new_clip"] <= 0.01 and v["detail"] >= 0.95
                      and v["detail"] >= h["detail"] - 0.02)
    results["gate_vs_handmade"] = gate
    print("gate:", gate, flush=True)

    torch.save({"state_dict": models["head"][0].state_dict(), "renderer": "per_channel", "look": "clean_cool"},
               Path("checkpoints") / f"pilot_a_clean_cool_head{args.tag}.pt")
    write_cube(bake_lut(r, theta.mean(0)), args.out / "clean_cool_average.cube", title="Clean Cool (average)")
    write_cube(bake_lut(r, hm[0]), args.out / "clean_cool_handmade.cube", title="Clean Cool (hand-made)")

    # Report figure: AdobeMIT-licensed test images only, a mix of regimes and subjects.
    showcase = [it for it in test if it["meta"]["license"] == "AdobeMIT"]
    showcase = sorted(showcase, key=lambda it: (it["regime"], it["meta"]["subject"]))
    pick = [it for g in ("night", "day") for it in [x for x in showcase if x["regime"] == g][:3]]
    with torch.no_grad():
        rows = []
        for it in pick:
            i = test.index(it)
            rows.append([it["src"], r(it["src"][None], hm)[0], r(it["src"][None], models["preset"][1][i:i + 1])[0],
                         r(it["src"][None], models["head"][1][i:i + 1])[0]])
    grid(rows, ["input (FiveK, AdobeMIT)", "hand-made preset", "learned preset", "adaptive (head)"],
         args.out / "pilot_a_grid.jpg")

    # Owner photos (private): unedited-or-unknown ones, full-size proxies at 1024 px.
    man = args.owner / "manifest.csv"
    if man.exists():
        with man.open() as f:
            own = [row for row in csv.DictReader(f) if row["edited"] != "yes"]
        rows = []
        for row in own:
            src = shrink(to_tensor(Image.open(args.owner / "srgb" / f"{row['id']}.jpg")), 1024)
            with torch.no_grad():
                th = no_vignette(models["head"][0](fx(src)[None]))
                out = r(src[None], th)[0]
            write_cube(bake_lut(r, th[0]), args.private_out / f"{row['id']}.cube", title=f"{row['id']} Clean Cool")
            Image.fromarray((out.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
                args.private_out / f"{row['id']}_clean_cool.jpg", quality=92)
            rows.append([src, r(src[None], hm)[0], out])
        grid(rows, ["your photo", "hand-made preset", "adaptive Clean Cool"], args.private_out / "grid.jpg", tw=360)

    for it in test:
        it.pop("_out", None)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
