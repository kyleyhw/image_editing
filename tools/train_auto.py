"""Phase 21 task 18: train a slider model on MIT-Adobe FiveK expert edits.

The base correction model predicts the ten correction sliders (white balance, exposure, contrast,
the four tone regions, vibrance, saturation) for each photo; with --colour it also predicts the 24
HSL sliders (an "expert style"). Training renders through the differentiable pipeline and
minimises the Oklab distance to the expert plus an L1 penalty on the sliders (photostyle/auto.py).

Validation: the FiveK validation split when it is on disk, else the last 10 % of train. lambda
(the L1 weight) is chosen on validation from --lams. The final model is evaluated on the FiveK
test split when present.

    uv run python tools/train_auto.py --expert c --edge 128 --epochs 40 --lams 0.003 0.01 0.03
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
from photostyle import pipeline as P  # noqa: E402
from photostyle.colour_diff import ciede2000, srgb_to_lab  # noqa: E402

DATA = ROOT / "data/fivek"
IMG = (".png", ".jpg", ".jpeg")


def misaligned() -> set[tuple[str, str]]:
    """Pairs whose original and expert frames do not register (Phase 20 report: align_r < 0.6)."""
    import csv
    man = ROOT / "data/manifests/fivek.csv"
    if not man.exists():
        return set()
    return {(r["split"], f"img_{int(r['idx']):04d}") for r in csv.DictReader(man.open()) if float(r["align_r"] or 1) < 0.6}


def split_files(split: str, expert: str, limit: int | None) -> list[Path]:
    bad = misaligned()
    src = sorted(f for f in (DATA / split / "original").iterdir() if f.suffix.lower() in IMG) if (DATA / split / "original").exists() else []
    tdir = DATA / split / f"expert_{expert}"
    return [f for f in src if (split, f.stem) not in bad and any((tdir / (f.stem + e)).exists() for e in IMG)][:limit]


def de00(a: torch.Tensor, b: torch.Tensor) -> float:
    A_ = a.clamp(0, 1).permute(0, 2, 3, 1).double().numpy()
    B_ = b.clamp(0, 1).permute(0, 2, 3, 1).double().numpy()
    return float(ciede2000(srgb_to_lab(A_), srgb_to_lab(B_)).mean())


def score(model: A.SliderModel, feats: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> dict[str, float]:
    s = model.predict(feats)
    with torch.no_grad():
        out = torch.cat([P.render(xs[i:i + 64], A.to_params(s[i:i + 64], model.names)) for i in range(0, len(xs), 64)])
    return {"psnr": float(np.mean([A.psnr(o, y) for o, y in zip(out, ys)])), "de00": de00(out, ys),
            "dok": float(A.oklab_distance(out, ys).mean() * 100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="c")
    ap.add_argument("--edge", type=int, default=128)
    ap.add_argument("--n", type=int, default=None, help="limit on training pairs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    ap.add_argument("--colour", action="store_true", help="also predict the 24 HSL sliders")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(3)     # the FiveK download shares the CPU
    names = A.CORRECTION + (A.COLOUR if args.colour else [])
    tag = f"{'style' if args.colour else 'auto'}_{args.expert}"
    out = args.out or ROOT / "photostyle/models" / f"{tag}.pt"
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)

    train_f = split_files("train", args.expert, args.n)
    val_f = split_files("validation", args.expert, None)
    test_f = split_files("test", args.expert, None)
    if len(val_f) < 50:                       # validation split not downloaded yet: hold out 10 % of train
        k = max(50, len(train_f) // 10)
        train_f, val_f = train_f[:-k], train_f[-k:]
    print(f"train {len(train_f)}, validation {len(val_f)}, test {len(test_f)}", flush=True)

    def load(files: list[Path], split_hint: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = cache / f"pairs_{split_hint}_{args.expert}_{args.edge}_{len(files)}.pt"
        if key.exists():
            d = torch.load(key)
            return d["x"], d["y"], d["f"]
        out = _load(files)
        torch.save({"x": out[0], "y": out[1], "f": out[2]}, key)
        return out

    def _load(files: list[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        by_split: dict[str, list[Path]] = {}
        for f in files:
            by_split.setdefault(f.parent.parent.name, []).append(f)
        xs, ys, fs = [], [], []
        for sp, fl in by_split.items():
            x, y, stems = A.load_pairs(DATA, sp, args.expert, args.edge, None)
            keep = [i for i, s in enumerate(stems) if s in {f.stem for f in fl}]
            xs.append(x[keep])
            ys.append(y[keep])
            fs.append(A.descriptors([DATA / sp / "original" / (stems[i] + ".png") for i in keep], cache / f"desc_{sp}_{len(keep)}_{stems[keep[0]]}.pt"))
        return torch.cat(xs), torch.cat(ys), torch.cat(fs)

    xtr, ytr, ftr = load(train_f, "train")
    xva, yva, fva = load(val_f, "validation")
    print(f"loaded in {time.time() - t0:.0f} s; {xtr.shape[2]}x{xtr.shape[3]} px", flush=True)

    report: dict = {"expert": args.expert, "sliders": names, "edge": args.edge, "n_train": len(xtr), "n_val": len(xva),
                    "identity_val": {"psnr": float(np.mean([A.psnr(x, y) for x, y in zip(xva, yva)])), "de00": de00(xva, yva)}}
    print("identity on validation", report["identity_val"], flush=True)
    best = None
    for lam in args.lams:
        model, hist = A.train(ftr, xtr, ytr, names, lam=lam, epochs=args.epochs, val=(fva, xva, yva))
        sc = score(model, fva, xva, yva)
        report[f"lam_{lam}"] = {**sc, "history": hist}
        print(f"lambda {lam}: {sc}", flush=True)
        if best is None or sc["de00"] < best[1]["de00"]:
            best = (lam, sc, model)
    lam, sc, model = best
    report["chosen_lambda"] = lam
    report["val"] = sc
    if test_f:
        xte, yte, fte = load(test_f, "test")
        report["identity_test"] = {"psnr": float(np.mean([A.psnr(x, y) for x, y in zip(xte, yte)])), "de00": de00(xte, yte)}
        report["test"] = score(model, fte, xte, yte)
        print("test", report["test"], "identity", report["identity_test"], flush=True)
    model.save(out)
    pv = model.predict(fva)
    report["slider_stats_val"] = {n: {"median": round(float(pv[:, i].median()), 2), "p05": round(float(pv[:, i].quantile(0.05)), 2),
                                      "p95": round(float(pv[:, i].quantile(0.95)), 2)} for i, n in enumerate(names) if n in A.CORRECTION}
    report["runtime_s"] = round(time.time() - t0, 1)
    (ROOT / "tests/reports" / f"phase21_{tag}.json").write_text(json.dumps(report, indent=1))
    print(f"saved {out}; {report['runtime_s']} s")


if __name__ == "__main__":
    main()
