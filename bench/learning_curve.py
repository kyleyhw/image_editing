"""Phase 7 learning curve on paired data (FiveK landscape subset by default).

For each training-set size n and seed, fits:

  head/<renderer>    content-adaptive: frozen features -> Head -> renderer
  preset/<renderer>  one static parameter vector for all images (baseline)

and evaluates on the fixed test split against:

  identity           the unedited input
  histmatch          per-channel histogram specification to the mean
                     training-target CDF (a classical, non-learned look)
  oracle/<renderer>  parameters optimised per test image against its own
                     target: the renderer's ceiling

Protocol (fixes the Phase 6 confounds):
  * the pool is the train+validation splits; a fixed validation set of 30
    images is held out for early stopping, and the test split is never
    touched during training;
  * n training images are sampled from the rest of the pool with the seed;
  * frozen, ImageNet-normalised features are cached; there is no BatchNorm
    anywhere; aspect ratio is preserved;
  * training runs at a 256 px long edge and evaluation at the stored 512 px.

Outputs: <out>/results.json (every run, every image), <out>/summary.csv, and
<out>/learning_curve.png.

Usage:
    uv run python -m bench.learning_curve --data data/fivek_landscape_c --expert c
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
import torch.nn.functional as F
from PIL import Image

from bench.metrics import all_metrics
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head, Preset
from photostyle.render import GlobalRenderer

TRAIN_EDGE = 256


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def shrink(t: torch.Tensor, edge: int) -> torch.Tensor:
    _, h, w = t.shape
    s = edge / max(h, w)
    if s >= 1:
        return t
    return F.interpolate(t[None], size=(round(h * s), round(w * s)), mode="bilinear",
                         antialias=True, align_corners=False)[0]


def align_pair(src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Make an (original, expert) pair pixel-aligned, or return None.

    In the FiveK mirror some originals are stored unrotated while the expert
    retouch is rotated (e.g. 512x340 vs 341x512), and independent resizing
    leaves 1-2 px size differences. We try the original's 0/90/180/270 degree
    rotations, keep the one whose shape is within 2 px of the target and whose
    content matches best on a thumbnail, then resample the target to the
    original's exact size. Pairs that cannot be matched (e.g. crops) are dropped.
    """
    best, best_err = None, float("inf")
    for k in range(4):
        cand = torch.rot90(src, k, dims=(1, 2))
        if abs(cand.shape[1] - tgt.shape[1]) > 2 or abs(cand.shape[2] - tgt.shape[2]) > 2:
            continue
        t = F.interpolate(tgt[None], size=cand.shape[1:], mode="bilinear", align_corners=False)[0]
        a = F.adaptive_avg_pool2d(cand.mean(0, keepdim=True)[None], 16)
        b = F.adaptive_avg_pool2d(t.mean(0, keepdim=True)[None], 16)
        # compare structure, not brightness: the edit changes tone, not layout
        a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
        err = float((a - b).abs().mean())
        if err < best_err:
            best, best_err = (cand.contiguous(), t), err
    if best is None or best_err > 0.5:
        return None
    return best


def as_u8(t: torch.Tensor) -> torch.Tensor:
    return (t * 255).round().to(torch.uint8)


class Pairs:
    """In-memory paired dataset with cached features.

    To fit ~2,700 pairs in RAM, training-size copies are stored as uint8 and
    full-size (512 px) copies are kept only for ``full_splits``.
    """

    def __init__(self, root: Path, expert: str, fx: FeatureExtractor,
                 full_splits: tuple[str, ...] = ("test",)):
        with (root / "meta.csv").open() as f:
            meta = list(csv.DictReader(f))
        self.items = []
        self.dropped = 0
        for m in meta:
            o = root / "original" / m["file"]
            e = root / f"expert_{expert}" / m["file"]
            if not (o.exists() and e.exists()):
                continue
            pair = align_pair(to_tensor(Image.open(o)), to_tensor(Image.open(e)))
            if pair is None:
                self.dropped += 1
                continue
            src, tgt = pair
            full = m["split"] in full_splits
            self.items.append({
                "file": m["file"], "split": m["split"], "meta": m,
                "src": src if full else None, "tgt": tgt if full else None,
                "src_s8": as_u8(shrink(src, TRAIN_EDGE)), "tgt_s8": as_u8(shrink(tgt, TRAIN_EDGE)),
                "feat": fx.cached(src, f"{o}|{o.stat().st_mtime}|{tuple(src.shape)}"),
            })

    def split(self, name: str) -> list[dict]:
        return [it for it in self.items if it["split"] == name]


def small(it: dict, key: str) -> torch.Tensor:
    return it[key + "_s8"].float() / 255.0


@torch.no_grad()
def render_items(model, renderer, items, small_size=False):
    feats = torch.stack([it["feat"] for it in items])
    theta = model(feats)
    outs = []
    for it, th in zip(items, theta):
        img = small(it, "src") if small_size else it["src"]
        outs.append(renderer(img[None], th[None])[0])
    return outs, theta


def l1_loss(model, renderer, items) -> torch.Tensor:
    feats = torch.stack([it["feat"] for it in items])
    theta = model(feats)
    loss = 0.0
    for it, th in zip(items, theta):
        loss = loss + (renderer(small(it, "src")[None], th[None])[0] - small(it, "tgt")).abs().mean()
    reg = renderer.regularizer() if hasattr(renderer, "regularizer") else 0.0
    return loss / len(items) + 1e-4 * theta.pow(2).mean() + reg


def train(model, renderer, train_items, val_items, steps=3000, batch=8, lr=3e-3,
          weight_decay=1e-2, patience=20, eval_every=25):
    """AdamW with early stopping on validation L1 (patience x eval_every steps)."""
    # Renderers with learnable state (e.g. the Phase 9 basis LUTs) train jointly.
    params = list(model.parameters()) + list(renderer.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    for step in range(1, steps + 1):
        model.train()
        idx = random.sample(range(len(train_items)), min(batch, len(train_items)))
        loss = l1_loss(model, renderer, [train_items[i] for i in idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                v = float(l1_loss(model, renderer, val_items))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = ({k: t.clone() for k, t in model.state_dict().items()},
                              {k: t.clone() for k, t in renderer.state_dict().items()})
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        renderer.load_state_dict(best_state[1])
    model.eval()
    return {"val_l1": best, "steps": step}


@torch.enable_grad()
def oracle_params(renderer, it, steps=200, lr=0.03) -> torch.Tensor:
    theta = torch.zeros(1, renderer.num_params, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        loss = (renderer(small(it, "src")[None], theta)[0] - small(it, "tgt")).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return theta.detach()


def mean_target_cdf(items, bins=256) -> np.ndarray:
    acc = np.zeros((3, bins))
    for it in items:
        t = small(it, "tgt").numpy()
        for c in range(3):
            h, _ = np.histogram(t[c], bins=bins, range=(0, 1))
            acc[c] += np.cumsum(h) / h.sum()
    return acc / len(items)


def hist_match(src: torch.Tensor, ref_cdf: np.ndarray) -> np.ndarray:
    x = src.numpy()
    bins = ref_cdf.shape[1]
    centers = (np.arange(bins) + 0.5) / bins
    out = np.empty_like(x)
    for c in range(3):
        h, _ = np.histogram(x[c], bins=bins, range=(0, 1))
        cdf = np.cumsum(h) / h.sum()
        src_q = np.interp(x[c], centers, cdf)
        out[c] = np.interp(src_q, ref_cdf[c], centers)
    return out.transpose(1, 2, 0)


def hwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().permute(1, 2, 0).numpy().astype(np.float64)


def evaluate(outs, items) -> list[dict]:
    return [{"file": it["file"], **all_metrics(np.clip(o, 0, 1), hwc(it["tgt"]))}
            for o, it in zip(outs, items)]


def summarise(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "delta_e", "ssim", "l1")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--expert", default="c")
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase7_fivek_landscape_c"))
    ap.add_argument("--sizes", default="10,25,50,100,250,500,1000,all")
    ap.add_argument("--test", type=int, default=200, help="Fixed random test subset size.")
    ap.add_argument("--oracle", type=int, default=100, help="Test images used for the oracle ceiling.")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--large-n", type=int, default=250,
                    help="Sizes >= this use only the first seed (large-n runs are slow and low-variance).")
    ap.add_argument("--resume", action="store_true", help="Skip runs already in <out>/results.json.")
    ap.add_argument("--renderers", default="shared,per_channel")
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, args.expert, fx)
    pool = data.split("train") + data.split("validation")
    test = data.split("test")
    test = random.Random(99).sample(test, min(args.test, len(test)))
    rng = random.Random(1234)
    rng.shuffle(pool)
    val, pool = pool[: args.val], pool[args.val:]
    print(f"pool={len(pool)} val={len(val)} test={len(test)} dropped={data.dropped} "
          f"(loaded in {time.time() - t0:.0f}s)", flush=True)

    results = {"config": {k: str(v) for k, v in vars(args).items()},
               "counts": {"pool": len(pool), "val": len(val), "test": len(test),
                          "dropped_unalignable": data.dropped}, "runs": []}

    prev = None
    if args.resume and (args.out / "results.json").exists():
        prev = json.loads((args.out / "results.json").read_text())
        results["runs"] = prev["runs"]
    done = {(r["n"], r["seed"], r["renderer"], r["model"]) for r in results["runs"]}

    # --- Non-learned references ------------------------------------------------
    if prev and "identity" in prev:
        for k in [k for k in prev if k in ("identity", "histmatch") or k.startswith("oracle/")]:
            results[k] = prev[k]
    else:
        compute_references(results, test, pool, args)

    # --- Learning curve ----------------------------------------------------------
    seeds = list(map(int, args.seeds.split(",")))
    sizes = [len(pool) if s == "all" else int(s) for s in args.sizes.split(",")]
    for n in sizes:
        n = min(n, len(pool))
        for seed in (seeds if n < args.large_n else seeds[:1]):
            seed_all(seed)
            train_items = random.Random(seed).sample(pool, n)
            for rname in args.renderers.split(","):
                r = GlobalRenderer(rname)
                for kind in ("head", "preset"):
                    if (n, seed, rname, kind) in done:
                        continue
                    seed_all(seed)
                    model = Head(FEATURE_DIM, r.num_params) if kind == "head" else Preset(r.num_params)
                    if kind == "head":
                        model.set_norm(torch.stack([it["feat"] for it in pool]))
                    info = train(model, r, train_items, val)
                    outs, theta = render_items(model, r, test)
                    rows = evaluate([hwc(o) for o in outs], test)
                    spread = float(theta.std(0).mean()) if kind == "head" else 0.0
                    run = {"n": n, "seed": seed, "renderer": rname, "model": kind, **info,
                           "theta_spread": spread, "summary": summarise(rows), "per_image": rows}
                    results["runs"].append(run)
                    s = run["summary"]
                    print(f"n={n:4d} seed={seed} {kind}/{rname:11s} psnr={s['psnr']:.2f} "
                          f"dE={s['delta_e']:.2f} spread={spread:.4f} steps={info['steps']} "
                          f"[{time.time() - t0:.0f}s]", flush=True)
                    (args.out / "results.json").write_text(json.dumps(results))

    write_summary(results, args.out)
    plot(results, args.out)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


def compute_references(results: dict, test: list[dict], pool: list[dict], args) -> None:
    results["identity"] = evaluate([hwc(it["src"]) for it in test], test)
    cdf = mean_target_cdf(pool)
    results["histmatch"] = evaluate([hist_match(it["src"], cdf) for it in test], test)
    for rname in args.renderers.split(","):
        r = GlobalRenderer(rname)
        otest = test[: args.oracle]
        with torch.no_grad():
            outs = [r(it["src"][None], oracle_params(r, it))[0] for it in otest]
        results[f"oracle/{rname}"] = evaluate([hwc(o) for o in outs], otest)
        print(f"oracle/{rname}: {summarise(results[f'oracle/{rname}'])}", flush=True)
    for k in ("identity", "histmatch"):
        print(f"{k}: {summarise(results[k])}", flush=True)


def write_summary(results: dict, out: Path) -> None:
    rows = []
    for k in [k for k in results if k in ("identity", "histmatch") or k.startswith("oracle/")]:
        rows.append({"method": k, "n": "", **summarise(results[k]), "std_psnr": ""})
    groups: dict[tuple, list] = {}
    for r in results["runs"]:
        groups.setdefault((f"{r['model']}/{r['renderer']}", r["n"]), []).append(r["summary"])
    for (m, n), ss in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append({"method": m, "n": n,
                     **{k: float(np.mean([s[k] for s in ss])) for k in ("psnr", "delta_e", "ssim", "l1")},
                     "std_psnr": float(np.std([s["psnr"] for s in ss]))})
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "psnr", "std_psnr", "delta_e", "ssim", "l1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def plot(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {"head/per_channel": ("#1f6feb", "-", "o"), "head/shared": ("#1f6feb", "--", "s"),
              "preset/per_channel": ("#d97706", "-", "o"), "preset/shared": ("#d97706", "--", "s")}
    groups: dict[str, dict[int, list]] = {}
    for r in results["runs"]:
        groups.setdefault(f"{r['model']}/{r['renderer']}", {}).setdefault(r["n"], []).append(r["summary"]["psnr"])
    for m, byn in groups.items():
        ns = sorted(byn)
        mu = [np.mean(byn[n]) for n in ns]
        sd = [np.std(byn[n]) for n in ns]
        c, ls, mk = styles.get(m, ("gray", "-", "o"))
        ax.errorbar(ns, mu, yerr=sd, color=c, ls=ls, marker=mk, capsize=3, label=m)
    refs = {"identity": "#6b7280", "histmatch": "#9333ea"}
    for k, c in refs.items():
        ax.axhline(summarise(results[k])["psnr"], color=c, ls=":", label=k)
    for k in [k for k in results if k.startswith("oracle/")]:
        ax.axhline(summarise(results[k])["psnr"], color="#059669", ls="-." if "shared" in k else "-",
                   lw=1, label=k + " (ceiling)")
    ax.set_xscale("log")
    ax.set_xlabel("training pairs (log scale)")
    ax.set_ylabel("test PSNR (dB), higher is better")
    ax.set_title("FiveK landscapes: content-adaptive head vs static preset")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)


if __name__ == "__main__":
    main()
