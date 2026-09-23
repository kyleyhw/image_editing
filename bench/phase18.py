"""Phase 18 (atmosphere, A2.4): depth-aware haze / dehaze / clarity, checked and timed.

There is no ground truth for "right amount of haze", so this is a
behavioural check plus a demo, not a learning benchmark:

  clipping     largest increase in clipped pixels per preset (target <= 1 %)
  identity     all strengths 0 returns the input bit-exactly
  direction    +haze lowers far-region local contrast more than near-region;
               dehaze raises far-region contrast; clarity_near raises near
               contrast more than far (ratios reported per image)
  depth        agreement (Spearman) between Depth Anything V2 Small and the
               transparent fallback (position + dark channel)
  runtime      depth inference and the full scene edit at 1024 px and at the
               photo's full size, on this CPU
  figure       AdobeMIT-licensed FiveK nature test images: input, depth,
               +haze, dehaze, near clarity / far softening

Owner nature photos, if present, are processed into data/owner/outputs/phase18
(private, not committed).

Usage:
    uv run python -m bench.phase18 --out bench/results/phase18
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench.pilot_a import grid
from photostyle.atmosphere import SceneParams, apply_scene, depth_map
from photostyle.io import load_image
from photostyle.looks import clip_fraction
from photostyle.regional import box_filter
from photostyle.train import shrink

PRESETS = {
    "+haze": SceneParams(haze=0.5),
    "dehaze": SceneParams(haze=-0.5),
    "near clarity / far soft": SceneParams(clarity_near=0.6, clarity_far=-0.4),
}


def local_contrast(x: torch.Tensor, mask: torch.Tensor) -> float:
    y = x.mean(1, keepdim=True)
    d = (y - box_filter(y, max(2, min(x.shape[2:]) // 100))).abs()
    return float((d * mask).sum() / mask.sum().clamp_min(1))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().numpy(), b.flatten().numpy()
    idx = np.random.default_rng(0).choice(a.size, min(a.size, 20000), replace=False)
    ra, rb = a[idx].argsort().argsort(), b[idx].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def to_t(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(load_image(path)[0], dtype=np.float32) / 255).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase18"))
    ap.add_argument("--owner", type=Path, default=Path("data/owner"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.data / "meta.csv").open() as f:
        meta = [r for r in csv.DictReader(f) if r["split"] == "test" and r["subject"] == "nature"
                and r["license"] == "AdobeMIT" and r["location"] == "outdoor"]
    meta = meta[: args.n]
    res = {"images": [], "identity_exact": True}
    rows = []
    for m in meta:
        full = to_t(args.data / "original" / m["file"])
        img = shrink(full, 1024)[None]
        rec = {"file": m["file"], "size": list(full.shape[1:])}
        t = time.time()
        near = depth_map(img)
        rec["depth_s_1024"] = time.time() - t
        fb = depth_map(img, use_model=False)
        rec["spearman_model_vs_fallback"] = spearman(near, fb)
        res["identity_exact"] &= bool(torch.equal(apply_scene(img, SceneParams(), near=near), img))
        nm, fm = (near > 0.6).float(), (near < 0.3).float()
        base = (local_contrast(img, nm), local_contrast(img, fm))
        outs = {}
        for name, p in PRESETS.items():
            t = time.time()
            o = apply_scene(img, p, near=near)
            rec[f"edit_s_{name}"] = time.time() - t
            c = (local_contrast(o, nm), local_contrast(o, fm))
            rec[f"contrast_ratio_{name}"] = {"near": c[0] / max(base[0], 1e-6), "far": c[1] / max(base[1], 1e-6)}
            rec[f"new_clip_{name}"] = float(clip_fraction(o) - clip_fraction(img))
            outs[name] = o[0]
        t = time.time()
        big = full[None]
        near_big = F.interpolate(near, size=big.shape[2:], mode="bilinear", align_corners=False)
        apply_scene(big, PRESETS["+haze"], near=near_big)
        rec["edit_s_native_haze"] = time.time() - t
        res["images"].append(rec)
        print(f"{m['file']} depth {rec['depth_s_1024']:.1f}s rho={rec['spearman_model_vs_fallback']:.2f} "
              + " ".join(f"{k}:{v['near']:.2f}/{v['far']:.2f}" for k, v in rec.items()
                         if k.startswith("contrast_ratio")), flush=True)
        if len(rows) < 5:
            rows.append([img[0], near[0].expand(3, -1, -1), *outs.values()])
        (args.out / "results.json").write_text(json.dumps(res, indent=1))

    ims = res["images"]
    checks = {
        "haze_far_drops_more": float(np.mean([r["contrast_ratio_+haze"]["far"] < r["contrast_ratio_+haze"]["near"]
                                              for r in ims])),
        "dehaze_far_rises": float(np.mean([r["contrast_ratio_dehaze"]["far"] > 1 for r in ims])),
        "clarity_near_gt_far": float(np.mean([r["contrast_ratio_near clarity / far soft"]["near"]
                                              > r["contrast_ratio_near clarity / far soft"]["far"] for r in ims])),
    }
    res["summary"] = {
        "n": len(ims), "identity_exact": res["identity_exact"], **checks,
        "spearman_median": float(np.median([r["spearman_model_vs_fallback"] for r in ims])),
        "depth_s_median": float(np.median([r["depth_s_1024"] for r in ims])),
        **{f"new_clip_max_{k}": float(max(r[f"new_clip_{k}"] for r in ims)) for k in PRESETS},
        "edit_s_native_haze_median": float(np.median([r["edit_s_native_haze"] for r in ims])),
    }
    # 12 MP runtime (the FiveK copies here are 512 px): depth on the 1024 px proxy, edit at full size.
    big = torch.rand(1, 3, 3000, 4000)
    t = time.time()
    near = depth_map(shrink(big[0], 1024)[None])
    res["summary"]["depth_s_12mp_proxy"] = time.time() - t
    near = F.interpolate(near, size=big.shape[2:], mode="bilinear", align_corners=False)
    for name, p in PRESETS.items():
        t = time.time()
        apply_scene(big, p, near=near)
        res["summary"][f"edit_s_12mp_{name}"] = time.time() - t
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    grid(rows, ["input (FiveK, AdobeMIT)", "nearness", *PRESETS], args.out / "phase18_grid.jpg", tw=260)
    print("summary:", res["summary"], flush=True)

    # Owner nature photos: private outputs only.
    man = args.owner / "manifest.csv"
    if man.exists():
        priv = args.owner / "outputs" / "phase18"
        priv.mkdir(parents=True, exist_ok=True)
        with man.open() as f:
            own = [r for r in csv.DictReader(f) if r["scene"] == "nature"]
        prow = []
        for r in own:
            img = shrink(to_t(args.owner / "srgb" / f"{r['id']}.jpg"), 1024)[None]
            near = depth_map(img)
            prow.append([img[0], near[0].expand(3, -1, -1), *(apply_scene(img, p, near=near)[0]
                                                              for p in PRESETS.values())])
        if prow:
            grid(prow, ["your photo", "nearness", *PRESETS], priv / "grid.jpg", tw=260)
            print(f"owner grid: {len(prow)} photos -> {priv / 'grid.jpg'}", flush=True)


if __name__ == "__main__":
    main()
