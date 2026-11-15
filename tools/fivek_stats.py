"""Statistics of the FiveK proxies for the Phase 20 data report.

- counts per split and expert, and disk use;
- from the per-row records: out-of-sRGB-gamut fraction of the experts,
  orientation corrections, alignment correlation;
- identity baseline on a seeded sample of rows: the original compared with
  each expert, as mean CIEDE2000 (Lab from *unclipped* linear sRGB via XYZ,
  so wide-gamut expert colours are not clipped before comparison) and PSNR
  on clipped 8-bit sRGB (the convention of published FiveK results).

Usage::

    uv run python tools/fivek_stats.py --out tests/reports/phase20_fivek_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.color import deltaE_ciede2000, xyz2lab
from skimage.color.colorconv import xyz_from_rgb  # sRGB (D65) -> XYZ matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fetch_fivek import SPLITS, decode_linear_srgb, srgb_encode  # noqa: E402

RNG_SEED = 5000  # any fixed seed; selects the baseline sample
EXPERTS = "abcde"


def lab(lin: NDArray[np.float32]) -> NDArray[np.float64]:
    return xyz2lab(lin.astype(np.float64) @ xyz_from_rgb.T)


def psnr(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    qa = np.round(srgb_encode(np.clip(a, 0, 1)) * 255)
    qb = np.round(srgb_encode(np.clip(b, 0, 1)) * 255)
    mse = float(np.mean((qa - qb) ** 2))
    return 10 * np.log10(255.0**2 / max(mse, 1e-12))


def du(path: Path) -> int:
    # hidden folders (.tmp: sparse in-flight shards; .state: bookkeeping) excluded
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file() and not any(q.startswith(".") for q in p.relative_to(path).parts)
    )


def quantiles(x: list[float]) -> dict[str, float]:
    if not x:
        return {}
    q = np.percentile(x, [0, 5, 50, 95, 100])
    return {
        k: round(float(v), 4)
        for k, v in zip(["min", "p5", "p50", "p95", "max"], q, strict=True)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/fivek")
    ap.add_argument(
        "--sample", type=int, default=200, help="rows per split for the baseline"
    )
    ap.add_argument("--out", default="tests/reports/phase20_fivek_stats.json")
    args = ap.parse_args()
    root = Path(args.root)
    out: dict[str, Any] = {"splits": {}, "disk_bytes": du(root)}
    rng = np.random.default_rng(RNG_SEED)
    for split in SPLITS:
        d = root / split
        s: dict[str, Any] = {
            "original": len(list((d / "original").glob("*.png"))) if d.exists() else 0,
            "disk_bytes": du(d) if d.exists() else 0,
        }
        for x in EXPERTS:
            recs = [
                json.loads(p.read_text())
                for p in sorted((d / f"expert_{x}").glob("*.json"))
            ]
            if not recs:
                continue
            e: dict[str, Any] = {
                "rows": len(recs),
                "oog": quantiles([r["oog"] for r in recs]),
            }
            e["has_crop"] = sum(bool(r["has_crop"]) for r in recs)
            if x == "c":
                e["rot90"] = {
                    k: sum(r.get("rot90") == k for r in recs) for k in range(4)
                }
                e["aligned"] = sum(bool(r.get("aligned")) for r in recs)
                e["orig_smaller"] = sum(bool(r.get("orig_smaller")) for r in recs)
                e["align_r"] = quantiles([r["align_r"] for r in recs if "align_r" in r])
                e["low_r_rows"] = [
                    (r["idx"], r["fivek_id"], r["align_r"])
                    for r in recs
                    if r.get("align_r", 1) < 0.8
                ]
            have = [
                r["idx"]
                for r in recs
                if (d / "original" / f"img_{r['idx']:04d}.png").exists()
            ]
            pick = (
                rng.choice(have, size=min(args.sample, len(have)), replace=False)
                if have
                else []
            )
            de, ps = [], []
            for i in pick:
                name = f"img_{int(i):04d}.png"
                a = decode_linear_srgb((d / "original" / name).read_bytes(), clip=False)
                b = decode_linear_srgb(
                    (d / f"expert_{x}" / name).read_bytes(), clip=False
                )
                de.append(float(deltaE_ciede2000(lab(a), lab(b)).mean()))
                ps.append(psnr(a, b))
            if de:
                e["identity_baseline"] = {
                    "n": len(de),
                    "de00_mean": round(float(np.mean(de)), 2),
                    "psnr_mean_db": round(float(np.mean(ps)), 2),
                }
            s[f"expert_{x}"] = e
        out["splits"][split] = s
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
