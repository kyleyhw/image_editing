"""Collect an openly licensed, unpaired style set from Openverse.

Builds a stand-in dataset for a target look without using the reference
photographer's own (copyrighted) images:

  1. Query the Openverse API (https://api.openverse.org) with style keywords,
     restricted to licences that allow reuse and adaptation
     (CC0, Public Domain Mark, CC BY, CC BY-SA).
  2. Download each candidate at <= 1024 px (search responses and images are
     cached under data/cache/openverse/, so re-filtering costs no API calls),
     deduplicate by perceptual hash, crop letterbox bars, drop monochrome.
  3. Keep only images with one or two people, one of them prominent
     (torchvision Faster R-CNN, COCO "person" class).
  4. Measure CIELAB colour statistics plus a depth-of-field proxy (subject vs.
     background sharpness), score each image against a target look profile
     (see LOOK_PROFILES), and keep the best matches per regime (night / day).
  5. Write a manifest (CSV) with licence, creator, source URL, attribution
     text, SHA-256, statistics and score. Images stay in data/ (gitignored);
     the manifest is committed under data/manifests/.

Usage:
    uv run python tools/collect_style_set.py --profile neon_street_portrait \
        --out data/styles/neon_street_portrait --max-keep 150

Openverse anonymous limits: 20 requests/min, 200/day. The script sleeps
between pages to stay under the burst limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import skimage as ski
from PIL import Image

API = "https://api.openverse.org/v1/images/"
USER_AGENT = "image_editing-style-research/0.1 (github.com/kyleyhw/image_editing)"
ALLOWED_LICENSES = "cc0,pdm,by,by-sa"


@dataclass
class LookProfile:
    """Target colour statistics of a look, measured in CIELAB.

    Each entry maps a statistic name to (target, tolerance) or
    (target, tolerance, ">=") for one-sided targets. The score is the mean
    of exp(-((x - target) / tolerance)^2) over the statistics (one-sided
    terms are 1 once x >= target), so 1.0 is a perfect match and each term
    falls to ~0.37 one tolerance away.
    """

    queries: list[str]
    night: dict[str, tuple]
    day: dict[str, tuple]
    notes: str = ""
    extra: dict = field(default_factory=dict)


# Profile derived from the colour statistics in research_notes/styles/bleg.md
# (deep true blacks, cool shadows, warm midtones, neutral highlights;
# saturated at night, airy at day). Only summary statistics are used.
LOOK_PROFILES: dict[str, LookProfile] = {
    "neon_street_portrait": LookProfile(
        queries=[
            "street portrait night neon",
            "neon portrait",
            "night portrait city lights bokeh",
            "portrait neon lights street",
            "night market portrait",
            "tokyo street portrait night",
            "seoul street portrait",
            "hong kong portrait night",
            "street portrait bokeh",
            "environmental portrait street",
            "street portrait stranger",
            "urban portrait evening",
            "portrait cherry blossom street",
            "street portrait city daylight",
            # extra night coverage
            "neon sign portrait",
            "night street portrait bokeh",
            "portrait night lights city",
            "shibuya night portrait",
            "rainy night portrait street",
            "cyberpunk portrait",
            "night market street food vendor portrait",
            "portrait at night neon light",
            "street portrait shallow depth of field",
            "portrait bokeh city",
        ],
        night={
            "L_p01": (1.5, 3.0),          # true black point
            "L_p50": (17.0, 10.0),        # low-key
            "shadow_b": (-7.0, 5.0),      # cool shadows
            "mid_a": (8.0, 8.0),          # warm-ish mids
            "high_b": (0.0, 6.0),         # neutral highlights
            "sat_mean": (0.5, 0.15),      # rich colour
            "dof_log_ratio": (1.0, 0.8, ">="),  # subject sharper than background
        },
        day={
            "L_p01": (4.0, 4.0),
            "L_p50": (58.0, 15.0),
            "shadow_b": (-8.0, 5.0),
            "mid_a": (3.0, 6.0),
            "high_b": (1.0, 5.0),
            "sat_mean": (0.25, 0.12),     # airy, low saturation
            "dof_log_ratio": (1.0, 0.8, ">="),
        },
        notes="Stand-in for the look studied in research_notes/styles/bleg.md",
    ),
}


# ---------------------------------------------------------------------------
# Openverse
# ---------------------------------------------------------------------------


def search(query: str, page: int, session: requests.Session, cache: Path) -> list[dict]:
    key = cache / "search" / (hashlib.sha1(f"{query}|{page}|{ALLOWED_LICENSES}".encode()).hexdigest() + ".json")
    if key.exists():
        return json.loads(key.read_text())
    params = {
        "q": query,
        "license": ALLOWED_LICENSES,
        "page_size": 20,
        "page": page,
        "mature": "false",
    }
    r = session.get(API, params=params, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("Openverse rate limit reached; rerun later (200 requests/day).")
    r.raise_for_status()
    results = r.json().get("results", [])
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(results))
    time.sleep(3.2)  # stay under 20 requests/min (only for real API calls)
    return results


def download(item: dict, session: requests.Session, cache: Path) -> Image.Image | None:
    path = cache / "images" / f"{item['id']}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    try:
        r = session.get(item["url"], timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    img.thumbnail((1024, 1024))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    return img


def crop_letterbox(img: Image.Image, thresh: float = 0.03) -> Image.Image:
    """Remove near-black bars on any edge (letterbox/pillarbox).

    A row/column counts as bar if >= 90 % of its pixels are near-black, which
    tolerates the caption text often printed inside the bar. A bar is only
    cropped if it ends in a sharp edge (the first photo row/column is mostly
    not black), so genuinely dark night skies are left alone.
    """
    g = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    dark = g < thresh
    row_frac, col_frac = dark.mean(1), dark.mean(0)

    def edge(frac: np.ndarray, from_start: bool) -> int:
        n = len(frac)
        idx = range(n) if from_start else range(n - 1, -1, -1)
        k = 0
        for i in idx:
            if frac[i] < 0.9:
                break
            k += 1
        if k < 3 or k >= n // 2:
            return 0
        first_photo = frac[k] if from_start else frac[n - 1 - k]
        return k if first_photo < 0.6 else 0

    top, bottom = edge(row_frac, True), edge(row_frac, False)
    left, right = edge(col_frac, True), edge(col_frac, False)
    if top == bottom == left == right == 0:
        return img
    return img.crop((left, top, img.width - right, img.height - bottom))


# ---------------------------------------------------------------------------
# Filters and statistics
# ---------------------------------------------------------------------------


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class PersonDetector:
    """Largest-person area fraction via torchvision Faster R-CNN (COCO)."""

    def __init__(self) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        self.torch = torch
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()
        self.preprocess = weights.transforms()

    def people(self, img: Image.Image) -> list[tuple[float, tuple[float, float, float, float]]]:
        """Return [(area_fraction, (x0, y0, x1, y1)), ...] sorted largest first."""
        import torchvision.transforms.functional as F

        x = self.preprocess(F.pil_to_tensor(img))
        with self.torch.no_grad():
            out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] > 0.7)
        res = []
        for box in out["boxes"][keep].tolist():
            area = (box[2] - box[0]) * (box[3] - box[1]) / (img.width * img.height)
            res.append((area, tuple(box)))
        return sorted(res, reverse=True)


def dof_log_ratio(img: Image.Image, box: tuple[float, float, float, float]) -> float:
    """log(sharpness inside the subject box / sharpness outside it).

    Sharpness = mean absolute Laplacian on a 512 px greyscale copy. Large
    values mean a sharp subject on a blurred background (shallow depth of
    field); ~0 means everything is equally sharp.
    """
    scale = 512 / max(img.size)
    g = np.asarray(img.convert("L").resize((round(img.width * scale), round(img.height * scale))),
                   dtype=np.float64) / 255.0
    lap = np.abs(ski.filters.laplace(g))
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    inside = np.zeros_like(lap, dtype=bool)
    inside[y0:y1, x0:x1] = True
    if inside.all() or not inside.any():
        return 0.0
    return float(np.log((lap[inside].mean() + 1e-4) / (lap[~inside].mean() + 1e-4)))


def colour_stats(img: Image.Image) -> dict[str, float]:
    small = img.copy()
    small.thumbnail((512, 512))
    rgb = np.asarray(small, dtype=np.float64) / 255.0
    lab = ski.color.rgb2lab(rgb)
    hsv = ski.color.rgb2hsv(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    def masked_mean(x: np.ndarray, m: np.ndarray) -> float:
        return float(x[m].mean()) if m.any() else 0.0

    shadows, mids, highs = L < 25, (L >= 25) & (L < 70), L >= 70
    return {
        "L_p01": float(np.percentile(L, 1)),
        "L_p50": float(np.percentile(L, 50)),
        "L_p99": float(np.percentile(L, 99)),
        "shadow_frac": float(shadows.mean()),
        "shadow_a": masked_mean(a, shadows),
        "shadow_b": masked_mean(b, shadows),
        "mid_a": masked_mean(a, mids),
        "mid_b": masked_mean(b, mids),
        "high_a": masked_mean(a, highs),
        "high_b": masked_mean(b, highs),
        "sat_mean": float(hsv[..., 1].mean()),
        "chroma_mean": float(np.hypot(a, b).mean()),
    }


def score(stats: dict[str, float], profile: dict[str, tuple]) -> float:
    terms = []
    for k, spec in profile.items():
        t, tol = spec[0], spec[1]
        if len(spec) > 2 and spec[2] == ">=" and stats[k] >= t:
            terms.append(1.0)
        else:
            terms.append(np.exp(-(((stats[k] - t) / tol) ** 2)))
    return float(np.mean(terms))


def attribution(item: dict) -> str:
    lic = item.get("license", "").upper()
    ver = item.get("license_version") or ""
    lic_txt = "CC0 1.0" if lic == "CC0" else ("Public Domain Mark" if lic == "PDM" else f"CC {lic} {ver}")
    return f'"{item.get("title") or "Untitled"}" by {item.get("creator") or "unknown"}, {lic_txt}, via {item.get("source")}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="neon_street_portrait", choices=sorted(LOOK_PROFILES))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--pages", type=int, default=3, help="Result pages (20 each) per query.")
    ap.add_argument("--min-person", type=float, default=0.06,
                    help="Minimum area fraction of the largest detected person.")
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--min-side", type=int, default=320,
                    help="Minimum short side in px after letterbox cropping.")
    ap.add_argument("--max-keep", type=int, default=150,
                    help="Total images kept, split evenly between night and day where possible.")
    ap.add_argument("--min-chroma", type=float, default=6.0,
                    help="Reject near-monochrome images (mean CIELAB chroma).")
    ap.add_argument("--max-people", type=int, default=2,
                    help="Reject crowds: max people with area >= 3%% of the frame.")
    ap.add_argument("--cache", type=Path, default=Path("data/cache/openverse"))
    ap.add_argument("--exclude", type=Path, default=None,
                    help="Hand-review exclusion list: one '<openverse_id> # reason' per line. "
                         "Default: data/manifests/<profile>.exclude.txt if it exists.")
    args = ap.parse_args()

    profile = LOOK_PROFILES[args.profile]
    out = args.out or Path("data/styles") / args.profile
    manifest = args.manifest or Path("data/manifests") / f"{args.profile}.csv"
    (out / "images").mkdir(parents=True, exist_ok=True)
    for stale in (out / "images").glob("*.jpg"):  # output reflects this run only
        stale.unlink()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    exclude_file = args.exclude or Path("data/manifests") / f"{args.profile}.exclude.txt"
    excluded: set[str] = set()
    if exclude_file.exists():
        for line in exclude_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                excluded.add(line)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    detector = PersonDetector()

    seen_ids: set[str] = set()
    seen_hashes: list[int] = []
    rows: list[dict] = []
    stats_log = {"candidates": 0, "download_failed": 0, "too_small": 0, "duplicate": 0,
                 "monochrome": 0, "no_person": 0, "crowd": 0, "low_score": 0}

    for query in profile.queries:
        for page in range(1, args.pages + 1):
            try:
                results = search(query, page, session, args.cache)
            except RuntimeError as e:
                print(e)
                break
            if not results:
                break
            for item in results:
                if item["id"] in seen_ids or item["id"] in excluded:
                    continue
                seen_ids.add(item["id"])
                stats_log["candidates"] += 1
                img = download(item, session, args.cache)
                if img is not None:
                    img = crop_letterbox(img)
                if img is None:
                    stats_log["download_failed"] += 1
                    continue
                if min(img.size) < args.min_side:
                    stats_log["too_small"] += 1
                    continue
                h = dhash(img)
                if any(bin(h ^ s).count("1") <= 6 for s in seen_hashes):
                    stats_log["duplicate"] += 1
                    continue
                seen_hashes.append(h)
                st = colour_stats(img)
                if st["chroma_mean"] < args.min_chroma:
                    stats_log["monochrome"] += 1
                    continue
                people = detector.people(img)
                if not people or people[0][0] < args.min_person:
                    stats_log["no_person"] += 1
                    continue
                if sum(a >= 0.03 for a, _ in people) > args.max_people:
                    stats_log["crowd"] += 1
                    continue
                person = people[0][0]
                st["dof_log_ratio"] = dof_log_ratio(img, people[0][1])
                s_night, s_day = score(st, profile.night), score(st, profile.day)
                regime = "night" if st["L_p50"] < 35 else "day"
                s = s_night if regime == "night" else s_day
                if s < args.min_score:
                    stats_log["low_score"] += 1
                    continue
                rows.append({
                    "openverse_id": item["id"], "query": query, "regime": regime,
                    "score": round(s, 4), "person_frac": round(person, 3),
                    "title": item.get("title"), "creator": item.get("creator"),
                    "creator_url": item.get("creator_url"),
                    "license": item.get("license"), "license_version": item.get("license_version"),
                    "license_url": item.get("license_url"),
                    "source": item.get("source"), "landing_url": item.get("foreign_landing_url"),
                    "image_url": item.get("url"), "attribution": attribution(item),
                    "_img": img,
                    **{k: round(v, 3) for k, v in st.items()},
                })
            print(f"[{query!r} p{page}] kept so far: {len(rows)}  {stats_log}", flush=True)

    # Balance regimes: take the best of each, then fill any shortfall.
    rows.sort(key=lambda r: r["score"], reverse=True)
    half = args.max_keep // 2
    night = [r for r in rows if r["regime"] == "night"][:half]
    day = [r for r in rows if r["regime"] == "day"][: args.max_keep - len(night)]
    rest = [r for r in rows if r not in night and r not in day]
    rows = sorted(night + day + rest[: args.max_keep - len(night) - len(day)],
                  key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        img = r.pop("_img")
        fname = f"{i:04d}_{r['regime']}_{r['openverse_id'][:8]}.jpg"
        path = out / "images" / fname
        img.save(path, quality=92)
        r["file"] = fname
        r["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    fields = ["file", "sha256", "regime", "score", "person_frac", "dof_log_ratio", "license", "license_version",
              "license_url", "title", "creator", "creator_url", "source", "landing_url",
              "image_url", "attribution", "openverse_id", "query",
              "L_p01", "L_p50", "L_p99", "shadow_frac", "shadow_a", "shadow_b",
              "mid_a", "mid_b", "high_a", "high_b", "sat_mean", "chroma_mean"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    summary = {"profile": args.profile, "kept": len(rows),
               "night": sum(r["regime"] == "night" for r in rows),
               "day": sum(r["regime"] == "day" for r in rows),
               "licenses": {lic: sum(r["license"] == lic for r in rows)
                            for lic in sorted({r["license"] for r in rows})},
               **stats_log}
    (manifest.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
