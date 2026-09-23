"""Collect an openly licensed, unpaired style set from Openverse.

Builds a stand-in dataset for a target look without using the reference
photographer's own (copyrighted) images:

  1. Query the Openverse API (https://api.openverse.org) with style keywords,
     restricted to licences that allow reuse and adaptation
     (CC0, Public Domain Mark, CC BY, CC BY-SA).
  2. Download each candidate at <= 1024 px, deduplicate by perceptual hash.
  3. Keep only images where a person is a prominent subject (torchvision
     Faster R-CNN, COCO "person" class).
  4. Measure CIELAB colour statistics and score each image against a target
     look profile (see LOOK_PROFILES), then keep the best matches.
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

    Each entry maps a statistic name to (target, tolerance). The score is
    the mean of exp(-((x - target) / tolerance)^2) over the statistics, so
    1.0 is a perfect match and each term falls to ~0.37 one tolerance away.
    """

    queries: list[str]
    night: dict[str, tuple[float, float]]
    day: dict[str, tuple[float, float]]
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
        ],
        night={
            "L_p01": (1.5, 3.0),          # true black point
            "L_p50": (17.0, 10.0),        # low-key
            "shadow_b": (-7.0, 5.0),      # cool shadows
            "mid_a": (8.0, 8.0),          # warm-ish mids
            "high_b": (0.0, 6.0),         # neutral highlights
            "sat_mean": (0.5, 0.15),      # rich colour
        },
        day={
            "L_p01": (4.0, 4.0),
            "L_p50": (58.0, 15.0),
            "shadow_b": (-8.0, 5.0),
            "mid_a": (3.0, 6.0),
            "high_b": (1.0, 5.0),
            "sat_mean": (0.25, 0.12),     # airy, low saturation
        },
        notes="Stand-in for the look studied in research_notes/styles/bleg.md",
    ),
}


# ---------------------------------------------------------------------------
# Openverse
# ---------------------------------------------------------------------------


def search(query: str, page: int, session: requests.Session) -> list[dict]:
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
    return r.json().get("results", [])


def download(url: str, session: requests.Session) -> Image.Image | None:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    img.thumbnail((1024, 1024))
    return img


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

    def largest_person_fraction(self, img: Image.Image) -> float:
        import torchvision.transforms.functional as F

        x = self.preprocess(F.pil_to_tensor(img))
        with self.torch.no_grad():
            out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] > 0.7)
        if not keep.any():
            return 0.0
        boxes = out["boxes"][keep]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return float(areas.max() / (img.width * img.height))


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
    }


def score(stats: dict[str, float], profile: dict[str, tuple[float, float]]) -> float:
    terms = [np.exp(-(((stats[k] - t) / tol) ** 2)) for k, (t, tol) in profile.items()]
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
    ap.add_argument("--max-keep", type=int, default=150)
    args = ap.parse_args()

    profile = LOOK_PROFILES[args.profile]
    out = args.out or Path("data/styles") / args.profile
    manifest = args.manifest or Path("data/manifests") / f"{args.profile}.csv"
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    detector = PersonDetector()

    seen_ids: set[str] = set()
    seen_hashes: list[int] = []
    rows: list[dict] = []
    stats_log = {"candidates": 0, "download_failed": 0, "duplicate": 0,
                 "no_person": 0, "low_score": 0}

    for query in profile.queries:
        for page in range(1, args.pages + 1):
            try:
                results = search(query, page, session)
            except RuntimeError as e:
                print(e)
                break
            time.sleep(3.2)  # stay under 20 requests/min
            if not results:
                break
            for item in results:
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                stats_log["candidates"] += 1
                img = download(item["url"], session)
                if img is None or min(img.size) < 400:
                    stats_log["download_failed"] += 1
                    continue
                h = dhash(img)
                if any(bin(h ^ s).count("1") <= 6 for s in seen_hashes):
                    stats_log["duplicate"] += 1
                    continue
                seen_hashes.append(h)
                person = detector.largest_person_fraction(img)
                if person < args.min_person:
                    stats_log["no_person"] += 1
                    continue
                st = colour_stats(img)
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
            print(f"[{query!r} p{page}] kept so far: {len(rows)}  {stats_log}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[: args.max_keep]
    for i, r in enumerate(rows):
        img = r.pop("_img")
        fname = f"{i:04d}_{r['regime']}_{r['openverse_id'][:8]}.jpg"
        path = out / "images" / fname
        img.save(path, quality=92)
        r["file"] = fname
        r["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    fields = ["file", "sha256", "regime", "score", "person_frac", "license", "license_version",
              "license_url", "title", "creator", "creator_url", "source", "landing_url",
              "image_url", "attribution", "openverse_id", "query",
              "L_p01", "L_p50", "L_p99", "shadow_frac", "shadow_a", "shadow_b",
              "mid_a", "mid_b", "high_a", "high_b", "sat_mean"]
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
