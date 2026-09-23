"""Collect an openly licensed, unpaired style set from Openverse.

Builds a stand-in dataset for a target look without using the reference
photographer's own (copyrighted) images:

  1. Query the Openverse API (https://api.openverse.org) with style keywords,
     restricted to licences that allow reuse and adaptation
     (CC0, Public Domain Mark, CC BY, CC BY-SA).
  2. Download each candidate at <= 1024 px (search responses and images are
     cached under data/cache/openverse/, so re-filtering costs no API calls),
     deduplicate by perceptual hash, crop letterbox bars, drop monochrome.
  3. Subject filter (torchvision Faster R-CNN, COCO "person" class):
     portrait profiles keep images with one or two people, one of them
     prominent; scene profiles (landscapes) reject images where a person is
     prominent.
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
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.stats import colour_stats, regime_of  # noqa: E402,F401  (re-exported)

from photostyle.openverse import (  # noqa: E402,F401  (re-exported for tools)
    ALLOWED_LICENSES,
    API,
    USER_AGENT,
    PersonDetector,
    attribution,
    crop_letterbox,
    dhash,
    dof_log_ratio,
    download,
    score,
    search,
)


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
    # "portrait": require a prominent person; "scene": reject prominent people.
    subject: str = "portrait"
    # Hard pass/fail bounds per regime: {"night": {"shadow_b": (lo, hi)}, ...}.
    # Unlike the soft score, a single failed gate rejects the image, so a
    # candidate cannot trade a wrong colour cast for a good black point.
    gates: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
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
    # Amendment A2: the same clean, cool-shadow grade, applied to urban and
    # natural landscapes (no people). Gates enforce the split tone that the
    # portrait stand-in set failed to match.
    "clean_cool_landscape": LookProfile(
        queries=[
            # urban, night
            "city night neon street",
            "tokyo street night",
            "seoul street night",
            "hong kong street night neon",
            "city blue hour skyline",
            "rainy street night reflections",
            "alley neon night",
            "osaka dotonbori night",
            # urban, day
            "city street architecture daylight",
            "tokyo street day",
            "cherry blossom street city",
            "minimal urban architecture",
            # nature
            "mountain landscape blue hour",
            "misty forest",
            "lake mountains morning",
            "coastal cliffs",
            "snow mountains landscape",
            "cherry blossom park",
            "misty mountains",
            "fjord landscape",
            "desert landscape dusk",
        ],
        night={
            "L_p01": (1.0, 2.0),
            "L_p50": (18.0, 10.0),
            "shadow_b": (-7.0, 4.0),
            "high_b": (0.0, 5.0),
            "sat_mean": (0.45, 0.15),
        },
        day={
            "L_p01": (3.0, 3.0),
            "L_p50": (55.0, 15.0),
            "shadow_b": (-8.0, 4.0),
            "mid_b": (-2.0, 6.0),
            "high_b": (1.0, 4.0),
            "sat_mean": (0.25, 0.10),
        },
        subject="scene",
        gates={
            "night": {"shadow_b": (-99.0, -2.0), "high_b": (-8.0, 6.0), "L_p01": (0.0, 4.0)},
            "day": {"shadow_b": (-99.0, -2.0), "high_b": (-6.0, 5.0), "L_p01": (0.0, 8.0),
                    "sat_mean": (0.0, 0.45)},
        },
        notes="Clean cool-shadow grade on urban and natural landscapes (amendment A2).",
    ),
}


# ---------------------------------------------------------------------------
# Openverse
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filters and statistics
# ---------------------------------------------------------------------------


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
    ap.add_argument("--max-scene-person", type=float, default=0.05,
                    help="Scene profiles: reject if the largest person covers more than this.")
    ap.add_argument("--max-per-creator", type=int, default=10,
                    help="Cap images per creator in the final selection (diversity).")
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
                 "monochrome": 0, "no_person": 0, "has_person": 0, "crowd": 0,
                 "gated": 0, "low_score": 0}

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
                if profile.subject == "portrait":
                    if not people or people[0][0] < args.min_person:
                        stats_log["no_person"] += 1
                        continue
                    if sum(a >= 0.03 for a, _ in people) > args.max_people:
                        stats_log["crowd"] += 1
                        continue
                    st["dof_log_ratio"] = dof_log_ratio(img, people[0][1])
                elif people and people[0][0] > args.max_scene_person:
                    stats_log["has_person"] += 1
                    continue
                person = people[0][0] if people else 0.0
                regime = regime_of(st)
                gates = profile.gates.get(regime, {})
                if any(not (lo <= st[k] <= hi) for k, (lo, hi) in gates.items()):
                    stats_log["gated"] += 1
                    continue
                s = score(st, profile.night if regime == "night" else profile.day)
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

    # Cap images per creator (keep each creator's best), then balance regimes:
    # take the best of each, then fill any shortfall.
    rows.sort(key=lambda r: r["score"], reverse=True)
    per_creator: dict[str, int] = {}
    capped = []
    for r in rows:
        c = r.get("creator") or "unknown"
        if per_creator.get(c, 0) < args.max_per_creator:
            per_creator[c] = per_creator.get(c, 0) + 1
            capped.append(r)
    stats_log["creator_capped"] = len(rows) - len(capped)
    rows = capped
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
