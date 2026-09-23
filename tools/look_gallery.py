"""Build a gallery of candidate looks (openly licensed photos) for the owner to choose from.

For each look: a few Openverse queries (CC0 / PDM / CC BY / CC BY-SA only,
as in tools/collect_style_set.py), letterbox crop, dedupe, then rank by how
well the photo's colour statistics fit a rough target for that look, and keep
the best ``--per-look``. Writes:

  data/styles/look_gallery/<look>/<openverse id>.jpg   (local, gitignored)
  data/styles/look_gallery/look_gallery.jpg            contact sheet, labelled 1a, 1b, ...
  data/manifests/look_gallery.csv                      label, look, licence, creator, source, attribution

Usage:
    uv run python tools/look_gallery.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.stats import colour_stats  # noqa: E402
from tools.collect_style_set import (  # noqa: E402
    USER_AGENT,
    attribution,
    crop_letterbox,
    dhash,
    download,
    score,
    search,
)

# look -> (description, queries, stat targets (value, tolerance[, ">="]))
LOOKS = {
    "natural": ("True-to-life colour, gentle contrast, nothing pushed",
                ["mountain lake landscape", "japan street daylight", "countryside landscape photography", "city skyline daytime"],
                {"sat_mean": (0.35, 0.1), "chroma_mean": (18, 8), "L_p50": (50, 12), "mid_b": (6, 7), "mid_a": (0, 5), "L_p01": (4, 4)}),
    "vivid": ("Rich, saturated colour and punchy contrast",
              ["vibrant colorful landscape", "colorful autumn landscape", "colorful city street"],
              {"sat_mean": (0.6, 0.12), "chroma_mean": (35, 10, ">="), "L_p01": (2, 3)}),
    "golden_film": ("Warm golden light, film-like softness (Portra-ish)",
                    ["golden hour landscape", "kodak portra", "golden hour city street"],
                    {"mid_b": (22, 8), "mid_a": (6, 5), "high_b": (15, 8), "L_p01": (8, 6)}),
    "airy_pastel": ("Bright, soft, low contrast, pastel colours",
                    ["pastel landscape", "soft light minimal landscape", "bright airy street photography"],
                    {"L_p50": (72, 10), "L_p01": (25, 12), "sat_mean": (0.22, 0.1)}),
    "moody_dark": ("Dark, low key, deep shadows, muted greens and blues",
                   ["moody forest fog", "moody landscape dark", "rainy city street"],
                   {"L_p50": (28, 10), "sat_mean": (0.28, 0.12), "L_p01": (1, 2)}),
    "teal_orange": ("Cinematic teal shadows and orange highlights",
                    ["teal and orange", "cinematic city street", "cinematic landscape"],
                    {"shadow_b": (-10, 8), "high_b": (20, 10), "high_a": (8, 6)}),
    "faded_film": ("Muted, faded blacks, gentle film colour (Classic Chrome-ish)",
                   ["film photography landscape", "fujifilm classic chrome", "35mm film street"],
                   {"L_p01": (18, 8), "sat_mean": (0.25, 0.1), "L_p99": (88, 8)}),
    "neon_night": ("Night city with saturated neon",
                   ["neon night street", "tokyo night neon", "city night lights street"],
                   {"L_p50": (25, 12), "chroma_mean": (30, 10, ">="), "sat_mean": (0.55, 0.15)}),
    "black_white": ("Black and white, strong tonal contrast",
                    ["black and white landscape photography", "black and white city", "monochrome landscape"],
                    {"chroma_mean": (0, 2), "L_p01": (2, 3), "L_p99": (97, 4)}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/styles/look_gallery"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/openverse"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/look_gallery.csv"))
    ap.add_argument("--per-look", type=int, default=4)
    ap.add_argument("--exclude", type=Path, default=Path("data/manifests/look_gallery.exclude.txt"),
                    help="Openverse ids rejected on manual review (not photos of real scenes, watermarks, ...)")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--max-person", type=float, default=0.04,
                    help="reject photos whose largest person covers more than this fraction of the frame")
    args = ap.parse_args()
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    from tools.collect_style_set import PersonDetector

    det = PersonDetector()
    seen: set[int] = set()
    excluded = set()
    if args.exclude.exists():
        excluded = {ln.split("#")[0].strip() for ln in args.exclude.read_text().splitlines() if ln.split("#")[0].strip()}
    picked: list[tuple[str, dict, Image.Image, float]] = []
    for look, (_, queries, target) in LOOKS.items():
        cands = []
        for q in queries:
            for page in range(1, args.pages + 1):
                for item in search(q, page, s, args.cache):
                    if item["id"] in excluded:
                        continue
                    img = download(item, s, args.cache)
                    if img is None or min(img.size) < 320:
                        continue
                    img = crop_letterbox(img)
                    if min(img.size) < 320:
                        continue
                    h = dhash(img)
                    if h in seen:
                        continue
                    st = colour_stats(img)
                    mono = st["chroma_mean"] < 3
                    if mono != (look == "black_white"):
                        continue
                    sc = score(st, target)
                    cands.append((sc, item, img, h))
        cands.sort(key=lambda c: -c[0])
        creators: set[str] = set()
        n = 0
        for sc, item, img, h in cands:
            if n >= args.per_look:
                break
            if item.get("creator") in creators:  # variety: one per creator
                continue
            people = det.people(img)
            if people and people[0][0] > args.max_person:  # scenes, not portraits
                continue
            creators.add(item.get("creator"))
            seen.add(h)
            picked.append((look, item, img, sc))
            n += 1
        print(f"{look}: {len(cands)} candidates, kept {n}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    looks = list(LOOKS)
    tw, th, pad, lw = 300, 220, 8, 230
    sheet = Image.new("RGB", (lw + args.per_look * (tw + pad) + pad, len(looks) * (th + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        big = small = ImageFont.load_default()
    with args.manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "look", "file", "license", "license_version", "creator", "title", "source",
                    "landing_url", "attribution", "score"])
        for li, look in enumerate(looks):
            y = pad + li * (th + pad)
            desc = LOOKS[look][0]
            d.text((pad, y + 8), f"{li + 1}. {look.replace('_', ' ')}", fill="black", font=big)
            words, line, yy = desc.split(), "", y + 42
            for word in words:
                if d.textlength(line + " " + word, font=small) > lw - 20:
                    d.text((pad, yy), line.strip(), fill="#444", font=small)
                    line, yy = "", yy + 17
                line += " " + word
            d.text((pad, yy), line.strip(), fill="#444", font=small)
            row = [p for p in picked if p[0] == look]
            for k, (_, item, img, sc) in enumerate(row):
                label = f"{li + 1}{'abcdefgh'[k]}"
                fn = args.out / look / f"{item['id']}.jpg"
                fn.parent.mkdir(parents=True, exist_ok=True)
                img.save(fn, quality=92)
                t = img.copy()
                t.thumbnail((tw, th))
                x = lw + pad + k * (tw + pad)
                sheet.paste(t, (x + (tw - t.width) // 2, y + (th - t.height) // 2))
                d.rectangle([x, y, x + 44, y + 30], fill="black")
                d.text((x + 6, y + 3), label, fill="white", font=big)
                w.writerow([label, look, str(fn), item.get("license"), item.get("license_version"),
                            item.get("creator"), item.get("title"), item.get("source"),
                            item.get("foreign_landing_url"), attribution(item), f"{sc:.3f}"])
    sheet.save(args.out / "look_gallery.jpg", quality=88)
    print(f"sheet -> {args.out / 'look_gallery.jpg'}; manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
