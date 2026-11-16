"""Find unedited, openly licensed example photos on Wikimedia Commons.

The "before" panel of every example must be an unedited photo, so that the look is shown acting
on a camera's own picture rather than on someone else's edit. A Commons file qualifies when its
embedded metadata says it came straight from a camera:

- it is a JPEG with camera Make and Model tags (a scan, render or screenshot has none);
- its Software tag is absent or names camera firmware, not an editor (EDITORS below);
- its modification time (DateTime) is within a day of capture (DateTimeOriginal): an editor
  rewrites DateTime on export, a camera writes both at capture;
- its licence is CC0, public domain, CC BY or CC BY-SA;
- it is at least MIN_EDGE px on the long side.

Thumbnails of the qualifying files are saved for review, with the metadata, under --out.

    uv run python tools/find_unedited.py --out <dir> [--only look ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "IrodoriExampleFinder/1.0 (https://github.com/kyleyhw/image_editing)"
MIN_EDGE = 1600            # examples are shown up to 1200 px; leave room to crop
LICENCES = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [1-4]\.0)", re.I)
EDITORS = re.compile(r"lightroom|photoshop|gimp|capture ?one|dxo|darktable|rawtherapee|snapseed|luminar|affinity|aperture|"
                     r"picasa|paint\.net|acdsee|on1|silkypix|digikam|polarr|vsco|photomatix|aurora|hdr|lightzone|"
                     r"photoscape|pixelmator|camera raw|photo ?lab|corel|paintshop|irfanview|faststone|xnview|shotwell|"
                     r"windows photo|photos \d|picmonkey|fotor|befunky|canva", re.I)
# Commons search queries per look (and for the welcome-page samples), aimed at the look's subject
QUERIES: dict[str, list[str]] = {
    "moody_forest": ["foggy forest", "misty forest trees"],
    "golden_hour": ["sunset landscape mountains", "sunrise landscape"],
    "velvia": ["lake landscape autumn trees sunny", "mountain lake forest"],
    "milky_way": ["milky way landscape", "night sky stars mountain"],
    "blue_hour": ["blue hour sea", "twilight coast after sunset"],
    "teal_orange": ["city street daylight buildings", "downtown street architecture"],
    "kodachrome": ["market street colourful", "colourful street scene"],
    "matte_urban": ["old building facade street", "graffiti wall street"],
    "cinestill_800t": ["street at night shops", "city street night lights"],
    "tokyo_night": ["alley night lanterns Tokyo", "neon street night Japan"],
    "cyberpunk": ["neon signs street night", "shinjuku night"],
    "portra_400": ["portrait woman outdoors", "portrait man park"],
    "bright_airy": ["wedding bride portrait", "bride window light"],
    "moody_portrait": ["portrait indoor window light", "low key portrait"],
    "tri_x": ["street portrait man", "portrait elderly man street"],
    "golden_backlit": ["backlit portrait sunset", "portrait sunset field"],
    "bright_food": ["breakfast table food top view", "food plate table daylight"],
    "moody_food": ["coffee cup table dark", "bread still life"],
    "ektar_100": ["colourful houses street travel", "fruit market colourful"],
    "autumn": ["autumn forest leaves", "autumn foliage path"],
    "winter": ["snow landscape trees", "snowy field winter"],
    "samples": ["forest path", "harbour dusk", "mountain lake reflection", "valley mountains", "street city", "night city lights"],
}


def get(params: dict) -> dict:
    for attempt in range(6):
        r = requests.get(API, params={"format": "json", **params}, headers={"User-Agent": UA}, timeout=60)
        if r.status_code == 200 and r.text.startswith("{"):
            return r.json()
        time.sleep(30 * (attempt + 1))                     # rate limit: wait and retry
    raise OSError(f"Commons API refused: {r.status_code} {r.text[:120]}")


def exif_time(v: str | None) -> datetime | None:
    try:
        return datetime.strptime(str(v)[:19], "%Y:%m:%d %H:%M:%S") if v else None
    except ValueError:
        return None


def verdict(ii: dict) -> tuple[bool, str]:
    if ii.get("mime") != "image/jpeg":
        return False, "not a JPEG"
    if max(ii["width"], ii["height"]) < MIN_EDGE:
        return False, "too small"
    lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
    if not LICENCES.match(lic):
        return False, f"licence {lic}"
    md = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if not isinstance(m.get("value"), list)}
    if not md.get("Make") or not md.get("Model"):
        return False, "no camera tags"
    sw = str(md.get("Software") or "")
    if EDITORS.search(sw):
        return False, f"edited ({sw})"
    t0, t1 = exif_time(md.get("DateTimeOriginal")), exif_time(md.get("DateTime"))
    if t0 and t1 and abs((t1 - t0).total_seconds()) > 86400:
        return False, "modified after capture"
    if not t0:
        return False, "no capture time"
    return True, f"{md.get('Make')} {md.get('Model')}; software {sw or 'none'}; {lic}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--per-query", type=int, default=40)
    args = ap.parse_args()
    for look, qs in QUERIES.items():
        if args.only and look not in args.only:
            continue
        d = args.out / look
        d.mkdir(parents=True, exist_ok=True)
        found: list[dict] = []
        for q in qs:
            res = get({"action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": 6,
                       "gsrlimit": args.per_query, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata|metadata",
                       "iiurlwidth": 640})
            for page in res.get("query", {}).get("pages", {}).values():
                ii = (page.get("imageinfo") or [{}])[0]
                if not ii:
                    continue
                ok, why = verdict(ii)
                if not ok or any(f["title"] == page["title"] for f in found):
                    continue
                em = ii.get("extmetadata", {})
                found.append({"title": page["title"], "why": why, "thumb": ii["thumburl"], "url": ii["url"],
                              "page": ii.get("descriptionurl"), "width": ii["width"], "height": ii["height"],
                              "licence": em.get("LicenseShortName", {}).get("value"),
                              "artist": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()})
            time.sleep(2)                                    # be gentle with the API
        for k, f in enumerate(found):
            t = d / f"{k:02d}.jpg"
            if not t.exists():
                r = requests.get(f["thumb"], headers={"User-Agent": UA}, timeout=60)
                if r.ok:
                    t.write_bytes(r.content)
                time.sleep(0.5)
        (d / "candidates.json").write_text(json.dumps(found, indent=1))
        print(f"{look}: {len(found)} unedited candidates", flush=True)


if __name__ == "__main__":
    sys.exit(main())
