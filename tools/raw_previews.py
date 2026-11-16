"""Thumbnails of the raw.pixls.us collection (2,020 CC0 raw files), to find example subjects.

A raw file carries the camera's own JPEG rendering (a thumbnail and usually a larger preview),
mostly near the start of the file. This reads the first READ bytes of every file in the catalogue
over a range request, decodes the largest JPEG found there, and saves a THUMB px thumbnail with
the file's id, make and model, for review. Nothing else is kept.

    uv run python tools/raw_previews.py --out <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

CATALOGUE = "https://raw.pixls.us/json/getrepository.php?set=all"
READ = 1_500_000      # the first 1.5 MB: enough for the embedded thumbnail of most formats
THUMB = 200           # px, long edge: enough to judge a subject in a contact sheet
WORKERS = 8


def jpegs(blob: bytes) -> list[Image.Image]:
    out = []
    for m in re.finditer(rb"\xff\xd8\xff", blob):
        try:
            im = Image.open(io.BytesIO(blob[m.start():]))
            im.load()
            out.append(im)
        except Exception:                      # a truncated or false marker
            continue
    return out


def one(row: list, out: Path) -> dict | None:
    href = re.search(r"href='([^']+/nice/[^']+)'", row[7])
    if not href:
        return None
    url = href.group(1)
    m = re.search(r"getfile\.php/(\d+)/", url)
    if not m:
        return None
    fid = m.group(1)
    t = out / f"{fid}.jpg"
    rec = {"id": fid, "make": row[0], "model": row[1], "mode": row[2], "url": url}
    if t.exists():
        return rec
    try:
        r = requests.get(url, headers={"Range": f"bytes=0-{READ - 1}"}, timeout=60)
        ims = jpegs(r.content)
    except requests.RequestException:
        return None
    if not ims:
        return None
    im = max(ims, key=lambda i: i.width * i.height).convert("RGB")
    rec["preview"] = [im.width, im.height]
    im.thumbnail((THUMB, THUMB))
    im.save(t, quality=80)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = requests.get(CATALOGUE, timeout=120).json()["data"]
    with ThreadPoolExecutor(WORKERS) as ex:
        recs = [r for r in ex.map(lambda row: one(row, args.out), rows) if r]
    (args.out / "index.json").write_text(json.dumps(recs, indent=1))
    print(f"{len(recs)} of {len(rows)} files have a readable preview")


if __name__ == "__main__":
    main()
