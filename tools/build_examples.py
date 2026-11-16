"""Example panels and try-it samples from unedited photos.

``tools/example_picks.json`` maps each tutorial look, and each welcome-page sample, to one photo
that is unedited by construction:

- ``{"raw": id}``: a CC0 raw file from raw.pixls.us (found with tools/raw_previews.py). The
  picture used is the camera's own JPEG rendering embedded in the raw file, rotated as the camera
  recorded; where that preview is smaller than needed, the raw is developed neutrally with LibRaw
  (camera white balance, standard sRGB rendering, no further adjustment).
- ``{"title": "File:..."}``: a Wikimedia Commons file that passes the straight-out-of-camera test
  of tools/find_unedited.py, re-checked here.
For each look this writes ``studio/web/public/examples/<look>/before.jpg`` (the camera's picture,
1200 px on the long side) and ``after.jpg`` (the same through the look's slider settings in the
reference pipeline, photostyle/pipeline.py, exactly as the editor applies it), and updates the
look's example credit in ``styles.json``. Samples go to ``studio/web/public/samples`` (1600 px and
a 320 px thumbnail) with ``samples.json`` and ``ATTRIBUTION.md``.

    uv run python tools/build_examples.py [--only look ...]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import pipeline as P  # noqa: E402
from photostyle import slider_looks as SL  # noqa: E402
from tools.find_unedited import API, UA, verdict  # noqa: E402

RAW_DIR = ROOT / "data/raw_examples"        # downloaded raw files (not committed)
RAW_CATALOGUE = "https://raw.pixls.us/json/getrepository.php?set=all"
_catalogue: dict[str, dict] = {}

PICKS = ROOT / "tools/example_picks.json"
EXAMPLES = ROOT / "studio/web/public/examples"
SAMPLES = ROOT / "studio/web/public/samples"
STYLES = ROOT / "studio/web/pages/models/styles.json"
EXAMPLE_EDGE = 1200         # the example panel is at most about 600 css px wide: 2x for sharp displays
SAMPLE_EDGE = 1600          # a try-it photo: large enough to judge detail sliders, small to download
THUMB_EDGE = 320            # browser.js THUMB_EDGE


def licence(short: str) -> tuple[str, str]:
    """Commons LicenseShortName -> (code, version) as the credits use them."""
    s = short.lower()
    if s.startswith("cc0"):
        return "cc0", "1.0"
    if s.startswith("public domain") or s == "pd":
        return "pdm", ""
    m = re.match(r"cc (by(?:-sa)?) ([1-4]\.0)", s)
    if not m:
        raise ValueError(f"unexpected licence {short!r}")
    return m.group(1), m.group(2)


def fetch(title: str, width: int) -> tuple[Image.Image, dict]:
    r = requests.get(API, params={"format": "json", "action": "query", "titles": title, "prop": "imageinfo",
                                  "iiprop": "url|size|mime|extmetadata|metadata", "iiurlwidth": width},
                     headers={"User-Agent": UA}, timeout=60).json()
    ii = next(iter(r["query"]["pages"].values()))["imageinfo"][0]
    ok, why = verdict(ii)
    if not ok:                                         # re-checked at build time, not only when searching
        raise ValueError(f"{title}: not an unedited camera file ({why})")
    img = Image.open(io.BytesIO(requests.get(ii["thumburl"], headers={"User-Agent": UA}, timeout=120).content))
    return ImageOps.exif_transpose(img).convert("RGB"), ii


def raw_photo(fid: str, edge: int) -> tuple[Image.Image, dict]:
    """The camera's own picture from a raw.pixls.us file: its embedded JPEG if at least ``edge`` px,
    else a neutral LibRaw development. Returns the image and its credit."""
    import rawpy
    if not _catalogue:
        for row in requests.get(RAW_CATALOGUE, timeout=120).json()["data"]:
            m = re.search(r"href='([^']+/getfile\.php/(\d+)/nice/[^']+)'", row[7])
            if m:
                _catalogue[m.group(2)] = {"make": row[0], "model": row[1], "url": m.group(1)}
    info = _catalogue[fid]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    f = next(iter(RAW_DIR.glob(f"{fid}.*")), None)
    if f is None:
        f = RAW_DIR / f"{fid}.{info['url'].rsplit('.', 1)[1]}"
        f.write_bytes(requests.get(info["url"], timeout=300).content)
    with rawpy.imread(str(f)) as raw:
        img = None
        try:
            t = raw.extract_thumb()
            if t.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(bytes(t.data))).convert("RGB")
                # LibRaw's flip code: 3 = 180 degrees, 5 = 90 counter-clockwise, 6 = 90 clockwise
                rot = {3: Image.Transpose.ROTATE_180, 5: Image.Transpose.ROTATE_90, 6: Image.Transpose.ROTATE_270}.get(raw.sizes.flip)
                img = img.transpose(rot) if rot else img
        except (rawpy.LibRawError, OSError):
            img = None
        source = "the camera's embedded JPEG"
        if img is None or max(img.size) < edge:
            img = Image.fromarray(raw.postprocess(use_camera_wb=True, output_bps=8))
            source = "a neutral LibRaw development of the raw file"
    cam = f"{info['make']} {info['model']}"
    return img, {"title": f"{cam} sample {fid}", "creator": None, "license": "cc0", "license_version": "1.0",
                 "url": "https://raw.pixls.us/", "attribution": f"\"{cam} sample {fid}\", CC0 1.0, via raw.pixls.us",
                 "unedited": f"{source}, {cam}"}


def photo(pick: dict, edge: int) -> tuple[Image.Image, dict]:
    if "raw" in pick:
        return raw_photo(str(pick["raw"]), edge)
    img, ii = fetch(pick["title"], edge)
    return img, credit(pick["title"], ii)


def fit(img: Image.Image, edge: int) -> Image.Image:
    s = edge / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS) if s < 1 else img


def credit(title: str, ii: dict) -> dict:
    em = ii.get("extmetadata", {})
    code, ver = licence(em.get("LicenseShortName", {}).get("value", ""))
    name = re.sub(r"^File:|\.[A-Za-z]+$", "", title)
    artist = re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip() or None
    lic = {"cc0": "CC0 1.0", "pdm": "Public Domain"}.get(code, f"CC {code.upper()} {ver}")
    return {"title": name, "creator": artist, "license": code, "license_version": ver, "url": ii.get("descriptionurl"),
            "attribution": f"\"{name}\" by {artist or 'unknown'}, {lic}, via Wikimedia Commons", "unedited": True}


def render(img: Image.Image, name: str) -> Image.Image:
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        y = P.render(x, SL.look_sliders(name)).clamp(0, 1)
    return Image.fromarray((y.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    picks = json.loads(PICKS.read_text())
    styles = json.loads(STYLES.read_text())
    cards = {s["card"]["name"]: s for s in styles["styles"]}
    for name, pick in picks["looks"].items():
        if args.only and name not in args.only:
            continue
        img, cred = photo(pick, EXAMPLE_EDGE)
        img = fit(img, EXAMPLE_EDGE)
        out = EXAMPLES / name
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / "before.jpg", quality=88)
        render(img, name).save(out / "after.jpg", quality=88)
        cards[name]["card"]["example"] = {"before": f"examples/{name}/before.jpg", "after": f"examples/{name}/after.jpg",
                                          "photo": cred, "why": pick["why"]}
        print(f"{name}: {cred['title']}", flush=True)
        time.sleep(1)
    STYLES.write_text(json.dumps(styles))
    if args.only and "samples" not in args.only:
        return
    for old in SAMPLES.glob("*.jpg"):                 # the sample set is replaced as a whole
        old.unlink()
    rows, lines = [], ["# Sample photos", "", "Unedited camera pictures (each camera's own JPEG from a CC0 raw file, or a "
                       "straight-out-of-camera file), used as try-it samples. Made by tools/build_examples.py.", ""]
    for file, pick in picks["samples"].items():
        img, c = photo(pick, SAMPLE_EDGE)
        img = fit(img, SAMPLE_EDGE)
        if c["license"] not in ("cc0", "pdm"):
            raise ValueError(f"sample {file} must be public domain or CC0, not {c['license']}")
        img.save(SAMPLES / f"{file}.jpg", quality=88)
        fit(img, THUMB_EDGE).save(SAMPLES / f"{file}.thumb.jpg", quality=85)
        rows.append({"file": f"{file}.jpg", "thumb": f"{file}.thumb.jpg", "title": c["title"], "creator": c["creator"],
                     "license": c["license"], "landing_url": c["url"], "attribution": c["attribution"]})
        lines.append(f"- `{file}.jpg`: {c['attribution']} ({c['url']})")
        print(f"sample {file}: {c['title']}", flush=True)
        time.sleep(1)
    (SAMPLES / "samples.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (SAMPLES / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
