"""Ingest the project owner's own photos into a private, local dataset.

Owner photos have two possible roles (PROJECT_PLAN §17, amendment A3):

  * unedited camera output  -> *inputs* the model learns to edit;
  * already-edited photos   -> examples of the owner's own look, and, if the
                               unedited original is supplied too, a *paired*
                               (before, after) example: the most valuable kind.

What this script does:

  1. Copies each file into data/owner/originals/ (named by content hash).
  2. Colour-manages it: converts from its embedded ICC profile (iPhones use
     Display P3) to sRGB, writing data/owner/srgb/<id>.jpg. All statistics
     and training use the sRGB copy.
  3. Measures the CIELAB statistics used by the look profiles.
  4. Writes/updates data/owner/manifest.csv, keeping any labels already set.
     Labels (edited yes/no/unknown, scene urban/nature, pair_of) are meant to
     be confirmed by the owner, and suspected values can be passed on the
     command line.

Everything lives under data/ (gitignored): owner photos never enter git.
Location metadata is never read or stored.

Usage:
    uv run python tools/ingest_owner_photos.py photo1.jpg photo2.jpg ...
    uv run python tools/ingest_owner_photos.py --label <id> edited=yes scene=urban
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from collect_style_set import colour_stats  # noqa: E402

ROOT = Path("data/owner")
FIELDS = ["id", "source_name", "width", "height", "icc", "regime",
          "edited", "edited_source", "scene", "pair_of", "notes",
          "L_p01", "L_p50", "L_p99", "shadow_b", "mid_a", "mid_b", "high_b", "sat_mean"]
LABEL_FIELDS = {"edited", "edited_source", "scene", "pair_of", "notes"}


def to_srgb(img: Image.Image) -> tuple[Image.Image, str]:
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    if not icc:
        return img.convert("RGB"), "none (assumed sRGB)"
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    desc = ImageCms.getProfileDescription(src).strip()
    dst = ImageCms.createProfile("sRGB")
    out = ImageCms.profileToProfile(img.convert("RGB"), src, dst,
                                    renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
    return out, desc


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save_manifest(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})


def ingest(files: list[Path], rows: dict[str, dict]) -> None:
    for src in files:
        data = src.read_bytes()
        pid = hashlib.sha256(data).hexdigest()[:12]
        (ROOT / "originals").mkdir(parents=True, exist_ok=True)
        (ROOT / "srgb").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, ROOT / "originals" / f"{pid}{src.suffix.lower()}")
        srgb, icc = to_srgb(Image.open(src))
        srgb.save(ROOT / "srgb" / f"{pid}.jpg", quality=95)
        st = colour_stats(srgb)
        row = rows.get(pid, {"edited": "unknown", "edited_source": "", "scene": "",
                             "pair_of": "", "notes": ""})
        row.update({
            "id": pid, "source_name": src.name, "width": srgb.width, "height": srgb.height,
            "icc": icc, "regime": "night" if st["L_p50"] < 35 else "day",
            **{k: round(v, 2) for k, v in st.items() if k in FIELDS},
        })
        rows[pid] = row
        print(f"{pid}  {src.name}  {srgb.size}  icc={icc}  regime={row['regime']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--label", nargs="+", metavar=("ID", "KEY=VALUE"),
                    help="Set labels on an existing entry, e.g. --label ab12cd34ef56 edited=yes scene=urban")
    args = ap.parse_args()

    manifest = ROOT / "manifest.csv"
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)
    if args.files:
        ingest(args.files, rows)
    if args.label:
        pid, *pairs = args.label
        if pid not in rows:
            sys.exit(f"unknown id {pid}")
        for kv in pairs:
            k, v = kv.split("=", 1)
            if k not in LABEL_FIELDS:
                sys.exit(f"unknown label {k}; allowed: {sorted(LABEL_FIELDS)}")
            rows[pid][k] = v
    save_manifest(manifest, rows)
    print(f"{len(rows)} photos in {manifest}")


if __name__ == "__main__":
    main()
