"""Download the landscape subset of MIT-Adobe FiveK (amendment A2).

Streams ``logasja/mit-adobe-fivek`` (one config per expert) and keeps only
rows whose labels match a landscape filter (default: ``location == outdoor``
and ``subject in {man_made, nature}``). Both the original and the expert's
retouch are downsized to a fixed long edge and saved as JPEG in the layout
``MIT5KDataset`` expects:

    <out>/original/<split>_<row>.jpg
    <out>/expert_<x>/<split>_<row>.jpg
    <out>/meta.csv      one row per *kept* image with split, row, labels, licence

The mirror stores full-resolution images in ~2 GB parquet shards, so every
row is transferred even when it is skipped. Image columns are therefore
read undecoded, and only kept rows are decoded. Re-running skips rows whose
files already exist, so interrupted downloads resume.

Note: the mirror holds ~250/251/231 train/validation/test rows per expert,
not the full 5,000 (PROJECT_PLAN §17 A2.2).

Usage:
    uv run python tools/download_fivek_landscapes.py --expert c --out data/fivek_landscape_c
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

DATASET = "logasja/mit-adobe-fivek"


def save_jpeg(data: bytes, path: Path, max_edge: int, quality: int) -> None:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=list("abcde"), default="c")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--locations", default="outdoor")
    ap.add_argument("--subjects", default="man_made,nature")
    ap.add_argument("--max_edge", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    out = args.out or Path(f"data/fivek_landscape_{args.expert}")
    (out / "original").mkdir(parents=True, exist_ok=True)
    (out / f"expert_{args.expert}").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.csv"
    meta: dict[str, dict] = {}
    if meta_path.exists():
        with meta_path.open(newline="") as f:
            meta = {r["file"]: r for r in csv.DictReader(f)}

    want_loc = set(args.locations.split(","))
    want_sub = set(args.subjects.split(","))
    fields = ["file", "split", "row", "location", "time", "light", "subject", "license"]

    for split in args.splits.split(","):
        ds = load_dataset(DATASET, args.expert, split=split, streaming=True)
        feats = ds.features
        ds = ds.cast_column("original", HFImage(decode=False))
        ds = ds.cast_column("augmented", HFImage(decode=False))
        names = {k: feats[k].names for k in ("location", "time", "light", "subject", "license")}
        t0, seen, kept = time.time(), 0, 0
        for row, ex in enumerate(ds):
            seen += 1
            labels = {k: names[k][ex[k]] for k in names}
            if labels["location"] in want_loc and labels["subject"] in want_sub:
                fname = f"{split}_{row:04d}.jpg"
                o = out / "original" / fname
                e = out / f"expert_{args.expert}" / fname
                if not (o.exists() and e.exists()):
                    try:
                        save_jpeg(ex["original"]["bytes"], o, args.max_edge, args.quality)
                        save_jpeg(ex["augmented"]["bytes"], e, args.max_edge, args.quality)
                    except Exception as exc:  # corrupt row: skip, keep going
                        print(f"  skip {split}/{row}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                meta[fname] = {"file": fname, "split": split, "row": row, **labels}
                kept += 1
            if seen % 10 == 0:
                rate = seen / (time.time() - t0)
                print(f"  {split}: seen {seen}, kept {kept} ({rate:.2f} rows/s)", flush=True)
                with meta_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(meta.values())
        print(f"{split}: done, seen {seen}, kept {kept}", flush=True)

    with meta_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(meta.values())
    print(f"total kept: {len(meta)} -> {out}")


if __name__ == "__main__":
    main()
