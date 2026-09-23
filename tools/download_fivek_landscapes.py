"""Download the landscape subset of MIT-Adobe FiveK (amendment A2).

Source: ``logasja/mit-adobe-fivek`` on the Hugging Face Hub, one folder per
expert (``a`` ... ``e``). This is the **full** FiveK (~4,900 rows per expert:
44 train, 7 validation and 13 test parquet shards of 77 rows each,
~123 GB per expert at full resolution).

Rows whose labels match a landscape filter are kept (default:
``location == outdoor`` and ``subject in {man_made, nature}``). The
original and the expert retouch are both downsized to a fixed long edge and
saved as JPEG in the layout ``MIT5KDataset`` expects:

    <out>/original/<split>_<row>.jpg
    <out>/expert_<x>/<split>_<row>.jpg
    <out>/meta.csv     one row per kept image: split, row, labels, licence

Memory-safe and resumable. Each shard (~2 GB, one row group) is downloaded
to disk. The label columns are read first; then each image column is read
separately and only kept rows are decoded; then the shard is deleted.
Streaming with ``datasets`` buffered whole row groups for both image
columns at once and was OOM-killed on a 16 GB machine. Rows whose output
files already exist are skipped, so interrupted downloads resume.

Usage:
    uv run python tools/download_fivek_landscapes.py --expert c --out data/fivek_landscape_c
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "logasja/mit-adobe-fivek"
LABELS = ("location", "time", "light", "subject", "license")


def label_names(pf: pq.ParquetFile) -> dict[str, list[str]]:
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    feats = meta["info"]["features"]
    return {k: feats[k]["names"] for k in LABELS}


def save_jpeg(data: bytes, path: Path, max_edge: int, quality: int) -> None:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def write_meta(path: Path, meta: dict[str, dict]) -> None:
    fields = ["file", "split", "row", *LABELS]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(meta.values(), key=lambda r: r["file"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=list("abcde"), default="c")
    ap.add_argument("--splits", default="test,validation,train")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--locations", default="outdoor")
    ap.add_argument("--subjects", default="man_made,nature")
    ap.add_argument("--max_edge", type=int, default=512)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--max_shards", type=int, default=0, help="Per split; 0 = all.")
    ap.add_argument("--cache", type=Path, default=Path("data/cache/hf"))
    args = ap.parse_args()

    out = args.out or Path(f"data/fivek_landscape_{args.expert}")
    odir, edir = out / "original", out / f"expert_{args.expert}"
    odir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.csv"
    meta: dict[str, dict] = {}
    if meta_path.exists():
        with meta_path.open(newline="") as f:
            meta = {r["file"]: r for r in csv.DictReader(f)}
    want_loc, want_sub = set(args.locations.split(",")), set(args.subjects.split(","))

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"{args.expert}/") and f.endswith(".parquet"))
    t0 = time.time()
    for split in args.splits.split(","):
        shards = [f for f in files if f.split("/")[1].startswith(split + "-")]
        if args.max_shards:
            shards = shards[: args.max_shards]
        row0 = 0
        for si, shard in enumerate(shards):
            local = Path(hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=args.cache))
            pf = pq.ParquetFile(local)
            names = label_names(pf)
            labels = pf.read(columns=list(LABELS)).to_pydict()
            n = pf.metadata.num_rows
            keep = {}
            for i in range(n):
                lab = {k: names[k][labels[k][i]] for k in LABELS}
                if lab["location"] in want_loc and lab["subject"] in want_sub:
                    fname = f"{split}_{row0 + i:04d}.jpg"
                    keep[i] = (fname, lab)
            todo = {i: v for i, v in keep.items()
                    if not ((odir / v[0]).exists() and (edir / v[0]).exists())}
            for col, d in (("original", odir), ("augmented", edir)):
                if not todo:
                    break
                column = pf.read(columns=[col]).column(col)
                for i, (fname, _) in todo.items():
                    try:
                        save_jpeg(column[i]["bytes"].as_py(), d / fname, args.max_edge, args.quality)
                    except Exception as exc:  # corrupt row: skip it
                        print(f"  skip {shard}[{i}] {col}: {type(exc).__name__}: {exc}", flush=True)
                del column
            for i, (fname, lab) in keep.items():
                if (odir / fname).exists() and (edir / fname).exists():
                    meta[fname] = {"file": fname, "split": split, "row": row0 + i, **lab}
            row0 += n
            local.unlink(missing_ok=True)  # free disk: shards are ~2 GB each
            write_meta(meta_path, meta)
            print(f"{split} shard {si + 1}/{len(shards)}: rows {n}, kept {len(keep)}, "
                  f"total kept {len(meta)} [{time.time() - t0:.0f}s]", flush=True)
    write_meta(meta_path, meta)
    print(f"done: {len(meta)} pairs -> {out}")


if __name__ == "__main__":
    main()
