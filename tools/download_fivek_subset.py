"""Stream a paired (original, expert-retouched) subset of the MIT-Adobe
FiveK dataset from HuggingFace and save it in the layout that
``MIT5KDataset`` expects.

Source: ``logasja/mit-adobe-fivek`` on HuggingFace. The dataset is
organised into five configurations (``a`` ... ``e``), one per
photography-student expert. Each row contains an ``original`` and an
``augmented`` (expert-retouched) image.

We stream (no full download) and downsize both images to a fixed long
edge (default 512 px) before saving as JPEG quality 90 - the network
ultimately trains at 192 px, so a 512 px cache is more than enough
detail while keeping disk usage to ~100 KB per image.

Resumable
---------

The script saves files as ``img_NNNN.jpg`` where NNNN matches the
dataset row index. On startup it counts the largest contiguous prefix
of already-saved pairs and resumes the HuggingFace iterator with
``.skip(N)`` so previously-downloaded rows are not refetched. If the
HF connection drops mid-run, re-run the same command and it will pick
up at the next missing index.

Output layout::

    <out_dir>/
        original/img_0000.jpg
        original/img_0001.jpg
        ...
        expert_<x>/img_0000.jpg
        expert_<x>/img_0001.jpg

This is exactly the layout the existing ``MIT5KDataset`` consumes.

Run from anywhere::

    python tools/download_fivek_subset.py --expert c --count 500 --out_dir data/fivek_c_500
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from PIL import Image
from datasets import load_dataset


DATASET = "logasja/mit-adobe-fivek"


def _save_jpeg(img: Image.Image, path: str, max_edge: int, quality: int) -> None:
    """Downsize so the long edge is <= max_edge, then save as JPEG."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def _count_contiguous_pairs(out_dir: str, expert: str) -> int:
    """Largest N such that img_0000.jpg .. img_(N-1).jpg exist in both dirs."""
    original_dir = os.path.join(out_dir, "original")
    expert_dir = os.path.join(out_dir, f"expert_{expert}")
    if not (os.path.isdir(original_dir) and os.path.isdir(expert_dir)):
        return 0
    n = 0
    while True:
        name = f"img_{n:04d}.jpg"
        if os.path.exists(os.path.join(original_dir, name)) and os.path.exists(
            os.path.join(expert_dir, name)
        ):
            n += 1
        else:
            return n


def _save_pair(ex, out_dir: str, expert: str, idx: int, max_edge: int, quality: int) -> None:
    name = f"img_{idx:04d}.jpg"
    _save_jpeg(ex["original"], os.path.join(out_dir, "original", name), max_edge, quality)
    _save_jpeg(
        ex["augmented"], os.path.join(out_dir, f"expert_{expert}", name), max_edge, quality
    )


def _stream_pass(args, start_idx: int) -> int:
    """Stream from ``start_idx`` and save pairs until ``args.count`` is hit
    or an exception interrupts. Returns the new (resumable) index after the
    pass.
    """
    print(f"  streaming from row {start_idx} ...")
    ds = load_dataset(DATASET, args.expert, split=args.split, streaming=True)
    if start_idx > 0:
        ds = ds.skip(start_idx)

    cur = start_idx
    for ex in ds:
        if cur >= args.count:
            break
        try:
            _save_pair(ex, args.out_dir, args.expert, cur, args.max_edge, args.quality)
        except Exception as exc:
            # Per-row save failure (e.g. corrupt webp) is non-fatal; skip and
            # bump the index so we do not re-attempt this row.
            print(f"    skip row {cur}: {type(exc).__name__}: {exc}")
            cur += 1
            continue
        cur += 1
        if cur % 20 == 0:
            print(f"    saved {cur}/{args.count}")
    return cur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=list("abcde"), default="c")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation"])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--out_dir", default="data/fivek_c_500")
    parser.add_argument("--max_edge", type=int, default=512)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--max_attempts", type=int, default=8,
                        help="How many times to retry the HF stream on a transient error.")
    parser.add_argument("--retry_sleep", type=float, default=3.0,
                        help="Seconds to wait between connection retries.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, f"expert_{args.expert}"), exist_ok=True)

    print(f"Streaming {DATASET} config={args.expert} split={args.split} target={args.count}")
    existing = _count_contiguous_pairs(args.out_dir, args.expert)
    print(f"  found {existing} existing pair(s); resuming from row {existing}")

    cur = existing
    attempts = 0
    while cur < args.count and attempts < args.max_attempts:
        attempts += 1
        try:
            cur = _stream_pass(args, cur)
        except Exception as exc:
            print(f"  attempt {attempts}/{args.max_attempts} interrupted at row {cur}: "
                  f"{type(exc).__name__}: {exc}")
            time.sleep(args.retry_sleep)
            continue
        # Successful pass: we either hit `cur >= args.count` or exhausted the
        # iterator. The latter would be unusual on a 3500-row train split.
        break

    final = _count_contiguous_pairs(args.out_dir, args.expert)
    print(f"done. {final} contiguous pairs in {args.out_dir}")


if __name__ == "__main__":
    main()
