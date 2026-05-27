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

Output layout::

    <out_dir>/
        original/img_0000.jpg
        original/img_0001.jpg
        ...
        expert_<x>/img_0000.jpg
        expert_<x>/img_0001.jpg

This is exactly the layout the existing ``MIT5KDataset`` consumes.

Run from anywhere::

    python tools/download_fivek_subset.py --expert c --count 200 --out_dir data/fivek_c_200
"""

from __future__ import annotations

import argparse
import os
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=list("abcde"), default="c",
                        help="Which expert config to fetch.")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation"])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out_dir", default="data/fivek_c_200")
    parser.add_argument("--max_edge", type=int, default=512,
                        help="Long-edge resize before saving (px).")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--skip_existing", action="store_true",
                        help="If both paired files already exist, skip the example.")
    args = parser.parse_args()

    original_dir = os.path.join(args.out_dir, "original")
    expert_dir = os.path.join(args.out_dir, f"expert_{args.expert}")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(expert_dir, exist_ok=True)

    print(f"Streaming {DATASET} config={args.expert} split={args.split} count={args.count}")
    ds = load_dataset(DATASET, args.expert, split=args.split, streaming=True)

    written = 0
    for idx, ex in enumerate(ds):
        if written >= args.count:
            break
        name = f"img_{written:04d}.jpg"
        original_path = os.path.join(original_dir, name)
        expert_path = os.path.join(expert_dir, name)
        if args.skip_existing and os.path.exists(original_path) and os.path.exists(expert_path):
            written += 1
            continue
        try:
            _save_jpeg(ex["original"], original_path, args.max_edge, args.quality)
            _save_jpeg(ex["augmented"], expert_path, args.max_edge, args.quality)
        except Exception as exc:  # network hiccup, decode failure, etc.
            print(f"  [skip idx={idx}] {type(exc).__name__}: {exc}")
            continue
        written += 1
        if written % 20 == 0:
            print(f"  saved {written}/{args.count}")

    print(f"done. wrote {written} pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
