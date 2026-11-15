"""Contact sheet of FiveK (original, expert) pairs for the Phase 20 report.

Pairs are drawn with a seeded RNG from the first ``--pool`` rows of a split
that exist on disk. Each cell shows the original (left) and the expert's
edit (right); the expert proxy is ProPhoto RGB and is converted to sRGB for
display (out-of-gamut colours clipped), the original is already sRGB.

Usage::

    uv run python tools/fivek_contact_sheet.py --out tests/reports/assets/phase20_fivek_sheet.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fetch_fivek import decode_linear_srgb, srgb_encode  # noqa: E402

RNG_SEED = 2020  # any fixed seed; makes the sample reproducible
CELL = 150  # px, long edge of each thumbnail: 12 pairs fit in ~1000 px width
COLS = 3  # pairs per row
GAP, LABEL = 6, 14  # px between images; px of caption text under each pair


def thumb(path: Path) -> Image.Image:
    lin = decode_linear_srgb(path.read_bytes())
    im = Image.fromarray(np.round(srgb_encode(lin) * 255).astype(np.uint8))
    im.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/fivek")
    ap.add_argument("--split", default="train")
    ap.add_argument("--expert", default="c")
    ap.add_argument("--pool", type=int, default=240)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="tests/reports/assets/phase20_fivek_sheet.jpg")
    args = ap.parse_args()

    root = Path(args.root) / args.split
    have = [
        i
        for i in range(args.pool)
        if (root / "original" / f"img_{i:04d}.png").exists()
        and (root / f"expert_{args.expert}" / f"img_{i:04d}.png").exists()
    ]
    rng = np.random.default_rng(RNG_SEED)
    picks = sorted(
        rng.choice(have, size=min(args.n, len(have)), replace=False).tolist()
    )
    rows = (len(picks) + COLS - 1) // COLS
    pair_w, pair_h = 2 * CELL + GAP, CELL + LABEL
    sheet = Image.new(
        "RGB",
        (COLS * pair_w + (COLS + 1) * 2 * GAP, rows * (pair_h + 2 * GAP) + 2 * GAP),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for n, i in enumerate(picks):
        x0 = 2 * GAP + (n % COLS) * (pair_w + 2 * GAP)
        y0 = 2 * GAP + (n // COLS) * (pair_h + 2 * GAP)
        name = f"img_{i:04d}.png"
        a, b = (
            thumb(root / "original" / name),
            thumb(root / f"expert_{args.expert}" / name),
        )
        sheet.paste(a, (x0 + (CELL - a.width) // 2, y0 + (CELL - a.height) // 2))
        sheet.paste(
            b, (x0 + CELL + GAP + (CELL - b.width) // 2, y0 + (CELL - b.height) // 2)
        )
        draw.text(
            (x0, y0 + CELL + 1),
            f"{args.split} {i:04d}: original | expert {args.expert.upper()}",
            fill=(60, 60, 60),
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # quality 85: a viewing sheet, kept well under the 400 KB limit
    sheet.save(out, format="JPEG", quality=85, optimize=True)
    print(out, sheet.size, f"{out.stat().st_size / 1024:.0f} KB", "rows", picks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
