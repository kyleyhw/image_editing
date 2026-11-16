"""The README's overview figure: unedited photos through a few looks, with the current pipeline.

Rows are three of the unedited example photos (each camera's own picture, from CC0 raw files;
tools/build_examples.py); columns are the unedited photo and the same photo through four looks'
slider settings in the reference pipeline (photostyle/pipeline.py), exactly as the editor applies
them at full strength.

    uv run python tools/readme_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import pipeline as P  # noqa: E402
from photostyle import slider_looks as SL  # noqa: E402

ROWS = [("Mountain lake", "studio/web/public/samples/mountain-lake.jpg"),
        ("Night street", "studio/web/public/examples/tokyo_night/before.jpg"),
        ("Portrait outdoors", "studio/web/public/examples/portra_400/before.jpg")]
LOOKS = [("Portra 400", "portra_400"), ("Teal & Orange", "teal_orange"), ("Cyberpunk", "cyberpunk"), ("Tri-X B&W", "tri_x")]
OUT = ROOT / "tests/reports/assets/looks_overview.jpg"
EDGE = 520          # px per panel, long edge: sharp at the README's column width


def load(path: str) -> torch.Tensor:
    im = Image.open(ROOT / path).convert("RGB")
    im.thumbnail((EDGE, EDGE), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


def main() -> None:
    fig, axes = plt.subplots(len(ROWS), 1 + len(LOOKS), figsize=(3.0 * (1 + len(LOOKS)), 2.3 * len(ROWS)), dpi=150)
    for r, (label, path) in enumerate(ROWS):
        x = load(path)
        panels = [("Unedited", x)]
        for title, name in LOOKS:
            with torch.no_grad():
                panels.append((title, P.render(x, SL.look_sliders(name)).clamp(0, 1)))
        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, pil_kwargs={"quality": 88})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
