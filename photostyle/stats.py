"""Summary colour statistics (NumPy, non-differentiable) shared by the tools.

Used for data selection (tools/collect_style_set.py), owner-photo intake
(tools/ingest_owner_photos.py) and pilot evaluation. The differentiable
counterparts used in training live in photostyle/looks.py.
"""

from __future__ import annotations

import numpy as np
import skimage as ski
from PIL import Image


def colour_stats(img: Image.Image) -> dict[str, float]:
    small = img.copy()
    small.thumbnail((512, 512))
    rgb = np.asarray(small, dtype=np.float64) / 255.0
    lab = ski.color.rgb2lab(rgb)
    hsv = ski.color.rgb2hsv(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    def masked_mean(x: np.ndarray, m: np.ndarray) -> float:
        return float(x[m].mean()) if m.any() else 0.0

    shadows, mids, highs = L < 25, (L >= 25) & (L < 70), L >= 70
    return {
        "L_p01": float(np.percentile(L, 1)),
        "L_p50": float(np.percentile(L, 50)),
        "L_p99": float(np.percentile(L, 99)),
        "shadow_frac": float(shadows.mean()),
        "high_frac": float(highs.mean()),
        "shadow_a": masked_mean(a, shadows),
        "shadow_b": masked_mean(b, shadows),
        "mid_a": masked_mean(a, mids),
        "mid_b": masked_mean(b, mids),
        "high_a": masked_mean(a, highs),
        "high_b": masked_mean(b, highs),
        "sat_mean": float(hsv[..., 1].mean()),
        "chroma_mean": float(np.hypot(a, b).mean()),
    }


def regime_of(stats: dict[str, float]) -> str:
    """Night = dark median AND little bright area.

    The median alone misclassifies dark-foliage daytime scenes (a red maple
    against a bright sky has median L* ~33 but ~30 % of pixels above L* 70).
    """
    return "night" if stats["L_p50"] < 35 and stats["high_frac"] < 0.12 else "day"


