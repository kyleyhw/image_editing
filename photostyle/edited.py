"""Does a photo look already edited? A tiny classifier used to pick honest example photos.

Logistic regression on colour statistics, fitted on MIT-Adobe FiveK AdobeMIT pairs (the unedited
camera render vs expert C's edit of the same photo; about 85 % cross-validated accuracy).
Rough: a normal camera JPEG already leans "edited" compared with a flat raw render. Only the
fitted coefficients are stored (photostyle/models/edited_clf.json).

    uv run --with scikit-learn python -m photostyle.edited fit      # refit (needs data/fivek_landscape_c)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from photostyle.stats import colour_stats

KEYS = ["L_p01", "L_p50", "L_p99", "shadow_a", "shadow_b", "mid_a", "mid_b", "high_a", "high_b", "sat_mean", "chroma_mean"]
MODEL = Path(__file__).parent / "models" / "edited_clf.json"


def features(img) -> list[float]:
    im = (img if isinstance(img, Image.Image) else Image.open(img)).convert("RGB")
    im.thumbnail((384, 384))
    st = colour_stats(im)
    a = np.asarray(im, np.float32) / 255
    return [st[k] for k in KEYS] + [float((a.max(2) - a.min(2)).std()), float((a > 0.98).mean()), float((a < 0.02).mean())]


def p_edited(imgs) -> list[float]:
    m = json.loads(MODEL.read_text())
    X = (np.array([features(i) for i in imgs]) - m["mu"]) / m["sd"]
    z = X @ np.array(m["coef"]) + m["intercept"]
    return list(1 / (1 + np.exp(-z)))


def fit(root: Path = Path("data/fivek_landscape_c"), n: int = 400) -> None:
    import csv
    import random

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rows = [r for r in csv.DictReader(open(root / "meta.csv")) if r["license"] == "AdobeMIT"]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    X = np.array([features(root / "original" / r["file"]) for r in rows] + [features(root / "expert_c" / r["file"]) for r in rows])
    y = np.r_[np.zeros(len(rows)), np.ones(len(rows))]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=0.5)
    acc = cross_val_score(clf, (X - mu) / sd, y, cv=5).mean()
    clf.fit((X - mu) / sd, y)
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_text(json.dumps({"mu": mu.tolist(), "sd": sd.tolist(), "coef": clf.coef_[0].tolist(),
                                 "intercept": float(clf.intercept_[0]), "cv_accuracy": float(acc), "n_pairs": len(rows),
                                 "data": "MIT-Adobe FiveK AdobeMIT subset, original vs expert C"}, indent=1))
    print(f"cv accuracy {acc:.3f} -> {MODEL}")


if __name__ == "__main__" and sys.argv[1:] == ["fit"]:
    fit()
