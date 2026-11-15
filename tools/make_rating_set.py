"""Phase 20 task 14: the evaluation set and the images of the blind A/B rating page.

Sixty FiveK test photos are drawn, stratified over seven groups that approximate the editor's look
categories from the FiveK labels (subject, time, location). For each, the unedited photo and the
Auto-correct result (photostyle/models/auto_c.pt) are written as JPEGs under random pair ids, with
the side (A or B) of the Auto-correct version drawn at random. The key (which side is which) goes
to a separate file that the rating page never sees, so the owner rates blind.

The images are FiveK photos (research licence): they are written outside the repository (--out)
and must not be committed. The key is written to data/rating/key.json.

    uv run python tools/make_rating_set.py --out <scratch dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import auto as A  # noqa: E402
from photostyle import pipeline as P  # noqa: E402

SEED = 1415                                   # fixed: the same evaluation set on every run
N = 60                                        # A6.6: "about 60 photos"
GROUPS = {                                    # group -> predicate on a manifest row
    "Day landscape": lambda r: r["subject"] == "nature" and r["time"] == "day",
    "Dusk and night": lambda r: r["time"] in ("dusk", "night"),
    "Day city": lambda r: r["subject"] == "man_made" and r["location"] == "outdoor" and r["time"] == "day",
    "Interiors and objects": lambda r: r["subject"] in ("man_made", "abstract") and r["location"] == "indoor",
    "Portrait, outdoors": lambda r: r["subject"] == "people" and r["location"] == "outdoor",
    "Portrait, indoors": lambda r: r["subject"] == "people" and r["location"] == "indoor",
    "Animals": lambda r: r["subject"] == "animals",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows = [r for r in csv.DictReader((ROOT / "data/manifests/fivek.csv").open()) if r["split"] == "test"]
    per = N // len(GROUPS)                     # 8 per group; the remainder is drawn from the largest groups
    chosen: list[tuple[str, dict]] = []
    used: set[str] = set()
    for g, pred in GROUPS.items():
        cand = [r for r in rows if pred(r) and r["idx"] not in used]
        for i in rng.permutation(len(cand))[:per]:
            chosen.append((g, cand[i]))
            used.add(cand[i]["idx"])
    for g in ("Day city", "Day landscape", "Portrait, outdoors", "Interiors and objects"):
        if len(chosen) >= N:
            break
        cand = [r for r in rows if GROUPS[g](r) and r["idx"] not in used]
        chosen.append((g, cand[int(rng.integers(len(cand)))]))
        used.add(chosen[-1][1]["idx"])

    test_files = sorted((ROOT / "data/fivek/test/original").glob("*.png"))
    cache = torch.load(sorted((ROOT / "data/cache").glob("desc_test_1000_*.pt"))[0])
    feats = {Path(f).stem: cache["feats"][i] for i, f in enumerate(cache["files"])}
    model = A.SliderModel.load(ROOT / "photostyle/models/auto_c.pt")
    (args.out / "img").mkdir(parents=True, exist_ok=True)
    pairs, key = [], {}
    ids = rng.permutation(np.arange(100, 1000))[: len(chosen)]   # 3-digit pair ids, in random order
    for (g, r), pid in zip(chosen, ids):
        stem = f"img_{int(r['idx']):04d}"
        assert (ROOT / "data/fivek/test/original" / f"{stem}.png") in test_files
        x = torch.from_numpy(np.asarray(Image.open(ROOT / "data/fivek/test/original" / f"{stem}.png").convert("RGB")).copy()).permute(2, 0, 1).float() / 255
        s = model.params(feats[stem][None])[0]
        with torch.no_grad():
            y = P.render(x, s).clamp(0, 1)
        auto_side = "a" if rng.random() < 0.5 else "b"
        imgs = {auto_side: y, ("b" if auto_side == "a" else "a"): x}
        for side, t in imgs.items():
            Image.fromarray((t.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)).save(
                args.out / "img" / f"{pid}{side}.jpg", quality=90)
        pairs.append({"id": str(pid), "group": g})
        key[str(pid)] = {"stem": stem, "fivek_id": r["fivek_id"], "group": g, "auto": auto_side,
                         "sliders": {k: round(v, 2) for k, v in s.items()}}
    pairs.sort(key=lambda p: p["id"])
    (args.out / "pairs.json").write_text(json.dumps(pairs))
    (ROOT / "data/rating").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/rating/key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(pairs)} pairs -> {args.out}; key -> data/rating/key.json")


if __name__ == "__main__":
    main()
