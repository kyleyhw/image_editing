"""Build the built-in style packs from trained checkpoints (Phases 12-13).

    uv run python tools/build_stylepacks.py [--only clean_cool,fivek_c_landscape]

clean_cool           Pilot A v2 head (unpaired Clean Cool look, landscapes), default strength 0.7.
                     Needs checkpoints/pilot_a_clean_cool_head_v2.pt from
                     bench/pilot_a.py --keep-chroma 1.0 --tag _v2.
fivek_c_landscape    Paired head trained here on the FiveK expert-C landscape pool.
                     Research-licence data (MIT-Adobe FiveK); see the style card.

OOD statistics come from the features of the images each head was trained on.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.learning_curve import Pairs, small  # noqa: E402
from photostyle.engine import save_stylepack  # noqa: E402
from photostyle.features import FEATURE_DIM, FeatureExtractor  # noqa: E402
from photostyle.head import Head  # noqa: E402
from photostyle.render import GlobalRenderer  # noqa: E402
from photostyle.train import calibrate_shrinkage, fit, paired_loss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="clean_cool,fivek_c_landscape")
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--out", type=Path, default=Path("stylepacks"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    want = set(args.only.split(","))
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, "c", fx)
    pool = data.split("train") + data.split("validation")
    r = GlobalRenderer("per_channel")

    if "clean_cool" in want:
        ck = torch.load("checkpoints/pilot_a_clean_cool_head_v2.pt", map_location="cpu", weights_only=False)
        head = Head(FEATURE_DIM, r.num_params)
        head.load_state_dict(ck["state_dict"])
        feats = torch.stack([it["feat"] for it in random.Random(7).sample(pool, min(800, len(pool)))])
        save_stylepack(args.out / "clean_cool", "clean_cool", head, r, feats, {
            "source": "unpaired",
            "description": "Deep blacks, cool shadows, neutral highlights; clean neon at night, airy by day. "
                           "Landscapes (urban and natural).",
            "inspiration": "grade statistics from a public style study (research_notes/styles/bleg.md); "
                           "no images of that photographer were used",
            "training_data": "FiveK landscape originals (inputs only) + 31 openly licensed seed images "
                             "(data/manifests/clean_cool_landscape.csv)",
            "report": "tests/reports/pilotA_clean_cool_landscapes.md",
            "note": "vignette is disabled for this look",
            "version": "v2 (chroma-keep), chosen by the owner at 70 % strength (PROJECT_PLAN A4.5)",
            "default_strength": 0.7,
            "order": 3,
        })
        print("clean_cool -> stylepacks/clean_cool")

    if "fivek_c_landscape" in want:
        rng = random.Random(1234)
        items = list(pool)
        rng.shuffle(items)
        val, tr = items[:30], items[30:]
        for it in items:
            it["src_t"], it["tgt_t"] = small(it, "src"), small(it, "tgt")
        head = Head(FEATURE_DIM, r.num_params)
        head.set_norm(torch.stack([it["feat"] for it in items]))
        torch.manual_seed(0)
        info = fit(head, r, tr, val, paired_loss)
        info["shrinkage_alpha"] = calibrate_shrinkage(head, r, tr, val)
        save_stylepack(args.out / "fivek_c_landscape", "fivek_c_landscape", head, r,
                       torch.stack([it["feat"] for it in tr]), {
            "source": "paired",
            "description": "Expert C's retouching of FiveK landscapes (content-adaptive tone and colour).",
            "training_data": f"{len(tr)} MIT-Adobe FiveK landscape pairs, expert C",
            "licence": "MIT-Adobe FiveK: research use (Adobe / AdobeMIT per image); "
                       "do not redistribute this pack commercially",
            "report": "tests/reports/phase7_learning_curve.md",
            "order": 4,
            "train": info,
        })
        print(f"fivek_c_landscape -> stylepacks/fivek_c_landscape ({info})")


if __name__ == "__main__":
    main()
