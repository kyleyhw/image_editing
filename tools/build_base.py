"""Train the shared style base: one StyleHead + StyleEncoder over FiveK experts A-E (landscapes).

Phase 10 showed that a new style fitted as a code on a shared base beats a
separate model trained on 6.6x more pairs. The base is FiveK-derived
(research licence), so it stays local: checkpoints/ is not in git.

Usage:
    uv run python tools/build_base.py [--n 400] [--steps 4000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.phase10 import load, test_psnr, train_joint  # noqa: E402
from photostyle.condition import BASE_PATH  # noqa: E402
from photostyle.features import FEATURE_DIM, FeatureExtractor  # noqa: E402
from photostyle.render import GlobalRenderer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="a,b,c,d,e")
    ap.add_argument("--n", type=int, default=400, help="max training pairs per expert")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    styles = args.experts.split(",")
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = {x: load(x, fx) for x in styles}
    r = GlobalRenderer("per_channel")
    head, enc, info = train_joint(data, r, args.n, steps=args.steps, styles=styles)
    per = {x: test_psnr(lambda f, i=i: head(f, head.embed.weight[i][None].expand(len(f), -1)), r, data[x]["test"])
           for i, x in enumerate(styles)}
    info = {**info, "styles": styles, "n_per_style": {x: min(args.n, len(data[x]["pool"])) for x in styles},
            "test_psnr": per, "licence": "MIT-Adobe FiveK derived: research / personal use only"}
    Path(BASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "encoder": enc.state_dict(), "in_dim": FEATURE_DIM,
                "out_dim": r.num_params, "n_styles": len(styles), "renderer": r.kind, "info": info}, BASE_PATH)
    print(f"base -> {BASE_PATH}: {info} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
