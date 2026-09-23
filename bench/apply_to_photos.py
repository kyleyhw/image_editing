"""Train on the full paired pool and apply the model to arbitrary photos.

Used for the visual check on the owner's photos (amendment A3): the model
never sees these photos during training. Outputs a side-by-side grid
(input | model edit) and the predicted parameters per photo. Owner photos
and outputs stay under data/ or a scratch directory; nothing here is
committed.

Usage:
    uv run python -m bench.apply_to_photos --renderer per_channel \
        --photos data/owner/srgb --only-unedited data/owner/manifest.csv --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from bench.learning_curve import Pairs, seed_all, shrink, to_tensor, train
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.render import GlobalRenderer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fivek_landscape_c"))
    ap.add_argument("--expert", default="c")
    ap.add_argument("--renderer", default="per_channel")
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--only-unedited", type=Path, default=None,
                    help="Owner manifest; if given, skip photos labelled edited=yes.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edge", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = Pairs(args.data, args.expert, fx)
    pool = data.split("train") + data.split("validation")
    rng = random.Random(1234)
    rng.shuffle(pool)
    val, train_items = pool[:30], pool[30:]
    seed_all(args.seed)
    r = GlobalRenderer(args.renderer)
    model = Head(FEATURE_DIM, r.num_params)
    model.set_norm(torch.stack([it["feat"] for it in pool]))
    info = train(model, r, train_items, val)
    ckpt = Path("checkpoints") / f"phase7_head_{args.renderer}_fivek_landscape_{args.expert}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "renderer": args.renderer,
                "knots": r.K, "train": info, "n_train": len(train_items)}, ckpt)
    print(f"trained on {len(train_items)} pairs: {info} -> {ckpt}")

    skip = set()
    if args.only_unedited:
        with args.only_unedited.open() as f:
            skip = {row["id"] for row in csv.DictReader(f) if row["edited"] == "yes"}
    photos = sorted(p for p in args.photos.glob("*.jpg") if p.stem not in skip)

    tiles, params = [], {}
    for p in photos:
        src = shrink(to_tensor(Image.open(p)), args.edge)
        with torch.no_grad():
            theta = model(fx(src)[None])
            out = r(src[None], theta)[0]
        params[p.stem] = theta[0].tolist()
        a = Image.fromarray((src.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
        b = Image.fromarray((out.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
        b.save(args.out / f"{p.stem}_edit.jpg", quality=92)
        tiles.append((p.stem, a, b))

    tw = 420
    rows = []
    for name, a, b in tiles:
        a.thumbnail((tw, tw))
        b.thumbnail((tw, tw))
        row = Image.new("RGB", (2 * tw + 30, max(a.height, b.height) + 22), "white")
        row.paste(a, (10, 18))
        row.paste(b, (20 + tw, 18))
        ImageDraw.Draw(row).text((10, 3), f"{name}: input | model (FiveK expert-{args.expert} landscapes, "
                                          f"{args.renderer})", fill="black")
        rows.append(row)
    grid = Image.new("RGB", (max(r_.width for r_ in rows), sum(r_.height for r_ in rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(args.out / "grid.jpg", quality=88)
    (args.out / "params.json").write_text(json.dumps(params, indent=1))
    print(f"{len(tiles)} photos -> {args.out / 'grid.jpg'}")


if __name__ == "__main__":
    main()
