"""CLIP-embed every openly licensed image already downloaded from Openverse (data/cache/openverse), so
looks can find subject photos locally when Openverse's daily limit is reached. Resumable."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.photo_filter import embed_images  # noqa: E402

OUT = Path("data/cache/clip_cache_embeddings.pt")


def main() -> None:
    torch.set_num_threads(4)
    meta = set()
    for f in glob.glob("data/cache/openverse/search/*.json"):
        meta.update(it["id"] for it in json.loads(Path(f).read_text()))
    files = [p for p in sorted(glob.glob("data/cache/openverse/images/*.jpg")) if Path(p).stem in meta]
    done = torch.load(OUT) if OUT.exists() else {"files": [], "emb": torch.zeros(0, 512)}
    todo = [f for f in files if f not in set(done["files"])]
    print(f"{len(files)} images, {len(todo)} to embed", flush=True)
    for i in range(0, len(todo), 256):
        chunk = todo[i:i + 256]
        e = embed_images(chunk, batch=16)
        done = {"files": done["files"] + chunk, "emb": torch.cat([done["emb"], e])}
        torch.save(done, OUT)
        print(f"  {len(done['files'])}/{len(files)}", flush=True)
    print("saved", tuple(done["emb"].shape), flush=True)


if __name__ == "__main__":
    main()
