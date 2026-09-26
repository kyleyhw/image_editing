"""Build the twenty-look library (photostyle/recipe_library.py) end to end.

For each look:
1. Search Openverse for the subject its tutorial was written for (openly licensed only).
2. Train a content-adaptive style on photos graded by the tutorial recipe
   (--recipe teacher, open-only: those photos first, then a general sample).
3. Pack it with the tutorial citations.
4. Pick the example: the candidate that best matches the subject (CLIP) and is a
   real photo. Render its before/after with the trained style for Studio's
   "example" panel, with the photo's attribution and the tutorial links.

    uv run python tools/build_library.py [--only NAME ...] [--skip-search]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle import newstyle as ns  # noqa: E402
from photostyle.edited import p_edited  # noqa: E402
from photostyle.recipe_library import LOOKS, NOT_MODELLED  # noqa: E402
from photostyle.recipes import SOURCES  # noqa: E402

EXAMPLES = Path("studio/web/public/examples")
FIRST_ORDER = 3                      # fujifilm = 1, cyberpunk = 2
GOOD_LICENCES = {"cc0": 0, "pdm": 0, "by": 1, "by-sa": 2}


def tutorials(rows):
    keys = list(dict.fromkeys(k for row in rows for k in row[3]))
    return [{"title": SOURCES[k][0], "url": SOURCES[k][1]} for k in keys]


def local_candidates(look: dict, out: Path, n: int = 60) -> list[dict]:
    """Fallback when Openverse refuses (daily limit): the n openly licensed images already downloaded
    (data/cache/openverse) that best match the look's subject and queries, by CLIP similarity."""
    import glob
    import shutil

    import torch

    from photostyle import openverse as ov
    from photostyle.photo_filter import embed_texts, photo_score
    from photostyle.stats import colour_stats

    meta = {}
    for f in glob.glob(str(ns.CACHE / "search" / "*.json")):
        for it in json.loads(Path(f).read_text()):
            meta[it["id"]] = it
    d = torch.load("data/cache/clip_cache_embeddings.pt")
    texts = [f"a photo of {look['subject']}"] + [f"a photo of {q}" for q in look["queries"]]
    sim = (d["emb"] @ embed_texts(texts).T).mean(1)
    out.mkdir(parents=True, exist_ok=True)
    cands = []
    for k in sim.argsort(descending=True).tolist():
        f = Path(d["files"][k])
        item = meta.get(f.stem)
        img = Image.open(f).convert("RGB")
        if item is None or min(img.size) < 320:
            continue
        mono = colour_stats(img)["chroma_mean"] < 3
        if mono and look["title"] != "Tri-X B&W":
            continue
        dst = out / f.name
        shutil.copy(f, dst)
        cands.append({"id": item["id"], "file": str(dst), "query": "local cache (CLIP subject match)",
                      "stats": colour_stats(img), "photo_score": photo_score(img),
                      "license": item.get("license"), "license_version": item.get("license_version"),
                      "creator": item.get("creator"), "title": item.get("title"), "source": item.get("source"),
                      "landing_url": item.get("foreign_landing_url"), "attribution": ov.attribution(item)})
        if len(cands) >= n:
            break
    return cands


def colourful(file: str, min_chroma: float = 0.04) -> bool:
    a = np.asarray(Image.open(file).convert("RGB").resize((128, 128)), dtype=np.float32) / 255
    return float((a.max(-1) - a.min(-1)).mean()) >= min_chroma


def pick_example(p: ns.Project, subject: str) -> int:
    idx = [i for i, c in enumerate(p.candidates) if not ns.flagged(p, i) and min(Image.open(c["file"]).size) >= 480
           and colourful(c["file"])]            # a monochrome "before" cannot show a look (even a B&W one)
    from photostyle.photo_filter import embed_images, embed_texts

    files = [p.candidates[i]["file"] for i in idx]
    E = embed_images(files)
    T = embed_texts([f"a photo of {subject}", "a photo with a watermark, a signature, a logo or text on it",
                     "a heavily edited, over-processed photo with a filter"])
    sim = (E @ T.T) * 100                  # CLIP logit scale
    pe = p_edited(files)                   # an example should start from an unedited-looking photo

    def key(k):
        c = p.candidates[idx[k]]
        return (float(sim[k, 0]) - 0.6 * float(sim[k, 1]) - 0.4 * float(sim[k, 2])
                + 2.0 * (c.get("photo_score") or 0) - 6.0 * pe[k] - 0.5 * GOOD_LICENCES.get(c["license"], 3))
    return idx[max(range(len(idx)), key=key)]


def build(name: str, look: dict, order: int, skip_search: bool, examples_only: bool = False) -> None:

    proj = ns.ROOT / name / "project.json"
    if not proj.exists():
        ns.new(name, look["subject"], queries=look["queries"])
    p = ns.Project.load(name)
    if not p.candidates and not skip_search:
        portrait = look["category"] == "Portrait"
        p.candidates = ns._collect(look["queries"], 2, None if name == "tri_x" else False, p.dir / "candidates",
                                   max_person=1.0 if portrait else 0.04)
        if len(p.candidates) < 20:           # Openverse limit reached: use the local cache of open images
            print(f"[{name}] Openverse gave {len(p.candidates)}; using the local cache", flush=True)
            p.candidates = local_candidates(look, p.dir / "candidates")
        p.save()
    print(f"[{name}] {len(p.candidates)} candidates", flush=True)
    if not p.candidates:
        raise SystemExit(f"{name}: no candidates (Openverse limit?)")
    card_f = Path("stylepacks") / name / "style.json"
    if not (examples_only and card_f.exists()):
        p.refs = p.picks = []
        p.save()
        ns.train(name, "teacher", open_only=True)
        ns.pack(name, strength=1.0, order=order)
    card = json.loads(card_f.read_text())
    card.update(title=look["title"], category=look["category"], subject=look["subject"], grade=look["grade"],
                description=f"{look['title']}: for {look['subject']}. From tutorials (see sources).",
                tutorials=tutorials(look["rows"]), not_modelled=NOT_MODELLED,
                references="training photos found by subject search on Openverse (openly licensed); "
                           "attribution in ATTRIBUTION.csv")
    card_f.write_text(json.dumps(card, indent=1))
    write_example(name, p, look["subject"], look["title"])


def write_example(name: str, p: ns.Project, subject: str, title: str) -> None:
    """Pick the candidate that best shows the look's ideal subject; write its before/after and credit."""
    from photostyle.engine import Engine

    card_f = Path("stylepacks") / name / "style.json"
    card = json.loads(card_f.read_text())
    i = pick_example(p, subject)
    c = p.candidates[i]
    out = EXAMPLES / name
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(c["file"]).convert("RGB")
    img.thumbnail((1200, 1200))
    img.save(out / "before.jpg", quality=88)
    eng = Engine(roots=[Path("stylepacks")])
    eng.render(img, eng.predict(img, name)).save(out / "after.jpg", quality=88)
    example = {"before": f"examples/{name}/before.jpg", "after": f"examples/{name}/after.jpg",
               "photo": {"title": c.get("title"), "creator": c.get("creator"), "license": c.get("license"),
                         "license_version": c.get("license_version"), "url": c.get("landing_url"),
                         "attribution": c.get("attribution"),
                         "p_edited": round(float(p_edited([c["file"]])[0]), 2)},
               "why": f"chosen as the candidate that best matches the tutorial's subject: {subject}"}
    card["example"] = example
    card_f.write_text(json.dumps(card, indent=1))
    (out / "example.json").write_text(json.dumps({**example, "tutorials": card.get("tutorials"), "title": title}, indent=1))
    print(f"[{name}] packed; example: {c.get('title')!r} by {c.get('creator')} ({c.get('license')})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--examples-only", action="store_true", help="re-pick and re-render examples of built looks")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="also re-pick examples of tutorial looks outside the library (e.g. cyberpunk)")
    args = ap.parse_args()
    for name in args.extra:
        card = json.loads((Path("stylepacks") / name / "style.json").read_text())
        write_example(name, ns.Project.load(name), card["subject"], card.get("title") or name)
    for k, (name, look) in enumerate(LOOKS.items()):
        if args.only and name not in args.only:
            continue
        build(name, look, FIRST_ORDER + k, args.skip_search or args.examples_only, args.examples_only)
    print("LIBRARY DONE", flush=True)


if __name__ == "__main__":
    main()
