"""New-style pipeline: from an idea to a style pack, in resumable steps.

    photostyle style new     NAME --describe "..." [--query ...] [--mono]
    photostyle style search  NAME [--pages 3]          openly licensed candidates + numbered sheets
    photostyle style pick    NAME 3 7 12 ...           your picks -> "more like these" -> reference set
    photostyle style exclude NAME 5 9 ...              drop references you do not want
    photostyle style restore NAME 12 ...               keep candidates the digital-art filter greyed out
    photostyle style flag    NAME                      run the digital-art filter on an older project
    photostyle style train   NAME [--recipe gentle|strong|instant|paired] [--pairs BEFORE AFTER]
    photostyle style preview NAME                      before/after on your photos and public samples
    photostyle style pack    NAME [--strength 0.8]     write the style pack (with licences/attribution)
    photostyle style status  NAME

State lives in ``data/styles/<name>/project.json`` (plus images, sheets and
checkpoints next to it; ``data/`` is not in git). The reference manifest
with licence and attribution is also written to
``data/manifests/<name>.csv`` so it can be committed.

Design notes (see PROJECT_PLAN A5):

* **References** come only from Openverse with reuse-and-adapt licences
  (CC0, PDM, CC BY, CC BY-SA), or from photos you own.
* **Picking.** You pick a few examples; the rest of the reference set is
  chosen as "more like these". Candidates are ranked by the distance of
  their CIELAB colour statistics to the picks, so selection follows the
  look, not the subject.
* **Training (Phase 11 finding).**
  - Unpaired: the default "gentle" recipe uses distort-and-recover
    pseudo-pairs only. The "strong" recipe adds per-image sliced-Wasserstein
    and fidelity terms for looks far from natural (e.g. cyberpunk). It needs
    input photos: an openly licensed generic-scene pool is used, so a pack
    does not depend on research-licence data.
  - Instant (Phase 10): no training. The shared base's encoder reads a
    style code from the references. It only covers looks within the span of
    the base's training styles.
  - Paired: with before/after pairs (``--pairs``). A code fitted on the
    shared base (Phase 10: 20 pairs beat a separate model on 132); without
    a base, a separate head with shrinkage (Phase 7b).
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from photostyle import openverse as ov
from photostyle.photo_filter import is_likely_art, photo_score
from photostyle.stats import colour_stats

ROOT = Path("data/styles")
MANIFESTS = Path("data/manifests")
CACHE = Path("data/cache/openverse")
INPUT_POOL = "_inputs"          # generic openly licensed scenes, input domain for "strong" training
INPUT_QUERIES = ["landscape", "city street", "night city", "forest", "beach", "mountains",
                 "countryside", "harbour", "street at dusk", "park"]
STAT_KEYS = ["L_p01", "L_p50", "L_p99", "shadow_a", "shadow_b", "mid_a", "mid_b", "high_a", "high_b",
             "sat_mean", "chroma_mean"]
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass
class Project:
    name: str
    description: str = ""
    queries: list[str] = field(default_factory=list)
    mono: bool = False                       # a black-and-white look
    candidates: list[dict] = field(default_factory=list)
    picks: list[int] = field(default_factory=list)
    refs: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)
    restored: list[int] = field(default_factory=list)   # flagged as digital art, but kept by the owner
    train: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    @property
    def dir(self) -> Path:
        return ROOT / self.name

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "project.json").write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(name: str) -> Project:
        p = ROOT / name / "project.json"
        if not p.exists():
            raise SystemExit(f"no style project {name!r}; start with: photostyle style new {name} --describe ...")
        return Project(**json.loads(p.read_text()))


# --------------------------------------------------------------------------- idea


def new(name: str, describe: str, queries: list[str] | None = None, mono: bool = False,
        overwrite: bool = False) -> Project:
    if not name.replace("_", "").isalnum():
        raise SystemExit("style names use letters, digits and underscores")
    if (ROOT / name / "project.json").exists() and not overwrite:
        raise SystemExit(f"style project {name!r} already exists (its candidates and picks would be lost); "
                         "choose another name or pass --overwrite")
    q = queries or [describe, f"{describe} landscape", f"{describe} city street"]
    p = Project(name=name, description=describe, queries=q, mono=mono)
    p.save()
    return p


# --------------------------------------------------------------------------- search


def _collect(queries: list[str], pages: int, mono: bool | None, out: Path, max_person: float = 0.04,
             min_side: int = 320, progress=None) -> list[dict]:
    """Search, download, clean and describe candidates. ``mono`` None keeps both colour and B&W."""
    s = ov.session()
    det = ov.PersonDetector()
    seen, cands = set(), []
    out.mkdir(parents=True, exist_ok=True)
    for q in queries:
        for page in range(1, pages + 1):
            try:
                items = ov.search(q, page, s, CACHE)
            except RuntimeError as e:          # daily API limit: keep what we have
                print(f"  {e}")
                break
            for item in items:
                img = ov.download(item, s, CACHE)
                if img is None or min(img.size) < min_side:
                    continue
                img = ov.crop_letterbox(img)
                if min(img.size) < min_side:
                    continue
                h = ov.dhash(img)
                if h in seen:
                    continue
                st = colour_stats(img)
                if mono is not None and (st["chroma_mean"] < 3) != mono:
                    continue
                people = det.people(img)
                if people and people[0][0] > max_person:
                    continue
                seen.add(h)
                fn = out / f"{item['id']}.jpg"
                img.save(fn, quality=92)
                cands.append({"id": item["id"], "file": str(fn), "query": q, "stats": st,
                              "photo_score": photo_score(img),
                              "license": item.get("license"), "license_version": item.get("license_version"),
                              "creator": item.get("creator"), "title": item.get("title"),
                              "source": item.get("source"), "landing_url": item.get("foreign_landing_url"),
                              "attribution": ov.attribution(item)})
        print(f"  {q!r}: {len(cands)} candidates so far", flush=True)
        if progress:
            progress(queries.index(q) + 1, len(queries), len(cands))
    return cands


def search(name: str, pages: int = 3, progress=None) -> Project:
    p = Project.load(name)
    p.candidates = _collect(p.queries, pages, p.mono, p.dir / "candidates", progress=progress)
    p.picks, p.refs, p.excluded = [], [], []
    p.save()
    sheets = contact_sheets(p, list(range(len(p.candidates))), "candidates")
    print(f"{len(p.candidates)} candidates -> {', '.join(map(str, sheets))}")
    print(f"next: photostyle style pick {name} <numbers of the photos whose look you like>")
    return p


def contact_sheets(p: Project, idx: list[int], stem: str, per_sheet: int = 48, cols: int = 8,
                   tw: int = 240, th: int = 180) -> list[Path]:
    """Numbered contact sheets (numbers are 1-based candidate indices)."""
    try:
        font = ImageFont.truetype(_FONT, 20)
    except OSError:
        font = ImageFont.load_default()
    paths = []
    for si in range(0, max(1, len(idx)), per_sheet):
        chunk = idx[si:si + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (tw + 6) + 6, max(1, rows) * (th + 6) + 6), "white")
        d = ImageDraw.Draw(sheet)
        for k, i in enumerate(chunk):
            im = Image.open(p.candidates[i]["file"])
            im.thumbnail((tw, th))
            x, y = 6 + (k % cols) * (tw + 6), 6 + (k // cols) * (th + 6)
            sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
            lab = str(i + 1)
            d.rectangle([x, y, x + 12 + 12 * len(lab), y + 26], fill="black")
            d.text((x + 5, y + 2), lab, fill="white", font=font)
            if flagged(p, i):                     # likely digital art: greyed badge, not removed
                d.rectangle([x + tw - 58, y, x + tw, y + 24], fill="#b3261e")
                d.text((x + tw - 54, y + 2), "ART?", fill="white", font=font)
        path = p.dir / f"{stem}_{si // per_sheet + 1}.jpg"
        sheet.save(path, quality=85)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- pick / exclude


def _stat_matrix(cands: list[dict]) -> np.ndarray:
    return np.array([[c["stats"][k] for k in STAT_KEYS] for c in cands], dtype=np.float64)


def rank_like(cands: list[dict], picks: list[int]) -> list[int]:
    """Candidate indices ordered by colour-statistics distance to the picks (closest first)."""
    X = _stat_matrix(cands)
    z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    P = z[picks]
    # distance to the nearest pick (a look can have a few "modes", e.g. night and day)
    d = np.sqrt(((z[:, None, :] - P[None]) ** 2).mean(-1)).min(1)
    return [int(i) for i in np.argsort(d)]


def flagged(p: Project, i: int) -> bool:
    """Candidate i looks like digital art and the owner has not restored it."""
    return is_likely_art(p.candidates[i].get("photo_score")) and i not in p.restored


def pick(name: str, numbers: list[int], n_refs: int = 40) -> Project:
    p = Project.load(name)
    if not p.candidates:
        raise SystemExit("run search first")
    picks = [i - 1 for i in numbers if 1 <= i <= len(p.candidates)]
    if not picks:
        raise SystemExit("pick at least one candidate number")
    p.picks = picks
    p.restored = sorted(set(p.restored) | {i for i in picks if flagged(p, i)})   # picking one restores it
    ex = set(p.excluded) | {i for i in range(len(p.candidates)) if flagged(p, i)}
    order = [i for i in rank_like(p.candidates, picks) if i not in ex]
    refs = list(dict.fromkeys(picks + order))[:max(n_refs, len(picks))]
    p.refs = refs
    p.save()
    _write_manifest(p)
    sheets = contact_sheets(p, refs, "references")
    print(f"{len(refs)} references (your {len(picks)} picks + the closest matches) -> "
          f"{', '.join(map(str, sheets))}")
    print(f"next: photostyle style exclude {name} <numbers to drop>, or photostyle style train {name}")
    return p


def exclude(name: str, numbers: list[int]) -> Project:
    p = Project.load(name)
    drop = {i - 1 for i in numbers}
    p.excluded = sorted(set(p.excluded) | drop)
    if p.picks:
        n = len(p.refs)
        p.picks = [i for i in p.picks if i not in drop]
        p.save()
        return pick(name, [i + 1 for i in p.picks], n_refs=n) if p.picks else p
    p.save()
    return p


def restore(name: str, numbers: list[int]) -> Project:
    """Keep candidates the digital-art filter flagged (they become eligible again)."""
    p = Project.load(name)
    p.restored = sorted(set(p.restored) | {i - 1 for i in numbers})
    p.save()
    return pick(name, [i + 1 for i in p.picks], n_refs=len(p.refs) or 40) if p.picks else p


def flag(name: str) -> Project:
    """Score existing candidates with the photo-vs-art filter (projects searched before it existed)."""
    p = Project.load(name)
    for c in p.candidates:
        if c.get("photo_score") is None:
            c["photo_score"] = photo_score(c["file"])
    p.save()
    n = sum(flagged(p, i) for i in range(len(p.candidates)))
    print(f"{n} of {len(p.candidates)} candidates look like digital art (greyed out; `style restore` keeps any)")
    return p


def _write_manifest(p: Project) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / f"{p.name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "picked", "openverse_id", "license", "license_version", "creator", "title",
                    "source", "landing_url", "attribution"])
        for i in p.refs:
            c = p.candidates[i]
            w.writerow([i + 1, i in p.picks, c["id"], c["license"], c["license_version"], c["creator"],
                        c["title"], c["source"], c["landing_url"], c["attribution"]])
    return path


# --------------------------------------------------------------------------- train


def _tensor(path: str | Path, edge: int = 512) -> torch.Tensor:
    from photostyle.train import shrink

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return shrink(torch.from_numpy(arr).permute(2, 0, 1), edge)


def input_pool(exclude: str | None = None, pages: int = 2, min_size: int = 100,
               include_owner: bool = True) -> list[Path]:
    """Typical photos to be edited: the input domain for "strong" training.

    Reuses photos already downloaded by other style projects (all openly licensed; mostly
    ordinary scenes) plus the owner's unedited photos (local only), and tops up from
    Openverse's generic-scene queries only if fewer than ``min_size`` are available.
    ``include_owner=False`` keeps the owner's photos out (for packs that will be published)."""
    files: list[Path] = []
    for proj in sorted(ROOT.glob("*/project.json")):
        if proj.parent.name in (exclude, INPUT_POOL):
            continue
        for c in json.loads(proj.read_text()).get("candidates", []):
            if Path(c["file"]).exists():
                files.append(Path(c["file"]))
    man = Path("data/owner/manifest.csv")
    if include_owner and man.exists():
        with man.open() as f:
            files += [Path("data/owner/srgb") / f"{r['id']}.jpg" for r in csv.DictReader(f) if r.get("edited") == "no"]
    d = ROOT / INPUT_POOL
    listing = d / "pool.json"
    if listing.exists():
        files += [Path(f) for f in json.loads(listing.read_text())]
    elif len(files) < min_size:
        cands = _collect(INPUT_QUERIES, pages, False, d / "images")
        if cands:                                   # never cache an empty pool (e.g. API limit reached)
            d.mkdir(parents=True, exist_ok=True)
            listing.write_text(json.dumps([c["file"] for c in cands]))
            with (MANIFESTS / "style_input_pool.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["openverse_id", "license", "creator", "source", "landing_url", "attribution"])
                for c in cands:
                    w.writerow([c["id"], c["license"], c["creator"], c["source"], c["landing_url"], c["attribution"]])
            files += [Path(c["file"]) for c in cands]
    files = list(dict.fromkeys(files))
    if not files:
        raise SystemExit("no input photos available for the 'strong' recipe (Openverse limit reached?); "
                         "use --recipe gentle, or try again tomorrow")
    return files


def train(name: str, recipe: str = "gentle", pairs: tuple[Path, Path] | None = None, steps: int = 1200,
          seed: int = 0, progress=None, open_only: bool = False) -> Project:
    """Recipes:
    gentle   unpaired, pseudo-pairs only (Phase 11 default; looks near natural)
    strong   unpaired, pseudo-pairs + per-image SWD + fidelity (looks far from natural)
    instant  no training: the shared base's encoder reads a style code from the references
             (Phase 10; needs checkpoints/base_stylehead.pt)
    teacher  a tutorial recipe (photostyle/recipes.py) applied to input photos gives the pairs;
             for looks defined by how they are made (e.g. cyberpunk split toning)
    paired   ``pairs=(before_dir, after_dir)``: a code fitted on the shared base if present
             (Phase 10: 20 pairs suffice), else a separate head with shrinkage (Phase 7b)

    ``open_only``: train only on openly licensed photos (no owner photos in the "strong" input
    pool), so the pack can be published (e.g. on the hosted Studio). Paired/instant styles on the
    FiveK-derived base are never publishable.
    """
    from photostyle.condition import CodedHead, fit_code, load_base
    from photostyle.features import FeatureExtractor
    from photostyle.render import GlobalRenderer
    from photostyle.train import TRAIN_EDGE, align_pair, learn_paired, learn_unpaired, shrink

    p = Project.load(name)
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    t0 = time.time()
    base = load_base()
    kind, extra = "head", {}
    if pairs or recipe == "paired":
        if not pairs:
            raise SystemExit("--recipe paired needs --pairs BEFORE_DIR AFTER_DIR")
        before, after = pairs
        names = sorted(f.name for f in before.iterdir() if (after / f.name).exists())
        pr = [(_tensor(before / n), _tensor(after / n)) for n in names]
        if base is not None and len(pr) >= 5:
            sh, _, binfo = base
            r = GlobalRenderer("per_channel")
            items = []
            for a, b in pr:
                al = align_pair(a, b)
                if al is not None:
                    items.append({"feat": fx(al[0]), "src_t": shrink(al[0], TRAIN_EDGE), "tgt_t": shrink(al[1], TRAIN_EDGE)})
            code = fit_code(sh, r, items, seed=seed)
            head, feats, kind = CodedHead(sh, code), torch.stack([it["feat"] for it in items]), "coded"
            info, mode = {"n_pairs": len(items)}, "paired/base-code"
            extra = {"base_n_styles": sh.embed.weight.shape[0], "base": binfo.get("licence", "")}
        else:
            head, r, feats, info = learn_paired(pr, fx, seed=seed, progress=progress)
            mode = "paired"
    elif recipe == "instant":
        if base is None:
            raise SystemExit("recipe 'instant' needs the shared base: uv run python tools/build_base.py")
        sh, enc, binfo = base
        tf = torch.stack([fx(_tensor(p.candidates[i]["file"])) for i in p.refs])
        with torch.no_grad():
            code = enc(tf)
        r = GlobalRenderer("per_channel")
        head, feats, kind, info, mode = CodedHead(sh, code), tf, "coded", {"n_refs": len(p.refs)}, "instant/base-encoder"
        extra = {"base_n_styles": sh.embed.weight.shape[0], "base": binfo.get("licence", "")}
    elif recipe == "teacher":
        from photostyle.recipes import RECIPES

        if name not in RECIPES:
            raise SystemExit(f"no tutorial recipe for {name!r}; available: {sorted(RECIPES)}")
        # the look's own candidates (photos of the subject its tutorial was written for) come first,
        # then a general sample so the style still behaves on other photos
        own = [c["file"] for i, c in enumerate(p.candidates) if not flagged(p, i)]
        pool = [f for f in input_pool(exclude=name, include_owner=not open_only) if str(f) not in set(own)]
        files = own[:150] + random.Random(seed).sample(pool, min(300 - min(150, len(own)), len(pool)))
        ins = [_tensor(f) for f in files]
        # hue_weight: keep colourful pixels' hue (the global renderer otherwise trades hue for brightness
        # on strong highlight/shadow moves, e.g. orange skies turning green; tools/check_library.py)
        head, r, feats, info = learn_paired([(x, RECIPES[name](x)) for x in ins], fx, seed=seed, progress=progress,
                                            hue_weight=1.0)
        info = {**info, "n_own": min(150, len(own))}
        mode = "teacher/recipe"
    else:
        if not p.refs:
            raise SystemExit("no references yet: run search and pick first")
        ex = [_tensor(p.candidates[i]["file"]) for i in p.refs]
        ins = []
        if recipe == "strong":
            pool = input_pool(exclude=name, include_owner=not open_only)
            ins = [_tensor(f) for f in random.Random(seed).sample(pool, min(300, len(pool)))]
        kw = dict(use_pseudo=True, use_swd=recipe == "strong", use_fidelity=recipe == "strong")
        head, r, feats, info = learn_unpaired(ex, ins, fx, steps=steps, seed=seed, progress=progress, **kw)
        mode = f"unpaired/{recipe}"
    ck = p.dir / "head.pt"
    torch.save({"kind": kind, "state_dict": head.state_dict(), "renderer": r.kind, "feats": feats, **extra}, ck)
    p.train = {"mode": mode, "recipe": recipe, "n_refs": len(p.refs), "steps": steps, "seed": seed,
               "seconds": round(time.time() - t0), "open_only": open_only, **{k: v for k, v in info.items() if isinstance(v, int | float)}}
    p.save()
    print(f"trained {mode} in {p.train['seconds']} s -> {ck}")
    print(f"next: photostyle style preview {name}")
    return p


def _load_head(p: Project):
    from photostyle.features import FEATURE_DIM
    from photostyle.head import Head
    from photostyle.render import GlobalRenderer

    ck = torch.load(p.dir / "head.pt", map_location="cpu", weights_only=False)
    r = GlobalRenderer(ck["renderer"])
    if ck.get("kind") == "coded":
        from photostyle.condition import CodedHead, StyleHead

        sh = StyleHead(FEATURE_DIM, r.num_params, n_styles=ck["base_n_styles"])
        head = CodedHead(sh, torch.zeros(sh.embed.weight.shape[1]))
    else:
        head = Head(FEATURE_DIM, r.num_params)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, r, ck["feats"], ck


# --------------------------------------------------------------------------- preview / pack


def preview(name: str, photos: list[Path] | None = None, strengths: tuple[float, ...] = (0.7, 1.0)) -> Path:
    """Before/after sheet. Default photos: the owner's unedited photos (data/owner, private) if
    present, else a sample of the generic input pool."""
    from photostyle.features import FeatureExtractor

    p = Project.load(name)
    head, r, _, _ = _load_head(p)
    fx = FeatureExtractor(cache_dir=None)
    if photos is None:
        man = Path("data/owner/manifest.csv")
        if man.exists():
            with man.open() as f:
                photos = [Path("data/owner/srgb") / f"{row['id']}.jpg" for row in csv.DictReader(f)
                          if row.get("edited") == "no"]
        if not photos:
            pool = input_pool(exclude=name)
            photos = random.Random(0).sample(pool, min(6, len(pool)))
    rows = []
    with torch.no_grad():
        for ph in photos:
            x = _tensor(ph, 768)
            th = head(fx(x)[None])
            rows.append([x] + [r(x[None], th * s)[0] for s in strengths])
    tw, pad = 300, 6
    ims = [[Image.fromarray((t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")) for t in row]
           for row in rows]
    for row in ims:
        for im in row:
            im.thumbnail((tw, tw))
    hs = [max(im.height for im in row) for row in ims]
    sheet = Image.new("RGB", (len(ims[0]) * (tw + pad) + pad, sum(hs) + pad * (len(hs) + 1) + 24), "white")
    d = ImageDraw.Draw(sheet)
    for j, lab in enumerate(["original"] + [f"{name} {int(s * 100)} %" for s in strengths]):
        d.text((pad + j * (tw + pad), 5), lab, fill="black")
    y = 24 + pad
    for row, h in zip(ims, hs):
        for j, im in enumerate(row):
            sheet.paste(im, (pad + j * (tw + pad), y))
        y += h + pad
    out = p.dir / "preview.jpg"
    sheet.save(out, quality=88)
    print(f"preview -> {out}")
    print(f"next: photostyle style pack {name} [--strength 0.8]")
    return out


def pack(name: str, strength: float = 1.0, out_root: Path = Path("stylepacks"), order: int | None = None) -> Path:
    from photostyle.engine import save_stylepack

    p = Project.load(name)
    head, r, feats, ck = _load_head(p)
    lic = sorted({f"{p.candidates[i]['license']}" for i in p.refs}) if p.refs else []
    card = {
        "source": p.train.get("mode", "unknown"),
        "description": p.description,
        "default_strength": strength,
        "references": f"{len(p.refs)} openly licensed photos ({', '.join(lic)}); attribution in ATTRIBUTION.csv",
        "training_data": ("openly licensed references (and inputs) only; no research-licence data"
                          if ck.get("kind") != "coded" else
                          "a code on the shared base, which is MIT-Adobe FiveK derived: research / personal use only"),
        "pipeline": "photostyle style (photostyle/newstyle.py)",
        "train": p.train,
    }
    if ck.get("kind") == "coded":
        card.update(head_type="coded", base_n_styles=ck["base_n_styles"])
    # Publishable = trained on openly licensed photos only (refs are always open; the "strong"
    # input pool must have excluded the owner's photos; nothing FiveK-derived).
    mode = p.train.get("mode", "")
    card["publishable"] = (ck.get("kind") != "coded" and mode.startswith(("unpaired", "teacher"))
                           and (p.train.get("recipe") == "gentle" or bool(p.train.get("open_only"))))
    if mode.startswith("teacher"):
        from photostyle.recipes import RECIPE_TABLES, SOURCES

        card["training_data"] = ("a recipe transcribed from grading tutorials (photostyle/recipes.py), applied to "
                                 "openly licensed input photos; the references chose and checked the look")
        used = sorted({s for row in RECIPE_TABLES.get(name, []) for s in row[3]})
        card["tutorials"] = [{"title": SOURCES[k][0], "url": SOURCES[k][1]} for k in used]
    old = out_root / name / "style.json"
    if order is None and old.exists():                  # repacking keeps the style's place in the list
        order = json.loads(old.read_text()).get("order")
    if order is not None:
        card["order"] = order
    folder = save_stylepack(out_root / name, name, head, r, feats, card)
    if p.refs:
        man = _write_manifest(p)
        (folder / "ATTRIBUTION.csv").write_text(man.read_text())
    print(f"style pack -> {folder} (default strength {strength})")
    return folder


def status(name: str) -> dict:
    p = Project.load(name)
    s = {"name": p.name, "description": p.description, "queries": p.queries, "candidates": len(p.candidates),
         "likely_digital_art": [i + 1 for i in range(len(p.candidates)) if flagged(p, i)],
         "picks": [i + 1 for i in p.picks], "references": len(p.refs), "excluded": [i + 1 for i in p.excluded],
         "trained": p.train or None, "packed": (Path("stylepacks") / name / "style.json").exists()}
    return s
