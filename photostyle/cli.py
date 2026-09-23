"""``photostyle`` command-line interface (PROJECT_PLAN Phase 12).

    photostyle styles list
    photostyle apply  --style clean_cool [--strength 0.8] photos/*.jpg -o out/ [--cube] [--xmp] [--json]
    photostyle learn  --name my_look --pairs before/ after/          # paired
    photostyle learn  --name neon --examples inspo/ --inputs mine/   # unpaired
    photostyle export-cube --style my_look --average photos/*.jpg -o my_look.cube
    photostyle serve  [--port 8765]                                  # Studio web app
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".dng", ".cr2", ".cr3", ".nef",
           ".arw", ".raf", ".orf", ".rw2"}


def _images(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def cmd_styles(args, eng) -> None:
    for c in eng.styles():
        print(f"{c['name']:24s} {c.get('source', ''):10s} n={c.get('n_train', '?'):>5}  {c.get('description', '')}")


def cmd_apply(args, eng) -> None:
    from photostyle.io import load_image, save_image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in _images(args.photos):
        img, info = load_image(p)
        params = eng.predict(img, args.style, strength=args.strength)
        save_image(eng.render(img, params), out / f"{p.stem}_{args.style}.jpg", info)
        if args.cube:
            params.to_cube(out / f"{p.stem}_{args.style}.cube")
        if args.xmp:
            params.to_xmp(out / f"{p.stem}_{args.style}.xmp")
        if args.json:
            params.to_json(out / f"{p.stem}_{args.style}.json")
        flag = "  (unusual photo for this style)" if params.ood_score > 1.5 else ""
        print(f"{p.name} -> {args.style}  ood={params.ood_score:.2f}{flag}")


def cmd_learn(args, eng) -> None:
    from photostyle.engine import save_stylepack
    from photostyle.io import load_image, to_tensor
    from photostyle.train import learn_paired, learn_unpaired

    def prog(step, total, v):
        print(f"\r  step {step}/{total}  loss={v:.4f}", end="", flush=True)

    def t(p):
        return to_tensor(load_image(p)[0])

    if args.pairs:
        before, after = map(Path, args.pairs)
        names = {p.stem: p for p in _images([str(before)])}
        pairs = [(t(names[q.stem]), t(q)) for q in _images([str(after)]) if q.stem in names]
        print(f"{len(pairs)} matched pairs")
        head, r, feats, info = learn_paired(pairs, eng.fx, progress=prog)
        source = "paired"
    else:
        ex = [t(p) for p in _images([args.examples])]
        ins = [t(p) for p in _images([args.inputs])] if args.inputs else []
        print(f"{len(ex)} examples, {len(ins)} inputs")
        head, r, feats, info = learn_unpaired(ex, ins, eng.fx, progress=prog)
        source = "unpaired"
    print()
    folder = save_stylepack(Path(args.root) / args.name, args.name, head, r, feats,
                            {"source": source, "description": args.description or "", "train": info})
    print(f"saved style pack -> {folder}")


def cmd_export_cube(args, eng) -> None:
    from photostyle.engine import EditParams

    ps = [eng.predict(p, args.style) for p in _images(args.photos)]
    if not ps:
        sys.exit("no photos given: the average LUT is the mean edit over the photos you pass")
    mean = torch.tensor([p.theta for p in ps]).mean(0).tolist()
    avg = EditParams(style=args.style, renderer=ps[0].renderer, knots=ps[0].knots, theta=mean, strength=ps[0].strength)
    avg.to_cube(args.out, size=args.size)
    print(f"average of {len(ps)} edits -> {args.out}")


def cmd_serve(args, eng) -> None:
    from studio.server import run

    run(host=args.host, port=args.port)


def cmd_style(args) -> None:
    import json

    from photostyle import newstyle as ns

    if args.step == "new":
        p = ns.new(args.name, args.describe, args.query, args.mono, args.overwrite)
        print(f"style project {p.name!r}: queries {p.queries}")
        print(f"next: photostyle style search {p.name}")
    elif args.step == "search":
        ns.search(args.name, args.pages)
    elif args.step == "pick":
        ns.pick(args.name, args.numbers, args.n_refs)
    elif args.step == "exclude":
        ns.exclude(args.name, args.numbers)
    elif args.step == "train":
        ns.train(args.name, args.recipe, tuple(Path(d) for d in args.pairs) if args.pairs else None, args.steps)
    elif args.step == "preview":
        ns.preview(args.name, [Path(f) for f in args.photos] or None)
    elif args.step == "pack":
        ns.pack(args.name, args.strength, Path(args.root))
    elif args.step == "status":
        print(json.dumps(ns.status(args.name), indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="photostyle")
    ap.add_argument("--root", default="stylepacks", help="style pack folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("styles")
    s.add_argument("action", choices=["list"])
    a = sub.add_parser("apply")
    a.add_argument("--style", required=True)
    a.add_argument("--strength", type=float, default=None, help="default: the style's own default (usually 1)")
    a.add_argument("-o", "--out", default="out")
    a.add_argument("--cube", action="store_true")
    a.add_argument("--xmp", action="store_true")
    a.add_argument("--json", action="store_true")
    a.add_argument("photos", nargs="+")
    lr = sub.add_parser("learn")
    lr.add_argument("--name", required=True)
    lr.add_argument("--description")
    g = lr.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", nargs=2, metavar=("BEFORE_DIR", "AFTER_DIR"))
    g.add_argument("--examples", metavar="DIR")
    lr.add_argument("--inputs", metavar="DIR", help="unpaired only: typical unedited photos")
    e = sub.add_parser("export-cube")
    e.add_argument("--style", required=True)
    e.add_argument("--size", type=int, default=33)
    e.add_argument("-o", "--out", required=True)
    e.add_argument("photos", nargs="+")
    st = sub.add_parser("style", help="create a new style: idea -> references -> training -> pack")
    ss = st.add_subparsers(dest="step", required=True)
    x = ss.add_parser("new", help="start a style project")
    x.add_argument("name")
    x.add_argument("--describe", required=True, help='the idea in words, e.g. "neon cyberpunk night city"')
    x.add_argument("--query", action="append", help="search query (repeatable); default: from --describe")
    x.add_argument("--mono", action="store_true", help="a black-and-white look")
    x.add_argument("--overwrite", action="store_true", help="replace an existing project of that name")
    x = ss.add_parser("search", help="find openly licensed candidate photos")
    x.add_argument("name")
    x.add_argument("--pages", type=int, default=3)
    x = ss.add_parser("pick", help="the candidates whose look you like; more like them are added")
    x.add_argument("name")
    x.add_argument("numbers", nargs="+", type=int)
    x.add_argument("--n-refs", type=int, default=40)
    x = ss.add_parser("exclude", help="drop references")
    x.add_argument("name")
    x.add_argument("numbers", nargs="+", type=int)
    x = ss.add_parser("train")
    x.add_argument("name")
    x.add_argument("--recipe", choices=["gentle", "strong", "instant", "paired"], default="gentle",
                   help="gentle: looks near natural; strong: looks far from natural (e.g. cyberpunk); "
                        "instant: no training, via the shared base's encoder; paired: with --pairs")
    x.add_argument("--pairs", nargs=2, metavar=("BEFORE_DIR", "AFTER_DIR"), help="train on your before/after pairs")
    x.add_argument("--steps", type=int, default=1200)
    x = ss.add_parser("preview")
    x.add_argument("name")
    x.add_argument("photos", nargs="*")
    x = ss.add_parser("pack")
    x.add_argument("name")
    x.add_argument("--strength", type=float, default=1.0, help="the style's default strength")
    x = ss.add_parser("status")
    x.add_argument("name")
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.cmd == "style":
        return cmd_style(args)

    from photostyle.engine import Engine

    eng = Engine(roots=[Path(args.root)])
    {"styles": cmd_styles, "apply": cmd_apply, "learn": cmd_learn, "export-cube": cmd_export_cube,
     "serve": cmd_serve}[args.cmd](args, eng)


if __name__ == "__main__":
    main()
