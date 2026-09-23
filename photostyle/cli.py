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
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    from photostyle.engine import Engine

    eng = Engine(roots=[Path(args.root)])
    {"styles": cmd_styles, "apply": cmd_apply, "learn": cmd_learn, "export-cube": cmd_export_cube,
     "serve": cmd_serve}[args.cmd](args, eng)


if __name__ == "__main__":
    main()
