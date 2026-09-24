"""Export what the hosted (browser-only) Studio needs.

The GitHub Pages build of Studio runs the model in the browser: the frozen
ResNet-18 backbone through ONNX Runtime Web, the per-style head (a tiny MLP) in
plain JavaScript. Two parts:

  styles    publishable style packs -> studio/web/pages/models/styles.json (committed), from
            stylepacks/ and webpacks/ (web-only packs, e.g. an open-data version of a local style).
            Only packs whose card says ``"publishable": true`` (trained on openly
            licensed photos only; see ``photostyle style train --open-only``). Packs
            built on MIT-Adobe FiveK data or on the owner's photos are never exported.
  backbone  ResNet-18 (torchvision ImageNet weights) -> ONNX with fp16-stored weights
            and fp32 compute (22 MB, 0.1 % feature error vs PyTorch). Built in CI, not
            committed.

  hero      welcome-screen before/after images (public-domain samples x publishable packs).

Usage:
    uv run python tools/export_web.py styles
    uv run python tools/export_web.py hero
    uv run --with onnx --with onnxruntime python tools/export_web.py backbone --out site/models/rn18.onnx
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CARD_KEYS = ("name", "description", "default_strength", "order", "renderer", "knots", "references", "source")


def _b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().float().contiguous().numpy().astype("<f4").tobytes()).decode()


def export_styles(roots: list[Path], out: Path) -> None:
    from photostyle.engine import StylePack
    from photostyle.head import Head

    folders = {}                              # a later root overrides a style of the same name
    for root in roots:
        if root.exists():
            folders.update({f.name: f for f in sorted(root.iterdir()) if f.is_dir()})
    styles = []
    for folder in folders.values():
        card_f = folder / "style.json"
        if not card_f.exists() or not (folder / "head.pt").exists():
            continue
        card = json.loads(card_f.read_text())
        if not card.get("publishable"):
            print(f"skip {folder.name}: not publishable")
            continue
        pack = StylePack.load(folder)
        assert isinstance(pack.head, Head) and pack.card["renderer"] == "per_channel"
        sd = pack.head.state_dict()
        attribution = []
        if (folder / "ATTRIBUTION.csv").exists():
            with (folder / "ATTRIBUTION.csv").open() as f:
                attribution = [r.get("attribution", "") for r in csv.DictReader(f)]
        styles.append({
            "card": {k: card[k] for k in CARD_KEYS if k in card},
            "attribution": attribution,
            "head": {
                "mu": _b64(sd["mu"]), "sigma": _b64(sd["sigma"]),
                "w1": _b64(sd["net.0.weight"]), "b1": _b64(sd["net.0.bias"]),
                "ln_w": _b64(sd["net.1.weight"]), "ln_b": _b64(sd["net.1.bias"]),
                "w2": _b64(sd["net.4.weight"]), "b2": _b64(sd["net.4.bias"]),
                "theta_mean": _b64(sd["theta_mean"]), "alpha": float(sd["alpha"]),
                "hidden": int(sd["net.0.weight"].shape[0]), "out": int(sd["net.4.weight"].shape[0]),
            },
            "ood": {"mean": _b64(pack.feat_mean), "std": _b64(pack.feat_std), "ref": float(pack.ood_ref)},
        })
        print(f"export {folder.name}")
    styles.sort(key=lambda s: (s["card"].get("order", 1000), s["card"]["name"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "feature_dim": 1280, "styles": styles}))
    print(f"{len(styles)} styles -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def export_backbone(out: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from photostyle.features import IMAGENET_MEAN, IMAGENET_STD, FeatureExtractor

    class Backbone(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb
            self.register_buffer("mean", IMAGENET_MEAN)
            self.register_buffer("std", IMAGENET_STD)

        def forward(self, x):          # (1, 3, H, W) in [0, 1] -> (1, 512)
            return self.bb((x - self.mean) / self.std).flatten(1)

    m = Backbone(FeatureExtractor().backbone).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".fp32.onnx")
    x = torch.rand(1, 3, 224, 336)
    torch.onnx.export(m, (x,), str(tmp), input_names=["x"], output_names=["f"],
                      dynamic_axes={"x": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)
    g = (model := onnx.load(str(tmp))).graph
    casts, halves = [], []
    for init in list(g.initializer):       # store large weights as fp16, cast back to fp32 at load
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size > 64:
            halves.append(numpy_helper.from_array(a.astype(np.float16), init.name + "_h"))
            casts.append(helper.make_node("Cast", [init.name + "_h"], [init.name], to=TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(halves)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    tmp.unlink()
    try:
        import onnxruntime as ort

        ref = m(x).detach().numpy()
        got = ort.InferenceSession(str(out)).run(None, {"x": x.numpy()})[0]
        print(f"backbone -> {out} ({out.stat().st_size / 1e6:.1f} MB), "
              f"relative error {np.abs(got - ref).mean() / np.abs(ref).mean():.2e}")
    except ImportError:
        print(f"backbone -> {out}")


HERO = [("valley", "fujifilm"), ("neon-night", "cyberpunk"), ("mountain-lake", "fujifilm"), ("lisbon-street", "cyberpunk")]


def export_hero(roots: list[Path], samples: Path) -> None:
    """Before/after pairs for the welcome screen: public-domain samples graded by publishable packs."""
    from photostyle.engine import Engine

    eng = Engine(roots=roots)                 # a later root overrides a style of the same name
    out, items = samples / "hero", []
    out.mkdir(parents=True, exist_ok=True)
    for sample, style in HERO:
        if not eng.style(style).card.get("publishable"):
            raise SystemExit(f"{style} is not publishable")
        src = samples / f"{sample}.jpg"
        eng.render(src, eng.predict(src, style)).save(out / f"{sample}.{style}.jpg", quality=88)
        items.append({"before": f"{sample}.jpg", "after": f"hero/{sample}.{style}.jpg", "style": style})
        print(f"hero {sample} x {style}")
    (out / "hero.json").write_text(json.dumps(items, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("styles")
    s.add_argument("--roots", type=Path, nargs="+", default=[Path("stylepacks"), Path("webpacks")],
                   help="style pack folders; webpacks/ holds web-only packs")
    s.add_argument("--out", type=Path, default=Path("studio/web/pages/models/styles.json"))
    h = sub.add_parser("hero")
    h.add_argument("--roots", type=Path, nargs="+", default=[Path("stylepacks"), Path("webpacks")])
    h.add_argument("--samples", type=Path, default=Path("studio/web/public/samples"))
    b = sub.add_parser("backbone")
    b.add_argument("--out", type=Path, default=Path("site/models/rn18.onnx"))
    args = ap.parse_args()
    if args.what == "styles":
        export_styles(args.roots, args.out)
    elif args.what == "hero":
        export_hero(args.roots, args.samples)
    else:
        export_backbone(args.out)


if __name__ == "__main__":
    main()
