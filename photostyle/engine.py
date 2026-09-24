"""Public engine API (PROJECT_PLAN Phase 12).

    from photostyle.engine import Engine
    eng = Engine()                              # discovers style packs
    params = eng.predict("IMG_0042.jpg", style="clean_cool")
    img = eng.render("IMG_0042.jpg", params.with_strength(0.8))
    params.to_cube("IMG_0042.cube")

A **style pack** is a folder ``<root>/<name>/`` holding ``style.json`` (the
style card) and ``head.pt`` (the head weights, plus learnable renderer state
if any). ``EditParams`` is the contract between model, UI and exporters. It is
plain data, JSON-serialisable, and scales toward identity with ``strength``
because every renderer parameter is an offset from the identity edit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from photostyle.export import apply_lut, bake_lut, write_cube
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.io import load_image, to_pil, to_tensor
from photostyle.render import GlobalRenderer, _vignette

DEFAULT_ROOTS = [Path("stylepacks"), Path.home() / ".photostyle" / "styles"]
PREDICT_EDGE = 512


# --------------------------------------------------------------------------- params


@dataclass
class EditParams:
    """Serializable edit. ``theta`` is the full renderer parameter vector."""

    style: str
    renderer: str
    knots: int
    theta: list[float]
    strength: float = 1.0
    overrides: dict = field(default_factory=dict)    # UI edits: {field: values}
    ood_score: float = 0.0
    version: int = 1
    scene: dict = field(default_factory=dict)        # SceneParams fields (Phase 18), applied before the grade

    def vector(self) -> torch.Tensor:
        """Effective parameters: model prediction x strength, then user overrides."""
        th = torch.tensor(self.theta, dtype=torch.float32)[None] * self.strength
        r = self.make_renderer()
        q = r.unpack(th)
        for k, v in self.overrides.items():
            if k in q:
                q[k].copy_(torch.tensor(v, dtype=torch.float32).view_as(q[k]))
        return th

    def make_renderer(self) -> GlobalRenderer:
        return GlobalRenderer(self.renderer, self.knots)

    def with_strength(self, s: float) -> EditParams:
        return EditParams(**{**asdict(self), "strength": float(s)})

    # human-readable views (read-only) --------------------------------------
    def curves(self) -> dict[str, list[float]]:
        r = self.make_renderer()
        y = r.curves(r.unpack(self.vector())["curve"])[0]
        return {c: y[i].clamp(0, 1).tolist() for i, c in enumerate("rgb")}

    @torch.no_grad()
    def derived(self) -> dict[str, float]:
        """Slider-like readings measured through the *whole* renderer (curves + matrix).

        warmth: mean R - B on a neutral grey ramp (positive = warmer);
        tint: mean G - (R + B) / 2 on the ramp (positive = greener);
        saturation: chroma ratio (out / in) on a set of saturated test colours;
        vignette: the vignette parameter.
        """
        r = self.make_renderer()
        th = self.vector()
        q = r.unpack(th)
        th0 = th.clone()
        r.unpack(th0)["vignette"].zero_()
        grey = torch.linspace(0.15, 0.85, 8).view(1, 1, 1, 8).expand(1, 3, 1, 8).contiguous()
        g = r(grey, th0)[0, :, 0]
        cols = torch.tensor([[0.8, 0.3, 0.3], [0.3, 0.7, 0.3], [0.3, 0.4, 0.8], [0.8, 0.7, 0.3],
                             [0.3, 0.7, 0.7], [0.7, 0.3, 0.7]]).T.reshape(1, 3, 1, 6)
        c = r(cols, th0)[0, :, 0]

        def chroma(x):
            return (x.max(0).values - x.min(0).values).mean()

        return {"warmth": float((g[0] - g[2]).mean()), "tint": float((g[1] - (g[0] + g[2]) / 2).mean()),
                "saturation": float(chroma(c) / chroma(cols[0, :, 0])), "vignette": float(q["vignette"][0, 0])}

    # serialisation / export -------------------------------------------------
    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps({**asdict(self), "curves": self.curves(), "derived": self.derived()}, indent=1)
        if path:
            Path(path).write_text(s)
        return s

    @staticmethod
    def from_json(s: str | Path) -> EditParams:
        d = json.loads(Path(s).read_text() if isinstance(s, Path) or str(s).endswith(".json") else s)
        d = {k: v for k, v in d.items() if k in EditParams.__dataclass_fields__}
        return EditParams(**d)

    def to_cube(self, path: str | Path, size: int = 33) -> None:
        write_cube(bake_lut(self.make_renderer(), self.vector()[0], size), Path(path),
                   title=f"{self.style} ({self.strength:.2f})")

    def to_xmp(self, path: str | Path) -> None:
        """Lightroom / ACR preset carrying the per-channel curves only.

        The colour matrix cannot be expressed as ACR sliders, so it is not
        included; use the .cube for the full grade.
        """
        def pts(y):
            xs = np.linspace(0, 255, len(y))
            return "".join(f"<rdf:li>{int(round(x))}, {int(round(v * 255))}</rdf:li>" for x, v in zip(xs, y))

        c = self.curves()
        xmp = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal" crs:Version="15.0" crs:ProcessVersion="11.0"
    crs:ToneCurveName2012="Custom">
   <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">{self.style} curves</rdf:li></rdf:Alt></crs:Name>
   <crs:ToneCurvePV2012Red><rdf:Seq>{pts(c['r'])}</rdf:Seq></crs:ToneCurvePV2012Red>
   <crs:ToneCurvePV2012Green><rdf:Seq>{pts(c['g'])}</rdf:Seq></crs:ToneCurvePV2012Green>
   <crs:ToneCurvePV2012Blue><rdf:Seq>{pts(c['b'])}</rdf:Seq></crs:ToneCurvePV2012Blue>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
        Path(path).write_text(xmp)


# --------------------------------------------------------------------------- styles


@dataclass
class StylePack:
    name: str
    card: dict
    head: Head
    renderer: GlobalRenderer
    feat_mean: torch.Tensor
    feat_std: torch.Tensor
    ood_ref: float

    @staticmethod
    def load(folder: Path) -> StylePack:
        card = json.loads((folder / "style.json").read_text())
        ck = torch.load(folder / "head.pt", map_location="cpu", weights_only=False)
        r = GlobalRenderer(card["renderer"], card.get("knots", 9))
        if card.get("head_type") == "coded":        # a style on the shared base (Phase 10)
            from photostyle.condition import CodedHead, StyleHead

            sh = StyleHead(FEATURE_DIM, r.num_params, n_styles=card["base_n_styles"])
            head = CodedHead(sh, torch.zeros(sh.embed.weight.shape[1]))
        else:
            head = Head(FEATURE_DIM, r.num_params, **card.get("head_kwargs", {}))
        head.load_state_dict(ck["state_dict"])
        head.eval()
        return StylePack(card["name"], card, head, r, ck["feat_mean"], ck["feat_std"], ck["ood_ref"])


def save_stylepack(folder: Path, name: str, head: Head, renderer: GlobalRenderer, train_feats: torch.Tensor,
                   card: dict) -> Path:
    """Write a style pack. ``train_feats`` (N, D) are the training features, used for OOD."""
    folder.mkdir(parents=True, exist_ok=True)
    mu, sd = train_feats.mean(0), train_feats.std(0).clamp_min(1e-4)
    d = (((train_feats - mu) / sd) ** 2).mean(1).sqrt()
    ood_ref = float(torch.quantile(d, 0.95)) if len(d) > 1 else 1.0
    torch.save({"state_dict": head.state_dict(), "feat_mean": mu, "feat_std": sd, "ood_ref": ood_ref},
               folder / "head.pt")
    card = {"name": name, "renderer": renderer.kind, "knots": renderer.K, "n_train": len(train_feats), **card}
    (folder / "style.json").write_text(json.dumps(card, indent=1))
    return folder


# --------------------------------------------------------------------------- engine


class Engine:
    def __init__(self, roots: list[Path] | None = None, cache_dir: Path | None = None):
        self.roots = roots or DEFAULT_ROOTS
        self.fx = FeatureExtractor(cache_dir=cache_dir)
        self._styles: dict[str, StylePack] = {}
        self.refresh()

    def refresh(self) -> None:
        self._styles = {}
        for root in self.roots:
            if root.exists():
                for f in sorted(root.iterdir()):
                    if (f / "style.json").exists() and (f / "head.pt").exists():
                        pack = StylePack.load(f)
                        self._styles[pack.name] = pack

    def styles(self) -> list[dict]:
        """Style cards in display order: the card's optional "order" (lower first), then name."""
        packs = sorted(self._styles.values(), key=lambda p: (p.card.get("order", 1000), p.name))
        return [p.card for p in packs]

    def style(self, name: str) -> StylePack:
        if name not in self._styles:
            raise KeyError(f"unknown style {name!r}; available: {sorted(self._styles)}")
        return self._styles[name]

    @staticmethod
    def _proxy(img: Image.Image, edge: int = PREDICT_EDGE) -> torch.Tensor:
        t = to_tensor(img)
        _, h, w = t.shape
        s = edge / max(h, w)
        if s < 1:
            t = F.interpolate(t[None], size=(round(h * s), round(w * s)), mode="bilinear",
                              antialias=True, align_corners=False)[0]
        return t

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, style: str, strength: float | None = None) -> EditParams:
        """Predict edit parameters. ``strength`` defaults to the pack's ``default_strength`` (else 1)."""
        img = load_image(image)[0] if not isinstance(image, Image.Image) else image
        pack = self.style(style)
        f = self.fx(self._proxy(img))
        theta = pack.head(f[None])[0]
        z = ((f - pack.feat_mean) / pack.feat_std).pow(2).mean().sqrt()
        if strength is None:
            strength = float(pack.card.get("default_strength", 1.0))
        return EditParams(style=style, renderer=pack.renderer.kind, knots=pack.renderer.K,
                          theta=theta.tolist(), strength=strength, ood_score=float(z / pack.ood_ref))

    @torch.no_grad()
    def render(self, image: str | Path | Image.Image, params: EditParams, lut_size: int = 65) -> Image.Image:
        """Full-resolution render: bake curves+matrix into a LUT, apply, then vignette."""
        img = load_image(image)[0] if not isinstance(image, Image.Image) else image
        r = params.make_renderer()
        th = params.vector()
        lut = bake_lut(r, th[0], lut_size)
        t = to_tensor(img)[None]
        if params.scene:
            from photostyle.atmosphere import SceneParams, apply_scene

            sp = SceneParams(**{k: v for k, v in params.scene.items() if k in SceneParams.__dataclass_fields__})
            if not sp.is_identity():
                t = apply_scene(t, sp)
        out = torch.cat([apply_lut(t[..., i:i + 512, :], lut) for i in range(0, t.shape[2], 512)], 2)
        out = _vignette(out, r.unpack(th)["vignette"])
        return to_pil(out[0])
