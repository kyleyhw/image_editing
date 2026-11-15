"""Slider-space models (A6.4, Phase 21): predict pipeline sliders from a photo.

A model is the existing ``Head`` (photostyle/head.py: standardised 1280-D descriptor -> MLP) whose
outputs are slider values in units of ``UNITS`` (stops for exposure, hundreds for the +-100
sliders). It is trained through the differentiable pipeline (photostyle/pipeline.py) to reproduce
expert edits:

    L = mean_pixels || Oklab(render(x; h(phi(x)))) - Oklab(y) ||_2  +  lambda * sum_i w_i |s_i| / range_i

The image term is the Euclidean distance in Oklab (a perceptual space: equal distances are roughly
equally visible). The L1 term is needed because the slider map is not injective (exposure, whites
and a curve point can produce nearly the same picture), so the image loss alone leaves the sliders
undetermined; the penalty selects the sparsest explanation. lambda is chosen on validation data.

Also here: per-image oracles (the best slider vector for one photo, found by gradient descent),
which bound what any model of this class can reach, and the evaluation metrics.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from photostyle import pipeline as P
from photostyle.head import Head

# The correction sliders the base model predicts (A6.4): white balance, exposure, the four tone
# regions, contrast, vibrance and saturation.
CORRECTION = ["temp", "tint", "exposure", "contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "saturation"]
COLOUR = [f"hsl_{k}_{b}" for k in ("hue", "sat", "lum") for b in P.BAND_NAMES]


def units(names: Sequence[str]) -> torch.Tensor:
    """Output scale per slider: one model unit is 1 stop of exposure, else 100 slider units."""
    return torch.tensor([1.0 if n == "exposure" else 100.0 for n in names])


def ranges(names: Sequence[str]) -> torch.Tensor:
    spec = {s["name"]: s for s in P.SLIDERS}
    return torch.tensor([float(spec[n]["max"]) for n in names])


def to_params(s: torch.Tensor, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """(B, n) slider values -> pipeline params with (B,) tensors, clipped to the slider ranges."""
    r = ranges(names).to(s)
    s = torch.maximum(torch.minimum(s, r), -r)
    return {n: s[:, i] for i, n in enumerate(names)}


def oklab_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel Euclidean distance in Oklab between display-encoded sRGB images (B, 3, H, W)."""
    la = P.to_oklab(P._mat(P.SRGB_TO_W, P.srgb_decode(a.clamp(0, 1))))
    lb = P.to_oklab(P._mat(P.SRGB_TO_W, P.srgb_decode(b.clamp(0, 1))))
    return torch.sqrt(((la - lb) ** 2).sum(1) + 1e-10)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())
    return 10 * math.log10(1 / max(mse, 1e-12))


def delta_e00(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean CIEDE2000 between two display sRGB images (3, H, W)."""
    from photostyle.colour_diff import ciede2000, srgb_to_lab
    A = a.clamp(0, 1).permute(1, 2, 0).double().numpy()
    Bn = b.clamp(0, 1).permute(1, 2, 0).double().numpy()
    return float(ciede2000(srgb_to_lab(A), srgb_to_lab(Bn)).mean())


@dataclass
class SliderModel:
    """A trained slider model: the head, the slider names it predicts, their units."""
    head: Head
    names: list[str]

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, 1280) descriptors -> (B, n) slider values (clipped)."""
        with torch.no_grad():
            s = self.head(feats) * units(self.names)
        r = ranges(self.names)
        return torch.maximum(torch.minimum(s, r), -r)

    def params(self, feats: torch.Tensor) -> list[dict[str, float]]:
        s = self.predict(feats)
        return [{n: float(v) for n, v in zip(self.names, row) if abs(float(v)) > 1e-3} for row in s]

    def save(self, path: Path) -> None:
        torch.save({"state": self.head.state_dict(), "names": self.names,
                    "hidden": self.head.net[0].out_features}, path)

    @staticmethod
    def load(path: Path) -> SliderModel:
        d = torch.load(path, weights_only=False)
        h = Head(d["state"]["mu"].numel(), len(d["names"]), hidden=d["hidden"])
        h.load_state_dict(d["state"])
        return SliderModel(h.eval(), d["names"])


def train(feats: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, names: Sequence[str], *,
          lam: float = 0.01, epochs: int = 40, batch: int = 32, lr: float = 3e-3, hidden: int = 256,
          dropout: float = 0.2, weight_decay: float = 1e-4, seed: int = 0,
          val: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
          log: Callable[[str], None] = print) -> tuple[SliderModel, list[dict]]:
    """Fit a slider model. ``xs``/``ys`` are (N, 3, h, w) inputs and expert targets at training
    resolution (display sRGB); ``feats`` (N, 1280) the descriptors of the inputs."""
    torch.manual_seed(seed)
    names = list(names)
    head = Head(feats.shape[1], len(names), hidden=hidden, dropout=dropout)
    head.mu.copy_(feats.mean(0))
    head.sigma.copy_(feats.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(xs) / batch))
    u, r = units(names), ranges(names)
    hist = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(xs))
        tot = 0.0
        for i in range(0, len(xs), batch):
            idx = perm[i:i + batch]
            s = head(feats[idx]) * u
            out = P.render(xs[idx], to_params(s, names))
            loss = oklab_distance(out, ys[idx]).mean() + lam * (s.abs() / r).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        row = {"epoch": ep + 1, "train_loss": tot / len(xs)}
        if val is not None and ((ep + 1) % 5 == 0 or ep + 1 == epochs):
            row.update(evaluate(SliderModel(head.eval(), names), *val))
        hist.append(row)
        log(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return SliderModel(head.eval(), names), hist


def evaluate(model: SliderModel, feats: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> dict[str, float]:
    """Mean Oklab distance (x100) and PSNR of the model's renders against the targets."""
    s = model.predict(feats)
    with torch.no_grad():
        out = torch.cat([P.render(xs[i:i + 64], to_params(s[i:i + 64], model.names)) for i in range(0, len(xs), 64)])
    d = oklab_distance(out, ys).mean(dim=(1, 2))
    ps = [psnr(o, y) for o, y in zip(out, ys)]
    return {"val_dok": float(d.mean() * 100), "val_psnr": float(np.mean(ps))}


def oracle(xs: torch.Tensor, ys: torch.Tensor, names: Sequence[str], steps: int = 150, lr: float = 0.05,
           curves: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image best slider values (no model): gradient descent on the Oklab distance from zero.
    With ``curves``, an RGB point curve with 8 free interior ordinates is optimised too. Returns
    (renders, slider values)."""
    names = list(names)
    u = units(names)
    z = torch.zeros(len(xs), len(names), requires_grad=True)
    knots = torch.linspace(0, 1, 10)[1:-1]
    cy = torch.zeros(len(xs), 8, requires_grad=True)
    opt = torch.optim.Adam([z, cy] if curves else [z], lr=lr)
    for _ in range(steps):
        out = _render_with(xs, to_params(z * u, names), cy if curves else None, knots)
        loss = oklab_distance(out, ys).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = _render_with(xs, to_params(z * u, names), cy if curves else None, knots)
    return out.detach(), (z * u).detach()


def _render_with(xs: torch.Tensor, p: dict, cy: torch.Tensor | None, knots: torch.Tensor) -> torch.Tensor:
    if cy is None:
        return P.render(xs, p)
    # a free monotone curve applied after the pipeline, per image (differentiable PCHIP)
    out = P.render(xs, p)
    xs_k = torch.cat([torch.zeros(1), knots, torch.ones(1)])
    ys_k = torch.cat([torch.zeros(len(xs), 1), knots + 0.25 * torch.tanh(cy), torch.ones(len(xs), 1)], 1)
    ys_k = P._monotone(ys_k)
    flat = out.flatten(1)
    return P.pchip(xs_k, ys_k, flat).view_as(out)


def _read_linear(path: Path) -> torch.Tensor:
    """(3, H, W) linear sRGB of an image file, colour-managed: FiveK's expert edits are 8-bit
    ProPhoto RGB with an embedded profile, so reading them as sRGB would desaturate them
    (tools/fetch_fivek.py, Phase 20 report). Out-of-sRGB colours are clipped."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_fivek", Path(__file__).resolve().parent.parent / "tools" / "fetch_fivek.py")
    mod = sys.modules.get("fetch_fivek")
    if mod is None:
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fetch_fivek"] = mod
        spec.loader.exec_module(mod)
    lin = mod.decode_linear_srgb(path.read_bytes(), clip=True)
    return torch.from_numpy(np.ascontiguousarray(lin, dtype=np.float32)).permute(2, 0, 1)


def load_pairs(root: Path, split: str, expert: str, edge: int, limit: int | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Inputs and expert targets from ``root/{split}/{original,expert_X}`` as display-encoded sRGB,
    resized in linear light so the long edge is ``edge``, portrait frames turned to landscape and
    stretched to one (edge * 3/4 x edge) frame for batching. Returns (xs, ys, file stems)."""
    IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    src = sorted(f for f in (root / split / "original").iterdir() if f.suffix.lower() in IMG)
    tdir = root / split / f"expert_{expert}"
    pairs = [(f, next((g for g in tdir.glob(f.stem + ".*") if g.suffix.lower() in IMG), None)) for f in src]
    pairs = [(a, b) for a, b in pairs if b is not None][:limit]
    H, W = edge * 3 // 4, edge
    xs = torch.zeros(len(pairs), 3, H, W)
    ys = torch.zeros(len(pairs), 3, H, W)
    for i, (a, b) in enumerate(pairs):
        for t, f in ((xs, a), (ys, b)):
            lin = _read_linear(f)
            if lin.shape[1] > lin.shape[2]:
                lin = lin.rot90(1, (1, 2))           # portrait frames turned to landscape: one batch shape
            lin = torch.nn.functional.interpolate(lin[None], size=(H, W), mode="bilinear", antialias=True, align_corners=False)[0]
            t[i] = P.srgb_encode(lin.clamp(0, 1)).clamp(0, 1)
    return xs, ys, [a.stem for a, _ in pairs]


def descriptors(xs_full: Sequence[Path], cache: Path) -> torch.Tensor:
    """1280-D descriptors of the input files (cached in one tensor file)."""
    from PIL import Image

    from photostyle.features import FeatureExtractor
    if cache.exists():
        d = torch.load(cache)
        if d["files"] == [str(f) for f in xs_full]:
            return d["feats"]
    fx = FeatureExtractor()
    out = []
    for f in xs_full:
        im = Image.open(f).convert("RGB")
        im.thumbnail((512, 512))
        out.append(fx(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255))
    feats = torch.stack(out)
    torch.save({"files": [str(f) for f in xs_full], "feats": feats}, cache)
    return feats


def np_oklab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Display sRGB (..., 3) -> Oklab (numpy), for reports."""
    t = torch.from_numpy(np.moveaxis(rgb, -1, 0)[None]).float()
    return np.moveaxis(P.to_oklab(P._mat(P.SRGB_TO_W, P.srgb_decode(t)))[0].numpy(), 0, -1)
