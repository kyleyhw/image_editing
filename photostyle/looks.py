"""Look profiles and the differentiable losses that pull edits toward them.

A *look profile* is a small set of target colour statistics per lighting
regime (night / day). It describes a look without copying any image: the
Clean Cool profile comes from summary statistics measured in the style
study (research_notes/styles/bleg.md). See PROJECT_PLAN §17 A2.2.

Statistics (CIELAB), each with a tolerance that sets its weight:

  L_p01, L_p50               black point and key (1st percentile / median L*)
  shadow_a/b, mid_a/b,       mean chroma inside soft luminance bands
  high_a/b                     (shadows L* < 25, highlights L* > 70)
  sat                        mean HSV saturation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from photostyle.color import hsv_saturation, rgb_to_lab

STATS = ("L_p01", "L_p50", "shadow_a", "shadow_b", "mid_a", "mid_b", "high_a", "high_b", "sat")


@dataclass(frozen=True)
class LookProfile:
    name: str
    night: dict[str, tuple[float, float]]
    day: dict[str, tuple[float, float]]

    def targets(self, regime: str) -> dict[str, tuple[float, float]]:
        return self.night if regime == "night" else self.day


CLEAN_COOL = LookProfile(
    name="clean_cool",
    night={
        "L_p01": (1.0, 2.0), "L_p50": (18.0, 12.0),
        "shadow_a": (0.0, 4.0), "shadow_b": (-7.0, 3.0),
        "mid_a": (3.0, 6.0), "mid_b": (0.0, 8.0),
        "high_a": (0.0, 4.0), "high_b": (0.0, 3.0),
        "sat": (0.45, 0.15),
    },
    day={
        "L_p01": (3.0, 3.0), "L_p50": (55.0, 15.0),
        "shadow_a": (0.0, 4.0), "shadow_b": (-8.0, 3.0),
        "mid_a": (1.0, 5.0), "mid_b": (-2.0, 6.0),
        "high_a": (0.0, 3.0), "high_b": (1.0, 3.0),
        "sat": (0.28, 0.15),
    },
)
PROFILES = {CLEAN_COOL.name: CLEAN_COOL}


def image_stats(img: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable look statistics for (B, 3, H, W) -> dict of (B,) tensors."""
    lab = rgb_to_lab(img)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    B = img.shape[0]
    Lf = L.reshape(B, -1)
    w_s = torch.sigmoid((25 - L) / 3)
    # Blue, bright pixels are most likely sky. "Neutral highlights" came from a
    # portrait study, so sky is excluded from the highlight band to keep blue
    # skies blue. This is a crude heuristic until Phase 16's sky mask replaces it.
    sky = torch.sigmoid((-b - 8) / 2) * torch.sigmoid((L - 50) / 5)
    w_h = torch.sigmoid((L - 70) / 3) * (1 - sky)
    w_m = (1 - w_s) * (1 - torch.sigmoid((L - 70) / 3))

    def band(x, w):
        return (x * w).reshape(B, -1).sum(1) / w.reshape(B, -1).sum(1).clamp_min(1.0)

    return {
        "L_p01": torch.quantile(Lf, 0.01, dim=1),
        "L_p50": torch.quantile(Lf, 0.50, dim=1),
        "shadow_a": band(a, w_s), "shadow_b": band(b, w_s),
        "mid_a": band(a, w_m), "mid_b": band(b, w_m),
        "high_a": band(a, w_h), "high_b": band(b, w_h),
        "sat": hsv_saturation(img).reshape(B, -1).mean(1),
        # band occupancy, used to ignore bands an image barely has
        "_w_s": w_s.reshape(B, -1).mean(1), "_w_h": w_h.reshape(B, -1).mean(1),
    }


KEY_TOL = 6.0  # L* units: how far the edit may move an image's key (median L*)


def profile_loss(img: torch.Tensor, regimes: list[str], profile: LookProfile,
                 ref: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared, tolerance-normalised distance to the profile.

    Band statistics are down-weighted for images that have almost no pixels
    in that band (e.g. no highlights at night), so sparse bands do not
    dominate. If ``ref`` (the unedited input) is given, the key target becomes
    *relative*: the edit may move the median L* by about KEY_TOL, rather than
    being forced toward one absolute brightness. The first pilot run showed
    that an absolute key target made the model blow out bright scenes.
    """
    s = image_stats(img)
    r = image_stats(ref) if ref is not None else None
    total = 0.0
    for i, reg in enumerate(regimes):
        t = dict(profile.targets(reg))
        if r is not None:
            t["L_p50"] = (float(r["L_p50"][i]), KEY_TOL)
        for k, (target, tol) in t.items():
            w = 1.0
            if k.startswith("shadow"):
                w = float(torch.clamp(s["_w_s"][i].detach() / 0.05, max=1.0))
            elif k.startswith("high"):
                w = float(torch.clamp(s["_w_h"][i].detach() / 0.05, max=1.0))
            total = total + w * ((s[k][i] - target) / tol) ** 2
    return total / (len(regimes) * len(STATS))


def luminance(img: torch.Tensor) -> torch.Tensor:
    return (0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2])


def clip_fraction(img: torch.Tensor, hi: float = 0.985, soft: float = 0.005) -> torch.Tensor:
    """Soft fraction of pixels with any channel at/above ``hi`` -> (B,)."""
    return torch.sigmoid((img.max(1).values - hi) / soft).flatten(1).mean(1)


def detail_similarity(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of luminance gradient magnitudes (B,). 1 = detail kept."""
    def grad_mag(x):
        y = luminance(x)
        gx = y[:, :, 1:] - y[:, :, :-1]
        gy = y[:, 1:, :] - y[:, :-1, :]
        return torch.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2 + 1e-8).flatten(1)
    a, b = grad_mag(out), grad_mag(ref)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def fidelity_loss(out: torch.Tensor, ref: torch.Tensor, clip_margin: float = 0.01) -> torch.Tensor:
    """Penalise new highlight clipping (beyond ``clip_margin``) and loss of detail."""
    new_clip = torch.relu(clip_fraction(out) - clip_fraction(ref) - clip_margin)
    return (20.0 * new_clip + 5.0 * (1 - detail_similarity(out, ref))).mean()


@torch.no_grad()
def profile_distance(img: torch.Tensor, regime: str, profile: LookProfile,
                     ref: torch.Tensor | None = None) -> dict[str, float]:
    """Per-statistic |x - target| / tolerance for one image (evaluation).

    With ``ref`` (the unedited input) the key is scored relative to the input,
    matching the training objective."""
    s = image_stats(img[None] if img.dim() == 3 else img)
    t = dict(profile.targets(regime))
    if ref is not None:
        r = image_stats(ref[None] if ref.dim() == 3 else ref)
        t["L_p50"] = (float(r["L_p50"][0]), KEY_TOL)
    return {k: float(abs(s[k][0] - target) / tol) for k, (target, tol) in t.items()}


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Sliced W1 between two point clouds (N, D) and (M, D); both resampled to min(N, M)."""
    n = min(x.shape[0], y.shape[0])
    x = x[torch.randperm(x.shape[0])[:n]]
    y = y[torch.randperm(y.shape[0])[:n]]
    proj = torch.randn(x.shape[1], n_proj)
    proj = proj / proj.norm(dim=0, keepdim=True)
    px, _ = torch.sort(x @ proj, 0)
    py, _ = torch.sort(y @ proj, 0)
    return (px - py).abs().mean()


def lab_pixels(img: torch.Tensor, n: int = 2048) -> torch.Tensor:
    """Random Lab pixel sample from (B, 3, H, W), scaled so L, a, b have similar ranges."""
    lab = rgb_to_lab(img).permute(0, 2, 3, 1).reshape(-1, 3) / torch.tensor([100.0, 60.0, 60.0])
    idx = torch.randint(0, lab.shape[0], (n,))
    return lab[idx]
