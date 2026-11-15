"""Invariants of the image pipeline (photostyle/pipeline.py), Phase 19 task 4 (A6.6).

Each test states a property a professional editor's control must have, on synthetic inputs chosen
to expose it: grey ramps (tone order), mid grey (neutrality), a hue wheel (hue preservation) and
seeded random images (identity and range).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from photostyle import pipeline as P

RNG_SEED = 7                      # any fixed seed; the properties hold for every image


def rand_img(h: int = 24, w: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(RNG_SEED)
    return torch.rand(3, h, w, generator=g)


def ramp(n: int = 256) -> torch.Tensor:
    return torch.linspace(0, 1, n).view(1, 1, n).expand(3, 1, n).clone()


def hue_wheel(n: int = 72, s: float = 0.6, v: float = 0.7) -> torch.Tensor:
    from photostyle.hsv import hsv_to_rgb
    h = torch.linspace(0, 1, n + 1)[:-1].view(1, n)
    return hsv_to_rgb(torch.stack([h, torch.full_like(h, s), torch.full_like(h, v)]))


def lin(x: torch.Tensor) -> torch.Tensor:
    return P.srgb_decode(x)


def test_identity() -> None:
    """Default sliders reproduce the input (round trips through linear Rec. 2020 and Oklab)."""
    x = rand_img()
    assert (P.render(x) - x).abs().max() * 255 < 0.01


def test_exposure_doubles_linear_light() -> None:
    """+1 stop doubles linear values (mid-tones well away from clipping)."""
    x = torch.full((3, 4, 4), 0.35)
    y = P.render(x, {"exposure": 1.0})
    assert torch.allclose(lin(y), 2 * lin(x), rtol=1e-4)


@pytest.mark.parametrize("p", [{"exposure": 0.7}, {"contrast": 80}, {"shadows": -60}, {"highlights": 70},
                               {"whites": -50}, {"blacks": 40}, {"saturation": 80}, {"vibrance": -100},
                               {"hsl_sat_red": 100}, {"curves": {"rgb": [[0, 0.1], [1, 0.9]]}}])
def test_neutral_stays_neutral(p: dict) -> None:
    """Tone, curve and colour controls must not tint a grey."""
    x = torch.full((3, 4, 4), 0.45)
    y = P.render(x, p)
    assert (y.amax(0) - y.amin(0)).max() * 255 < 0.2


@pytest.mark.parametrize("p", [{"temp": 70}, {"temp": -70}, {"tint": 70}, {"tint": -70}])
def test_white_balance_keeps_luminance_and_black(p: dict) -> None:
    """White balance scales channels: the luminance of a neutral is unchanged and black stays black."""
    x = torch.full((3, 2, 2), 0.5)
    y = P.render(x, p)
    Y = lambda t: (torch.as_tensor(P.SRGB_TO_XYZ[1], dtype=t.dtype).view(3, 1, 1) * lin(t)).sum(0)  # noqa: E731
    assert torch.allclose(Y(y), Y(x), atol=2e-4)
    assert P.render(torch.zeros(3, 2, 2), p).max() < 1e-6


def test_white_balance_direction() -> None:
    """Temperature + is warmer (R > B); tint + is magenta (G lowest), as in Lightroom."""
    g = torch.full((3, 1, 1), 0.5)
    w = P.render(g, {"temp": 50})[:, 0, 0]
    m = P.render(g, {"tint": 50})[:, 0, 0]
    assert w[0] > w[2]
    assert m[1] < m[0] and m[1] < m[2]


def test_saturation_minus_100_is_grey() -> None:
    y = P.render(hue_wheel(), {"saturation": -100})
    assert (y.amax(0) - y.amin(0)).max() * 255 < 0.5


@pytest.mark.parametrize("p", [{"contrast": 100}, {"contrast": -100}, {"shadows": 100, "highlights": -100},
                               {"blacks": -100, "whites": 100}, {"blacks": 100, "whites": -100},
                               {"curves": {"rgb": [[0, 0], [0.4, 0.6], [0.6, 0.62], [1, 1]]}}])
def test_tone_is_monotone(p: dict) -> None:
    """Tones never invert, whatever the sliders."""
    y = P.render(ramp(), p)[0, 0]
    assert bool((y[1:] - y[:-1] >= -1e-6).all())


def test_tone_preserves_hue() -> None:
    """Tone sliders scale colour by a luminance ratio: Oklab hue moves by less than 2 degrees
    where nothing clips (the 'green sky' failure of per-channel curves)."""
    x = hue_wheel(s=0.5, v=0.5)
    y = P.render(x, {"contrast": 60, "shadows": 40, "highlights": -40})

    def hue(t: torch.Tensor) -> torch.Tensor:
        lab = P.to_oklab(P._mat(P.SRGB_TO_W, lin(t.unsqueeze(0))))[0]
        return torch.atan2(lab[2], lab[1])

    d = torch.remainder(hue(y) - hue(x) + np.pi, 2 * np.pi) - np.pi
    assert d.abs().max() < np.radians(2)


def test_highlight_recovery() -> None:
    """Exposure +1 pushes the upper half over white; highlights -100 brings it back below white."""
    x = ramp()
    over = P.render(x, {"exposure": 1.0})[0, 0]
    rec = P.render(x, {"exposure": 1.0, "highlights": -100})[0, 0]
    assert (over > 0.999).sum() > 60
    assert (rec > 0.999).sum() < (over > 0.999).sum() / 4


def test_band_weights_partition_of_unity() -> None:
    h = torch.linspace(-np.pi, np.pi, 721)
    assert torch.allclose(P.band_weights(h).sum(0), torch.ones_like(h), atol=1e-5)


def test_vignette_signs_and_centre() -> None:
    x = torch.full((3, 31, 41), 0.4)
    dark, light = P.render(x, {"vig_amount": -100}), P.render(x, {"vig_amount": 100})
    assert dark[:, 0, 0].mean() < 0.4 - 0.1 and light[:, 0, 0].mean() > 0.4 + 0.1
    assert abs(float(dark[:, 15, 20].mean()) - 0.4) < 1e-3


def test_gradients_flow() -> None:
    """The pipeline is differentiable in its sliders (needed for training in Phase 21)."""
    t = torch.tensor([10.0], requires_grad=True)
    P.render(rand_img(), {"exposure": t / 50, "contrast": t, "vibrance": t, "hsl_hue_red": t}).mean().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs() > 0


def test_crop_matches_slice() -> None:
    x = rand_img(40, 60)
    y = P.render(x, {"crop": {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}})
    assert (y - x[:, 10:30, 15:45]).abs().max() < 1e-3


# ---------------------------------------------------------------- Phases 22-24 operators

def test_detail_sliders_at_zero_are_identity() -> None:
    """Spatial operators switch off exactly at zero (the local means are not even computed)."""
    x = rand_img()
    p = {"texture": 0, "clarity": 0, "dehaze": 0, "sharpen": 0, "nr_luma": 0, "nr_colour": 0, "halation": 0, "grain": 0}
    assert (P.render(x, p) - x).abs().max() * 255 < 0.01


def test_gaussian_blur_preserves_mean_and_constant() -> None:
    """Normalised kernel: a constant image is unchanged, and the mean is kept (reflect padding)."""
    c = torch.full((1, 1, 20, 30), 0.37)
    assert torch.allclose(P.gaussian_blur(c, 3.0), c, atol=1e-6)
    x = rand_img().unsqueeze(0)
    assert abs(float(P.gaussian_blur(x, 2.0).mean() - x.mean())) < 2e-3


def test_clarity_increases_local_contrast() -> None:
    """Clarity +80 raises the standard deviation of mid-tone detail; -80 lowers it."""
    g = torch.Generator().manual_seed(RNG_SEED)
    x = (0.45 + 0.08 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1)
    sd = lambda t: float(t.mean(0).std())  # noqa: E731
    assert sd(P.render(x, {"clarity": 80})) > sd(x) > sd(P.render(x, {"clarity": -80}))


def test_grain_is_deterministic_and_zero_mean() -> None:
    """Same input, same grain (seeded hash); the noise averages out over the image."""
    x = torch.full((3, 64, 64), 0.5)
    a, b = P.render(x, {"grain": 80}), P.render(x, {"grain": 80})
    assert torch.equal(a, b)
    assert abs(float(a.mean() - 0.5)) < 0.01 and float(a.std()) > 0.005


def test_linear_mask_affects_its_side_only() -> None:
    x = torch.full((3, 40, 60), 0.4)
    lin = {"type": "linear", "x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 0.5, "sliders": {"exposure": 1.0}}
    y = P.render(x, {"masks": [lin]})
    assert float(y[:, :4].mean()) > 0.5 and abs(float(y[:, -10:].mean()) - 0.4) < 1e-3


def test_inverted_mask_is_complement() -> None:
    """A mask and its inverse with the same exposure add up to the whole-photo edit (in stops)."""
    x = torch.full((3, 40, 60), 0.3)
    r = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2, "feather": 50, "sliders": {"exposure": 0.5}}
    both = P.render(x, {"masks": [r, {**r, "invert": True}]})
    assert torch.allclose(both, P.render(x, {"exposure": 0.5}), atol=2e-3)


def test_geometry_defaults_are_identity() -> None:
    x = rand_img()
    crop = {"x": 0, "y": 0, "w": 1, "h": 1, "angle": 0, "vertical": 0, "horizontal": 0, "distortion": 0, "scale": 100}
    assert (P.render(x, {"crop": crop}) - x).abs().max() * 255 < 0.01


def test_lens_blur_keeps_focus_plane_sharp() -> None:
    """Depth ramp left (far) to right (near), focus on the near side: the far side is blurred, the
    focus plane is not."""
    x = rand_img(48, 96)
    d = torch.linspace(0, 1, 96).view(1, 1, 96).expand(1, 48, 96).clone()
    y = P.render(x, {"lens_amount": 100, "lens_focus": 1.0, "lens_range": 10}, depth=d)
    far = float((y[..., :20] - x[..., :20]).abs().mean())
    near = float((y[..., -6:] - x[..., -6:]).abs().mean())
    assert far > 3 * near and near < 0.05
