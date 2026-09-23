"""Unit tests for the Phase 7 engine core (photostyle/)."""

import numpy as np
import pytest
import torch

from bench.metrics import all_metrics
from photostyle.head import Head, Preset
from photostyle.render import GlobalRenderer


@pytest.mark.parametrize("kind", ["shared", "per_channel"])
def test_identity_at_zero(kind):
    r = GlobalRenderer(kind)
    img = torch.rand(2, 3, 17, 23)
    out = r(img, torch.zeros(2, r.num_params))
    assert torch.allclose(out, img, atol=1e-6)


@pytest.mark.parametrize("kind", ["shared", "per_channel"])
def test_gradients_reach_all_parameters(kind):
    r = GlobalRenderer(kind)
    img = torch.rand(1, 3, 16, 16)
    p = (0.01 * torch.randn(1, r.num_params)).requires_grad_()
    r(img, p).sum().backward()
    assert (p.grad.abs() > 0).float().mean() > 0.9


def test_per_channel_curves_are_monotone():
    r = GlobalRenderer("per_channel")
    p = torch.randn(64, r.num_params) * 2
    y = r.curves(r.unpack(p)["curve"])
    assert (y[..., 1:] >= y[..., :-1]).all()


def test_black_point_crush_and_lift():
    r = GlobalRenderer("per_channel")
    p = torch.zeros(1, r.num_params)
    black_idx = [0, r.K, 2 * r.K]  # first curve parameter of each channel
    p[0, black_idx] = -0.1
    dark = torch.full((1, 3, 4, 4), 0.05)
    assert r(dark, p).max() < 0.05  # crushed
    p[0, black_idx] = 0.1
    assert r(dark, p).min() > 0.05  # lifted


def test_head_and_preset_start_as_identity():
    r = GlobalRenderer("per_channel")
    feats = torch.randn(5, 1280)
    h = Head(1280, r.num_params)
    h.set_norm(feats)
    assert torch.count_nonzero(h(feats)) == 0
    assert torch.count_nonzero(Preset(r.num_params)(feats)) == 0


def test_metrics_perfect_match():
    img = np.random.default_rng(0).random((8, 8, 3))
    m = all_metrics(img, img)
    assert m["l1"] == 0 and m["delta_e"] < 1e-6 and m["psnr"] > 100


def test_cube_round_trip(tmp_path):
    from photostyle.export import apply_lut, bake_lut, read_cube, write_cube

    torch.manual_seed(0)
    r = GlobalRenderer("per_channel")
    theta = 0.15 * torch.randn(1, r.num_params)
    theta[0, -1] = 0.0  # vignette is spatial and not part of the LUT
    img = torch.rand(1, 3, 32, 48)
    lut = bake_lut(r, theta, size=65)
    write_cube(lut, tmp_path / "t.cube")
    lut2 = read_cube(tmp_path / "t.cube")
    ref = r(img, theta)
    out = apply_lut(img, lut2)
    mse = float(((out - ref) ** 2).mean())
    assert 10 * np.log10(1 / mse) > 45  # plan target: .cube fidelity > 45 dB


def test_identity_lut_is_identity():
    from photostyle.export import apply_lut, bake_lut

    r = GlobalRenderer("per_channel")
    img = torch.rand(1, 3, 8, 8)
    out = apply_lut(img, bake_lut(r, torch.zeros(r.num_params), size=17))
    assert torch.allclose(out, img, atol=1e-5)
