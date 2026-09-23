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


def test_regional_identity_and_gradients():
    from photostyle.regional import RegionalRenderer

    r = RegionalRenderer()
    img = torch.rand(2, 3, 24, 32)
    assert torch.allclose(r(img, torch.zeros(2, r.num_params)), img, atol=1e-5)
    p = (0.05 * torch.randn(1, r.num_params)).requires_grad_()
    r(img[:1], p).sum().backward()
    q = r.unpack(p.grad)
    for k in ("grad", "tone", "sky"):
        assert q[k].abs().sum() > 0, k


def test_sky_probability_prefers_blue_top():
    from photostyle.regional import sky_probability

    img = torch.zeros(1, 3, 40, 40)
    img[:, :, :20] = torch.tensor([0.45, 0.6, 0.9]).view(1, 3, 1, 1)   # blue sky
    img[:, :, 20:] = torch.tensor([0.2, 0.35, 0.15]).view(1, 3, 1, 1)  # green ground
    p = sky_probability(img)
    assert p[0, 0, :15].mean() > 0.6 and p[0, 0, 25:].mean() < 0.2


def test_lut_renderer_identity_and_learnable():
    from photostyle.lut import LUTRenderer

    r = LUTRenderer()
    img = torch.rand(2, 3, 16, 16)
    assert torch.allclose(r(img, torch.zeros(2, r.num_params)), img, atol=1e-5)
    p = torch.zeros(2, r.num_params)
    loss = (r(img, p) - img.flip(1)).abs().mean() + r.regularizer()
    loss.backward()
    assert r.bases.grad.abs().sum() > 0


def test_engine_round_trip(tmp_path):
    from PIL import Image

    from photostyle.engine import EditParams, Engine, save_stylepack
    from photostyle.features import FEATURE_DIM

    r = GlobalRenderer("per_channel")
    head = Head(FEATURE_DIM, r.num_params)
    feats = torch.randn(10, FEATURE_DIM)
    head.set_norm(feats)
    torch.nn.init.normal_(head.net[-1].weight, std=0.01)  # a non-identity style
    save_stylepack(tmp_path / "packs" / "t", "t", head, r, feats, {"source": "test"})
    eng = Engine(roots=[tmp_path / "packs"])
    assert [c["name"] for c in eng.styles()] == ["t"]
    img = Image.fromarray((np.random.default_rng(0).random((40, 60, 3)) * 255).astype("uint8"))
    p = eng.predict(img, "t")
    out = eng.render(img, p)
    assert out.size == img.size
    ident = eng.render(img, p.with_strength(0.0))
    assert np.abs(np.asarray(ident, float) - np.asarray(img, float)).max() <= 2  # LUT + 8-bit rounding
    p2 = EditParams.from_json(p.to_json())
    assert p2.theta == p.theta
    p.to_cube(tmp_path / "x.cube")
    p.to_xmp(tmp_path / "x.xmp")
    assert "ToneCurvePV2012Red" in (tmp_path / "x.xmp").read_text()
