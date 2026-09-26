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
    save_stylepack(tmp_path / "packs" / "u", "u", head, r, feats, {"source": "test", "default_strength": 0.7})
    eng = Engine(roots=[tmp_path / "packs"])
    assert [c["name"] for c in eng.styles()] == ["t", "u"]
    save_stylepack(tmp_path / "packs" / "v", "v", head, r, feats, {"source": "test", "order": 1})
    eng.refresh()
    assert [c["name"] for c in eng.styles()] == ["v", "t", "u"]     # "order" first, then name
    img = Image.fromarray((np.random.default_rng(0).random((40, 60, 3)) * 255).astype("uint8"))
    p = eng.predict(img, "t")
    assert p.strength == 1.0
    assert eng.predict(img, "u").strength == 0.7            # the pack's own default
    assert eng.predict(img, "u", strength=0.3).strength == 0.3
    out = eng.render(img, p)
    assert out.size == img.size
    ident = eng.render(img, p.with_strength(0.0))
    assert np.abs(np.asarray(ident, float) - np.asarray(img, float)).max() <= 2  # LUT + 8-bit rounding
    p2 = EditParams.from_json(p.to_json())
    assert p2.theta == p.theta
    p.to_cube(tmp_path / "x.cube")
    p.to_xmp(tmp_path / "x.xmp")
    assert "ToneCurvePV2012Red" in (tmp_path / "x.xmp").read_text()


def test_style_head_identity_and_encoder_shape():
    from photostyle.condition import StyleEncoder, StyleHead

    h = StyleHead(1280, 40, n_styles=3)
    f = torch.randn(4, 1280)
    h.set_norm(f)
    assert torch.count_nonzero(h(f, h.embed.weight[0].expand(4, -1))) == 0   # identity at init
    enc = StyleEncoder(1280)
    assert enc(torch.randn(7, 1280)).shape == (32,)


def test_scene_in_edit_params_reaches_the_render():
    from PIL import Image

    from photostyle.engine import EditParams, Engine

    img = Image.fromarray((np.random.default_rng(1).random((48, 64, 3)) * 255).astype("uint8"))
    r = GlobalRenderer("per_channel")
    p = EditParams(style="t", renderer="per_channel", knots=r.K, theta=[0.0] * r.num_params)
    eng = Engine.__new__(Engine)                      # render() needs no style packs
    same = eng.render(img, p)
    hazy = eng.render(img, EditParams(**{**p.__dict__, "scene": {"clarity_near": 0.8}}))
    assert np.abs(np.asarray(same, float) - np.asarray(img, float)).max() <= 2
    assert np.abs(np.asarray(hazy, float) - np.asarray(img, float)).mean() > 0.5
    assert EditParams.from_json(p.to_json()).scene == {}


def test_newstyle_pick_exclude_offline(tmp_path, monkeypatch):
    """Pipeline bookkeeping without network: picks -> 'more like these' -> exclude -> manifest."""
    from PIL import Image

    from photostyle import newstyle as ns
    from photostyle.stats import colour_stats

    monkeypatch.setattr(ns, "ROOT", tmp_path / "styles")
    monkeypatch.setattr(ns, "MANIFESTS", tmp_path / "manifests")
    p = ns.new("look", "a test look")
    with pytest.raises(SystemExit):
        ns.new("look", "again")                # never silently replaces a project
    rng = np.random.default_rng(0)
    cands = []
    for i in range(12):                        # 6 bluish + 6 reddish photos
        base = np.array([40, 60, 200]) if i < 6 else np.array([200, 60, 40])
        arr = np.clip(base + rng.normal(0, 20, (40, 60, 3)), 0, 255).astype("uint8")
        f = tmp_path / f"c{i}.jpg"
        Image.fromarray(arr).save(f)
        cands.append({"id": f"id{i}", "file": str(f), "stats": colour_stats(Image.open(f)), "license": "cc0",
                      "license_version": "1.0", "creator": f"c{i}", "title": "t", "source": "s",
                      "landing_url": "u", "attribution": "a"})
    p.candidates = cands
    p.save()
    q = ns.pick("look", [1], n_refs=4)         # pick a blue one: the refs should be blue ones
    assert q.refs[0] == 0 and all(i < 6 for i in q.refs)
    q = ns.exclude("look", [q.refs[1] + 1])
    assert len(q.refs) == 4 and q.excluded and all(i < 6 for i in q.refs)
    assert (tmp_path / "manifests" / "look.csv").read_text().count("\n") == 5
    assert ns.status("look")["references"] == 4


def test_coded_pack_round_trip(tmp_path):
    """A style stored as a code on the shared base loads and predicts like any pack."""
    from PIL import Image

    from photostyle.condition import CodedHead, StyleHead
    from photostyle.engine import Engine, save_stylepack
    from photostyle.features import FEATURE_DIM

    r = GlobalRenderer("per_channel")
    sh = StyleHead(FEATURE_DIM, r.num_params, n_styles=5)
    torch.nn.init.normal_(sh.code_bias.weight, std=0.05)      # make the code matter
    head = CodedHead(sh, torch.randn(sh.embed.weight.shape[1]))
    feats = torch.randn(6, FEATURE_DIM)
    save_stylepack(tmp_path / "p" / "c", "c", head, r, feats, {"head_type": "coded", "base_n_styles": 5})
    eng = Engine(roots=[tmp_path / "p"])
    img = Image.fromarray((np.random.default_rng(2).random((32, 48, 3)) * 255).astype("uint8"))
    p = eng.predict(img, "c")
    with torch.no_grad():
        want = head(eng.fx(eng._proxy(img))[None])[0]
    assert np.allclose(p.theta, want.tolist(), atol=1e-5)


def test_flagged_art_is_skipped_unless_picked_or_restored(tmp_path, monkeypatch):
    from photostyle import newstyle as ns

    monkeypatch.setattr(ns, "ROOT", tmp_path / "styles")
    monkeypatch.setattr(ns, "MANIFESTS", tmp_path / "manifests")
    p = ns.new("art", "x")
    stats = {k: 0.0 for k in ns.STAT_KEYS}
    p.candidates = [{"id": f"i{i}", "file": "f", "stats": {**stats, "L_p50": float(i)}, "license": "cc0",
                     "license_version": "", "creator": f"c{i}", "title": "", "source": "", "landing_url": "",
                     "attribution": "", "photo_score": 0.01 if i in (1, 2) else 0.9} for i in range(6)]
    p.save()
    monkeypatch.setattr(ns, "contact_sheets", lambda *a, **k: [])
    q = ns.pick("art", [1], n_refs=4)                      # candidates 2 and 3 (index 1, 2) look like art
    assert 1 not in q.refs and 2 not in q.refs
    q = ns.restore("art", [2])                              # the owner keeps number 2 (index 1)
    assert 1 in q.refs and 2 not in q.refs
    assert ns.status("art")["likely_digital_art"] == [3]


def test_pack_publishable_only_for_open_training(tmp_path, monkeypatch):
    """Only packs trained on openly licensed photos alone are marked publishable (hosted Studio)."""
    import json

    from photostyle import newstyle as ns
    from photostyle.features import FEATURE_DIM
    from photostyle.head import Head

    monkeypatch.setattr(ns, "ROOT", tmp_path / "styles")
    monkeypatch.setattr(ns, "MANIFESTS", tmp_path / "manifests")
    r = GlobalRenderer("per_channel")
    cases = {("unpaired/strong", "strong", False): False, ("unpaired/strong", "strong", True): True,
             ("unpaired/gentle", "gentle", False): True, ("paired", "paired", False): False}
    for i, ((mode, recipe, open_only), want) in enumerate(cases.items()):
        p = ns.new(f"s{i}", "x")
        torch.save({"kind": "head", "state_dict": Head(FEATURE_DIM, r.num_params).state_dict(), "renderer": "per_channel",
                    "feats": torch.randn(4, FEATURE_DIM)}, p.dir / "head.pt")
        p.train = {"mode": mode, "recipe": recipe, "open_only": open_only}
        p.save()
        ns.pack(f"s{i}", out_root=tmp_path / "packs")
        assert json.loads((tmp_path / "packs" / f"s{i}" / "style.json").read_text())["publishable"] is want


def test_web_export_matches_torch_head(tmp_path):
    """tools/export_web.py: only publishable packs are exported, and the head maths the browser
    runs (numpy mirror of studio/web/src/lib/browser.js) reproduces the PyTorch head."""
    import base64
    import json
    import math

    from photostyle.engine import save_stylepack
    from photostyle.features import FEATURE_DIM
    from photostyle.head import Head
    from tools.export_web import export_styles

    r = GlobalRenderer("per_channel")
    torch.manual_seed(0)
    head = Head(FEATURE_DIM, r.num_params).eval()
    for prm in head.parameters():
        torch.nn.init.normal_(prm, std=0.05)
    feats = torch.randn(8, FEATURE_DIM)
    head.set_norm(feats)
    save_stylepack(tmp_path / "p" / "open", "open", head, r, feats, {"publishable": True})
    save_stylepack(tmp_path / "p" / "closed", "closed", head, r, feats, {})
    export_styles([tmp_path / "p"], tmp_path / "styles.json")
    data = json.loads((tmp_path / "styles.json").read_text())
    assert [s["card"]["name"] for s in data["styles"]] == ["open"]
    dt = "<f2" if data.get("dtype") == "float16" else "<f4"
    H = {k: (np.frombuffer(base64.b64decode(v), dt).astype(np.float64) if isinstance(v, str) else v)
         for k, v in data["styles"][0]["head"].items()}
    f = torch.randn(FEATURE_DIM)
    z = (f.numpy() - H["mu"]) / H["sigma"]
    h = H["w1"].reshape(H["hidden"], -1) @ z + H["b1"]
    h = (h - h.mean()) / np.sqrt(h.var() + 1e-5) * H["ln_w"] + H["ln_b"]
    h = 0.5 * h * (1 + np.vectorize(math.erf)(h / math.sqrt(2)))
    out = H["w2"].reshape(H["out"], -1) @ h + H["b2"]
    theta = H["theta_mean"] + H["alpha"] * (out - H["theta_mean"])
    with torch.no_grad():
        want = head(f[None])[0].numpy()
    assert np.allclose(theta, want, atol=2e-3)       # float16 weights


def test_cyberpunk_recipe_split_tones():
    """The tutorial recipe: blue shadows, pink highlights, greens pulled toward cyan."""
    from photostyle.recipes import _selfcheck, cyberpunk

    _selfcheck()
    ramp = torch.linspace(0.05, 0.95, 64).view(1, 1, 64).expand(3, 8, 64).contiguous()
    out = cyberpunk(ramp)
    dark, light = out[:, :, 4].mean(1), out[:, :, 60].mean(1)
    assert dark[2] > dark[0]                       # shadows lean blue
    assert light[0] > light[1] and light[2] > light[1]   # highlights lean magenta/pink
    green = torch.tensor([0.25, 0.6, 0.2]).view(3, 1, 1).expand(3, 4, 4)
    g = cyberpunk(green)[:, 2, 2]
    assert g[2] - g[0] > 0.2 - 0.25                # blue gains on red: green shifts toward cyan
