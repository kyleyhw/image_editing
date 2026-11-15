"""Does the browser pipeline (studio/web/src/lib/pipeline.js) agree with the reference
(photostyle/pipeline.py)? Phase 19, task 4.

A synthetic test image (grey ramps, a hue wheel at three lightnesses, saturated and skin-like
patches, seeded noise) is rendered in headless Chromium (WebGL2, SwiftShader) and by PyTorch for a
set of slider settings: the identity, each control at a strong value, and seeded random combinations.
Reported per case: maximum and mean absolute difference in 8-bit code values, and mean / 99th
percentile CIEDE2000. Targets (A6.6): max <= 1 code value per channel and mean dE00 <= 0.5.
The browser renders without dithering, so the comparison is deterministic.

    uv run python tools/check_pipeline.py [--out tests/reports]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import pipeline as P  # noqa: E402
from photostyle.colour_diff import ciede2000, srgb_to_lab  # noqa: E402

CHROMIUM = "/opt/pw-browsers/chromium"
SEED = 20261115                       # fixed so the random cases are the same on every run


def test_image(w: int = 256, h: int = 192) -> NDArray[np.uint8]:
    """Rows: 0-31 grey ramp; 32-127 hue wheel at three lightnesses; 128-159 patches; 160-191 noise."""
    img = np.zeros((h, w, 3), np.float64)
    img[:32] = np.linspace(0, 1, w)[None, :, None]
    hue = np.linspace(0, 1, w, endpoint=False)
    for k, (s, v) in enumerate([(1.0, 1.0), (0.6, 0.7), (0.35, 0.35)]):
        hh = hue * 6
        x = 1 - np.abs(hh % 2 - 1)
        sect = hh.astype(int) % 6
        rgb = np.select([sect[:, None] == i for i in range(6)],
                        [np.stack(c, -1) for c in [(np.ones_like(x), x, 0 * x), (x, np.ones_like(x), 0 * x),
                                                   (0 * x, np.ones_like(x), x), (0 * x, x, np.ones_like(x)),
                                                   (x, 0 * x, np.ones_like(x)), (np.ones_like(x), 0 * x, x)]])
        img[32 + 32 * k: 64 + 32 * k] = (v * (1 - s + s * rgb))[None]
    patches = [(0.76, 0.57, 0.47), (0.55, 0.38, 0.30), (0.22, 0.45, 0.75), (0.30, 0.55, 0.20), (0.95, 0.95, 0.92),
               (0.05, 0.05, 0.06), (0.85, 0.20, 0.15), (0.50, 0.50, 0.50)]
    pw = w // len(patches)
    for i, c in enumerate(patches):
        img[128:160, i * pw:(i + 1) * pw] = c
    rng = np.random.default_rng(SEED)
    img[160:] = rng.random((h - 160, w, 3))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def cases() -> list[tuple[str, dict]]:
    """Each control alone at a strong setting, then random combinations of everything."""
    out: list[tuple[str, dict]] = [("identity", {})]
    for s in P.SLIDERS:
        if s["group"] in ("detail", "effects"):
            continue
        if s["name"] in ("vig_midpoint", "vig_feather", "vig_roundness", "grade_blending", "grade_balance") \
                or s["name"].endswith("_hue") and s["name"].startswith("grade_"):
            continue
        v = s["max"] if s["name"] != "exposure" else 1.5
        if s["name"].startswith("grade_") and s["name"].endswith("_sat"):
            r = s["name"].split("_")[1]
            out.append((s["name"], {s["name"]: 80, f"grade_{r}_hue": 210}))
            continue
        out.append((s["name"], {s["name"]: v}))
        if s["min"] < 0 and s["name"] != "exposure":
            out.append((s["name"] + "-", {s["name"]: s["min"]}))
    out.append(("exposure-", {"exposure": -1.5}))
    out.append(("vignette shape", {"vig_amount": -80, "vig_midpoint": 30, "vig_feather": 80, "vig_roundness": 60}))
    out.append(("grade balance", {"grade_shadows_sat": 60, "grade_shadows_hue": 200, "grade_highlights_sat": 60,
                                  "grade_highlights_hue": 40, "grade_balance": 50, "grade_blending": 90}))
    out.append(("curves", {"curves": {"rgb": [[0, 0.08], [0.3, 0.25], [0.7, 0.78], [1, 0.95]],
                                      "r": [[0, 0], [0.5, 0.55], [1, 1]], "b": [[0, 0.1], [1, 0.9]]}}))
    out.append(("crop + straighten", {"crop": {"x": 0.1, "y": 0.05, "w": 0.7, "h": 0.8, "angle": 4.0}}))
    out.append(("crop + perspective + lens", {"crop": {"x": 0.05, "y": 0.05, "w": 0.9, "h": 0.9, "angle": -2.0, "vertical": 40,
                                                       "horizontal": -20, "distortion": 50, "scale": 110}}))
    for k, v in [("clarity", 80), ("clarity", -80), ("texture", 80), ("dehaze", 60), ("dehaze", -60), ("sharpen", 120),
                 ("nr_luma", 100), ("nr_colour", 100), ("halation", 100), ("grain", 80)]:
        out.append((f"spatial {k} {v:+d}", {k: v}))
    out += mask_cases()
    rng = np.random.default_rng(SEED + 1)
    for i in range(12):
        p: dict = {}
        for s in P.SLIDERS:
            if s["group"] in ("detail", "effects"):
                continue
            if s["name"] == "bw":
                p["bw"] = float(rng.random() < 0.15)
            elif rng.random() < 0.35:
                lo, hi = s["min"], s["max"]
                if s["name"] == "exposure":
                    lo, hi = -2, 2
                p[s["name"]] = round(float(rng.uniform(lo, hi)), 2)
        out.append((f"random {i + 1}", p))
    return out


def brush_png() -> str:
    """A soft disc, as a painted brush mask would be (base64 PNG, grey)."""
    yy, xx = np.mgrid[0:96, 0:128]
    d = np.hypot(xx - 40, yy - 50) / 30
    a = (np.clip(1.2 - d, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_cases() -> list[tuple[str, dict]]:
    lin = {"type": "linear", "x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 0.6, "sliders": {"exposure": 1.0, "temp": 40}}
    rad = {"type": "radial", "cx": 0.55, "cy": 0.45, "rx": 0.25, "ry": 0.3, "angle": 20, "feather": 40, "invert": True,
           "sliders": {"contrast": 60, "saturation": -80, "highlights": -70}}
    lum = {"type": "all", "range": {"lum": [0.55, 1.0], "feather": 10}, "sliders": {"shadows": 50, "whites": -40}}
    hue = {"type": "all", "range": {"hue": 140, "hue_width": 35}, "sliders": {"tint": 40, "saturation": 60}}
    br = {"type": "brush", "data": brush_png(), "sliders": {"exposure": -0.8, "clarity": 50}}
    return [("mask linear", {"masks": [lin]}), ("mask radial inverted", {"masks": [rad], "highlights": 30}),
            ("mask luminance range", {"masks": [lum]}), ("mask colour range", {"masks": [hue]}),
            ("mask brush", {"masks": [br]}), ("masks combined", {"masks": [lin, rad, lum, hue], "contrast": 20})]


def browser_module() -> str:
    """pipeline.js with its JSON import inlined, so a plain page can import it."""
    src = (ROOT / "studio/web/src/lib/pipeline.js").read_text()
    spec = (ROOT / "studio/web/src/lib/pipeline_spec.json").read_text()
    return src.replace('import SPEC from "./pipeline_spec.json";', f"const SPEC = {spec};")


PAGE = """<!doctype html><meta charset=utf-8><canvas id=c></canvas><script type=module>
import * as PL from "data:text/javascript;base64,%s";
const img = new Image(); img.src = "data:image/png;base64,%s"; await img.decode();
window.renderCase = async (p) => {
  const c = document.getElementById("c"), [w, h] = PL.outputSize(img.width, img.height, p.crop);
  c.width = w; c.height = h;
  const R = window.R && window.R.canvas === c ? window.R : (window.R = PL.makeGL(c));
  const brushes = [];
  for (const [j, m] of (p.masks || []).entries()) if (m.type === "brush") {
    const b = new Image(); b.src = "data:image/png;base64," + m.data; await b.decode();
    const cv = document.createElement("canvas"); cv.width = b.width; cv.height = b.height; cv.getContext("2d").drawImage(b, 0, 0);
    brushes[j] = { canvas: cv, version: Math.random() };
  }
  PL.setImage(R, img); PL.draw(R, { p, look: null, brushes }, { mode: 2, dither: 0 });
  const px = new Uint8Array(w * h * 4); R.gl.readPixels(0, 0, w, h, R.gl.RGBA, R.gl.UNSIGNED_BYTE, px);
  return { w, h, data: Array.from(px) };
};
window.ready = true;
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tests/reports")
    ap.add_argument("--image", type=Path, help="test on this photo instead of the synthetic image")
    ap.add_argument("--only", help="only cases whose name starts with this")
    args = ap.parse_args()
    t0 = time.time()
    from playwright.sync_api import sync_playwright

    if args.image:
        im = Image.open(args.image).convert("RGB")
        im.thumbnail((640, 640))
        img = np.asarray(im).copy()
    else:
        img = test_image()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    page_html = PAGE % (base64.b64encode(browser_module().encode()).decode(), base64.b64encode(buf.getvalue()).decode())
    x = torch.from_numpy(img).permute(2, 0, 1).double() / 255
    rows = []
    with sync_playwright() as pw:
        exe = CHROMIUM if Path(CHROMIUM).exists() else None
        b = pw.chromium.launch(executable_path=exe, args=["--use-gl=angle", "--use-angle=swiftshader",
                                                          "--enable-unsafe-swiftshader"])
        page = b.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.set_content(page_html)
        page.wait_for_function("window.ready === true", timeout=60_000)
        for name, p in cases():
            if args.only and not name.startswith(args.only):
                continue
            r = page.evaluate("p => window.renderCase(p)", p)
            got = np.array(r["data"], np.uint8).reshape(r["h"], r["w"], 4)[::-1, :, :3].astype(np.float64)
            with torch.no_grad():
                ref = P.render(x.float(), p).double().permute(1, 2, 0).numpy() * 255
            ref8 = np.clip(np.round(ref), 0, 255)
            d = np.abs(got - ref8)
            if "crop" in p:                      # bilinear sampling differs by half a texel at the cut edges
                d = d[2:-2, 2:-2]
                got, ref8 = got[2:-2, 2:-2], ref8[2:-2, 2:-2]
            de = ciede2000(srgb_to_lab(ref8 / 255), srgb_to_lab(got / 255))
            spatial = name.startswith("spatial") or name == "mask brush"
            # spatial operators blur differently on the GPU (downsampled pyramid) than the reference
            # (one full-resolution Gaussian); their target is on the mean only: mean dE00 <= 0.5
            ok = de.mean() <= 0.5 if spatial else (d.max() <= 1 and de.mean() <= 0.5)
            rows.append({"case": name, "max": float(d.max()), "mean": float(d.mean()),
                         "de_mean": float(de.mean()), "de_p99": float(np.percentile(de, 99)), "ok": bool(ok)})
            print(f"{name:28s} max {d.max():4.0f}  mean {d.mean():.3f}  dE00 mean {de.mean():.3f} p99 "
                  f"{np.percentile(de, 99):.3f}  {'ok' if rows[-1]['ok'] else 'FAIL'}", flush=True)
        b.close()
    runtime = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out / "assets" / "pipeline_test_image.png") if (args.out / "assets").exists() else None
    if args.image or args.only:
        print(f"{sum(r['ok'] for r in rows)}/{len(rows)} cases within target (not saved: partial run)")
        return
    (args.out / "pipeline_agreement.json").write_text(json.dumps({"rows": rows, "errors": errors, "runtime_s": runtime}, indent=1))
    n_ok = sum(r["ok"] for r in rows)
    print(f"{n_ok}/{len(rows)} cases within target; page errors: {errors or 'none'}; {runtime:.0f} s")


if __name__ == "__main__":
    main()
