"""Phase 22 task 28: does tiled export (pipeline.js renderTiled) agree with the one-pass render?

A real 24-megapixel photo (the CC0 Leica M10 DNG from raw.pixls.us, developed to 8-bit sRGB with
rawpy) is rendered in headless Chromium twice: in one pass (renderPixels) and in tiles
(renderTiled, small tiles so that there are many seams). Both use 16-bit output, which has no
dithering, so the comparison is deterministic. Cases: per-pixel controls only (expected exact up to
float rounding), the spatial operators (the coarse means come from a 2048 px copy in the tiled
path, so small differences are expected), masks, geometry with rotation and perspective, grain.
Reported per case: max and mean absolute difference in 8-bit code values, mean and 99th
percentile CIEDE2000, and the largest difference along tile seams.

    uv run python tools/check_tiles.py [--raw FILE] [--tile 700]
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
from numpy.typing import NDArray
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle.colour_diff import ciede2000, srgb_to_lab  # noqa: E402
from tools.check_pipeline import CHROMIUM, browser_module  # noqa: E402

RAW = ROOT / "data/camera_looks/raw/Leica__M10__f5381888.dng"
BASE = {"exposure": 0.3, "contrast": 25, "highlights": -40, "shadows": 30, "vibrance": 20, "hsl_sat_blue": -20,
        "grade_shadows_hue": 210, "grade_shadows_sat": 15, "vig_amount": -30}
CASES: dict[str, dict] = {
    "per-pixel": BASE,
    "spatial": {**BASE, "texture": 30, "clarity": 30, "sharpen": 50, "nr_luma": 30, "nr_colour": 40, "dehaze": 25, "halation": 40},
    "masks": {**BASE, "masks": [{"type": "linear", "x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 0.5, "sliders": {"exposure": -0.6, "clarity": 30}},
                                {"type": "radial", "cx": 0.4, "cy": 0.6, "rx": 0.2, "ry": 0.25, "feather": 60, "sliders": {"shadows": 40}}]},
    "geometry": {**BASE, "crop": {"x": 0.1, "y": 0.08, "w": 0.8, "h": 0.84, "angle": 3.5, "vertical": 20, "distortion": 15, "scale": 100}},
    "grain": {**BASE, "grain": 40, "grain_size": 30},
}

PAGE = """<!doctype html><meta charset=utf-8><script type=module>
import * as PL from "data:text/javascript;base64,%s";
const img = new Image(); img.src = "data:image/png;base64,%s"; await img.decode();
const bmp = await createImageBitmap(img);
window.run = (p, tile) => {
  const e = { p, look: null, brushes: [], depth: null };
  const t0 = performance.now();
  const one = PL.renderPixels(bmp, e, { float: true });
  const t1 = performance.now();
  const til = PL.renderTiled(PL.bitmapSource(bmp), e, { float: true, tile });
  const t2 = performance.now();
  // 8-bit code values (float output * 255) as RGB, top row first, for transfer
  const q = (px, ch) => { const n = px.width * px.height, o = new Uint8Array(3 * n), s = px.data instanceof Uint16Array ? 255 / 65535 : 255;
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) o[3 * i + k] = Math.round(px.data[ch * i + k] * s); return o; };
  const b64 = (u) => { let s = ""; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode(...u.subarray(i, i + 32768)); return btoa(s); };
  return { w: one.width, h: one.height, w2: til.width, h2: til.height, one: b64(q(one, 4)), til: b64(q(til, til.channels ?? 4)),
           t_one: t1 - t0, t_til: t2 - t1 };
};
window.ready = true;
</script>"""


def develop(raw: Path) -> NDArray[np.uint8]:
    import rawpy
    with rawpy.imread(str(raw)) as r:
        return r.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--tile", type=int, default=700, help="small tiles: many seams to test")
    ap.add_argument("--out", type=Path, default=ROOT / "tests/reports/phase22_tiles.json")
    args = ap.parse_args()
    t0 = time.time()
    img = develop(args.raw)
    H, W = img.shape[:2]
    print(f"photo {W} x {H} ({W * H / 1e6:.1f} MP)", flush=True)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", compress_level=1)
    from playwright.sync_api import sync_playwright
    rows = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROMIUM if Path(CHROMIUM).exists() else None,
                               args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_content(PAGE % (base64.b64encode(browser_module().encode()).decode(), base64.b64encode(buf.getvalue()).decode()))
        page.wait_for_function("window.ready === true", timeout=600_000)
        for name, p in CASES.items():
            r = page.evaluate("([p, t]) => window.run(p, t)", [p, args.tile])
            assert (r["w"], r["h"]) == (r["w2"], r["h2"]), (r["w"], r["h"], r["w2"], r["h2"])
            w, h = r["w"], r["h"]
            a = np.frombuffer(base64.b64decode(r["one"]), np.uint8).reshape(h, w, 3).astype(np.float64)
            t = np.frombuffer(base64.b64decode(r["til"]), np.uint8).reshape(h, w, 3).astype(np.float64)
            d = np.abs(a - t)
            step = 4                                   # CIEDE2000 on every 4th pixel each way (6 % of a 24 MP image)
            de = ciede2000(srgb_to_lab(a[::step, ::step] / 255), srgb_to_lab(t[::step, ::step] / 255))
            seams = [x for x in range(args.tile, w, args.tile)]
            seam_d = float(max([d[:, max(0, x - 2):x + 2].max() for x in seams] + [0]))
            rows.append({"case": name, "size": [w, h], "max": float(d.max()), "mean": float(d.mean()), "de_mean": float(de.mean()),
                         "de_p99": float(np.percentile(de, 99)), "seam_max": seam_d,
                         "one_pass_s": round(r["t_one"] / 1000, 1), "tiled_s": round(r["t_til"] / 1000, 1)})
            print(f"{name:10s} {w}x{h}  max {d.max():3.0f}  mean {d.mean():.4f}  dE00 mean {de.mean():.4f} p99 {np.percentile(de, 99):.3f}  "
                  f"seam max {seam_d:.0f}  one pass {r['t_one'] / 1000:.1f} s  tiled {r['t_til'] / 1000:.1f} s", flush=True)
        b.close()
    out = {"photo": [W, H], "tile": args.tile, "rows": rows, "page_errors": errors, "runtime_s": round(time.time() - t0, 1)}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"page errors: {errors or 'none'}; {out['runtime_s']} s")


if __name__ == "__main__":
    main()
