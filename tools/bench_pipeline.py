"""Phase 26 task 41: render time of the editor pipeline per operator group.

The browser pipeline (studio/web/src/lib/pipeline.js) is timed in headless Chromium at the editor's
preview size (1600 px long edge, browser.js PREVIEW_EDGE) and at a 12-megapixel export size, for
groups of settings that switch on progressively more passes. Each timing is the median of REPEATS
draws, each followed by gl.finish() so the GPU work is included. The reference renderer
(photostyle/pipeline.py, PyTorch on the CPU) is timed at the preview size for comparison.

The test machine has no GPU: Chromium renders WebGL2 with SwiftShader, a CPU emulation. The
absolute times are therefore an upper bound, far slower than a phone or laptop GPU; the ratios
between groups show which passes cost most.

    uv run python tools/bench_pipeline.py
"""

from __future__ import annotations

import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import pipeline as P  # noqa: E402
from tools.check_pipeline import CHROMIUM, browser_module  # noqa: E402

SEED = 26                                  # any fixed seed: the content does not affect the timing
REPEATS = 5                                # median of 5 draws; the first draw (shader compile) is discarded
SIZES = {"preview 1600 x 1067": (1600, 1067), "export 4000 x 3000 (12 MP)": (4000, 3000)}
GROUPS: dict[str, dict] = {
    "per-pixel (tone, curves, colour, grading)": {"exposure": 0.3, "contrast": 30, "highlights": -40, "shadows": 30,
                                                  "vibrance": 20, "hsl_sat_blue": 20, "grade_shadows_sat": 20,
                                                  "curves": {"rgb": [[0, 0.02], [0.5, 0.52], [1, 0.98]]}},
    "+ detail (texture, clarity, sharpen, NR)": {"texture": 20, "clarity": 20, "sharpen": 40, "nr_luma": 20, "nr_colour": 30},
    "+ dehaze, grain, halation": {"dehaze": 20, "grain": 30, "halation": 30},
    "+ two masks": {"masks": [{"type": "linear", "x0": 0.5, "y0": 0, "x1": 0.5, "y1": 0.5, "sliders": {"exposure": -0.5}},
                              {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.3, "ry": 0.3, "feather": 50,
                               "sliders": {"clarity": 30}}]},
}

PAGE = """<!doctype html><meta charset=utf-8><canvas id=c></canvas><script type=module>
import * as PL from "data:text/javascript;base64,%s";
const img = new Image(); img.src = "data:image/png;base64,%s"; await img.decode();
const c = document.getElementById("c"); c.width = img.width; c.height = img.height;
const R = PL.makeGL(c); PL.setImage(R, img);
window.time = (p, n) => {
  const t = [];
  for (let i = 0; i <= n; i++) {
    const t0 = performance.now();
    PL.draw(R, { p, look: null, brushes: [], depth: null }, { mode: 2, dither: 0 }); R.gl.finish();
    t.push(performance.now() - t0);
  }
  return t.slice(1);
};
window.ready = true;
</script>"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    t_start = time.time()
    rng = np.random.default_rng(SEED)
    cumulative: list[tuple[str, dict]] = []
    acc: dict = {}
    for name, g in GROUPS.items():
        acc = {**acc, **g}
        cumulative.append((name, acc))
    rows = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROMIUM if Path(CHROMIUM).exists() else None,
                               args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for size_name, (w, h) in SIZES.items():
            img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "PNG")
            page = b.new_page()
            page.set_content(PAGE % (base64.b64encode(browser_module().encode()).decode(), base64.b64encode(buf.getvalue()).decode()))
            page.wait_for_function("window.ready === true", timeout=300_000)
            for name, p in cumulative:
                t = page.evaluate("([p, n]) => window.time(p, n)", [p, REPEATS])
                rows.append({"size": size_name, "group": name, "browser_ms": round(statistics.median(t), 1)})
                print(f"{size_name:28s} {name:42s} {rows[-1]['browser_ms']:8.1f} ms", flush=True)
            page.close()
        b.close()
    w, h = SIZES["preview 1600 x 1067"]
    x = torch.from_numpy((rng.random((h, w, 3))).astype(np.float32)).permute(2, 0, 1)
    for name, p in cumulative:
        with torch.no_grad():
            P.render(x, p)
            t0 = time.time()
            P.render(x, p)
        ms = (time.time() - t0) * 1000
        for r in rows:
            if r["size"].startswith("preview") and r["group"] == name:
                r["reference_cpu_ms"] = round(ms, 1)
        print(f"reference preview {name:42s} {ms:8.1f} ms", flush=True)
    out = {"rows": rows, "repeats": REPEATS, "renderer": "SwiftShader (CPU emulation of WebGL2)", "threads": torch.get_num_threads(),
           "runtime_s": round(time.time() - t_start, 1)}
    (ROOT / "tests/reports/phase26_bench.json").write_text(json.dumps(out, indent=1))
    print(f"done in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
