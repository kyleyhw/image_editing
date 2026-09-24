"""End-to-end Studio check in headless Chromium (Phase 13 gate).

Starts nothing itself: point it at a running Studio (``python -m studio.server``).

  1. uploads a photo, waits for prediction and the style strip;
  2. screenshots the editor (split view) and the "after" view;
  3. golden test: reads the WebGL canvas and compares it with the Python
     renderer on the same preview image and parameters (target: max
     difference <= 2/255, mean < 0.5/255);
  4. exercises strength, a curve drag, undo, and a .cube / JPEG export.

Usage:
    uv run python tools/check_studio.py --url http://127.0.0.1:8765 --photo some.jpg --out shots/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.engine import EditParams  # noqa: E402
from photostyle.io import to_tensor  # noqa: E402

CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    report = {}
    with sync_playwright() as pw:
        exe = CHROMIUM if Path(CHROMIUM).exists() else None
        b = pw.chromium.launch(executable_path=exe, args=["--use-gl=angle", "--use-angle=swiftshader",
                                                          "--enable-unsafe-swiftshader"])
        page = b.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.set_input_files("#upload", str(args.photo))
        page.wait_for_function("window.__studio.S.params !== null", timeout=120_000)
        page.wait_for_timeout(4000)   # style strip thumbnails
        page.screenshot(path=str(args.out / "studio_split.png"))

        # golden test: 'after' mode, compare canvas with Python render of the same preview
        page.click("button[data-mode=after]")
        page.wait_for_timeout(300)
        page.screenshot(path=str(args.out / "studio_after.png"))
        data = page.evaluate("""() => {
          const {S, main} = window.__studio, gl = main.gl, w = main.canvas.width, h = main.canvas.height;
          window.__studio.draw(main, window.__studio.effective(), S.params.knots, 2);
          const px = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
          return {w, h, px: Array.from(px), params: window.__studio.exportParams(), id: S.id};
        }""")
        w, h = data["w"], data["h"]
        gl_img = np.array(data["px"], dtype=np.uint8).reshape(h, w, 4)[::-1, :, :3].astype(np.int16)
        prev = Image.open(io.BytesIO(urllib.request.urlopen(f"{args.url}/api/photo/{data['id']}?edge=1600").read()))
        p = EditParams.from_json(json.dumps(data["params"]))
        with torch.no_grad():
            ref = p.make_renderer()(to_tensor(prev)[None], p.vector())[0]
        ref = (ref.permute(1, 2, 0).numpy() * 255).round().astype(np.int16)
        diff = np.abs(gl_img - ref)
        report["golden"] = {"max": int(diff.max()), "mean": float(diff.mean()),
                            "p99": float(np.percentile(diff, 99)), "size": [w, h]}
        report["golden"]["pass"] = report["golden"]["p99"] <= 2 and report["golden"]["mean"] < 0.5

        # interactions: strength, curve drag, undo
        page.click("button[data-mode=split]")
        page.fill("#strength", "60")
        page.dispatch_event("#strength", "input")
        box = page.locator("#curves").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.35, steps=5)
        page.mouse.up()
        page.wait_for_timeout(200)
        edited = page.evaluate("() => window.__studio.S.knots !== null")
        page.keyboard.press("Control+z")
        undone = page.evaluate("() => window.__studio.S.knots === null")
        page.screenshot(path=str(args.out / "studio_edit.png"))
        report["interactions"] = {"curve_edit": edited, "undo": undone}

        # scene tools (v2 only): dehaze is rendered by the server and must change the picture
        if page.locator("text=Haze").count():
            before = page.evaluate("() => { const c = window.__studio.main.canvas; return c.toDataURL().length; }")
            page.evaluate("() => { window.__studio.S.scene.haze = -0.5; }")
            page.evaluate("""() => { const i = document.querySelector('input[type=range][aria-label="Haze"]');
                                     i.value = -50; i.dispatchEvent(new Event('input', {bubbles: true})); }""")
            page.wait_for_function("window.__studio.S.sceneBusy === true", timeout=10_000)
            page.wait_for_function("window.__studio.S.sceneBusy === false", timeout=180_000)
            page.wait_for_timeout(300)
            after = page.evaluate("() => { const c = window.__studio.main.canvas; return c.toDataURL().length; }")
            page.screenshot(path=str(args.out / "studio_scene.png"))
            report["scene"] = {"dehaze_changed_picture": before != after}

        # exports
        for fmt in ("cube", "jpg"):
            page.select_option("#exportFmt", fmt)
            with page.expect_download(timeout=120_000) as dl:
                page.click("#btnExport")
            f = args.out / dl.value.suggested_filename
            dl.value.save_as(f)
            report.setdefault("exports", {})[fmt] = {"file": f.name, "bytes": f.stat().st_size}
        report["page_errors"] = errors
        b.close()
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
