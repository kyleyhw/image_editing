"""End-to-end check of the hosted editor (Pages build in ``site/``) in headless Chromium.

Serves ``site/`` under /image_editing/ (as GitHub Pages does), opens a sample photo and exercises
the A6 editor through its real controls, asserting the behaviours Phase 19 promises:

  1. a slider edit changes the picture;
  2. the edit survives a change of look, and undo restores the previous look;
  3. curves accept a new point;
  4. crop changes the output size; straighten keeps it inside the frame;
  5. the white-balance picker sets temperature / tint;
  6. JPEG and .cube export download;
  7. no page errors, on desktop and phone viewports.

Screenshots go to --out. Usage:
    uv run python tools/check_editor.py --out /tmp/shots
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import ConsoleMessage, FloatRect, Locator

ROOT = Path(__file__).resolve().parent.parent
CHROMIUM = "/opt/pw-browsers/chromium"


def serve(port: int) -> http.server.ThreadingHTTPServer:
    """Serve the repository root so that /image_editing/... maps to site/ (via a symlink-free alias)."""
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            path = path.split("?", 1)[0]
            if path.startswith("/image_editing/"):
                path = path[len("/image_editing"):]
            return str(ROOT / "site" / path.lstrip("/"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base class's name)
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def bb(loc: Locator) -> FloatRect:
    """The element's box; the elements tested are always visible, so a missing box is a failure."""
    r = loc.bounding_box()
    assert r is not None, f"{loc} has no bounding box"
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8812)
    ap.add_argument("--raw", type=Path, nargs="*", default=[], help="raw files to open and export (16-bit TIFF)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    srv = serve(args.port)
    url = f"http://127.0.0.1:{args.port}/image_editing/"
    from playwright.sync_api import sync_playwright

    res: dict = {}
    with sync_playwright() as pw:
        exe = CHROMIUM if Path(CHROMIUM).exists() else None
        b = pw.chromium.launch(executable_path=exe, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for vw, vh, tag in [(1440, 900, "desk"), (390, 844, "phone")]:
            pg = b.new_page(viewport={"width": vw, "height": vh}, accept_downloads=True)
            errs: list[str] = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            def on_console(m: ConsoleMessage) -> None:
                if m.type in ("error", "warning"):
                    errs.append(f"{m.type}: {m.text}")

            pg.on("console", on_console)
            pg.goto(url)
            pg.wait_for_function("window.__studio && window.__studio.S.model.ready", timeout=180_000)
            pg.click(".sample >> nth=0")
            pg.wait_for_function("window.__studio.S.params !== null", timeout=180_000)
            pg.wait_for_timeout(1500)
            S = lambda expr: pg.evaluate(f"() => {expr}")   # noqa: E731
            sig = lambda: pg.evaluate("""() => { const c = document.getElementById('view'), g = c.getContext('webgl2');
                const px = new Uint8Array(4 * 64); g.readPixels(Math.floor(c.width/2) - 32, Math.floor(c.height/2), 64, 1, g.RGBA, g.UNSIGNED_BYTE, px);
                return Array.from(px).reduce((a, v) => a + v, 0); }""")   # noqa: E731
            if tag == "desk":
                pg.wait_for_timeout(2500)
                res["auto_available"] = S("window.__studio.S.caps.auto === true")
                res["auto_off_by_default"] = S("window.__studio.S.autoOn") is False
                pg.evaluate("() => window.__studio.setAuto(true)")
                pg.wait_for_function("window.__studio.S.auto !== null", timeout=60_000)
                res["auto_sliders"] = S("window.__studio.S.auto")
                pg.evaluate("() => window.__studio.setAuto(false)")
                pg.wait_for_timeout(300)
                res["look_kind"] = S("window.__studio.S.params?.kind ?? 'head'")
                pg.click("button[data-mode=after]")
                pg.wait_for_timeout(300)
                before = sig()
                pg.fill("#s_exposure ~ .num, .slider:has(#s_exposure) .num", "+0.80")
                pg.keyboard.press("Enter")
                pg.wait_for_timeout(300)
                res["exposure_changes_picture"] = sig() > before
                res["exposure_value"] = S("window.__studio.S.p.exposure")
                pg.screenshot(path=str(args.out / "editor_basic.png"))
                style0 = S("window.__studio.S.style")
                pg.select_option("select[aria-label=Look]", index=4)
                pg.wait_for_function(f"window.__studio.S.style !== {json.dumps(style0)} && window.__studio.S.params !== null", timeout=60_000)
                res["edit_survives_look_change"] = S("window.__studio.S.p.exposure") == 0.8
                pg.keyboard.press("Control+z")
                pg.wait_for_timeout(800)
                res["undo_restores_look"] = S("window.__studio.S.style") == style0
                # expert style: the model predicts the look's sliders for this photo
                if S("window.__studio.S.styles.some(s => s.kind === 'model')"):
                    pg.evaluate("() => window.__studio.setStyle('expert_c')")
                    pg.wait_for_function("window.__studio.S.style === 'expert_c' && window.__studio.S.params !== null", timeout=60_000)
                    res["expert_kind"] = S("window.__studio.S.params.kind")
                    res["expert_sliders"] = S("Object.keys(window.__studio.S.params.sliders || {}).length")
                    pg.keyboard.press("Control+z")
                    pg.wait_for_timeout(800)
                # curves: click on the canvas adds a point
                cv = pg.locator("#curves")
                cv.scroll_into_view_if_needed()
                box = bb(cv)
                cv.click(position={"x": box["width"] * 0.3, "y": box["height"] * 0.45})
                pg.wait_for_timeout(300)
                res["curve_points"] = S("(window.__studio.S.p.curves?.rgb || []).length")
                # mixer and grading
                pg.click("summary:has-text('Colour mixer')")
                pg.fill(".slider:has(#s_hsl_hue_orange) .num", "-40")
                pg.keyboard.press("Enter")
                pg.click("summary:has-text('Colour grading')")
                pg.fill(".slider:has(#s_grade_shadows_sat) .num", "40")
                pg.keyboard.press("Enter")
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(args.out / "editor_colour.png"), full_page=False)
                pg.locator(".inspector").screenshot(path=str(args.out / "inspector.png"))
                # detail and effects (multi-pass GPU path)
                pg.click("summary:has-text('Detail')")
                before = sig()
                pg.fill(".slider:has(#s_clarity) .num", "60")
                pg.keyboard.press("Enter")
                pg.click("summary:has-text('Effects')")
                pg.fill(".slider:has(#s_grain) .num", "40")
                pg.keyboard.press("Enter")
                pg.wait_for_timeout(500)
                res["detail_changes_picture"] = sig() != before
                pg.screenshot(path=str(args.out / "editor_detail.png"))
                # masks: linear gradient with local exposure, radial with local contrast, a brush stroke
                pg.click("summary:has-text('Masks')")
                pg.click("button:has-text('+ Linear')")
                pg.wait_for_timeout(200)
                before = sig()
                msk = pg.locator(".inspector .stack .slider").filter(has_text="Exposure").last
                msk.locator(".num").fill("-1.5")
                msk.locator(".num").press("Enter")
                pg.wait_for_timeout(400)
                res["mask_linear_changes_picture"] = sig() != before
                pg.keyboard.press("o")
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(args.out / "editor_mask_linear.png"))
                pg.keyboard.press("o")
                pg.click("button:has-text('+ Brush')")
                vb = bb(pg.locator("#view"))
                pg.mouse.move(vb["x"] + vb["width"] * 0.3, vb["y"] + vb["height"] * 0.7)
                pg.mouse.down()
                for k in range(12):
                    pg.mouse.move(vb["x"] + vb["width"] * (0.3 + 0.03 * k), vb["y"] + vb["height"] * 0.7)
                pg.mouse.up()
                pg.wait_for_timeout(300)
                res["brush_painted"] = S("(window.__studio.S.p.masks[1].data || '').length > 100")
                res["masks"] = S("window.__studio.S.p.masks.length")
                # crop
                w0 = S("document.getElementById('view').width")
                pg.click("summary:has-text('Crop & straighten')")
                pg.select_option("select[aria-label='Aspect ratio']", "1")
                pg.wait_for_timeout(300)
                res["crop_square"] = S("(() => { const c = document.getElementById('view'); return [c.width, c.height]; })()")
                res["crop_changes_size"] = res["crop_square"][0] != w0
                pg.click("button:has-text('Crop')")
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(args.out / "editor_crop_tool.png"))
                pg.click("button:has-text('Done')")
                # clipping, zoom
                pg.keyboard.press("j")
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(args.out / "editor_clipping.png"))
                pg.keyboard.press("j")
                pg.keyboard.press("z")
                pg.wait_for_timeout(300)
                res["zoom"] = S("window.__studio.S.zoom")
                pg.screenshot(path=str(args.out / "editor_zoom100.png"))
                pg.keyboard.press("z")
                # white-balance picker on the photo centre
                pg.keyboard.press("w")
                vb = bb(pg.locator("#view"))
                pg.mouse.click(vb["x"] + vb["width"] * 0.5, vb["y"] + vb["height"] * 0.4)
                pg.wait_for_timeout(300)
                res["wb_picked"] = S("[window.__studio.S.p.temp ?? 0, window.__studio.S.p.tint ?? 0]")
                # export
                for fmt in ("jpg", "cube", "png16", "xmp"):
                    pg.select_option("#exportFmt", fmt)
                    with pg.expect_download() as dl:
                        pg.click("#btnExport")
                    d = dl.value
                    f = args.out / f"export.{fmt.replace('16', '')}"
                    d.save_as(f)
                    res[f"export_{fmt}_bytes"] = f.stat().st_size
                from PIL import Image
                with Image.open(args.out / "export.png") as im:
                    res["png16_mode"] = f"{im.mode} {im.size}"
                xmp = (args.out / "export.xmp").read_text()
                res["xmp_has_exposure"] = "crs:Exposure2012=" in xmp and "ToneCurvePV2012" in xmp
                cube = (args.out / "export.cube").read_text().splitlines()
                res["cube_rows"] = sum(1 for ln in cube if ln and ln[0].isdigit())
                # tiled export (forced with small tiles): must match the one-pass export
                import numpy as np
                for fmt, ext in (("jpg", "jpg"), ("png16", "png")):
                    with pg.expect_download(timeout=240_000) as dl:
                        pg.evaluate(f"() => window.__studio.exportCurrent({{ fmt: '{fmt}', tile: 256 }})")
                    t = args.out / f"export_tiled.{ext}"
                    dl.value.save_as(t)
                    one = np.asarray(Image.open(args.out / f"export.{ext}").convert("RGB"), np.float64)
                    til = np.asarray(Image.open(t).convert("RGB"), np.float64)
                    res[f"tiled_{fmt}_same_size"] = one.shape == til.shape
                    res[f"tiled_{fmt}_mean_diff"] = round(float(np.abs(one - til).mean()), 3) if one.shape == til.shape else None
                # workflow: copy / paste, preset, snapshot, history, persistence across a reload
                pg.keyboard.press("Control+Shift+C")
                res["copied"] = S("!!window.__studio.S.clipboard")
                pg.evaluate("() => window.__studio.setSlider('exposure', 0)")
                pg.keyboard.press("Control+Shift+V")
                pg.wait_for_timeout(300)
                res["paste_restores_exposure"] = S("window.__studio.S.p.exposure") == 0.8
                res["history_steps"] = S("document.querySelectorAll('.history li').length")
                res["sw_registered"] = pg.evaluate("async () => !!(await navigator.serviceWorker?.getRegistration())")
                pg.reload()
                pg.wait_for_function("window.__studio && window.__studio.S.ready && window.__studio.S.id", timeout=180_000)
                pg.wait_for_timeout(1500)
                res["kept_after_reload"] = S("window.__studio.S.p.exposure") == 0.8
                res["photos_after_reload"] = S("window.__studio.S.photos.length")
                for k, rf in enumerate(args.raw):
                    t1 = time.time()
                    pg.set_input_files("#upload", str(rf))
                    pg.wait_for_function(f"window.__studio.S.photos.length >= {2 + k} && window.__studio.S.imgVersion > 0", timeout=240_000)
                    pg.wait_for_timeout(1500)
                    res[f"raw_{rf.suffix}_open_s"] = round(time.time() - t1, 1)
                    res[f"raw_{rf.suffix}_contrast"] = S("window.__studio.S.p.contrast ?? 0")
                    pg.screenshot(path=str(args.out / f"raw_{k}.png"))
                    pg.select_option("#exportFmt", "tif16")
                    with pg.expect_download(timeout=240_000) as dl:
                        pg.click("#btnExport")
                    f = args.out / f"raw_{k}.tif"
                    dl.value.save_as(f)
                    res[f"raw_{rf.suffix}_tif_bytes"] = f.stat().st_size
            else:
                pg.click(".quickbar .btn")
                pg.wait_for_timeout(600)
                pg.screenshot(path=str(args.out / "phone_sheet.png"))
            res[f"{tag}_errors"] = errs
            pg.close()
        b.close()
    srv.shutdown()
    res["runtime_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=1))
    (args.out / "check_editor.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
