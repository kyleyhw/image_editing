"""Record a narrated Studio walkthrough (video + storyboard) with Playwright.

Start Studio first (e.g. PHOTOSTYLE_STUDIO_DATA=/tmp/demo uv run photostyle serve), then:

    uv run python tools/demo_studio.py --photos a.jpg b.jpg ... --out demo/

Writes demo/studio_demo.webm and demo/storyboard.jpg (key frames with captions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
CAPTION_JS = """(t) => {
  let d = document.getElementById('__cap');
  if (!d) { d = document.createElement('div'); d.id = '__cap';
    Object.assign(d.style, {position: 'fixed', left: '50%', bottom: '140px', transform: 'translateX(-50%)',
      background: 'rgba(0,0,0,.78)', color: '#fff', padding: '10px 18px', borderRadius: '8px',
      font: '600 18px system-ui, sans-serif', zIndex: 99999, pointerEvents: 'none', maxWidth: '80vw', textAlign: 'center'});
    document.body.appendChild(d); }
  d.textContent = t; d.style.display = t ? 'block' : 'none';
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--photos", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene-photo", help="file name of the photo to show the scene tools on (default: the last)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[str, Path]] = []

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = b.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=str(args.out),
                            record_video_size={"width": 1280, "height": 800})
        page = ctx.new_page()

        def cap(text: str, wait: int = 1600, shot: bool = True) -> None:
            page.evaluate(CAPTION_JS, text)
            page.wait_for_timeout(wait)
            if shot and text:
                f = args.out / f"frame_{len(frames):02d}.png"
                page.screenshot(path=str(f))
                frames.append((text, f))

        def slider(label: str, value: int) -> None:
            page.evaluate("""([label, v]) => {
              const l = [...document.querySelectorAll('#panel label')].find(x => x.textContent.trim().startsWith(label));
              const i = l.querySelector('input'); i.value = v;
              i.dispatchEvent(new Event('input', {bubbles: true})); i.dispatchEvent(new Event('change', {bubbles: true}));
            }""", [label, value])

        page.goto(args.url)
        cap("photostyle Studio: everything runs on this computer", 1500)
        page.set_input_files("#upload", [str(x) for x in args.photos])
        page.wait_for_function("window.__studio.S.params !== null", timeout=180_000)
        page.wait_for_timeout(5000)
        cap("Upload photos → the first style is predicted for THIS photo")
        cap("The strip previews every style on your photo", 2200)

        for name in ("cyberpunk", "clean_cool", "fujifilm"):
            page.locator(".thumb", has_text=name).click()
            page.wait_for_timeout(1800)
            cap(f"Style: {name} (content-adaptive: the edit depends on the photo)", 1500)

        cap("Strength: 0 % is the original, 150 % pushes further")
        for v in (20, 60, 100, 140, 100):
            page.fill("#strength", str(v))
            page.dispatch_event("#strength", "input")
            page.wait_for_timeout(400)

        cap("Drag the divider to compare before / after")
        box = page.locator("#view").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.mouse.down()
        for fr in (0.3, 0.2, 0.7, 0.8, 0.5):
            page.mouse.move(box["x"] + box["width"] * fr, box["y"] + box["height"] * 0.5, steps=8)
        page.mouse.up()
        page.click("button[data-mode=after]")
        cap("After", 900, shot=False)
        page.click("button[data-mode=before]")
        cap("Before", 900, shot=False)
        page.click("button[data-mode=split]")

        cap("Curves stay editable: drag a knot (G channel)")
        page.click(".tab:has-text('G')")
        cb = page.locator("#curves").bounding_box()
        page.mouse.move(cb["x"] + cb["width"] * 0.5, cb["y"] + cb["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(cb["x"] + cb["width"] * 0.5, cb["y"] + cb["height"] * 0.38, steps=8)
        page.mouse.up()
        cap("A filled square marks what you changed. Ctrl+Z undoes it", 1800)
        page.keyboard.press("Control+z")

        # scene tools on the last (landscape) photo
        item = (page.locator(f".libitem[aria-label='{args.scene_photo}']") if args.scene_photo
                else page.locator(".libitem").last)
        item.click()
        page.wait_for_function("window.__studio.S.params !== null", timeout=60_000)
        page.wait_for_timeout(2500)
        cap("Scene tools: depth-aware haze and sky light (applied before the style)")
        page.evaluate("document.querySelector('#panel').scrollTo(0, 500)")
        slider("Haze", -60)
        page.wait_for_function("window.__studio.S.sceneBusy === false", timeout=180_000)
        page.wait_for_timeout(800)
        cap("Dehaze −60: the distance clears, the sky is left alone", 2000)
        slider("Sky exposure", 35)
        page.wait_for_function("window.__studio.S.sceneBusy === false", timeout=180_000)
        page.wait_for_timeout(800)
        cap("Sky exposure +: only the sky (learned sky mask)", 2000)
        page.evaluate("document.querySelector('#panel').scrollTo(0, 0)")

        cap("⚙ Customise: reorder, collapse or hide panels", 800, shot=False)
        page.click("button[aria-label='Customise panels']")
        page.wait_for_timeout(700)
        page.click("button[aria-label='Move Scene up']")
        page.wait_for_timeout(300)
        page.click("button[aria-label='Move Scene up']")
        page.wait_for_timeout(300)
        cap("Scene moved up (remembered in this browser)", 1600)
        page.click("dialog[open] button:has-text('Close')")

        cap("+ Create style: from an idea, your before/after edits, or example photos", 900, shot=False)
        page.click("button:has-text('+ Create style')")
        page.wait_for_timeout(600)
        page.check("input[value=idea]")
        page.fill("dialog[open] input[placeholder='cyberpunk_night']", "misty_forest")
        page.fill("dialog[open] input[placeholder^='neon cyberpunk']", "soft misty forest mornings, muted greens")
        cap("Describe a look → it finds openly licensed photos → you pick → it trains", 2400)
        page.click("dialog[open] button:has-text('Close')")

        cap("Batch: one style across a series, with a consistency slider", 800, shot=False)
        page.click("button:has-text('Batch')")
        page.wait_for_timeout(500)
        page.click("dialog[open] button:has-text('Preview')")
        page.wait_for_timeout(6000)
        cap("Batch preview (0 % = each photo on its own, 100 % = one shared edit)", 2000)
        page.click("dialog[open] button:has-text('Close')")

        page.select_option("#exportFmt", "cube")
        with page.expect_download(timeout=120_000):
            page.click("#btnExport")
        cap("Export: full-res JPEG, .cube LUT (Resolve, Premiere, …), Lightroom XMP, JSON", 2200)
        cap("", 300, shot=False)
        video = page.video.path()
        ctx.close()
        b.close()
    Path(video).rename(args.out / "studio_demo.webm")

    # storyboard
    tw = 640
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    ims = []
    for text, f in frames:
        im = Image.open(f).convert("RGB")
        im.thumbnail((tw, tw))
        ims.append((text, im))
    cols, h = 2, ims[0][1].height
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (h + 34) + 10), "white")
    d = ImageDraw.Draw(sheet)
    for k, (text, im) in enumerate(ims):
        x, y = 10 + (k % cols) * (tw + 10), 10 + (k // cols) * (h + 34)
        d.text((x, y), f"{k + 1}. {text}", fill="black", font=font)
        sheet.paste(im, (x, y + 22))
    sheet.save(args.out / "storyboard.jpg", quality=85)
    print(f"-> {args.out / 'studio_demo.webm'}, {args.out / 'storyboard.jpg'} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
