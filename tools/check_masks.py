"""Phase 23 task 31: learned masks (sky, subject) in the built editor, on real photos.

The Pages build is served locally and driven in headless Chromium. For each photo (unedited
example photos chosen to have, or to lack, sky and people), the editor's "+ Sky" or "+ Subject"
action runs the in-browser segmentation model and the guided-filter refinement; the stored mask
is saved with an overlay for review. Reported per photo: the share of the frame selected, and the
time taken. With --reference, the sky masks are compared with the Python side's learned sky mask
(photostyle/sky.py, UperNet with a ConvNeXt backbone, a much larger model) by intersection over
union.

    uv run python tools/check_masks.py --out <dir> [--reference]
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.check_editor import CHROMIUM, serve  # noqa: E402

EX = ROOT / "studio/web/public"
# (photo, mask kind): sky on landscapes, cities and a night skyline; subject on portraits; and
# photos without the class, which should give an (almost) empty mask
CASES = [("samples/mountain-lake.jpg", "sky"), ("samples/valley.jpg", "sky"), ("samples/pier-dusk.jpg", "sky"),
         ("examples/teal_orange/before.jpg", "sky"), ("examples/winter/before.jpg", "sky"),
         ("examples/cyberpunk/before.jpg", "sky"), ("examples/moody_food/before.jpg", "sky"),
         ("examples/portra_400/before.jpg", "subject"), ("examples/bright_airy/before.jpg", "subject"),
         ("examples/tri_x/before.jpg", "subject"), ("examples/moody_portrait/before.jpg", "subject"),
         ("examples/velvia/before.jpg", "subject")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8813)
    ap.add_argument("--reference", action="store_true", help="IoU of the sky masks against photostyle/sky.py")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    serve(args.port)
    from playwright.sync_api import sync_playwright
    rows: list[dict] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROMIUM if Path(CHROMIUM).exists() else None,
                               args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:{args.port}/image_editing/")
        pg.wait_for_function("window.__studio && window.__studio.S.model.ready", timeout=180_000)
        for k, (rel, kind) in enumerate(CASES):
            n0 = pg.evaluate("() => window.__studio.S.photos.length")
            pg.set_input_files("#upload", str(EX / rel))
            pg.wait_for_function(f"window.__studio.S.photos.length > {n0} && window.__studio.S.params !== null", timeout=120_000)
            pg.wait_for_timeout(800)
            t1 = time.time()
            pg.evaluate(f"() => window.__studio.addLearnedMask('{kind}')")
            pg.wait_for_function("(window.__studio.S.p.masks || []).some(m => m.label) && !window.__studio.S.seg.loading", timeout=300_000)
            secs = time.time() - t1
            data = pg.evaluate("() => window.__studio.S.p.masks.find(m => m.label).data")
            m = np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("L"), np.float64) / 255
            photo = Image.open(EX / rel).convert("RGB").resize((m.shape[1], m.shape[0]))
            red = np.asarray(photo, np.float64) * (1 - 0.55 * m[..., None]) + np.array([255, 40, 60]) * 0.55 * m[..., None]
            name = f"{k:02d}_{kind}_{Path(rel).parent.name if 'examples' in rel else Path(rel).stem}"
            Image.fromarray(red.clip(0, 255).astype(np.uint8)).save(args.out / f"{name}.jpg", quality=85)
            Image.fromarray((m * 255).astype(np.uint8)).save(args.out / f"{name}_mask.png")
            row = {"photo": rel, "kind": kind, "selected": round(float(m.mean()), 3), "seconds": round(secs, 1)}
            rows.append(row)
            print(row, flush=True)
        b.close()
    if args.reference:
        import torch

        from photostyle import sky as SKY
        if SKY._load() is None:
            raise SystemExit("the reference sky model could not be loaded")
        for row in rows:
            if row["kind"] != "sky":
                continue
            name = next(args.out.glob(f"*_sky_*{Path(row['photo']).parent.name if 'examples' in row['photo'] else Path(row['photo']).stem}_mask.png"))
            m = np.asarray(Image.open(name), np.float64) / 255
            img = Image.open(EX / row["photo"]).convert("RGB").resize((m.shape[1], m.shape[0]))
            x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad():
                ref = SKY.sky_mask(x)[0, 0].numpy().astype(np.float64)
            a, r = m > 0.5, ref > 0.5
            row["iou_vs_upernet"] = round(float((a & r).sum() / max(1, (a | r).sum())), 3) if (a | r).any() else 1.0
            print(row["photo"], "IoU", row["iou_vs_upernet"], flush=True)
    out = {"rows": rows, "page_errors": errs, "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "tests/reports/phase23_masks.json").write_text(json.dumps(out, indent=1))
    print(f"page errors: {errs or 'none'}; {out['runtime_s']} s")


if __name__ == "__main__":
    main()
