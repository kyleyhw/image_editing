# User guide

photostyle learns the edit that takes *each* photo toward a look, and hands
it back as settings you can still change: per-channel curves, a colour
balance, a small vignette, and a `.cube` LUT for other apps. Everything runs
locally on a laptop CPU. Nothing is uploaded anywhere.

## Install

```
uv sync                # CPU-only PyTorch on Linux/Windows
uv run photostyle styles list
```

The built-in style packs live in `stylepacks/`. Build or refresh them from
the trained checkpoints:

```
uv run python tools/build_stylepacks.py
```

## Studio (the app)

```
uv run photostyle serve        # http://127.0.0.1:8765
```

| Area | What it does |
|---|---|
| **Library** (left) | Upload photos (JPEG, PNG, TIFF, HEIC with `pillow-heif`, camera RAW). Colour profiles such as iPhone Display P3 are converted to sRGB on import. |
| **Canvas** | Split view with a draggable divider, or *After* / *Before*. Hold `\` to see the original. |
| **Style strip** (bottom) | Every style previewed on *your* photo. Click one, or press `1`–`9`. |
| **Style panel** | Strength from 0–150 % (0 % is exactly the original). The *Explain* line says in words what the edit does. A yellow warning appears when a photo is unlike anything the style learned from. |
| **Curves** | R / G / B tabs; drag a knot. Curves stay monotone, so tones never invert. The dashed line shows the model's curve once you have changed it. |
| **Colour** | Warmth, tint and saturation on top of the model's colour balance. |
| **Markers** | A hollow circle means the value came from the model; a filled square means you changed it. |
| **Histogram** | Grey is before; coloured lines are after. |
| **Export** | JPEG (full resolution, EXIF kept, sRGB), `.cube` (curves + colour, for Resolve, Premiere, Photoshop, OBS…), XMP (Lightroom / ACR preset carrying the per-channel curves only), JSON (the full edit, re-loadable). |
| **Remember this edit** | Stores your corrected version as a training example for that style. |
| **Personalise style** | Fine-tunes a copy of the style (`<style>_personal`) on your remembered edits. The original style is never changed. |
| **+ Create style** | *Before/after pairs* (best; same file names in both sets; 20+ pairs recommended), or *photos in a look* (20+ examples, plus a few of your unedited photos). Training runs in the background on the CPU. |
| **Batch** | Apply the current style to the whole library. The *series consistency* slider pulls every photo's edit toward the set's average: 0 % edits each photo on its own; 100 % applies one shared edit. |

Keyboard shortcuts: `\` original · `Y` cycle compare mode · `[` `]`
strength ±5 % · `R` reset to model · `Ctrl/Cmd+Z` undo (`+Shift` redo) · `E`
export · `←` `→` previous/next photo · `1`–`9` styles.

## Command line

```
photostyle apply --style clean_cool --strength 0.8 photos/*.jpg -o out/ --cube --json
photostyle learn --name my_look --pairs before/ after/            # your before/after edits
photostyle learn --name neon --examples inspo/ --inputs mine/     # photos in a look (use ones you own or openly licensed)
photostyle export-cube --style my_look photos/*.jpg -o my_look.cube   # one average LUT for a set of photos
```

## Scene tools (Python API, Phase 18)

Depth-aware haze and clarity, and sky light. They adjust existing pixels only
and never generate content:

```python
from photostyle.atmosphere import SceneParams, apply_scene
out = apply_scene(img_tensor, SceneParams(haze=-0.4, clarity_far=-0.3, clarity_near=0.4))
```

## What the edits can and cannot do

- **Global colour and tone** (curves, colour balance, saturation, black
  point): yes. This is where the model is strongest.
- **Region-aware light** (graduated filter, shadow/midtone/highlight toning,
  sky): available in the regional renderer (`photostyle.regional`). Studio
  exposes the global stage for now.
- **Generative changes** (removing objects, replacing skies, relighting a
  face): no, by design.

## Privacy and data

- Photos, style packs and corrections stay under `data/` and `stylepacks/`.
  Both are local, and `data/` is not tracked by git.
- Train styles only on photos you own, have permission for, or that carry an
  open licence that permits it (PROJECT_PLAN §17 A1.1).
