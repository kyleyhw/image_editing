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

The built-in style packs live in `stylepacks/`. They are not in git: both
are derived from MIT-Adobe FiveK, which is licensed for research. Build or
refresh them from the data and checkpoints:

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
| **Panels** (right) | Click a panel title to collapse it; **⚙** reorders or hides panels (remembered in this browser). |
| **Style panel** | Strength from 0–150 % (0 % is exactly the original). The *Explain* line says in words what the edit does. A yellow warning appears when a photo is unlike anything the style learned from. |
| **Curves** | R / G / B tabs; drag a knot. Curves stay monotone, so tones never invert. The dashed line shows the model's curve once you have changed it. |
| **Colour** | Warmth, tint and saturation on top of the model's colour balance. |
| **Scene** | Haze (+ add / − remove), near and far clarity, and sky exposure / warmth / saturation. Uses a depth model and a learned sky mask, computed once per photo (about 2 s). Applied to the photo before the style, and carried into exports. |
| **Markers** | A hollow circle means the value came from the model; a filled square means you changed it. |
| **Histogram** | Grey is before; coloured lines are after. |
| **Export** | JPEG (full resolution, EXIF kept, sRGB), `.cube` (curves + colour, for Resolve, Premiere, Photoshop, OBS…), XMP (Lightroom / ACR preset carrying the per-channel curves only), JSON (the full edit, re-loadable). |
| **Remember this edit** | Stores your corrected version as a training example for that style. |
| **Personalise style** | Fine-tunes a copy of the style (`<style>_personal`) on your remembered edits. The original style is never changed. |
| **+ Create style** | Three routes. **An idea:** describe the look; Studio searches openly licensed photos, you click the ones whose look you like, it adds more like them, you drop any you don't want, then it trains (see *Creating a new style* below). **Before/after pairs** (best; same file names in both sets; 20+ pairs recommended). **Photos in a look** you already have (20+). Training runs in the background on the CPU. Pairs give clearly better styles than example photos alone (`tests/reports/phase11_unpaired.md`). |
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

## Creating a new style (idea → references → style)

The same steps in Studio (**+ Create style → An idea**) or on the command
line. A project is resumable at every step; its state is in
`data/styles/<name>/`.

```
photostyle style new cyberpunk --describe "neon cyberpunk night city" \
    --query "neon city night rain" --query "cyberpunk street"
photostyle style search cyberpunk            # openly licensed candidates -> numbered contact sheets
photostyle style pick cyberpunk 3 8 14 21    # the ones whose look you like -> + closest matches
photostyle style exclude cyberpunk 17        # drop references you don't want (optional)
photostyle style train cyberpunk --recipe strong
photostyle style preview cyberpunk           # before/after on your unedited photos
photostyle style pack cyberpunk --strength 0.8
```

**Choosing a recipe**

| recipe | use when | notes |
|---|---|---|
| `gentle` | the look is close to a natural photo (film tones, soft colour) | pseudo-pairs only; stable; subtle |
| `strong` | the look is far from natural (neon, teal & orange) | adds colour-distribution matching on a pool of openly licensed input photos; stronger but can homogenise |
| `instant` | you want a preview in seconds | uses the shared base's encoder (build it with `tools/build_base.py`); only covers looks near the FiveK experts' |
| `paired` | you have before/after edits (`--pairs BEFORE AFTER`) | best; with the shared base, 20 pairs are enough |

**Licences.** References come only from Openverse with CC0, public
domain, CC BY or CC BY-SA licences. Each pack carries `ATTRIBUTION.csv`,
and the manifest is written to `data/manifests/<name>.csv`. Packs made with
`gentle` or `strong` use no research-licence data. Packs made with
`instant`, or with `paired` on the shared base, depend on the FiveK-derived
base: keep them personal.

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
- **Region-aware light**: Studio's Scene panel covers sky light and
  depth-aware haze and clarity. A graduated filter and
  shadow/midtone/highlight toning are in the regional renderer
  (`photostyle.regional`, Python only for now).
- **Generative changes** (removing objects, replacing skies, relighting a
  face): no, by design.

## Privacy and data

- Photos, style packs and corrections stay under `data/` and `stylepacks/`.
  Both are local, and `data/` is not tracked by git.
- Train styles only on photos you own, have permission for, or that carry an
  open licence that permits it (PROJECT_PLAN §17 A1.1).
