# User guide (Irodori)

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

Two ways to run it, one interface:

- **In your browser:** <https://kyleyhw.github.io/image_editing/>. The model runs on
  your device (ResNet-18 through ONNX Runtime Web, the style head in JavaScript; a
  22 MB download, cached by the browser). Photos are never uploaded. It has the
  publishable looks only (trained on openly licensed photos; see *Hosting* below), and no
  scene tools, batch or style training.
- **Locally, with everything:** `uv run photostyle serve` (http://127.0.0.1:8765).
  All your style packs, the Scene panel, batch, create-a-style and personalisation.

| Area | What it does |
|---|---|
| **Welcome** | Open a photo, drop one anywhere, or start from a public-domain sample. |
| **Looks** (left) | Grouped by the kind of photo they suit (landscapes, cities, portraits, food, seasons). The chevron next to a look opens its example: an openly licensed photo of the subject its tutorial was written for, before/after the look, with links to the tutorials the recipe comes from (the authors' own before/after is on their pages). The same panel is under *Example & sources* in the Adjust panel. Every look is previewed on *your* photo. **Hover** a look to see it full size without committing; click it (or press `1`–`9`) to choose it. `0` / *Original* shows the unedited photo. On a phone the looks are a row under the photo. |
| **Photos** (top) | Your open photos; **+** adds more (JPEG, PNG, TIFF, HEIC with `pillow-heif` and camera RAW when running locally). `←` `→` step through them. |
| **Canvas** | Before / Split / After. Drag the divider; hold `\` to see the original. Changing look animates from the old edit to the new one. The room around the photo is lit by it, and the accent colour follows the edit. |
| **Look card** (right) | Strength 0–150 % (0 % is exactly the original). The chips below say what the edit does (shadows, midtones, highlights, warmth, tint, saturation, vignette), measured through the renderer, so they stay true as you edit. A note appears when a photo is unlike what the look learned from. |
| **Tone** | Histogram of the edited photo, and R / G / B curves: drag a point. Curves stay monotone, so tones never invert. The dashed line is the model's curve once you have changed it; *Model curve* puts it back. |
| **Colour**, **Light** | Warmth, tint, saturation and vignette on top of the model's edit. Double-click a slider to reset it. |
| **Scene** (local) | Haze (+ add / − remove), near and far clarity, and sky exposure / warmth / saturation. Uses a depth model and a learned sky mask, computed once per photo (about 2 s). Applied to the photo before the look, and carried into exports. |
| **Export** | JPEG at full resolution; `.cube` (curves + colour, for Resolve, Premiere, Photoshop, OBS…); JSON (the full edit). Locally also XMP (Lightroom / ACR preset carrying the per-channel curves only); in the browser also PNG. |
| **Remember / Personalise** (local) | *Remember* stores your corrected version as a training example for that look; *Personalise* fine-tunes a copy (`<style>_personal`) on those. The original look is never changed. |
| **New style** (local) | Three routes. **An idea:** describe the look; Studio searches openly licensed photos, you click the ones whose look you like, it adds more like them, you drop any you don't want, then it trains (see *Creating a new style* below). **Before/after pairs** (best; same file names in both sets; 20+ pairs recommended). **Photos in a look** you already have (20+). Training runs in the background on the CPU. |
| **Batch** (local) | Apply the current look to the whole library. *Series consistency* pulls every photo's edit toward the set's average: 0 % edits each photo on its own; 100 % applies one shared edit. |

Keyboard shortcuts: `\` original · `Y` cycle compare mode · `[` `]`
strength ±5 % · `R` reset to model · `Ctrl/Cmd+Z` undo (`+Shift` redo) · `E`
export · `←` `→` previous/next photo · `1`–`9` looks · `0` original.

### Hosting (GitHub Pages)

`.github/workflows/pages.yml` builds and deploys the browser version on every push that
touches the front end. It exports the ResNet-18 backbone to ONNX (fp16-stored weights,
fp32 maths; 0.1 % feature error), builds with `npm run build:pages` and publishes `site/`.
The looks it ships are in `studio/web/pages/models/styles.json`, written by

```
uv run python tools/export_web.py styles     # packs whose style.json says "publishable": true
uv run python tools/build_library.py         # the twenty tutorial looks, with their examples
```

A pack is publishable only if it was trained on openly licensed photos alone:
`photostyle style train NAME --recipe strong --open-only` (keeps your own photos out of the
input pool) or the `gentle` recipe, then `pack`. Anything on the FiveK-derived base
(`instant`, `paired`) or FiveK data (`clean_cool`, `fivek_c_landscape`) is never exported.
A web-only version of a local look can live in `webpacks/`
(`uv run photostyle --root webpacks style pack NAME`); it overrides a local pack of the same name
in the export. The browser's
predictions match PyTorch to 0.35/255 on average (p99 1.8/255).

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
photostyle style restore cyberpunk 12        # keep a photo the digital-art filter greyed out (optional)
photostyle style train cyberpunk --recipe strong
photostyle style preview cyberpunk           # before/after on your unedited photos
photostyle style pack cyberpunk --strength 0.8
```

**Digital art is filtered out automatically.** Searches for stylised looks
return many renders and illustrations. Each candidate is scored by a small
image model (LAION CLIP, MIT licence), and likely digital art is greyed out
(marked **ART?** on contact sheets): it is skipped unless you pick or
restore it. In a test it caught 30 of 32 renders; about 1 in 9 real photos
(graffiti, glossy cars) gets flagged too, which is why it only greys out.
`photostyle style flag NAME` scores a project searched before the filter
existed.

**Choosing a recipe**

| recipe | use when | notes |
|---|---|---|
| `gentle` | the look is close to a natural photo (film tones, soft colour) | pseudo-pairs only; stable; subtle |
| `strong` | the look is far from natural (neon, teal & orange) | adds colour-distribution matching on a pool of openly licensed input photos; stronger but can homogenise |
| `teacher` | the look is defined by how people make it (e.g. cyberpunk split toning) | a tutorial recipe in `photostyle/recipes.py` grades openly licensed input photos, and the style is trained on those pairs; the references are only used to choose and check the look |
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
