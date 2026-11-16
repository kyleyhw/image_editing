# Changelog

## 0.3.0 (2026-11-15)

Plan amendment A6: a professional editor. Every control is a slider of one image pipeline with a
PyTorch reference (`photostyle/pipeline.py`) and a WebGL2 port (`studio/web/src/lib/pipeline.js`)
that agrees with it (147 of 151 cases within 1 code value or mean ΔE00 0.5). Reports:
`tests/reports/phase19_editor.md`, `phase20_data.md`, `phase21_models.md`, `phase22_25_editor.md`,
`phase26_release.md`.

**Pipeline** ([docs/pipeline.md](docs/pipeline.md))
- Linear Rec. 2020 float working space; white balance on the Planckian locus; exposure in stops.
- Hue-preserving tone (blacks, shadows, contrast, highlights, whites) and RGB/R/G/B point curves.
- Oklab colour: vibrance, saturation, 8-band colour mixer, colour grading, black and white mix,
  calibration.
- Detail: texture, clarity, dehaze, sharpening with masking, luminance and colour noise reduction.
- Effects: grain, halation, vignette. Lens blur from an on-device depth model.
- Geometry: crop, straighten, perspective, lens distortion, scale.
- Up to four local masks (linear, radial, brush, luminance and colour range) with twelve local
  sliders each.
- Output in sRGB or Display P3, with a luminance-preserving gamut clip.

**Editor**
- Raw files decoded in the browser (LibRaw in WebAssembly) to linear float data.
- Exports: JPEG with EXIF, PNG, 16-bit PNG and TIFF, `.cube`, Lightroom preset (`.xmp`), JSON.
- Workflow: photos and edits kept on the device, offline and installable (PWA), copy, paste and
  sync, presets, snapshots, history, match to a reference photo.

**Looks and models** ([docs/looks.md](docs/looks.md), [docs/data.md](docs/data.md))
- *Auto-correct*: ten correction sliders predicted per photo by a model trained on FiveK expert C
  (test set: 20.4 → 22.4 dB PSNR, ΔE00 9.95 → 7.92). Off by default: the owner's blind ratings
  found no preference for it (21–22, 17 ties).
- Tutorial looks are now exact slider settings; `styles.json` version 3 (5.3 MB → 0.5 MB).
- Twelve film stocks fitted to film colour tables (held-out ΔE00 0.7–1.7).
- Expert styles A–E: per-photo slider predictions trained on each FiveK retoucher.

**Tools**: `check_pipeline.py` (browser against reference), `check_editor.py` (end to end),
`bench_pipeline.py`, `train_auto.py`, `oracle_study.py`, `fit_film.py`, `fetch_fivek.py`,
`make_rating_set.py`.

## 0.2.0 (2026-09-23)

The project's purpose was redefined: learn the edit that takes *each* photo
to a look, as editable parameters (PROJECT_PLAN §1). This release carries
out the plan end to end. Every phase report is in `tests/reports/`, and the
gate outcomes are in PROJECT_PLAN amendment A4.

**Engine (`photostyle/`)**
- Content-adaptive head (frozen ResNet-18 + CDF features → MLP) predicting
  per-channel curves, a colour matrix, bias and a vignette.
- Shrinkage calibration for small data.
- Paired and unpaired learning (the unpaired default is pseudo-pairs, from
  Phase 11).
- Style conditioning: `StyleHead`, `StyleEncoder`, and `CodedHead` (a style
  as a code on a shared base).
- Regional tools (graduated filter, range masks, sky).
- Learned sky segmenter (UperNet-ConvNeXt-tiny, MIT).
- Depth-aware scene tools (Depth Anything V2 Small): haze, dehaze,
  near/far clarity, sky light.
- `EditParams` with strength and user overrides, and `scene`.
- Exports: `.cube` LUT, Lightroom XMP (curves), JSON.
- Style packs with OOD scores and a per-pack default strength.

**Studio**
- v2 front end in Svelte:
  - customisable panels;
  - WebGL preview matching the Python renderer to 1/255;
  - curves with monotone knots;
  - Scene panel;
  - style strip;
  - batch with series consistency;
  - create-a-style;
  - "remember" and "personalise".
- The v1 page stays at `/legacy`.

**Styles and data**
- `photostyle style …`: the pipeline from an idea to a style pack. It
  searches openly licensed references, lets you pick, finds more like your
  picks, trains, previews and packs with attribution.
- Clean Cool, a pilot look for landscapes.
- Placeholder styles `natural` and `cyberpunk`.
- The FiveK landscape tools and the benchmark harness (`bench/`).

**Licence**
- MIT for the code. Datasets and weights keep their own licences; see the
  README and the commercial checklist in PROJECT_PLAN A4.5.

## 0.1.0

The Phases 1–6 groundwork: fixed-function styles (Classic Chrome,
Cyberpunk, Tilt-shift), a Streamlit UI and a first FiveK expert-C model. It
is kept in `legacy/` and the root scripts.
