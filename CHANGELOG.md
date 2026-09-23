# Changelog

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
