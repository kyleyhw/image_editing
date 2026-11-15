# Phase 26 report: verification and release

*2026-11-15. Plan amendment A6, Phase 26 (tasks 41–43), release 0.3.0.*

## What was done

1. **Accuracy suite** (task 41): the browser pipeline against the PyTorch reference, the pipeline
   invariants and unit tests, and the end-to-end editor test on the built site, all rerun on the
   release build.
2. **Performance** (task 41): render time per operator group, browser and reference
   (`tools/bench_pipeline.py`, new).
3. **Documentation** (task 42): [`docs/data.md`](../../docs/data.md) (new: datasets and slider
   models), [`docs/looks.md`](../../docs/looks.md) (film stocks and expert styles; generator
   updated), the user guide (Auto-correct, look groups, publishing rule), the README index and
   directory diagram, and the changelog.
4. **Toolchain** (task 43): ruff, ty and detect-secrets against the project standard; ty
   configuration added to `pyproject.toml`.

**Why.** A6.6 sets the release gates: agreement of the two implementations, the invariants, the
editor working end to end on desktop and phone viewports, and measured performance.

## Results

### Accuracy

| Check | Inputs and why | Result |
|---|---|---|
| Agreement, browser against reference (`tools/check_pipeline.py`, 151 cases) | Synthetic image (grey ramps, hue wheel at three lightnesses, skin and saturated patches, seeded noise), chosen to expose tone order, hue shifts and spatial filtering; each control at a strong value, masks, geometry and seeded random combinations | **147 / 151** within target (max 1 code value and mean ΔE00 ≤ 0.5; spatial operators mean ΔE00 ≤ 0.5). The four misses are dehaze +60, colour NR +100, halation +100 and lens blur, on the synthetic noise rows only; on a photograph they agree (largest mean ΔE00 0.32; [Phase 22–25 report](phase22_25_editor.md)) |
| Unit tests (`pytest`, 65 tests) | Invariants on grey ramps, mid grey, a hue wheel and seeded random images (identity, neutrality, monotone tone, hue preservation, highlight recovery, white balance direction and luminance, mask complement, lens-blur focus plane, gradients), plus the colour-table loader and the older engine tests | **65 passed** |
| End to end (`tools/check_editor.py`, built site, Chromium, desktop 1440 × 900 and phone 390 × 844 viewports) | A public-domain sample photo and a CC0 Leica M10 DNG (raw.pixls.us), so both the 8-bit and the raw path are exercised | **All checks pass; no page errors** (table below) |

End-to-end checks:

| Check | Result |
|---|---|
| Auto-correct available and on by default; predicts ten sliders | ✓ |
| Exposure, curves, detail, linear mask, brush mask, crop, white-balance picker change the picture | ✓ |
| A user edit survives a look change; undo restores the look | ✓ |
| Expert style (`expert_c`) predicts its sliders for the photo (new check) | ✓ 30 non-zero sliders, kind `sliders` |
| Exports: JPEG, `.cube` (35,937 rows = 33³), 16-bit PNG, Lightroom preset with `Exposure2012` | ✓ |
| Copy and paste, history (14 steps), service worker, photos and edits kept after reload | ✓ |
| Leica M10 DNG: opens, starts at contrast +20, 16-bit TIFF export | 8.1 s; 143 MB TIFF ✓ |
| Page errors, desktop and phone | none |

### Performance

The test machine has no GPU. Chromium renders WebGL2 with SwiftShader, a CPU emulation, so these
times are far above a real device's and only the ratios between groups carry over. Settings are
cumulative down the table (each row adds passes to the one above). Median of 5 frames.

| Settings (cumulative) | Browser, preview 1600 × 1067 | Browser, export 4000 × 3000 | Reference (PyTorch, CPU), preview |
|---|---|---|---|
| Per-pixel: tone, curves, colour mixer, grading | 1,182 ms | 7,852 ms | 795 ms |
| + texture, clarity, sharpening, noise reduction | 2,622 ms | 19,802 ms | 1,748 ms |
| + dehaze, grain, halation | 2,710 ms | 19,356 ms | 1,839 ms |
| + two masks | 2,774 ms | 20,621 ms | 2,088 ms |

**Interpretation.**

- The detail passes (three local-mean scales) more than double the cost of a frame; dehaze,
  grain, halation and masks add little on top. The export frame costs about 7× the preview for
  7.0× the pixels, so the cost is proportional to pixel count.
- The A6.6 targets (16.7 ms per frame on a desktop, 33 ms on a phone, at the preview size) cannot
  be tested here. As an order of magnitude: SwiftShader is typically two to three orders of
  magnitude slower than an integrated GPU for fragment work, which would put the full set near or
  under the desktop target. This is an estimate, not a measurement. The owner's phone checklist
  ([docs/phone_checklist.md](../../docs/phone_checklist.md)) is where the real figure comes from.
- **Measurement fault found and fixed.** The first run reported 0.1–0.4 ms per frame because
  `gl.finish()` does not block in Chromium's GPU process. The benchmark now ends each frame with a
  one-pixel `readPixels`, which does block, and checks that the frame is not black.

### Toolchain (task 43)

| Tool | Scope | Result |
|---|---|---|
| ruff 0.15.14 | A6 modules and tools | clean (one unused import fixed in `tools/fit_film.py`) |
| ty 0.0.39 | A6 modules and tools (18 files) | clean after fixes: Playwright's optional bounding box asserted (`tools/check_editor.py`), `log_message` signature, annotations on mixed-type dicts, an unreachable `None` asserted in `tools/train_auto.py` |
| ty 0.0.39 | whole repository | **125 diagnostics, all in pre-A6 code** (`tools/demo_studio.py` 17, `tests/test_photostyle.py` 12, `tools/check_studio.py` 10, `photostyle/engine.py` 9, `bench/phase18.py` 9, `photostyle/{train,head,condition}.py` 7 each, `legacy/` and older `bench/` and `tools/` files the rest). Left as they are: outside this phase's scope |
| detect-secrets 1.5.0 | repository, excluding `data/`, `site/`, build output and generated JSON | no findings |

**Configuration added** (`pyproject.toml`): `[tool.ty.environment] python-version = "3.11"` (the
project venv), `[tool.ty.terminal] error-on-warning = true`, and `onnx` / `onnxruntime` as allowed
unresolved imports (they are run-time extras, `uv run --with onnx --with onnxruntime`).

**Finding for the owner.** `.pre-commit-config.yaml` follows the project template, but the hooks
are not installed in this clone (`.git/hooks` has no `pre-commit`), and its `ty` hook checks the
whole repository, so it would currently block every commit on the 125 legacy diagnostics. Two
options: fix the legacy diagnostics, or limit the hook to the maintained paths. Not changed here.

## Release

- Version 0.3.0 (`pyproject.toml`, `studio/web/package.json`); [`CHANGELOG.md`](../../CHANGELOG.md).
- `styles.json` version 3: 39 looks (Fujifilm head, 21 tutorial slider looks, 12 film stocks,
  5 expert styles) and 6 slider models; 1.7 MB.
- Deployed to GitHub Pages from `master` (the workflow builds the site and the ONNX backbone).

## Not done

- **Owner-dependent items:** the blind A/B ratings (Phase 20 task 14; the page is built), the phone
  checklist (task 39), and the performance figures on real hardware.
- **PPR10K** (task 11): its host still refuses downloads.
- **Tiled export and memory limits** (task 28) and **learned masks in the browser** (task 31):
  open, the latter waiting on the download-budget decision (A6.10).

## Runtime

| Run | Runtime |
|---|---|
| Agreement test, 151 cases | 170 s |
| Unit tests, 65 | 10 s |
| End to end, with the raw file | 143 s |
| Benchmark | 506 s |
| detect-secrets scan | under 60 s |
