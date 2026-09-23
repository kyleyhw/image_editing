# Project Plan: Content-Adaptive, Editable Photo Styles

> **One-line pitch.** Show it a photographer's edits or a folder of photos
> in a look you like, and it learns to edit *your* photos toward that look,
> differently for each photo, as sliders and curves you can still adjust,
> and as a `.cube` LUT you can take into any editor.

Rationale and literature review:
[`reports/project_direction/project_direction.pdf`](reports/project_direction/project_direction.pdf)
(one-page research report) and
[`reports/Content adaptive photo edit models.md`](reports/Content%20adaptive%20photo%20edit%20models.md)
(full comparison of 18 methods). Source notes are in [`research_notes/`](research_notes/).

> **Amended.** Amendment A1 (§17, 2026-09-23) adds the first target look
> (*Neon Street Portrait*, a pilot run right after Phase 7), a new
> Phase 18 for depth-of-field and light-shaping emulation, and a
> project-wide style-data policy. Where A1 and earlier text disagree, A1 wins.

---

## Contents

1. [Purpose](#1-purpose)
2. [Design principles](#2-design-principles)
3. [Users, use cases and user stories](#3-users-use-cases-and-user-stories)
4. [Success metrics](#4-success-metrics)
5. [Status: Phases 1–6 (groundwork, complete)](#5-status-phases-16-groundwork-complete)
6. [Target system architecture](#6-target-system-architecture)
7. [Roadmap overview](#7-roadmap-overview)
8. [Detailed phases (7–17)](#8-detailed-phases) — Pilot A and Phase 18 are specified in §17 (A1)
9. [UI / UX specification](#9-ui--ux-specification)
10. [Data strategy](#10-data-strategy)
11. [Evaluation protocol](#11-evaluation-protocol)
12. [Engineering standards](#12-engineering-standards)
13. [Risks and mitigations](#13-risks-and-mitigations)
14. [Non-goals](#14-non-goals)
15. [Open questions for the project owner](#15-open-questions-for-the-project-owner)
16. [Backlog and housekeeping](#16-backlog-and-housekeeping)
17. [Amendments](#17-amendments) — **A1: first target look, capture emulation, style-data policy**

---

## 1. Purpose

Learn **content-adaptive photo edits toward a target look**.

- A **style** is a target appearance, not a fixed operation. A dark street
  and a bright beach need *different* edits to reach the same look;
  predicting that per-image edit is the model's job.
- A look is specified by **paired examples** (a photographer's before/after
  edits, e.g. MIT-Adobe FiveK experts, PPR10K retouchers, a user's own
  Lightroom catalogue, RAW+JPEG pairs from a camera) or by **unpaired
  examples** (a folder of photos already in the style, e.g. cyberpunk,
  Portra-like film).
- **One model, many looks.** A single network conditioned on a *style code*:
  a learned embedding for known styles, or one computed from a handful of
  example images for new styles.
- **Editable, trustworthy output.** The model outputs parameters (per-channel
  curves, colour matrix, a small learned LUT, grain, vignette) that render
  deterministically, can be adjusted by hand, and bake to a `.cube` file. It
  never generates pixels, so it cannot invent or remove content.
- **Constraints.** Trains on a CPU. Useful with tens to hundreds of examples
  per style. Interactive on a laptop.

### Why this is worth building

Existing tools cover two ends: static presets and LUTs (same edit for every
photo) and generative AI editors (powerful, but they change content and are
not editable). Research models such as 3D-LUT enhancers are adaptive and
fast but single-style and not packaged for people. The gap is an
**adaptive, personal, editable** look engine that runs locally.

---

## 2. Design principles

1. **Edits, not pixels.** Every output is a parameter set applied by a
   known renderer. If a user can't see it as a slider, curve or LUT, it
   doesn't ship.
2. **Identity is always one click away.** Every parameter has a neutral
   value; strength 0 % returns the original exactly.
3. **Adaptive by default, controllable always.** The model proposes; the
   user disposes. Hand adjustments are never overwritten.
4. **Small data first.** Every feature must work with 10–100 examples. Big
   datasets are for benchmarks and pretraining, not a user requirement.
5. **Local and private.** Photos never leave the machine unless the user
   chooses. No telemetry by default.
6. **Measure before building.** Each research phase ends with a numeric
   gate; a feature that fails its gate is dropped or redesigned.
7. **Honest uncertainty.** The system says when a photo is unlike anything
   the style was learned from.
8. **Instant feedback.** Slider changes preview in under 50 ms; a new style
   applied to a photo in under 1 s on CPU.

---

## 3. Users, use cases and user stories

### Personas

| Persona | Who | What they want | What they bring |
|---|---|---|---|
| **The hobbyist** | Shoots on phone or mirrorless, edits occasionally | "Make my photos look like *that*" without learning curves | A few Pinterest / Instagram images of a look |
| **The working photographer** | Weddings, events, portraits; edits thousands of photos | Their *own* look applied consistently to a whole shoot, then fine-tune | Hundreds of before/after pairs in Lightroom |
| **The colourist / video editor** | Uses Resolve / Premiere | A per-shot adaptive LUT that drops into their grading pipeline | Stills from reference films, `.cube` workflows |
| **The researcher / student** | Studies learned enhancement | A clean, CPU-friendly codebase with honest baselines | FiveK / PPR10K, experiment ideas |

### Core use cases

1. **Apply a built-in look** (e.g. "Expert C", "Warm film", "Cyberpunk") to a
   photo, adjust strength and sliders, export.
2. **Learn my style from my edits**: point at a folder of originals + edited
   versions (or a Lightroom export), get a personal style in minutes.
3. **Learn a style from inspiration images**: drop 20–300 images in a look,
   no originals needed.
4. **Batch-apply a style to a shoot** with series consistency (similar
   frames get similar edits).
5. **Export a LUT** for use in Resolve, Premiere, Photoshop, Lightroom
   profiles, OBS or a camera's LUT slot.
6. **Blend styles** ("70 % Expert C, 30 % film") and explore a style space.
7. **Research**: reproduce the benchmark table, run ablations, add a renderer.

### User stories (acceptance-level)

- *As a hobbyist*, I can drag a photo into the app and see six styles
  previewed on **my** photo within 2 seconds, so I can pick one by looking.
- *As a hobbyist*, I can drag a strength slider from 0 to 150 % and the
  preview updates continuously.
- *As a photographer*, I can create a style from 50 before/after pairs, and
  the app shows me held-out previews of *my own* photos, so I can judge
  quality before trusting it on a shoot.
- *As a photographer*, after the model edits a photo I can nudge the
  shadows curve, and choose "remember this", so future predictions shift
  toward my correction.
- *As a photographer*, I can apply a style to 800 photos and export
  full-resolution JPEGs/TIFFs with EXIF preserved, unattended.
- *As a colourist*, I can export (a) the style's average LUT, or (b) one
  LUT per image, as 33³ `.cube` files that render identically to the app
  (≤ 1/255 max error).
- *As any user*, I get a visible warning when a photo is out of
  distribution for the chosen style (e.g. a night photo for a style
  learned only from beach shots).
- *As a researcher*, `make benchmark` reproduces every number in the README
  from a clean checkout with a fixed seed.

---

## 4. Success metrics

### Research (measured on held-out data, 3 seeds, ≥ 100 test images)

| Metric | Target | Why |
|---|---|---|
| FiveK expert C PSNR, 480p protocol, 4,500 train pairs | ≥ 25.0 dB (Zeng et al. 25.21) | Competitive with published global methods |
| Same, 100 train pairs | ≥ 23.5 dB | Small-data usefulness |
| Same, 25 train pairs | beats "average edit" baseline by ≥ 1 dB | Few-shot usefulness |
| Learning curve | monotone non-decreasing in #pairs | Fixes the 80 → 500 regression |
| Held-out expert E via style encoder (n = 20 examples) | within 1 dB of a model trained on E's pairs | Few-shot style transfer works |
| Unpaired expert C (disjoint images) | ≥ 22.5 dB (Zeng unpaired 22.86) | Unpaired route viable |
| Parameter spread for unpaired styles | significantly > 0 across diverse inputs | Proves content adaptivity |
| `.cube` export fidelity | ≥ 45 dB PSNR vs. in-app render | Export is trustworthy |

### Product

| Metric | Target |
|---|---|
| Predict + render, 12 MP photo, laptop CPU | < 1 s (predict at 256 px, apply LUT at full res) |
| Slider interaction latency (browser, WebGL) | < 50 ms / frame at 2 K preview |
| Train a personal style from 100 pairs, CPU | < 5 min (cached features) |
| Train a style from 100 unpaired images, CPU | < 15 min |
| Blind preference vs. static preset of the same look (small user study) | adaptive preferred in ≥ 60 % of trials |
| Crash-free batch export of 1,000 photos | 100 % |

---

## 5. Status: Phases 1–6 (groundwork, complete)

These phases built the infrastructure under the original, broader goal ("a
general-purpose stylization engine"). Their role has changed: the synthetic
styles are now **sanity tests** for the renderer, not deliverables.

- [x] **Phase 1 – Fujifilm data pipeline.** `StyleGenerator` base,
  `FujifilmGenerator`, `generate_dataset.py`.
- [x] **Phase 2 – Hybrid model.** Differentiable CDF + ResNet-18 features, 7-D
  Fujifilm head and renderer. *Legacy: effect ratio ≈ 0 %.*
- [x] **Phase 3 – Generic renderer.** Shared tone curve, 3×3 colour matrix +
  bias, grain, vignette (21 params, identity at init); L1 + VGG + CDF loss.
- [x] **Phase 4 – Cyberpunk** trained without model changes.
- [x] **Phase 5 – Tilt-shift** with a 3-parameter focus band.
- [x] **Phase 6 – UI and real data.** Streamlit UI; FiveK streaming
  downloader; expert-C model.

**Lessons carried forward**

1. Fixed-function styles don't need a network; the network earns its place
   only when the right edit depends on the image.
2. FiveK expert C: 80 pairs → held-out mean L1 0.0917 → 0.0741 (≈ 19 %
   closer, 4/5 improved); 500 pairs → 0.1030 (1/5). Five test images, one
   seed, confounded training ⇒ **not** evidence of an architectural ceiling.
3. Image adaptivity carries most of the FiveK gain (fixed LUT 20.4 dB →
   adaptive LUT 25.2 dB); spatial processing adds ≈ 0.3–0.5 dB.

---

## 6. Target system architecture

```
                         ┌──────────────────────── Style store ─────────────────────────┐
                         │ style cards (JSON) · embeddings · basis LUTs · model weights │
                         └───────────────▲───────────────────────────┬──────────────────┘
                                         │ create / update           │ load
 ┌───────────┐   ┌─────────────────┐   ┌─┴─────────────────┐   ┌─────▼──────────────┐
 │  Ingest   │──▶│ Feature cache   │──▶│ Style learner     │   │  Predictor         │
 │ RAW/JPEG/ │   │ ResNet-18 GAP + │   │ paired · unpaired │   │  head(feat, s) → θ │
 │ TIFF/HEIC │   │ CDF (+ OOD stats)│   │ · few-shot encode │   └─────┬──────────────┘
 └───────────┘   └─────────────────┘   └───────────────────┘         │ θ (EditParams)
                                                                     ▼
            ┌──────────────────────────────── Renderer ─────────────────────────────────┐
            │ per-channel curves → 3×3 matrix + bias → Σ wₙ·LUTₙ (9³ residual) →         │
            │ grain → vignette           (PyTorch · NumPy · GLSL — bit-matched)          │
            └──────┬───────────────────────────┬─────────────────────────┬──────────────┘
                   ▼                           ▼                         ▼
             Studio (web UI)            CLI / Python API          Exporters: .cube, XMP,
             + WebGL preview            batch, benchmark          JSON, JPEG/TIFF
```

### Components

| Component | Responsibility | Notes |
|---|---|---|
| `photostyle.io` | Decode RAW (rawpy), JPEG, PNG, TIFF 16-bit, HEIC (optional); colour-manage to linear/sRGB working space; preserve EXIF/ICC | Neutral RAW develop defines "before" for camera pairs |
| `photostyle.features` | Frozen ResNet-18 (ImageNet-normalised) + differentiable CDF; disk cache keyed by file hash | Makes CPU training fast |
| `photostyle.render` | Renderer primitives with identity-at-zero; three bit-matched backends: PyTorch (training), NumPy (CLI), GLSL (browser) | Golden-image tests keep backends in sync |
| `photostyle.params` | `EditParams` dataclass: typed, versioned, serialisable, interpolable (`lerp`, `scale`) | The contract between model, UI and exporters |
| `photostyle.model` | Head (MLP + FiLM), style embedding table, style encoder | No BatchNorm; LayerNorm or none |
| `photostyle.train` | Paired, unpaired, few-shot and fine-tune loops; early stopping; seeds; logging | Configs in YAML |
| `photostyle.styles` | Style store: create, list, version, delete, blend; style cards | Plain files on disk |
| `photostyle.ood` | Distance of an image's features from a style's training distribution | Drives the UI warning |
| `photostyle.export` | `.cube` (per image, per style average), Lightroom XMP (approximate), JSON, images | Round-trip tests |
| `photostyle.server` | FastAPI: predict, render, train jobs (background queue), export | Localhost only by default |
| `studio/` | Web front end (see §9) | WebGL renderer for instant sliders |
| `bench/` | Benchmark harness, baselines, report generation | Reproduces README tables |

### `EditParams` schema (v1, draft)

```yaml
version: 1
style_id: expert_c@3            # style name @ version
strength: 1.0                   # global blend toward identity, 0..1.5
curves:                         # per-channel monotone curves, K knots each
  knots_x: [0, 0.125, ..., 1]   # fixed grid
  r: [0, 0.11, ..., 1]
  g: [...]
  b: [...]
  master: [...]                 # optional luminance curve (UI convenience)
color:
  matrix: [[1.02, 0.01, -0.03], [...], [...]]
  bias: [0.004, -0.002, 0.01]
  derived:                      # human-readable views of matrix (read-only)
    temperature_shift_k: +310
    tint_shift: -4
    saturation: 1.08
look:
  basis_weights: [0.62, 0.21, -0.05, 0.12]   # over the style's shared basis LUTs
  lut_strength: 1.0
grain: {amount: 0.02, size: 1.0}
vignette: {amount: 0.15, midpoint: 0.5}
provenance:
  predicted: true
  user_overrides: [curves.r]    # fields the user touched
  ood_score: 0.8                # 0 = typical, >2 = unusual
```

### Public Python API (sketch)

```python
from photostyle import Engine

eng = Engine.load()                       # built-in styles + user styles
params = eng.predict("IMG_0042.CR3", style="expert_c")
params = params.with_strength(0.8).override(vignette={"amount": 0})
img = eng.render("IMG_0042.CR3", params)  # full-res, colour-managed
params.to_cube("IMG_0042.cube", size=33)

style = eng.learn_style(
    name="my_wedding_look",
    pairs="exports/lightroom/",           # or examples="inspo/" for unpaired
)
```

### CLI (sketch)

```
photostyle styles list
photostyle apply  --style expert_c --strength 0.8 photos/*.jpg -o out/
photostyle learn  --name my_look --pairs before/ after/     # paired
photostyle learn  --name neon    --examples inspo/          # unpaired
photostyle export-cube --style my_look --average -o my_look.cube
photostyle bench  --protocol fivek-480p --train-sizes 25,100,500,4500 --seeds 3
photostyle serve  # launches Studio on http://localhost:8765
```

---

## 7. Roadmap overview

Work runs in four tracks that interleave. **R** = research, **E** = engine,
**P** = product/UI, **I** = infrastructure. Effort: S ≈ days, M ≈ 1–2 weeks,
L ≈ 3–6 weeks (part-time pace).

| Phase | Track | Title | Effort | Depends on | Gate / exit criterion |
|---|---|---|---|---|---|
| 7 | R | Evaluation harness & confound removal | M | – | Learning curve measured; ceiling hypothesis accepted or retired |
| 8 | I | Engineering foundations | M | – (parallel with 7) | Package, tests, CI green |
| 9 | R/E | Renderer v2 (SepLUT-style cascade) | M | 7, 8 | LUT stage kept only if +0.3 dB at ≤ 250 pairs |
| 10 | R | Style conditioning & few-shot styles | L | 9 | Shared model ≥ separate models |
| 11 | R | Unpaired styles | L | 9 (10 helpful) | Simplest loss within 0.5 dB of best |
| 12 | E | Engine API, exporters, performance | M | 9 | Latency + `.cube` fidelity targets met |
| 13 | P | Studio v1 (web app) | L | 12 | Core use cases 1, 5 usable end to end |
| 14 | P/R | Style creation wizard & personal styles | L | 10, 11, 13 | Use cases 2, 3 usable; held-out preview shown |
| 15 | P/R | Personalisation loop, series consistency, blending | L | 14 | Measurable gain from user corrections |
| 16 | R | Regional edits (conditional) | L | 7–11 | Only if oracle gap shows local structure |
| 17 | I/P | Packaging, docs, release | M | 13 | Installable release + public demo |
| **A** | R | **Pilot: Neon Street Portrait look** *(A1)* | M | 7 (8 helpful) | Adaptive beats fixed preset; night/day parameter spread > 0 |
| **18** | R/E/P | **Capture emulation: depth of field + light shaping** *(A1)* | L | A, 12 | Preference ≥ 60 % vs. no-DoF; halo-free in ≥ 80 % |

Suggested sequencing (amended by A1): **7 ∥ 8 → Pilot A → 9 → (10 ∥ 12) →
11 ∥ 13 → 18 → 14 → 15 → 17**, with 16 decided after 11. Phase 18 can be
prototyped in the research lab straight after Pilot A.

---

## 8. Detailed phases

### Phase 7: Evaluation harness and confound removal  *(R, M)*

**Goal.** A trustworthy measurement of how accuracy scales with data, and
of how far the current renderer is from its own ceiling.

Tasks
- [ ] **Data splits.** FiveK expert C at the 480p protocol used by Zeng et
      al. / AdaInt (4,500 train / 500 test, short side 480 px). Freeze file
      lists as manifests with SHA-256 hashes in `bench/manifests/`.
- [ ] **Metrics.** PSNR, SSIM, ΔE2000 (CIELAB), L1, LPIPS (optional);
      per-image CSV + summary JSON; bootstrap 95 % CIs.
- [ ] **Baselines.**
  - identity (no edit);
  - mean-parameter edit (average θ over the train set, same for all images);
  - per-channel histogram matching to the train-target mean CDF;
  - **per-image oracle**: optimise θ directly against each test target
        with Adam (renderer's ceiling);
  - published numbers for context (Zeng 25.21, SepLUT 25.42, CSRNet 25.17).
- [ ] **Training fixes.**
  - ImageNet normalisation before ResNet;
  - frozen ResNet, features cached to disk (`.npy`/`.safetensors`);
  - remove BatchNorm from the head (LayerNorm or none);
  - validation split (10 % of train) + early stopping on val PSNR;
  - aspect-preserving resize (short side 256 for features; loss at 480p);
  - global seed control (`torch`, `numpy`, `random`), deterministic loaders.
- [ ] **Learning curve.** 25 / 50 / 100 / 250 / 500 / 1,000 / 4,500 pairs × 3
      seeds; plot PSNR vs. log(#pairs) with baselines and oracle as lines.
- [ ] **Loss ablation.** L1 only vs. L1 + CDF vs. full composite (VGG is
      expensive on CPU; keep only if it helps).
- [ ] **Report.** `tests/reports/phase7_learning_curve.md` with figure.

Deliverables: `bench/` harness, `photostyle bench` command, learning-curve
figure, report.

Gate: if accuracy rises with data after fixes → retire the ceiling story
and adopt the curve as the small-data baseline. If it still falls →
investigate optimisation/labels before any renderer work.

### Phase 8: Engineering foundations  *(I, M, parallel with 7)*

**Goal.** Turn a research script collection into a maintainable package.

Tasks
- [ ] Restructure to `src/photostyle/` (see §6 components); keep thin
      compatibility shims for `train.py`, `inference.py` until Phase 12.
- [ ] Config system: dataclasses + YAML (`configs/*.yaml`), CLI overrides.
- [ ] Logging: structured JSON logs per run in `runs/<timestamp>_<name>/`
      (config, git SHA, metrics, checkpoints); optional TensorBoard.
- [ ] Tests with `pytest`:
  - renderer identity at zero parameters (exact);
  - monotonicity of curves and LUTs after regularisation;
  - gradient flow to every parameter;
  - `EditParams` serialisation round trip and `lerp` properties;
  - golden images for each renderer backend;
  - dataset loaders on a 3-image fixture.
- [ ] CI (GitHub Actions): `ruff`, `ty`, `pytest` on CPU, `detect-secrets`;
      cache uv environment; < 10 min.
- [ ] `Makefile` / `just` targets: `setup`, `test`, `lint`, `bench-small`,
      `serve`.
- [ ] Remove stubs (`main.py`, `test.py`, print in `misc_funcs.py`); move
      legacy Fujifilm architecture to `legacy/` with a clear label.

Gate: CI green on main; `uv sync && make test` works from a clean clone.

### Phase 9: Renderer v2 — SepLUT-style cascade  *(R/E, M)*

**Goal.** A more expressive yet still editable global renderer.

Tasks
- [ ] **Per-channel monotone curves** (K = 9–17 knots each), parameterised as
      cumulative softplus increments so monotonicity holds by construction;
      optional master luminance curve.
- [ ] Keep **3×3 matrix + bias**; add read-only derived views (temperature,
      tint, saturation) for the UI.
- [ ] **Residual 9³ LUT** = identity + Σₙ wₙ·Bₙ with N ∈ {4, 8} shared
      bases; trilinear lookup via `torch.nn.functional.grid_sample`
      (CPU-friendly, no custom CUDA); Zeng's TV (1e-4) and monotonicity (10)
      regularisers.
- [ ] Grain (amount, size) and vignette (amount, midpoint) as explicit
      sliders; grain excluded from the training loss or trained with a
      noise-aware loss (it is not learnable from single pairs).
- [ ] Head resized for ~40–60 outputs (e.g. 1280 → 256 → P).
- [ ] **Ablation grid** at 100 / 250 / 4,500 pairs: current renderer →
      per-channel curves → + matrix → + LUT (N = 4, 8) → Zeng 33³ × 3 →
      CSRNet-style baseline (36K params).
- [ ] Re-run the Phase 7 oracle for each renderer (ceiling per renderer).
- [ ] Re-validate synthetic styles as sanity tests: the model must recover
      Fujifilm/cyberpunk parameters on held-out images (PSNR ≥ 35 dB).

Gate: keep the LUT stage only if it beats curves + matrix by > 0.3 dB at
≤ 250 pairs; otherwise ship curves + matrix (fully slider-editable).

### Phase 10: Style conditioning and few-shot styles  *(R, L)*

**Goal.** One model for many looks; new looks from a few examples.

Tasks
- [ ] **FiLM** conditioning of every head layer on a 32–64-D style code.
- [ ] **Style embedding table** for known styles: FiveK experts A–E,
      PPR10K experts a–c (if adopted, §10), synthetic styles.
- [ ] **Style encoder**: maps cached features of n "after" images to a code;
      mean-pool; trained on random style subsets (n ∈ [1, 32]) so it
      tolerates small n (StarEnhancer / PieNet recipe).
- [ ] **Contrastive auxiliary loss** on style codes (same style closer than
      different styles) to sharpen subtle expert differences.
- [ ] **Few-shot protocols** (train on A–D, test on E):
  - (a) encoder code from n = 5 / 10 / 20 / 50 unpaired "after" images;
  - (b) embedding-only fine-tune on n pairs (CSRNet-style);
  - (c) FiLM-only fine-tune on n pairs;
  - (d) separate model trained from scratch on n pairs.
- [ ] **Style-space analysis**: t-SNE/UMAP of codes; interpolation sanity
      (blending two codes gives an edit between the two).
- [ ] Report: `tests/reports/phase10_conditioning.md`.

Gate: conditioning kept if the shared model ≥ separate models at equal data
per style, and few-shot (a) or (b) at n = 20 is within 1 dB of (d) at
n = 500.

### Phase 11: Unpaired styles  *(R, L)*

**Goal.** Learn looks like cyberpunk or film from example photos only.

Tasks
- [ ] **Losses** (ablated individually and together):
  - batch-level **CDF / sliced-Wasserstein** distance between rendered
        outputs and the style set in RGB and Lab;
  - luminance-conditioned colour statistics (e.g. shadow / mid / highlight
        chroma means) to capture split-toning;
  - **identity anchor** (‖R(I) − I‖ small in structure; VGG fidelity);
  - **distort-and-recover pseudo-pairs**: perturb in-style images with
        random global edits, learn to recover them;
  - optional small **WGAN-GP critic** on 64×64 thumbnails.
- [ ] **Measurable protocol**: FiveK expert-C retouches of *disjoint* images
      as the "style set" (Zeng's unpaired protocol) at 25 / 100 / 500 /
      2,250 images → PSNR remains computable.
- [ ] **Real style sets**: collect ~300 images each for "neon/cyberpunk",
      "warm film", "cool matte" (licence-clean sources, §10).
- [ ] **Content-adaptivity metric**: variance of predicted θ across a
      diverse input set (must be > 0 and correlated with input statistics).
- [ ] Small **preference study** (2AFC): adaptive model vs. the old
      fixed-function style vs. histogram matching.

Gate: adopt the simplest loss combination within ~0.5 dB of the best on
the FiveK protocol and preferred ≥ 60 % over the fixed preset.

### Phase 12: Engine API, exporters and performance  *(E, M)*

**Goal.** A fast, stable engine that everything else builds on.

Tasks
- [ ] `EditParams` v1 (schema above), with JSON Schema file and migrations.
- [ ] `Engine` API and CLI (§6); deprecate `train.py` / `inference.py`.
- [ ] **Full-resolution path**: predict θ on a 256 px proxy; bake curves ∘
      matrix ∘ LUT into one 33³ (or 65³) LUT; apply to full-res image with
      trilinear interpolation (NumPy, vectorised, tiled); add grain and
      vignette at full res.
- [ ] **16-bit and colour management**: process TIFF 16-bit without
      banding; honour embedded ICC (convert to sRGB working space, tag
      output); preserve EXIF/XMP metadata.
- [ ] **RAW ingest** with rawpy: neutral, documented develop settings (camera
      WB, no auto-bright, linear → sRGB tone); cache developed proxies.
- [ ] **Exporters**:
  - `.cube` per image and per-style average (`TITLE`, `LUT_3D_SIZE`,
        `DOMAIN_MIN/MAX`); verify in DaVinci Resolve and Photoshop;
  - Lightroom/ACR **XMP preset** (approximate): per-channel tone curves map
        to `ToneCurvePV2012Red/Green/Blue`; the LUT stage cannot be expressed
        as sliders, so offer "curves-only preset" + separate LUT;
  - JSON (`EditParams`) sidecar for re-editing.
- [ ] **Performance**: predict + render 12 MP < 1 s on a 4-core laptop CPU;
      optional ONNX export of the head + encoder; benchmark script.
- [ ] **OOD score**: Mahalanobis distance of image features to the style's
      training feature distribution (stored in the style card).

Gate: all product latency targets met; `.cube` ≥ 45 dB vs. in-app render;
ONNX output matches PyTorch within 1e-4.

### Phase 13: Studio v1 — web app  *(P, L)*

**Goal.** A polished local app for applying and adjusting styles
(use cases 1 and 5). Full specification in §9.

Tasks
- [ ] Stack decision (recommendation): **FastAPI** backend + **Svelte or
      React** front end + **WebGL2** renderer (curves as 1D textures, LUT as
      a 3D texture) so slider changes never round-trip to Python. Keep the
      Streamlit app as `lab/` for research.
- [ ] Screens: Library, Editor, Style gallery, Export (§9.2).
- [ ] Before/after: split slider, toggle (`\` key), side-by-side.
- [ ] Style strip with live thumbnails of each style on the current photo.
- [ ] Adjustment panels bound to `EditParams`; "predicted" vs. "edited"
      indicators per field; reset per field / per panel.
- [ ] Histogram + per-channel CDF overlay (before vs. after) — the project's
      CDF heritage as a user-facing tool.
- [ ] Undo/redo history; non-destructive edits saved as JSON sidecars.
- [ ] Export dialog (image formats, `.cube`, XMP, JSON).
- [ ] Golden tests: GLSL render equals NumPy render (≤ 1/255).
- [ ] Usability test with 3–5 people; fix top issues.

Gate: a first-time user can apply a style, adjust it and export a JPEG and
a `.cube` without instructions in under 3 minutes.

### Phase 14: Style creation wizard and personal styles  *(P/R, L)*

**Goal.** Users create their own styles (use cases 2 and 3).

Tasks
- [ ] Wizard flow (§9.3): choose source → import → quality check → train →
      review → save.
- [ ] **Importers**: before/after folders (match by filename/EXIF time);
      Lightroom export pairs; a folder of unpaired inspiration images;
      RAW+JPEG from the same camera (neutral RAW develop as "before").
- [ ] **Dataset quality checks**: duplicates (perceptual hash), mismatched
      pairs (geometry check via downsampled correlation), too few images,
      low diversity (feature-space coverage), crops/rotations between
      before and after (warn: renderer can't learn geometry).
- [ ] **Background training job** with progress, ETA, cancel; runs on CPU
      using cached features; uses Phase 10 few-shot path when n is small
      and fine-tunes when n is large.
- [ ] **Honest review screen**: held-out examples (before / target / model)
      and a quality score vs. baselines on the user's own data.
- [ ] **Style cards**: name, description, cover image, source type, #examples,
      training date, metrics, OOD reference stats, version history.
- [ ] Style import/export as a single `.photostyle` (zip) file for sharing.

Gate: a photographer creates a usable style from 50 pairs in < 10 minutes
end to end, and the review screen's metrics match an offline benchmark.

### Phase 15: Personalisation loop, series consistency, blending  *(P/R, L)*

**Goal.** Features that make the tool feel like it learns *with* the user.

Tasks
- [ ] **Learn from corrections**: when a user edits a predicted photo and
      clicks "remember", store (image features, final params) as a new
      pair; periodically fine-tune the style's embedding (cheap, seconds).
      Measure: prediction error on the user's next photos decreases.
- [ ] **Series consistency** for a shoot: cluster photos by time/scene;
      regularise θ within a cluster (shrink toward the cluster mean) with a
      user-controlled "consistency" slider. PPR10K's group-level consistency
      metric can serve as the benchmark.
- [ ] **Style blending**: interpolate style codes (and basis weights);
      2-D "style map" UI where the user drags a point between styles.
- [ ] **Match a reference photo**: pick one reference image; the encoder
      makes a one-shot style code; show confidence.
- [ ] **Auto-strength**: predict a per-image strength so already-in-style
      photos get lighter edits (evaluated on the identity-anchor data).

Gate: corrections reduce error on subsequent photos in a simulated
user study (replaying FiveK expert edits as "corrections").

### Phase 16 (conditional): Regional edits  *(R, L)*

**Goal.** Region-dependent edits (sky vs. face) — only if Phases 7–11 show
that global edits leave a clear gap, e.g. the per-image oracle's residual
error is concentrated in specific regions.

Candidates, in order of editability:
- [ ] StarEnhancer-style **x/y spatial curves** (smooth gradients);
- [ ] a few explicit **graduated / radial filters** (DeepLPF-style) exposed as
      draggable masks in the UI;
- [ ] **semantic masks** (sky, skin, foreground) with per-mask curve offsets
      (RSFNet-style), masks from a small pretrained segmenter;
- [ ] trade-off documented: per-image `.cube` export no longer captures the
      whole edit; exporters must fall back to image export + mask files.

Gate: ≥ 0.5 dB gain on FiveK and a visible win in the preference study.

### Phase 17: Packaging, documentation and release  *(I/P, M)*

Tasks
- [ ] Installable package (`uv tool install photostyle` / `pipx`); optional
      desktop wrapper (Tauri or PyInstaller) with bundled weights.
- [ ] Documentation site (MkDocs Material): quick start, user guide with
      screenshots, style-creation guide, API reference, research notes,
      reproducibility guide.
- [ ] **Model cards** and **style cards** for built-in styles: data source,
      licence, metrics, known failure cases.
- [ ] Public demo (e.g. Hugging Face Space) with built-in styles only; no
      uploads stored.
- [ ] Versioning (SemVer), CHANGELOG, release workflow building wheels.
- [ ] Short technical write-up / paper draft from Phases 7–11 results.

---

## 9. UI / UX specification

### 9.1 Information architecture

```
Studio
├── Library        (import, browse, select photos; batch actions)
├── Editor         (one photo: style, strength, panels, compare, export)
├── Styles         (gallery of built-in + personal styles; create, blend, share)
│   └── Create style wizard
├── Batch          (apply to many; series consistency; progress; export queue)
└── Settings       (performance, colour management, privacy, shortcuts)
```

### 9.2 Editor layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ◀ Library   IMG_0042.CR3                         [Before|After ▾]  Export ▸ │
├───────────────────────────────────────────────────────┬──────────────────────┤
│                                                       │ STYLE                │
│                                                       │ [Expert C ▾] ⓘ       │
│                 image canvas                          │ Strength ━━━●━━ 85 % │
│         (split-view divider draggable ⇆)              │ ⚠ Unusual photo for  │
│                                                       │   this style (why?)  │
│                                                       ├──────────────────────┤
│                                                       │ ▸ Curves   R G B L   │
│                                                       │ ▸ Colour   temp tint │
│                                                       │            sat  mat. │
│                                                       │ ▸ Look     LUT str.  │
│                                                       │            ingredients│
│                                                       │ ▸ Grain & vignette   │
│                                                       │ ▸ Histogram / CDF    │
├───────────────────────────────────────────────────────┴──────────────────────┤
│ Style strip:  [orig] [Expert C] [Warm film] [Neon] [My wedding] [+ Create]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

Interaction details
- **Style strip** renders every style on the current photo as thumbnails
  (predict is cheap on cached features; thumbnails render in WebGL).
  Hover = temporary preview on the canvas; click = apply.
- **Strength** scales all parameters toward identity (0–150 %); 100 % is
  the model's prediction. Double-click resets.
- **Curves panel**: R, G, B and luminance tabs; draggable knots; predicted
  curve shown as a faint ghost line after the user edits, so they can see
  how far they moved from the model.
- **Colour panel**: temperature, tint, saturation sliders derived from the
  matrix; "advanced" reveals the raw 3×3 matrix and bias.
- **Look panel**: LUT strength plus *look ingredients* — the basis-LUT
  weights shown as named sliders (names learned by inspecting what each
  basis does, e.g. "teal shadows", "warm highlights"); a colour-wheel
  visualisation of the LUT's hue shifts.
- **Per-field provenance**: a dot next to each control — hollow = model
  prediction, filled = user override; "Reset to model" per panel.
- **Compare modes**: split (drag divider), toggle (hold `\`), side-by-side,
  and "difference" (amplified |after − before|) for pixel peepers.
- **Histogram/CDF panel**: before (grey) and after (colour) per channel;
  optional target-style CDF band for unpaired styles ("how close to the
  look am I?").
- **OOD warning**: shown when the OOD score exceeds a threshold, with an
  explanation ("This style was learned from daylight portraits; this photo
  is a night scene") and a suggestion (lower strength, or pick another
  style).
- **Explain panel** (ⓘ): a plain-language summary of the edit, generated
  from θ: "Lifted shadows (+12 %), warmed ≈ 300 K, reduced saturation in
  greens, light vignette."

### 9.3 Create-style wizard

1. **Source**: "My before/after edits" · "Photos in a style I like" · "My
   camera's RAW+JPEG" · "Import a .photostyle file".
2. **Import**: drag folders; automatic pairing preview; counts; problems
   highlighted inline (unpaired files, duplicates, crops).
3. **Check**: diversity map (feature-space scatter coloured by cluster),
   warnings ("All your examples are outdoors — indoor photos may be edited
   poorly"), recommended minimum counts.
4. **Train**: progress with ETA; live preview on 3 held-out photos updating
   each epoch; cancel/resume.
5. **Review**: held-out grid (before / your edit / model), score vs.
   baselines ("closer to your edits than a single preset by X"), accept or
   add more examples.
6. **Save**: name, cover image auto-chosen, description, optional tags.

### 9.4 Batch

- Select photos → choose style → optional **consistency** slider (0 = fully
  per-image, 1 = one shared edit per scene cluster) → preview grid of 12
  random samples → export queue with progress and per-file status →
  summary report (failures, OOD images flagged for manual review).

### 9.5 Styles gallery

- Cards with cover, source badge (paired / unpaired / built-in), #examples,
  quality score, version.
- Actions: apply, duplicate, rename, retrain with more data, export
  `.photostyle`, export average `.cube`, delete (with undo).
- **Style map**: 2-D projection of style codes; drag a point to blend; save
  a blend as a new style.

### 9.6 Keyboard shortcuts (initial)

| Key | Action |
|---|---|
| `\` (hold) | Show original |
| `Y` | Cycle compare mode |
| `1`–`9` | Apply style *n* from the strip |
| `[` / `]` | Strength −/+ 5 % |
| `R` | Reset all to model prediction |
| `Ctrl/Cmd+Z`, `Shift+Ctrl/Cmd+Z` | Undo / redo |
| `E` | Export dialog |
| `←` / `→` | Previous / next photo |

### 9.7 Accessibility and quality bar

- WCAG 2.1 AA contrast for chrome; UI chrome is neutral grey so it doesn't
  bias colour judgement (industry convention for editors).
- All controls keyboard-reachable; sliders accept typed values; ARIA
  labels on every control.
- No information conveyed by colour alone (provenance dots also differ in
  shape).
- Reduced-motion setting honoured.
- Responsive down to a 1280 px-wide laptop; read-only viewer on tablet.

### 9.8 Performance budget (UI)

| Interaction | Budget |
|---|---|
| Open 24 MP JPEG to first preview | < 1.5 s |
| Style strip thumbnails (8 styles) | < 2 s |
| Slider drag frame | < 16 ms (60 fps) at preview size |
| Apply new style (predict + upload θ) | < 300 ms |
| Full-res export 24 MP | < 2 s |

---

## 10. Data strategy

| Source | Type | Size | Use | Notes |
|---|---|---|---|---|
| **MIT-Adobe FiveK** (Bychkovsky et al. 2011) | Paired, 5 experts | 5,000 RAW + 5 × 5,000 edits | Main benchmark; multi-style training; few-shot hold-out (expert E) | Research licence; streamed via `logasja/mit-adobe-fivek` |
| **PPR10K** (Liang et al., CVPR 2021) | Paired portraits, 3 experts, grouped by shoot, human masks | ~11K photos | Portrait styles; **series consistency** benchmark; masks for Phase 16 | Check licence before use |
| **HDR+ burst** (Hasinoff et al. 2016) | Paired (merged RAW → Google's rendering) | ~3.6K bursts | Robustness; a "camera look" style | Used by Zeng et al. |
| **User's own Lightroom exports** | Paired | 10s–1000s | Personal styles (the product's core) | Stays local |
| **RAW+JPEG from cameras** (e.g. Fujifilm film simulations) | Paired, content-adaptive camera processing | Any | "Camera look" styles — real, adaptive Fujifilm styles replace the synthetic one | Needs a documented neutral RAW develop |
| **Unpaired style sets** (cyberpunk, film, matte…) | Unpaired | ~300 per style | Phase 11 | Use CC-licensed sources (Openverse, Wikimedia Commons, Unsplash licence); record licences in a manifest |
| **Synthetic fixed-function styles** (existing generators) | Paired | Unlimited | Unit/sanity tests of renderer and training | Content-independent by construction |
| **Unedited photo pool** (picsum / Unsplash / COCO) | Inputs only | 1–10K | Inputs for unpaired training; OOD calibration | |
| **Neon Street Portrait stand-in set** *(A1)* | Unpaired, openly licensed (CC0 / PDM / BY / BY-SA) via Openverse | ~100–150 | Pilot A style set | Collected by `tools/collect_style_set.py`; manifest + attribution in `data/manifests/neon_street_portrait.csv` |

Rules
- Every dataset gets a **manifest** (file list, hashes, licence, source
  URL, date) under `data/manifests/`; nothing enters training without one.
- **Splits are fixed** and versioned; test images never appear in style
  sets or feature caches used for training.
- Personal data stays on the user's machine; built-in styles use only
  licence-compatible data and ship with style cards stating provenance.

---

## 11. Evaluation protocol

- **Primary benchmark**: FiveK expert C, 480p protocol, 4,500/500 split;
  report PSNR / SSIM / ΔE2000 with bootstrap CIs; 3 seeds; mean ± std.
- **Learning curves**: 25 → 4,500 pairs, log-x axis, baselines + oracle.
- **Multi-style**: experts A–E; per-expert and mean; leave-one-expert-out.
- **Unpaired**: FiveK disjoint-image protocol (PSNR) + style-set SWD +
  parameter-spread + preference study.
- **Consistency**: PPR10K group-level metric (variance of edits within a
  shoot vs. expert).
- **Robustness**: OOD sets (night, indoor tungsten, snow, high-ISO noise);
  report failure modes with images.
- **Export fidelity**: `.cube` vs. in-app render PSNR and max error;
  GLSL vs. NumPy golden images.
- **Human evaluation**: 2AFC preference with ≥ 10 participants × 30 image
  pairs; randomised left/right; report binomial CIs.
- **Reporting**: every result table in the README is generated by
  `photostyle bench` and stored with config + git SHA under `bench/results/`.

---

## 12. Engineering standards

- Python ≥ 3.10, `uv`, `ruff` (lint + format), `ty` (types), `pytest`,
  `pre-commit`, `detect-secrets` (already configured).
- Type hints on public APIs; docstrings with shapes for tensor functions.
- No notebooks in `src/`; exploratory notebooks go in `notebooks/` and are
  not imported.
- Checkpoints and datasets never committed; built-in style weights
  distributed via GitHub Releases or Hugging Face Hub with checksums.
- Every PR: CI green, tests for new behaviour, README/doc updates if user
  facing.
- Reproducibility: runs record config, seed, git SHA, data manifest hash.

---

## 13. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Learning curve still degrades after fixes | Medium | High | Phase 7 gate stops renderer work; investigate label noise, loss, optimisation; try CSRNet baseline to separate model vs. data issues |
| Few-shot codes too noisy for subtle photographer styles (StarEnhancer Recall@1 as low as 24.6 %) | High | Medium | Contrastive code loss; fall back to embedding fine-tune on a few pairs; show quality score honestly |
| Unpaired styles look washed out or inconsistent | Medium | High | Combine distribution loss with pseudo-pairs; critic as last resort; preference study gate |
| Basis LUTs learned on FiveK don't span stylised looks (cyberpunk) | Medium | Medium | Per-style extra basis LUTs; larger N; fine-tune bases for unpaired styles |
| GLSL / NumPy / PyTorch renders drift apart | Medium | Medium | Golden-image tests in CI with tolerance 1/255 |
| CPU training too slow for good UX | Low | Medium | Feature caching; small heads; progress + preview during training |
| Dataset licences restrict distributing built-in styles | Medium | Medium | Ship FiveK-derived styles as research-only; build public styles from CC data |
| Lightroom XMP export can't represent the LUT stage | High | Low | Offer curves-only preset + separate LUT; document clearly |
| Scope creep (UI before research gates pass) | High | Medium | Track dependencies in §7; Studio v1 only needs Phase 12 |
| Colour-management bugs (wrong gamma/ICC) | Medium | High | Explicit working space; test images with known ICC profiles; compare with reference tools |

---

## 14. Non-goals

- Generative editing (adding/removing objects, generative relighting, sky
  replacement). *A1 clarification:* parametric, mask-based light shaping and
  synthetic depth of field (Phase 18) are in scope, because they only
  re-weight and blur existing pixels.
- Training on an identifiable artist's copyrighted work without licence or
  consent, or scraping platforms against their terms (A1 policy).
- Geometry changes (crop, rotate, lens correction, perspective).
- Denoising, sharpening, super-resolution (LUTs amplify noise; out of scope
  except the grain slider).
- Video-specific features beyond `.cube` export (no temporal model in v1).
- Cloud hosting of user photos or accounts.
- Matching Lightroom's full slider set one-to-one.

---

## 15. Open questions for the project owner

1. **Primary audience**: research codebase first, or product (Studio) first?
   This plan front-loads research gates; a product-first plan would start
   Phase 13 against the current renderer.
2. **Front-end stack**: FastAPI + Svelte/React + WebGL (recommended), stay
   on Streamlit, or a desktop app (Tauri)?
3. ~~**Which looks matter most?**~~ *Answered in A1:* first look is a
   night/day street-portrait look inspired by the photographer Bleg
   (Neon Street Portrait). Others remain open.
4. **Real Fujifilm data**: do you have access to a Fujifilm (or other)
   camera for RAW+JPEG pairs?
5. **Datasets**: is PPR10K's licence acceptable for your use?
6. **Distribution**: open-source release and public demo, or private?
7. **Compute**: strictly CPU, or is occasional GPU (e.g. Colab) acceptable
   for benchmarks at 4,500 pairs?

---

## 16. Backlog and housekeeping

Immediate (fold into Phase 8)
- [ ] Remove PyCharm stub `main.py`, matplotlib demo `test.py`, print in
      `misc_funcs.py`.
- [ ] Move the legacy 7-D Fujifilm architecture to `legacy/`.
- [ ] Replace square resize with aspect-preserving resize.
- [ ] Add `tests/` unit tests (currently only reports live there).

Ideas parking lot (not scheduled)
- Hue-vs-saturation and hue-vs-luminance curves (CURL-style) as an HSL panel.
- Adaptive LUT resolution (AdaInt) if accuracy gaps appear in saturated hues.
- In-browser inference (ONNX Runtime Web) for a zero-install demo.
- Mobile companion app that applies exported LUTs.
- Plug-ins: Lightroom Classic (Lua SDK) or Photoshop (UXP) calling the local
  server.
- "Explain this style" page: visualise what each basis LUT does on a colour
  checker and a skin-tone ramp.
- Style marketplace/sharing format with signed style cards.
- Temporal smoothing of θ for video sequences.

---

## 17. Amendments

### A1 — First target look, capture emulation, and style-data policy

*Adopted 2026-09-23. Status: active.*

**Why.** The project owner chose a first concrete look: the night/day
street-portrait style of the photographer **Bleg** (Bleg Bayraktar,
@itsbleg), studied in [`research_notes/styles/bleg.md`](research_notes/styles/bleg.md).
Three consequences did not fit the plan as written:

1. The look is only available as finished photos (no before/after pairs),
   and those photos are the photographer's copyrighted work.
2. Much of the look comes from capture, not grading: shallow depth of field
   with neon bokeh, and a subject that is brighter and warmer than a darker,
   cooler background. The plan had no place for these.
3. The plan front-loaded research gates and gave no early, visible end-to-end
   result on a look the owner actually cares about.

A1 answers each in turn.

#### A1.1 Style-data policy (project-wide)

These rules apply to every style from now on, built-in or experimental.

- **Only openly licensed or consented data is stored or trained on.**
  Allowed licences: CC0, Public Domain Mark, CC BY, CC BY-SA. Also allowed:
  data the user owns, or data with the author's written permission. NC and
  ND licences are excluded, so built-in styles stay redistributable.
- **No scraping against platform terms** (e.g. Instagram), and no bulk
  downloading of an identifiable artist's portfolio.
- **Reference artists may be studied, not copied.** Looking at a small
  number of publicly published images and recording *summary statistics*
  (black point, split-tone direction, saturation) is allowed; these numbers
  guide data selection. Their images are not stored in the repo or used as
  training data.
- **Descriptive names.** Styles are named after what they look like
  (*Neon Street Portrait*), not after people. An artist's name may appear
  only in research notes as "inspired by", unless the artist has consented.
- **Manifests are mandatory.** Every image has a row with licence, creator,
  source URL, attribution text and SHA-256 in `data/manifests/`. Built-in
  styles ship with a style card that lists the attributions.
- **Takedown.** A creator's request removes their images from the manifest,
  and any affected style is retrained.
- **Users' own data** (their photos, or images they save themselves for
  private study) stays local and is their responsibility. The app shows a
  short notice in the style-creation wizard.

#### A1.2 Pilot A: *Neon Street Portrait* (runs right after Phase 7)

**Goal.** A first end-to-end result on the chosen look, using the Phase 7
harness and the smallest renderer upgrade that can express the look. It
also serves as an early rehearsal of Phase 11.

**The look, as a target** (from the style study):
- deep, true blacks (L\* p1 ≈ 1) with no matte lift;
- cool shadows (b\* −4 to −10), warm skin and midtones, neutral highlights;
- **night:** low-key (median L\* ≈ 15–18) with saturated, glowing neon;
- **day:** airy, soft, low saturation, almost-white highlights.

The night/day contrast is the reason this look needs a *content-adaptive*
model.

**Data**
- [x] Style set: openly licensed night and day street portraits from
      Openverse. `tools/collect_style_set.py` filters for a person as the
      main subject and scores each image against the look's colour profile.
      Manifest: `data/manifests/neon_street_portrait.csv`; images in
      `data/styles/neon_street_portrait/` (gitignored).
- [ ] Hand-review the set: drop off-look images, check night/day balance,
      aim for ≥ 100 kept.
- [ ] Input pool (the photos to be edited): FiveK *original* renders, plus a
      disjoint set of openly licensed, un-stylised night street photos.
      Night coverage matters, because FiveK has few night scenes.
- [ ] Hold out 20 % of both sets for evaluation.

**Model** (minimum viable): the Phase 7 fixed trainer, frozen cached
features, and the current renderer with **per-channel curves** brought
forward from Phase 9 (the split tone needs them). No 3D LUT yet.

**Losses** (unpaired, from Phase 11):
- a luminance-conditioned colour-statistics loss: chroma means and
  variances in shadow, mid and highlight bands, matched to the style set;
- a batch CDF / sliced-Wasserstein loss on Lab;
- an identity anchor on structure;
- distort-and-recover pseudo-pairs built from style-set images.

**Baselines**
- (i) identity;
- (ii) a **hand-built static preset**: black point to 0, blue-lifted shadow
  curve, warm mid matrix;
- (iii) per-channel histogram matching to the style set's mean CDF.

**Metrics**
- Colour-statistics distance to the held-out style set, reported separately
  for night and day.
- **Parameter spread**: predicted θ must differ between night and day
  inputs (effect size > 0.8 on at least 3 parameters).
- A small blind preference test (≥ 5 people × 30 images) against baseline
  (ii).

**Gate.** The adaptive model beats the static preset on colour-statistics
distance in **both** regimes, and is preferred in ≥ 60 % of trials. If it
fails, the result still informs Phase 11's loss choice, and Phase 9 proceeds
unchanged.

**Deliverables**
- `tests/reports/pilotA_neon_street_portrait.md` with before/after grids
  (openly licensed images only).
- A `neon_street_portrait` style card.
- An exported average `.cube`.

#### A1.3 New Phase 18: Capture emulation — depth of field and light shaping  *(R/E/P, L)*

**Goal.** Emulate the capture side of the look with *parametric, editable*
spatial operations, without generating new content.

Building blocks (pretrained, frozen, used at inference and for training
targets):
- [ ] **Monocular depth**: Depth Anything V2 **Small** (Apache-2.0; the
      Base/Large/Giant checkpoints are CC BY-NC 4.0, so they are excluded
      by A1.1).
      Run at ≤ 518 px and upsample with an edge-aware (guided) filter.
- [ ] **Subject matte**: person segmentation plus a matting refinement
      around hair and glasses. The model is chosen by licence and CPU speed,
      and it is user-correctable with a brush in Studio.

Synthetic depth of field (the **Lens** controls):
- [ ] Parameters:
  - **focus depth**, set by clicking the subject to focus;
  - **aperture**: maximum blur radius as a fraction of the image diagonal;
  - **bokeh highlight gain**: how strongly the brightest lights bloom into
    discs;
  - **bokeh shape**: disc or slight cat's-eye toward the corners;
  - **transition softness**.
- [ ] Renderer: layered depth compositing, not a single variable blur.
  1. Slice the scene into K depth layers.
  2. Blur each layer with a disc kernel after **boosting linear-light
     highlights**, so neon lights become bright discs rather than grey
     smudges.
  3. Composite back to front with occlusion-aware alpha, so the background
     never bleeds onto the subject.
  4. Keep the subject matte sharp.
  Work in linear light; add matched grain after the blur so blurred regions
  don't look plasticky.
- [ ] Differentiable approximation (a soft layer assignment) so the head can
      *predict* default lens parameters. Adaptive example: wide street scenes
      get more blur, tight portraits less.

Light shaping (the **Light** controls):
- [ ] Parameters: **subject exposure, subject warmth, background exposure,
      background coolness, feather**, and an optional **directional
      gradient** (angle and strength) to suggest key-light direction.
- [ ] These are mask-weighted offsets to the existing curves and matrix, so
      they only re-weight existing pixels. True 3D relighting stays a
      non-goal.

Evaluation:
- [ ] **Halo audit**: manual review of 50 images for edge artefacts around
      hair, glasses, hands and transparent objects. Pass if ≥ 80 % show no
      visible halo at 100 % zoom.
- [ ] **Bokeh realism check**: compare blur-disc statistics of rendered
      night lights with real shallow-depth-of-field night photos in the
      openly licensed set.
- [ ] **Preference test**: Neon Street Portrait grade with Lens + Light vs.
      the grade alone, ≥ 60 % preferred.
- [ ] **Performance**: depth, matte and blur on a 12 MP photo in < 3 s on a
      laptop CPU (depth runs on a proxy; the blur is tiled).

Export consequences:
- Lens and Light are spatial, so a `.cube` file can no longer carry the full
  edit. Exporters write the graded image plus the `.cube` (colour only), and
  optionally the depth map and matte as 16-bit PNGs for use in other tools.
- The `EditParams` schema gains optional `lens` and `light` blocks (schema
  v1.1).

Studio (§9) additions:
- **Lens panel**: click-to-focus on the canvas, aperture and bokeh sliders,
  and a *depth overlay* toggle that tints near and far regions.
- **Light panel**: subject and background sliders, a gradient direction dial,
  and a *mask overlay* with refine and erase brushes.
- **Compare modes** gain a "grade only / grade + lens + light" toggle.

**Gate.** Halo audit and preference test pass. Otherwise Lens ships as
"experimental", with the controls off by default.

#### A1.4 Other changes made by A1

- **§7 roadmap:** Pilot A and Phase 18 rows added; sequencing updated.
- **§10 data:** stand-in style-set row added.
- **§14 non-goals:** clarifies that parametric light shaping and synthetic
  depth of field are in scope, and adds the style-data policy.
- **§15 open questions:** Q3 answered.
- **§13 risks**, added here:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Openly licensed stand-in set doesn't capture the look (licence filter shrinks the pool; Flickr-era images are graded differently) | Medium | High | Profile scoring plus hand review; widen queries; the user may add images they own; later, ask the photographer for consented examples |
| Depth or matte errors cause halos around hair and glasses | High | Medium | Guided upsampling; matting refinement; user brush; the halo audit gate |
| Synthetic bokeh looks fake (no highlight bloom, wrong occlusion) | Medium | Medium | Linear-light highlight boost; layered occlusion-aware compositing; grain matching |
| Style is perceived as a copy of a living artist | Low | High | A1.1 naming and data rules; no artist images in training; attribution only as inspiration |
