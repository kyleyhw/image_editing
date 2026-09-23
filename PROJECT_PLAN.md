# Project Plan: Content-Adaptive, Editable Photo Styles

## Purpose

Learn **content-adaptive photo edits toward a target look**. A *style* is a
target appearance, and the model predicts the per-image edit that takes a
given photo there: a dark street and a bright beach need different edits to
reach the same look.

- **Supervision.** A look is specified either by **paired examples** (a
  photographer's before/after edits, e.g. MIT-Adobe FiveK experts, or
  RAW+JPEG pairs) or by **unpaired examples** (a set of photos already in the
  style, e.g. cyberpunk or film).
- **One model, many looks.** A single network is conditioned on a style code:
  a learned embedding for known styles, or one computed from a few example
  images for new styles.
- **Editable output.** The model outputs parameters (per-channel curves,
  colour matrix, a small LUT, grain, vignette) that a user can inspect and
  adjust, and that bake to a `.cube` file. It never generates pixels, so it
  cannot invent content.
- **Constraints.** CPU-trainable; useful with tens to hundreds of examples
  per style.

The rationale and literature review behind this purpose are in
[`reports/project_direction.md`](reports/project_direction.md) (one-page
summary) and
[`reports/Content adaptive photo edit models.md`](reports/Content%20adaptive%20photo%20edit%20models.md)
(full comparison of 18 methods).

### What counts as success

- On held-out FiveK images (≥ 100, 3 seeds), the model beats the
  identity, "apply the average edit", and histogram-matching baselines on
  PSNR / ΔE, and is compared against published numbers under the 480p
  protocol of Zeng et al.
- Accuracy **does not fall** as training data grows, and a learning curve
  from 25 to 4,500 pairs shows how few pairs a photographer needs.
- For unpaired styles, predicted parameters **vary with input content**
  (measurable spread) and outputs match the style set's colour statistics
  better than a fixed-function preset.

## Status: Phases 1–6 (complete, groundwork)

These phases built the infrastructure under the original, broader goal
("a general-purpose stylization engine"). They remain useful, but their
role has changed: the synthetic styles are now **sanity tests** for the
renderer (can the model recover known, content-independent parameters?),
not deliverables.

- [x] **Phase 1 – Fujifilm data pipeline.** `StyleGenerator` base class,
  `FujifilmGenerator`, `generate_dataset.py`.
- [x] **Phase 2 – Hybrid model.** Differentiable CDF + ResNet-18 features,
  7-D Fujifilm head and renderer. *Legacy: its effect ratio is ~0 %.*
- [x] **Phase 3 – Generic renderer.** Tone curve, 3×3 colour matrix + bias,
  grain, vignette (21 parameters, identity at init); composite
  L1 + VGG + CDF loss.
- [x] **Phase 4 – Cyberpunk.** Trained with no model changes.
- [x] **Phase 5 – Tilt-shift.** Focus band with three global parameters;
  per-pixel parameter maps deferred.
- [x] **Phase 6 – UI and real data.** Streamlit UI; FiveK streaming
  downloader and loader; expert-C model.

**Lessons carried forward**

1. Fixed-function styles do not need a network: the network only earns its
   place when the right edit depends on the image.
2. FiveK expert C, 80 pairs: held-out mean L1 to expert 0.0917 → 0.0741
   (≈ 19 % closer, 4 / 5 images improved). At 500 pairs: 0.1030
   (1 / 5 improved). This used 5 test images, one seed and no baselines, and the
   500-pair run had confounds (see Phase 7), so it is **not** yet evidence of
   an architectural ceiling. See
   [`tests/reports/mit5k_expert_c_report.md`](tests/reports/mit5k_expert_c_report.md).
3. The literature shows image adaptivity carries most of the FiveK gain
   (fixed LUT 20.4 dB → adaptive global LUT 25.2 dB); spatial processing
   adds only ~0.3–0.5 dB. Global edits are the right first target.

## Roadmap

Each phase ends with a decision gate. Later phases may be dropped or
reordered depending on earlier results.

### Phase 7: Evaluation harness and confound removal
**Goal:** a trustworthy measurement of how accuracy scales with data.

- [ ] Evaluation script: fixed FiveK test split (≥ 100 images), PSNR,
      ΔE2000, L1; seeds; results written to a table.
- [ ] Baselines: identity, mean-parameter edit, per-channel histogram
      matching, and a per-image **oracle** (renderer parameters optimised
      directly against each target) to measure the renderer's ceiling.
- [ ] Training fixes: ImageNet normalisation; frozen ResNet with cached
      features; no BatchNorm in the head (or LayerNorm); validation split
      with early stopping; aspect-preserving resize; seeds.
- [ ] Learning curve: 25 / 50 / 100 / 250 / 500 / 4,500 pairs × 3 seeds.
- **Gate:** if accuracy now rises with data, retire the "capacity ceiling"
  explanation; the curve becomes the small-data baseline.

### Phase 8: Renderer upgrade (SepLUT-style cascade)
**Goal:** a more expressive but still editable global renderer.

- [ ] Per-channel monotone tone curves (replacing the single shared curve).
- [ ] Keep 3×3 matrix + bias.
- [ ] Residual 9³ 3D LUT as a blend of N ≈ 4–8 learned basis LUTs, with
      Zeng et al.'s smoothness and monotonicity regularisers; trilinear
      lookup via `torch.nn.functional.grid_sample` (CPU-friendly).
- [ ] Smaller head sized for the output (~40–60 parameters).
- [ ] Ablation: current renderer → per-channel curves → + LUT (N = 4, 8) →
      Zeng 33³ × 3; plus a CSRNet baseline.
- **Gate:** keep the LUT stage only if it beats curves + matrix by
  > 0.3 dB at ≤ 250 pairs.

### Phase 9: Style conditioning (one model, many photographers)
**Goal:** a single model serving multiple looks, and new looks from few
examples.

- [ ] FiLM conditioning of the head on a 32–64-D style code.
- [ ] Learned embeddings for FiveK experts A–D (+ synthetic styles).
- [ ] Style encoder: average embedding of n example "after" images.
- [ ] Hold out expert E: compare an encoded code (n = 5 / 10 / 20),
      embedding-only fine-tuning on n pairs, and a separate per-expert model.
- **Gate:** conditioning is kept if one shared model ≥ separate models at
  equal data per expert.

### Phase 10: Unpaired styles
**Goal:** learn looks such as cyberpunk or film from example photos only.

- [ ] Losses: batch-level CDF / sliced-Wasserstein colour distribution,
      identity / fidelity anchor, distort-and-recover pseudo-pairs; optional
      small WGAN-GP critic.
- [ ] Validate on FiveK with expert-C retouches of *disjoint* images, so
      PSNR stays measurable (Zeng et al.'s unpaired protocol).
- [ ] Collect ~300-image style sets (cyberpunk, film); evaluate parameter
      spread across inputs and a small preference test against the
      fixed-function presets.
- **Gate:** adopt the simplest loss within ~0.5 dB of the best.

### Phase 11: Editing and export
**Goal:** outputs a user can take into other tools.

- [ ] Bake curves ∘ matrix ∘ LUT to a 33³ `.cube`; check PyTorch-vs-`.cube`
      fidelity (> 45 dB PSNR).
- [ ] UI: style picker / "learn from these examples" upload; expose curves,
      matrix sliders, LUT strength, grain, vignette; `.cube` download.

### Phase 12 (conditional): Regional edits
**Goal:** edits that differ between regions (sky vs. face), only if
Phases 7–10 show global edits leave a clear gap (e.g. the oracle gap is
dominated by local differences).

- [ ] Candidates: StarEnhancer-style x/y spatial curves, a few explicit
      graduated / radial filters (DeepLPF), or region masks (RSFNet).
      These trade away single-`.cube` export.

## Housekeeping

- [ ] Remove the PyCharm stub `main.py`, the matplotlib demo `test.py` and
      the print-on-import in `misc_funcs.py`; add real unit tests under
      `tests/` (renderer identity at init, gradient flow, `.cube` round trip).
- [ ] Retire or clearly label the legacy Fujifilm-specific architecture.
