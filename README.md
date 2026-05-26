# Image Editing: CDF + CNN Stylization

A general-purpose image-stylization engine that combines global CDF
analysis with a CNN-driven differentiable renderer. The same network
architecture trains on any style; only the data generator changes.

Phases 1–6 of [`PROJECT_PLAN.md`](PROJECT_PLAN.md) are implemented:
generic style-agnostic primitives, a composite L1 / VGG / CDF loss,
three trained models (Fujifilm Classic Chrome, Cyberpunk, Tilt-Shift),
a multi-style Streamlit UI, and a MIT-Adobe FiveK data loader.

## Examples of every trained mode

The figure below was produced by `python tools/make_examples.py`. Rows
are deliberately diverse sample images; columns are the original input
and the trained-model output for each style. Sample images were chosen
to exercise different aspects of each style:

- *Foggy pine forest* — strong vertical sky/ground contrast for
  tilt-shift; saturated greens for tone-curve probing.
- *Misty pastel mountains* — already warm and low-saturation, so the
  cyberpunk grade has to push aggressively to differentiate.
- *Backlit bike portrait* — full photographic scene with a centred
  human subject, ideal for showcasing tilt-shift's focus band.

![Trained-model predictions across three sample inputs: rows are samples, columns are Original / Fujifilm Classic Chrome / Cyberpunk / Tilt-shift](tests/reports/assets/style_comparison.png)

How to read it:

- **Fujifilm Classic Chrome column.** Each output is a slightly darker,
  warm-shifted version of its row's original — exactly the recipe's
  signature (negative WB-blue shift, soft highlights, ~20% vignette).
  The effect is intentionally subtle; Classic Chrome is a "look", not a
  filter.
- **Cyberpunk column.** Each output reproduces the teal-and-orange
  S-curve: shadows crushed, highlights lifted, R/G channels pulled
  down and B pushed up. The transformation is most visible on the
  pastel mountains, where the warm fog is pushed into an unmistakable
  orange-teal sunset palette.
- **Tilt-shift column.** A horizontal sharp band remains in the middle
  of every output while sky and foreground are progressively blurred.
  The model recovers the focus-band geometry to within $|\Delta| < 0.01$
  on both centre and width; the blur amount saturates at roughly half
  of the data generator's, a known optimisation artefact documented in
  `models/tilt_shift.py`.

Full quantitative interpretation (per-channel mean shifts,
effect-magnitude ratios, target-vs-prediction comparison on a held-out
test image) is in
[`tests/reports/phase3_to_phase6_report.md`](tests/reports/phase3_to_phase6_report.md).

## Documentation index

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Mathematics of the feature extractor, every primitive, and the composite loss. Identity-at-init proof. |
| [`docs/training_and_inference.md`](docs/training_and_inference.md) | Operational guide: data generation, training each architecture, CLI inference, Streamlit UI. |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase-1 design rationale and historical context that motivated the architecture choices. |
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Status ledger for Phases 1 – 6. |
| [`tests/reports/phase3_to_phase6_report.md`](tests/reports/phase3_to_phase6_report.md) | End-to-end verification, including MCP-driven UI test. |

## Mathematical overview

For an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ the
network $\varphi$ predicts a parameter vector $\theta \in \mathbb{R}^P$
which the renderer $R$ converts to a styled image
$\tilde{\mathbf{I}} = R(\mathbf{I}; \theta)$.

The feature extractor concatenates two views of the image:

- a **differentiable CDF** per channel,
  $F_{c,k} = \sum_{k' \le k} p_{c, k'}$ with soft (Gaussian-binned)
  PDF $p_{c, k}$ — global tonal/colour statistics, 768 floats;
- a **ResNet-18 global average-pooled feature** — local spatial
  structure, 512 floats.

These are concatenated to a 1280-D descriptor and fed to a small MLP
head whose output dimension matches the chosen renderer:

| Architecture | $P$ | Parameter layout |
|---|---|---|
| Fujifilm (legacy) | 7 | Highlight, Shadow, Saturation, WB Red, WB Blue, Grain, Vignette |
| Generic | $K - 2 + 14$ (21 at $K = 9$) | Tone-curve interior knots ($K-2$), $3 \times 3$ colour matrix offset (9), colour bias (3), grain (1), vignette (1) |
| Tilt-shift | 24 | Generic 21-D + focus band (centre, width, blur strength) |

The renderer's primitives are designed to be identity at zero output,
so an untrained network produces $\tilde{\mathbf{I}} = \mathbf{I}$ and
any non-zero loss decrease unambiguously reflects learned behaviour.

The composite loss is

$$
\mathcal{L} = \lambda_{\text{pixel}} \,\|\tilde{\mathbf{I}} - \mathbf{I}^\star\|_1
   + \lambda_{\text{percep}}\,\frac{1}{|\mathcal{T}|}\sum_{\ell \in \mathcal{T}}\|\Phi_\ell(\tilde{\mathbf{I}}) - \Phi_\ell(\mathbf{I}^\star)\|_1
   + \lambda_{\text{cdf}}\,\|F(\tilde{\mathbf{I}}) - F(\mathbf{I}^\star)\|_1,
$$

where $\Phi_\ell$ are activations of a frozen ImageNet-pretrained VGG-16
at four canonical taps (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`) and
$F$ is the differentiable CDF from above. Full derivations and the
HSV-faithful colour identities used by the Fujifilm renderer are in
[`docs/architecture.md`](docs/architecture.md).

## Project structure

```
.
├── PROJECT_PLAN.md            # Status ledger for Phases 1 – 6
├── README.md                  # (this file)
├── pyproject.toml             # uv-managed deps (Python ≥ 3.10)
├── uv.lock                    # pinned resolution
├── .pre-commit-config.yaml    # ruff + ty + detect-secrets hooks
├── .secrets.baseline          # detect-secrets baseline
│
├── checkpoints/               # *.pth files (gitignored)
│   └── model_fujifilm_classic_chrome.pth   # legacy Phase 1/2 ship
│
├── data_generation/
│   ├── core.py                # StyleGenerator abstract base
│   ├── mit5k_loader.py        # MIT-Adobe FiveK paired loader
│   └── styles/
│       ├── film.py            # generic S-curve + grain + vignette
│       ├── fujifilm.py        # FujifilmGenerator + CHROME_STRENGTHS
│       ├── cyberpunk.py       # teal/orange S-curve grade
│       └── tilt_shift.py      # spatially variant focus band
│
├── models/
│   ├── feature_extractor.py   # DifferentiableCDF + SpatialEncoder
│   ├── transformation_head.py # 7-D Fujifilm-specific head
│   ├── style_net.py           # Fujifilm StyleNet
│   ├── differentiable_renderer.py # DifferentiableFujifilm
│   ├── generic_renderer.py    # ToneCurve / ColorMatrix / Grain / Vignette
│   ├── generic_head.py        # 21-D generic head
│   ├── generic_style_net.py   # GenericStyleNet
│   ├── tilt_shift.py          # spatial blur primitive + composite + net
│   ├── composite_loss.py      # L1 + VGG + CDF
│   └── checkpoint_io.py       # unified load/build helper
│
├── generate_dataset.py        # CLI: picsum download + style application
├── train.py                   # CLI: train any arch
├── inference.py               # CLI: render with any checkpoint
├── image_editor_ui.py         # Streamlit multi-style UI
│
├── images/
│   ├── original/              # picsum downloads (gitignored)
│   ├── styled/                # generated pairs (gitignored)
│   └── test_images/           # tracked test images + inference outputs
│
├── docs/
│   ├── architecture.md        # math + primitives + identity proof
│   ├── training_and_inference.md # operational guide
│   └── roadmap.md             # original Phase-1 design rationale
│
├── tests/
│   └── reports/
│       ├── phase3_to_phase6_report.md  # verification narrative
│       └── assets/                     # screenshots + comparison figure
│
└── tools/
    └── make_examples.py       # regenerate the comparison figure
```

## Quick start

```powershell
# 1. Install
uv sync
uv run pre-commit install

# 2. Generate a dataset and train one style end-to-end
uv run python generate_dataset.py --style cyberpunk --count 30
uv run python train.py --arch generic --style cyberpunk --epochs 8 --image_size 192

# 3. Apply it to an image
uv run python inference.py --image_path images/test_images/climbing_test_original.jpeg --checkpoint checkpoints/model_generic_cyberpunk.pth

# 4. Or launch the UI
uv run streamlit run image_editor_ui.py
```

See [`docs/training_and_inference.md`](docs/training_and_inference.md)
for all options and a multi-style reproduction recipe that matches the
verification report.

## References

<span id="ref-he-2016">[1]</span> He, K., Zhang, X., Ren, S. & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [Link](https://doi.org/10.1109/CVPR.2016.90)

<span id="ref-simonyan-2014">[2]</span> Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556. [Link](https://arxiv.org/abs/1409.1556)

<span id="ref-johnson-2016">[3]</span> Johnson, J., Alahi, A. & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* ECCV. [Link](https://arxiv.org/abs/1603.08155)

<span id="ref-bychkovsky-2011">[4]</span> Bychkovsky, V., Paris, S., Chan, E. & Durand, F. (2011). *Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs.* CVPR. [Link](https://doi.org/10.1109/CVPR.2011.5995332)
