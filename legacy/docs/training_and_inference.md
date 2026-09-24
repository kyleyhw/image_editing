# Training and inference

Operational guide: how to generate a dataset, train each architecture,
and run inference. The mathematics is in
[architecture.md](architecture.md); status is in
[../PROJECT_PLAN.md](../PROJECT_PLAN.md).

## 1. Environment

`pyproject.toml` declares Python ≥ 3.10 and the runtime dependencies.
After cloning:

```powershell
uv sync                              # build the .venv from pyproject + uv.lock
uv run pre-commit install            # register ruff / ty / detect-secrets hooks
```

On hardware without a CUDA GPU, training runs on CPU automatically.
All commands below were tested on Windows 10, Python 3.10.19, CPU only.

## 2. Generate a synthetic dataset

```powershell
python generate_dataset.py --style <name> [--recipe <name>] --count <N>
```

Available `--style` values: `film`, `fujifilm`, `cyberpunk`,
`tilt_shift`. The Fujifilm generator additionally takes `--recipe`
(`classic_chrome` (default), `velvia`).

Behaviour:

1. If `images/original/` is empty, `<N>` images are downloaded from
   `picsum.photos` (1600 x 1200) as `picsum_NNNN.jpg`. Cap is 100.
2. Each source image is run through the chosen `StyleGenerator`,
   producing `(_original.jpg, _styled.jpg)` pairs in
   `images/styled/<style>/[<recipe>/]`.

`images/original/` and `images/styled/` are git-ignored, so generated
data does not pollute the repo.

## 3. Train a model

```powershell
python train.py --arch <arch> [options]
```

### 3.1 Architecture choices

| `--arch` | Network | Renderer | Parameters | Loss |
|---|---|---|---|---|
| `fujifilm` (default, legacy) | `StyleNet` | `DifferentiableFujifilm` | 7 | MSE |
| `generic` | `GenericStyleNet` | `DifferentiableGenericRenderer` | 21 | Composite |
| `tilt_shift` | `TiltShiftStyleNet` | `DifferentiableTiltShiftComposite` | 24 | Composite |

The legacy `fujifilm` path is retained so the shipped Phase 1/2
checkpoint still loads; new training runs should use `generic` or
`tilt_shift`.

### 3.2 Common options

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `images/styled` | Root of the synthetic dataset. |
| `--style` | `fujifilm` | Sub-folder under `data_dir`. |
| `--recipe` | `classic_chrome` | Sub-folder under `<data_dir>/fujifilm/`. |
| `--batch_size` | 4 | |
| `--epochs` | 5 | |
| `--lr` | 1e-4 | Adam learning rate. |
| `--image_size` | 256 | Resize at the dataloader. |
| `--checkpoint` | (auto) | Override save path. |

### 3.3 `--arch generic` options

| Flag | Default | Meaning |
|---|---|---|
| `--tone_points` | 9 | Number of tone-curve control points $K$. |
| `--lambda_pixel` | 1.0 | Weight of the L1-pixel loss. |
| `--lambda_perceptual` | 0.05 | Weight of the VGG-feature loss. |
| `--lambda_cdf` | 1.0 | Weight of the CDF L1 loss. |

### 3.4 `--arch tilt_shift` options

Adds `--tilt_max_sigma` (default 8.0): the pre-baked Gaussian blur
sigma. Lower values keep more sharpness in the "blurred" region.

### 3.5 MIT-Adobe FiveK

Train on real-world expert retouches:

```powershell
python train.py --arch generic --mit5k_root <fivek_path> --mit5k_expert c --epochs 8
```

The dataset must be obtained from
[data.csail.mit.edu/graphics/fivek](https://data.csail.mit.edu/graphics/fivek/);
its licence forbids redistribution. The expected layout is

```
<fivek_path>/
    original/<filename>.jpg
    expert_c/<filename>.jpg
```

with identical filenames in both subdirectories. The loader
auto-discovers the common filename set.

### 3.6 Worked example

Reproduce the report in `tests/reports/phase3_to_phase6_report.md`:

```powershell
python generate_dataset.py --style fujifilm --recipe classic_chrome --count 30
python train.py --arch generic --style fujifilm --recipe classic_chrome --epochs 8 --batch_size 4 --lr 2e-4 --image_size 192

python generate_dataset.py --style cyberpunk --count 30
python train.py --arch generic --style cyberpunk --epochs 8 --batch_size 4 --lr 2e-4 --image_size 192

python generate_dataset.py --style tilt_shift --count 30
python train.py --arch tilt_shift --style tilt_shift --epochs 16 --batch_size 4 --lr 1e-4 --image_size 192 --tilt_max_sigma 6.0
```

Each run finishes in roughly 3–5 minutes on the reference hardware
(CPU only). Checkpoints land in `checkpoints/model_<arch>_<style>[_<recipe>].pth`;
the `*.pth` file is git-ignored.

## 4. Inference

### 4.1 CLI

```powershell
python inference.py --image_path <input.jpg> --checkpoint checkpoints/model_<...>.pth [--output_path <out.jpg>]
```

The checkpoint loader auto-detects the architecture (`fujifilm`,
`generic`, or `tilt_shift`) so the same command works for any model.
Predicted parameters are printed to stdout.

### 4.2 Streamlit UI

```powershell
streamlit run image_editor_ui.py
```

A browser opens at `http://localhost:8501`. The sidebar's dropdown is
populated by scanning `checkpoints/` at startup, so any new
checkpoint is picked up the next time the cache is cleared (use the
**Reload checkpoint** button).

Each upload triggers a single forward pass through the encoder + head
+ renderer; the UI then displays:

- the original image with per-channel histograms and overlaid CDFs;
- the styled output with its own histograms;
- the predicted parameter vector decomposed by the architecture
  (named Fujifilm scalars, generic primitives, or generic + tilt-shift);
- a download-as-JPEG button on the styled image.

### 4.3 Regenerate the comparison figure

The figure embedded in the README is produced by:

```powershell
python tools/make_examples.py
```

This script loads each trained checkpoint and the matching data
generator, runs both on `images/test_images/climbing_test_original.jpeg`,
and writes a 3 × 3 grid to `tests/reports/assets/style_comparison.png`.
