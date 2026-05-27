# Project Plan: Generalizable Deep Learning Image Editing

## Core Philosophy
Build a **general-purpose image stylization engine**. The system architecture (CNN + CDF analysis) remains constant; the *style* is determined solely by the training data. "Film", "Cyberpunk", and "Tilt-Shift" are simply data modules, not hardcoded logic.

## Master Checklist

### Phase 1: The Fujifilm Pipeline (Prototype)
**Goal**: Build the end-to-end infrastructure for a specific, complex style ("Fujifilm Classic Chrome") to validate the architecture.

- [x] **Infrastructure Setup**
    - [x] Create `data_generation/core.py`: Abstract base classes.
    - [x] Create `data_generation/styles/`: Directory for style-specific modules.

- [x] **"Fujifilm" Style Module**
    - [x] Implement `FujifilmGenerator` simulating camera recipe settings.
    - [x] **Deliverable**: Script `generate_dataset.py --style fujifilm` that outputs $(I_{original}, I_{styled})$ pairs.

### Phase 2: The Hybrid Model Architecture (Fujifilm Specific)
**Goal**: Design and implement the neural network that learns the mapping for the Fujifilm style.

- [x] **Feature Extraction**: `CDFExtractor` and `SpatialEncoder`.
- [x] **Transformation Head**: `ParameterPredictor` MLP that outputs 7 Fujifilm-specific parameters.
- [x] **Training Loop**: Implemented with `MSELoss` and a differentiable Fujifilm renderer.
- [x] **Deliverable**: Trained model (`model_fujifilm_classic_chrome.pth`) that applies the style.

---
### Phase 3: Generalization Refactor
**Goal**: Refactor the architecture from Phase 2 to be style-agnostic, enabling future generalization.

- [x] **"Film" Style as a Blueprint**
    - [x] Implement a generic `ToneCurve` generator (e.g., predicting control points).
    - [x] Implement a generic `ColorGrader` (e.g., predicting a 3x3 or 3D LUT).
    - [x] Ensure `Grain` and `Vignette` are compatible.

- [x] **Model Refactoring**
    - [x] Modify `TransformationHead` to predict parameters for the generic primitives (tone curve points, color matrix, etc.) instead of Fujifilm values.
    - [x] Implement a new `DifferentiableRenderer` that can apply these generic primitives.
    - [x] Update the `train.py` script to use the new generalized model and renderer.

- [x] **Loss Function Improvement**
    - [x] Implement the composite loss: $L = \lambda_{pixel} L_{1} + \lambda_{perceptual} L_{VGG} + \lambda_{cdf} L_{CDF}$.

---
### Phase 4: Generalization Verification (The "Cyberpunk" Test)
**Goal**: Prove the refactored system is general by training it on a completely different style *without changing the model code*.

- [x] **"Cyberpunk" Style Module**
    - [x] Create `data_generation/styles/cyberpunk.py`.
    - [x] Implement S-curve tone + teal/orange color grade using the *generic primitives* from Phase 3 (NeonGlow deferred -- it requires a Gaussian-bloom primitive which is not in the current renderer; the cyberpunk look without bloom is still distinct from Fujifilm).
    - [x] Generate "Cyberpunk" dataset.

- [x] **Retraining**
    - [x] Train the generalized architecture from Phase 3 on the Cyberpunk dataset.
    - [x] **Deliverable**: A second model file (`model_generic_cyberpunk.pth`) that applies the new style.

### Phase 5: Spatially Variant Styles (The "Tilt-Shift" Test)
**Goal**: Extend the model to handle styles that require spatially variant processing.

- [x] **"Tilt-Shift" Style Module**
    - [x] Create `data_generation/styles/tilt_shift.py`.
    - [x] Investigate modifying the `SpatialEncoder` or `TransformationHead` to output parameter maps instead of single values.
      *Chosen approach:* parametrise the *spatial structure* (a horizontal focus band) with three global scalars (center_y, width, blur strength) and let the renderer derive the per-pixel blur weight from them. True per-pixel parameter maps would require a U-Net-style decoder and are listed in the project's open work; see the docstring in `models/tilt_shift.py`.
    - [x] **Deliverable**: `model_tilt_shift_tilt_shift.pth` reproducing the spatially variant blur.

### Phase 6: Application and Polish
**Goal**: Create a user-facing application to showcase the trained models.

- [x] **Web UI**
    - [x] Streamlit interface (`image_editor_ui.py`).
    - [x] Allow users to upload a photo.
    - [x] Auto-discovered dropdown over every checkpoint in `checkpoints/`.
    - [x] Side-by-side original/styled view with per-channel histograms + CDFs.
    - [x] Arch-aware predicted-parameter readout.
    - [x] Download button on the styled output.

- [x] **Real-World Data (Adobe MIT-5K)**
    - [x] Implement a data loader for the MIT-5K dataset (`data_generation/mit5k_loader.py`).
    - [x] Implement a streaming downloader (`tools/download_fivek_subset.py`) using the `logasja/mit-adobe-fivek` HuggingFace mirror; no 4 GB up-front download needed.
    - [x] Train a model on Expert C's edits. **Result:** on a held-out 5-pair test set, the model's prediction is on average ≈10% closer to the expert's edit than the original; 4 / 5 test pairs move toward the expert (see [`tests/reports/mit5k_expert_c_report.md`](tests/reports/mit5k_expert_c_report.md)). This is the first regime in which the NN does something a closed-form data generator cannot — exactly the case for the architecture's existence per the viability analysis in the project's commit history.

---

### Verification report

End-to-end verification of Phases 3-6, including a Playwright/MCP-driven
test of the Streamlit UI, is recorded in
[tests/reports/phase3_to_phase6_report.md](tests/reports/phase3_to_phase6_report.md).
The four UI screenshots captured during verification live in
`tests/reports/assets/`.
