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

- [ ] **"Film" Style as a Blueprint**
    - [ ] Implement a generic `ToneCurve` generator (e.g., predicting control points).
    - [ ] Implement a generic `ColorGrader` (e.g., predicting a 3x3 or 3D LUT).
    - [ ] Ensure `Grain` and `Vignette` are compatible.

- [ ] **Model Refactoring**
    - [ ] Modify `TransformationHead` to predict parameters for the generic primitives (tone curve points, color matrix, etc.) instead of Fujifilm values.
    - [ ] Implement a new `DifferentiableRenderer` that can apply these generic primitives.
    - [ ] Update the `train.py` script to use the new generalized model and renderer.

- [ ] **Loss Function Improvement**
    - [ ] Implement the composite loss: $L = \lambda_{pixel} L_{1} + \lambda_{perceptual} L_{VGG} + \lambda_{cdf} L_{CDF}$.

---
### Phase 4: Generalization Verification (The "Cyberpunk" Test)
**Goal**: Prove the refactored system is general by training it on a completely different style *without changing the model code*.

- [ ] **"Cyberpunk" Style Module**
    - [ ] Create `data_generation/styles/cyberpunk.py`.
    - [ ] Implement `NeonGlow`, `ColorShift`, etc., using the *generic primitives* from Phase 3.
    - [ ] Generate "Cyberpunk" dataset.

- [ ] **Retraining**
    - [ ] Train the generalized architecture from Phase 3 on the Cyberpunk dataset.
    - [ ] **Deliverable**: A second model file (`model_cyberpunk.pth`) that applies the new style.

### Phase 5: Spatially Variant Styles (The "Tilt-Shift" Test)
**Goal**: Extend the model to handle styles that require spatially variant processing.

- [ ] **"Tilt-Shift" Style Module**
    - [ ] Create `data_generation/styles/tilt_shift.py`.
    - [ ] Investigate modifying the `SpatialEncoder` or `TransformationHead` to output parameter maps instead of single values.

### Phase 6: Application and Polish
**Goal**: Create a user-facing application to showcase the trained models.

- [ ] **Web UI**
    - [ ] Build a simple web interface using Gradio or Streamlit.
    - [ ] Allow users to upload a photo.
    - [ ] Provide a dropdown to select the desired style model (Fujifilm, Cyberpunk, etc.).
    - [ ] Display the final, styled image with a download button.

- [ ] **Real-World Data (Adobe MIT-5K)**
    - [ ] Implement a data loader for the MIT-5K dataset.
    - [ ] Train a model on an expert's edits to create a "professional retouch" style.
