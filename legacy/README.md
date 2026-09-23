# Legacy code

Kept for history, not used by the Phase 7+ engine (`photostyle/`, `bench/`).

- `exploration/`: the project's first scripts (RGB channel separation,
  per-channel histograms and CDF plots). They produced the images under
  `images/test_images/color_decompositions/` and `color_histograms/`.
  Run them from this folder: `python load_and_show.py`.

The Phase 1–6 model code (`models/`, `train.py`, `inference.py`,
`image_editor_ui.py`) is still at the repository root because existing
checkpoints and the Streamlit lab UI depend on it. The 7-parameter
Fujifilm-specific architecture (`--arch fujifilm`) is **legacy**: its
trained effect ratio is ≈ 0 % (see `tests/reports/phase3_to_phase6_report.md`).
