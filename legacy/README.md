# Legacy: Phases 1–6

The project's first incarnation, kept for history and for its checkpoints:

- fixed-function styles generated synthetically (Fujifilm Classic Chrome,
  Cyberpunk, Tilt-shift);
- a 21-parameter differentiable renderer and composite L1 / VGG / CDF loss;
- a Streamlit lab UI;
- a first FiveK expert-C model.

It is not used by the current engine (`photostyle/`), Studio or `bench/`.
The 7-parameter Fujifilm-specific architecture (`--arch fujifilm`) has a
trained effect ratio of about 0 % (see
`tests/reports/phase3_to_phase6_report.md`).

**Run everything from this folder**, so the imports and relative paths
resolve:

```
cd legacy
uv run python generate_dataset.py --style cyberpunk --count 30
uv run python train.py --arch generic --style cyberpunk --epochs 8 --image_size 192
uv run python inference.py --image_path images/test_images/climbing_test_original.jpeg \
    --checkpoint checkpoints/model_generic_cyberpunk.pth
uv run streamlit run image_editor_ui.py
uv run python tools/make_examples.py          # the comparison figure in the main README
```

| path | what |
|---|---|
| `models/`, `data_generation/` | Phase 1–6 networks, renderers and synthetic style generators |
| `train.py`, `inference.py`, `generate_dataset.py`, `image_editor_ui.py` | Phase 1–6 command-line tools and Streamlit UI |
| `tools/` | figure scripts for the Phase 1–6 reports |
| `docs/` | Phase 1–6 architecture, training guide and original roadmap |
| `images/test_images/` | test images and their outputs |
| `checkpoints/` | the Phase 1/2 Fujifilm checkpoint |
| `exploration/` | the very first scripts (channel separation, histograms, CDF plots); run with `python load_and_show.py` |
