# Contributing

Thanks for helping! This is a small personal research project released under
the [MIT licence](LICENSE).

## Setup

```
uv sync                      # Python env (CPU-only PyTorch on Linux/Windows)
uv run pre-commit install    # lint on commit
make test                    # unit tests (no data needed)
make lint
```

The Studio front end is Svelte in `studio/web/src`. The built files in
`studio/dist` are committed, so users do not need Node. If you change the
front end:

```
make web-dev                 # hot-reload dev server (run `make serve` alongside)
make web                     # rebuild studio/dist; commit it with your change (CI checks it)
```

## Ground rules

- **Data licences:**
  - Only use photos you own, have permission for, or that carry a licence
    allowing reuse and adaptation (CC0, public domain, CC BY, CC BY-SA),
    with attribution recorded.
  - Research-only datasets (MIT-Adobe FiveK, PPR10K) and anything trained
    on them must not be committed. `data/`, `checkpoints/` and
    `stylepacks/` are ignored for this reason.
  - See PROJECT_PLAN A1.1 and A4.5.
- **Measure before claiming:**
  - Changes to learning or rendering come with a benchmark run (`bench/`)
    and a short report in `tests/reports/`.
  - Keep baselines (identity, static preset) in every comparison.
- **Keep edits editable:** the model outputs parameters (curves, colour
  matrix, LUTs), never generated pixels.
- **Keep Python and WebGL in step:** the WebGL renderer must match the
  Python renderer. Run `tools/check_studio.py` (the golden test) after
  touching either.

## Layout

| path | what |
|---|---|
| `photostyle/` | engine: features, renderers, learning, exports, style packs, new-style pipeline |
| `studio/` | FastAPI server, Svelte front end (`web/`), built UI (`dist/`), v1 page (`static/`) |
| `bench/` | benchmarks and phase experiments |
| `tools/` | data collection, pack building, end-to-end checks |
| `tests/` | unit tests and phase reports |
