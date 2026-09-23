# Common tasks. All commands run inside the uv-managed environment.
.PHONY: setup test lint bench-small curve serve web web-dev check-studio

setup:
	uv sync
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run ruff check photostyle bench tools tests studio

# Quick harness smoke test (~4 min on a laptop CPU); needs data/fivek_landscape_c.
bench-small:
	uv run python -m bench.learning_curve --sizes 10,50 --seeds 0 --test 60 --oracle 10 \
		--out bench/results/smoke

# Full Phase 7 learning curve (~2-3 h on 4 CPU cores).
curve:
	uv run python -m bench.learning_curve

serve:
	uv run python -m studio.server

# Studio front end (Svelte). The build in studio/dist is committed; rebuild after editing studio/web/src.
web:
	cd studio/web && npm ci && npm run build

# Front-end dev server with hot reload (proxies /api to a running `make serve`).
web-dev:
	cd studio/web && npm run dev

# End-to-end Studio check in headless Chromium (needs `make serve` running and a photo).
check-studio:
	uv run python tools/check_studio.py --photo $(PHOTO) --out data/studio_check
