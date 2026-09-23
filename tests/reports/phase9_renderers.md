# Phase 9: renderer ablation (curves vs. residual LUTs vs. pixel MLP)

**Gate (a basis-LUT residual must beat per-channel curves by > 0.3 dB at
n ≤ 250): not met.** The engine keeps **per-channel curves + colour
matrix** as its renderer. It is the most editable option (curves you can
drag, exports to `.cube` and XMP) and is as accurate as the rest. The LUT
and CSR code stays in `photostyle/` for later styles that need
non-separable colour moves (e.g. hue-specific grades).

## Results (FiveK expert C landscapes, same 200-image test subset as Phase 7, seed 0; `bench/phase9.py`)

| renderer | params | PSNR n = 100 | ΔE n = 100 | PSNR n = 250 | ΔE n = 250 |
|---|---|---|---|---|---|
| shared curve + matrix | 20 | 21.43 | 8.34 | 21.62 | 8.04 |
| **per-channel curves + matrix** (default) | 40 | 21.42 | 8.39 | 21.50 | 7.99 |
| + 4 residual basis LUTs (9³) | 44 | 21.38 | 8.26 | 21.38 | 8.37 |
| + 8 residual basis LUTs (9³) | 48 | 21.28 | 8.39 | **21.68** | **7.91** |
| Zeng-style 3 × 33³ LUTs | 23 | 21.42 | **8.13** | 21.64 | 8.08 |
| CSR-lite (per-pixel FiLM MLP) | 128 | **21.46** | 8.46 | 21.52 | 7.93 |

"params" = numbers the head predicts per image. Reference points:
identity 20.45; static preset 21.11 (n = 100), 21.12 (n = 250).

## Reading

- **Everything lands within 0.4 dB.** The differences are about the size
  of seed noise (± 0.23 dB at n = 100 in Phase 7). The best LUT variant
  gains +0.18 dB at n = 250 and nothing at n = 100.
- This agrees with Phases 7 and 16. The same curves renderer can reach
  27.2 dB with per-image optimal parameters, so what is missing is
  **predicting** parameters from image content, not renderer capacity.
- **Implications for the plan:**
  - Better features or conditioning (Phase 10: style embeddings, a stronger
    or fine-tuned backbone) and more pairs per style are where accuracy
    will come from.
  - Renderer complexity only costs editability here.

## Caveats

- One seed per cell. The ± 0.2 dB differences are not significant.
- Only paired FiveK expert C, which is mostly a global retouch. A look with
  strong hue-selective moves (teal/orange, film emulation) could favour
  the LUT or CSR renderers. Re-run this ablation for any such style before
  choosing its renderer.
