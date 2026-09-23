# Phase 11: learning a look from example photos only (no pairs)

**Gate (adopt the simplest loss within 0.5 dB of the best): pseudo-pairs
only, at every size.** It is now the default for `learn_unpaired` (CLI
`photostyle learn --examples`, Studio "photos in a look"). The unpaired
route still falls **0.3–0.7 dB short of paired learning** at the same n.
For the look to actually move, pairs remain the better source when the
owner can supply them (amendment A3.2).

## Protocol (Zeng et al.'s disjoint split)

- **Pool:** the FiveK expert-C landscape pool, split in half.
- **Inputs:** 300 unedited originals from half A.
- **Style examples:** expert-C retouches of *different* photos (half B).
  The model never sees a pair.
- **Test:** the Phase 7 paired test subset (200 images, PSNR against
  expert C).
- **Command:** `uv run python -m bench.phase11`.

## Results (PSNR dB / ΔE2000)

| examples | SWD (+ fidelity) | **pseudo-pairs** | SWD + pseudo (+ fidelity) | paired head, same n (Phase 7) |
|---|---|---|---|---|
| 25 | 9.05 / 31.5 | **20.37** / 11.3 | 19.63 / 12.2 | 21.12 (with shrinkage) |
| 100 | 12.17 / 27.3 | **20.69** / 9.4 | 19.97 / 12.0 | 21.13 |
| 500 | 16.15 / 15.6 | 20.90 / 9.3 | **21.04** / 9.2 | 21.50 (n = 250) |

Identity (no edit): 20.45 / 9.16. The static preset, which needs pairs:
about 21.1.

## What went wrong with SWD, and what was fixed

1. **Bug: SWD was pooled over the batch.** With 8 photos per batch, the
   head learned to make some very dark and some very bright, so that their
   *union* matched the examples (first run: 9.7 dB at n = 25). The loss is
   now computed per image (`photostyle/train.py`).
2. **Still fails, and not because of a bug.** Per-image SWD pushes *every*
   photo toward the examples' pooled colour distribution: a dark forest and
   a bright street are asked for the same histogram.
   - The renderer's clamp to [0, 1] removes the gradient that would pull
     tones back, so training is also unstable: the same setup lands between
     9.7 and 16.5 dB depending on thread count.
   - Pilot A worked because a per-image **look profile with a relative
     key** and **fidelity** terms anchored it.
3. The vignette is now off by default in unpaired learning; distribution
   losses game it. This was not the cause of the failure (checked: 10.5 vs.
   12.0 dB).

## Reading

- **Pseudo-pairs (distort-and-recover) is the only robust signal without
  pairs.** It learns to undo random regrades of the examples, so it pulls
  photos toward the examples' tonal style without homogenising them. The
  gains are small but consistent: +0.3 / +0.5 / +0.9 dB over identity.
- **Its adaptivity is low.** Parameter spread across images is 0.03–0.05,
  against 0.10–0.25 for SWD variants and about 0.10 for paired heads. It
  behaves closer to a well-chosen preset.
- **Next steps (backlog, not needed for the gate):**
  - SWD relative to the input: match the *change* in distribution, or match
    per regime/key as the Pilot A profile does;
  - a perceptual or feature-space style loss;
  - more examples per style.
