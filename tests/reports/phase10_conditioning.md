# Phase 10: one style-conditioned model, and new styles from a few examples

**Both gates passed.**
1. One shared model conditioned on a style code beats four separate
   per-style models, on every known style.
2. A new, held-out style learned from **20 pairs** (shared weights frozen)
   beats a separate model trained on all 132 pairs of that style.

This is the architecture the product should use: a shared head, where each
style is a small code.

## Setup (`bench/phase10.py`)

- **Styles:** FiveK experts A–D are "known". Expert E is held out.
  Landscape subsets, 132 training pairs per expert (capped at E's pool so
  every style gets equal data). Test: 100 images per expert (49 for E).
- **Model:** `photostyle.condition.StyleHead`, the Phase 7 head with FiLM
  conditioning on a per-style embedding. A `StyleEncoder` is trained
  jointly: it maps the features of a style's *edited* photos to the same
  code space.
- **Renderer:** per-channel curves + colour matrix (Phase 9's choice).

## Joint vs. separate (PSNR dB; known styles, n = 132 each)

| expert | identity | separate head | **joint conditioned** |
|---|---|---|---|
| A | 18.74 | 19.31 | **20.80** |
| B | 21.36 | 22.83 | **23.02** |
| C | 20.23 | 20.86 | **21.71** |
| D | 19.24 | 20.05 | **20.21** |
| mean | 19.89 | 20.76 | **21.44** |

The joint model shares what all the experts agree on (exposure and
contrast fixes) and spends each style code on what differs. That is +0.7 dB
on average, and +1.5 dB for A, whose own 132 pairs are not enough alone.

## Few-shot: a new style (expert E, identity 20.20 dB)

Reference: a separate head trained on all 132 E pairs scores **21.18**.

| E examples | (a) encoder, edited photos only, no pairs | (b) new embedding fitted on pairs | (c) separate head on the same pairs |
|---|---|---|---|
| 5 | **20.97** | 20.31 | — |
| 20 | 20.92 | **21.68** | 20.87 |
| 50 | 20.86 | **21.75** | 20.82 |

## Reading

- **Twenty before/after pairs are enough for a new style.** Fitting only a
  new style code (a few dozen numbers) beats training a model from scratch
  on 6.6 times more pairs. For users this is the "create a style" path. It
  is also the right way to learn the owner's own look once they provide
  pairs (amendment A3.2).
- **With no pairs, the encoder gets most of the way from 5 edited photos**
  (0.2 dB below the full separate model). It does not improve with more
  photos: it summarises a style, it does not learn detail. It is a good
  instant preview while pairs are collected.
- **Caveats:**
  - one seed;
  - FiveK experts differ mostly in global tone and colour, so styles with
    strong hue-selective moves remain to be tested;
  - E's test set is small (49 images).

## Product implications (implemented or backlog)

- Styles are codes on a shared head. Adding a style needs no new model
  file. *Backlog:* ship a joint A–E head as the base for user styles in
  Studio's wizard; the current wizard trains a separate head (the path (c)
  measured above).
- Wizard defaults:
  - "photos in a look" gives an instant encoder preview;
  - after 20 pairs, fit an embedding.
