# Datasheet: `clean_cool_landscape` style set (seed)

*Built 2026-09-23 for Pilot A, re-targeted to landscapes (PROJECT_PLAN §17, amendment A2).*

## What it is

A **seed set** of 31 openly licensed urban and natural landscapes (14 night, 17
day) that show the *Clean Cool* grade: deep blacks, cool shadows, neutral
highlights, and clean neon at night. It contains no images by the photographer
who inspired the look. Images live in `data/styles/clean_cool_landscape/images/`
(gitignored). Per-image attribution is in `clean_cool_landscape.csv`, and 13
hand-review exclusions (with reasons) are in `clean_cool_landscape.exclude.txt`.

Regenerate from the cache (no API calls if `data/cache/openverse/` is present):

```
uv run python tools/collect_style_set.py --profile clean_cool_landscape --pages 2 --max-keep 150
```

## How it differs from the portrait stand-in set

The portrait set failed on the grade. This set applies **hard gates** before
scoring:
- shadow b\* ≤ −2;
- highlight b\* within about ±5;
- a deep black point;
- a daytime saturation ceiling.

The result matches the look profile:

| Median | L\* p1 | L\* p50 | shadow b\* | mid b\* | highlight b\* | saturation |
|---|---|---|---|---|---|---|
| **Night (14)** | 1.4 | 24.8 | **−7.5** | −12.1 | **−0.2** | 0.40 |
| **Day (17)** | 3.7 | 46.1 | **−5.1** | −10.6 | **−0.3** | 0.30 |
| Profile target, night | ≈1 | ≈18 | −7 | — | 0 | 0.45 |
| Profile target, day | ≈3 | ≈55 | −8 | −2 | +1 | 0.25 |
| *Portrait stand-in, night (for contrast)* | 0.9 | 22.0 | *+2.9* | *+11.2* | *+10.8* | 0.41 |

## Funnel

818 candidates from 21 Openverse queries × 2 pages:
- 547 failed the grade gates;
- 116 were monochrome;
- 63 had a prominent person;
- 47 failed to download, were too small, or were duplicates;
- 13 were excluded by hand;
- the remainder were ranked.

**Licences:** CC BY 20, CC BY-SA 9, CC0 1, PDM 1. There are 29 distinct
creators, none with more than 3 images.

## Known gaps

1. **Too small.** 31 images is a seed, not a training set (target ≥ 100).
   The gates are strict on purpose: most openly licensed city nights are
   warm and sodium-lit.
2. **Sources are exhausted for today.** Openverse's anonymous quota
   (200/day) is used up, and Wikimedia Commons rate-limits this environment's
   network. Next options: rerun Openverse with new queries tomorrow (results
   are cached); Wikimedia Commons *Quality images* from a different network;
   an Unsplash or Pexels API key (check their licence terms on ML use first);
   the owner's own landscape photos.
3. **Mid b\* is strongly negative** (−11 to −12). Many kept images are
   blue-hour scenes, which are cooler overall than the reference's
   warm-skin midtones. For landscapes this is probably acceptable, but the
   look profile, not this set, remains the authority on the grade.
4. Finish is better than the portrait set's, but a few images still show
   the heavier processing of older photos.
