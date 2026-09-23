# Datasheet: `neon_street_portrait` style set

*Built 2026-09-23 for Pilot A (PROJECT_PLAN §17, amendment A1).*

## What it is

142 openly licensed photographs of street portraits: 71 at **night** and 71 by
**day**. Each has a single prominent subject (at most two people), and most
have a shallow depth of field. The set is a *stand-in* for the look studied
in [`research_notes/styles/bleg.md`](../../research_notes/styles/bleg.md).
It contains **none** of that photographer's images.

| File | Contents |
|---|---|
| `neon_street_portrait.csv` | One row per image: file, SHA-256, regime, licence, creator, source/landing URL, **attribution text**, colour statistics, depth-of-field proxy, profile score |
| `neon_street_portrait.summary.json` | Counts per licence/regime and the filter funnel |
| `neon_street_portrait.exclude.txt` | 74 hand-review exclusions with reasons |
| images | `data/styles/neon_street_portrait/images/` (gitignored; regenerate with the command below) |

Regenerate (uses the cached search results and downloads in `data/cache/openverse/` when present):

```
uv run python tools/collect_style_set.py --profile neon_street_portrait --pages 3 --max-keep 142
```

## Provenance and licences

- **Source:** Openverse API, 24 queries × 3 pages, 871 unique candidates. All
  kept images originate from Flickr.
- **Licences:** CC BY 89, CC BY-SA 44, CC0 8, Public Domain Mark 1. The
  collector requests only licences that permit adaptation and commercial
  use; NC and ND licences are never requested (A1.1).
- **Creators:** 66 distinct creators. The most frequent is `colinlogan` with
  30 images (21 %; a "100 strangers" street-portrait series), then
  `Dick Thomas Johnson` with 16. See the concentration caveat below.
- **Attribution:** every CC BY / BY-SA image must be credited when shown
  (reports, figures, a style card). Use the `attribution` column verbatim.
  BY-SA images shown in modified form (e.g. before/after grids) must carry
  the BY-SA licence.

## Selection funnel

| Stage | Removed |
|---|---|
| Download failed | 9 |
| Short side < 320 px after letterbox crop | 29 |
| Near-duplicate (dHash) | 2 |
| Monochrome (mean chroma < 6) | 223 |
| No prominent person (Faster R-CNN, largest person < 6 % of frame) | 215 |
| Crowd (> 2 people ≥ 3 % of frame) | 97 |
| Profile score < 0.35 | 62 |
| Hand-review exclusions (art pieces, back views, tiny subjects, selfies, cosplay, mis-regimed day/night, …) | 74 |
| Ranked out (kept the top 71 per regime) | remainder |

## Measured statistics (medians)

| | L\* p1 | L\* p50 | shadow b\* | mid a\* | mid b\* | highlight b\* | saturation | DoF log-ratio |
|---|---|---|---|---|---|---|---|---|
| **Stand-in, night** | 0.9 | 22.0 | **+2.9** | 8.3 | 11.2 | **+10.8** | 0.41 | 0.57 |
| **Stand-in, day** | 2.4 | 54.3 | −0.2 | 2.9 | 4.1 | 2.5 | 0.23 | 0.96 |
| Reference look, night (style study) | 1.1 | 15–18 | **−4 … −10** | 5–16 | — | **≈ 0** | 0.47–0.59 | — |
| Reference look, day (style study) | 4.0 | 59.6 | −9.3 | 3.0 | −2.7 | 1.2 | 0.24 | — |

## Known gaps

1. **The colour grade differs from the reference.** The stand-in set matches
   the reference on black point, tonal key, saturation, subject framing and
   shallow depth of field. It does **not** match the split tone: its night
   images are warm throughout (sodium/tungsten street light, highlight
   b\* ≈ +11), while the reference has cool shadows and neutral highlights.
   A model trained purely to imitate this set's distribution would learn a
   warm night grade. Pilot A handles this by taking the *grade* targets from
   the look profile and using the set only for what it does match (A1.2
   update in PROJECT_PLAN).
2. **Creator concentration.** Two creators supply 32 % of the images. Cap
   future collections at ~10 images per creator.
3. **Era and camera bias (not measured).** By eye, the images are mostly
   older Flickr-era DSLR photos, while the reference is shot on current
   cameras. Grading conventions may differ as a result.
4. **Night scarcity.** The openly licensed pool of good night street
   portraits was exhausted at ~71. More needs new sources (Wikimedia
   Commons, Unsplash licence) or the user's own photos.
5. **Faces of real people.** These are public, licensed street portraits.
   Use them for training and small attributed figures only; do not use them
   for identity-related tasks.
