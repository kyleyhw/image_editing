# Style study: Bleg (Bleg Bayraktar, @itsbleg)

*Collected 2026-09-23. First target style for the project (unpaired style, Phase 11 route).*

## Who

Belgian street-portrait photographer (24 as of 2025), known on Instagram / TikTok /
YouTube as **@itsbleg** / "ItsBleg". He left a finance career and reached 1M
Instagram followers in about seven months. He approaches strangers politely,
asks for a few minutes of their time, and makes an environmental portrait,
usually published as a short video of the encounter plus the final still.
He has worked in Belgium, France, Turkey, Morocco, Japan and South Korea,
where he is now largely based.
Sources: Japan Times (2025-08-06), Korea Times interview (2025-09-03),
Japan Times Instagram post.

## Evidence used

Instagram and the Japan Times article were not machine-readable (HTTP 429/402).
The visual analysis uses three images published in the Korea Times interview
and credited "Courtesy of ItsBleg":

1. Night portrait, Seoul street market, pink beanie, neon signs (vertical).
2. Dusk/overcast portrait in front of a shop window, blue fleece (vertical).
3. Daytime photo of Ahn Seung-wan with Bleg under cherry blossoms (vertical;
   probably not shot by Bleg himself, so weighted less).

Images are **not** stored in the repo (copyright).

## Measured statistics (CIELAB, scikit-image)

| Image | L* p1 / p50 / p99 | Shadows (L<25) a*, b* | Mids a*, b* | Highs (L≥70) a*, b* | Mean HSV sat. |
|---|---|---|---|---|---|
| Night neon | 1.1 / 15.3 / 90.5 | +0.5, **−4.3** | **+15.8, +7.2** | −2.4, −2.5 | 0.47 |
| Shop window | 1.1 / 18.0 / 82.6 | +0.9, **−10.1** | +5.1, −24.0 (blue fleece) | +2.8, +3.2 | 0.59 |
| Cherry blossom (day) | 4.0 / 59.6 / 98.0 | +2.4, **−9.3** | +3.0, −2.7 | +2.9, +1.2 | 0.24 |

## Description of the look

- **Deep, clean blacks.** The black point sits at true 0 (L* p1 ≈ 1) with no
  matte/faded lift. Night frames are low-key: median L* ≈ 15–18, with
  ~60 % of pixels in shadow.
- **Cool shadows, warm skin.** Shadows lean blue (b* −4 to −10) in all three
  images. Skin and midtones stay warm and natural. This is a restrained
  teal/orange split tone, subtler than a "cinematic" grade.
- **Neutral highlights.** Highlights are close to neutral, and whites don't
  go yellow.
- **Saturation follows the scene.** Rich, punchy colour at night (neon pinks,
  reds, cyans preserved and glowing) and soft, airy, low saturation in
  daylight (blossoms nearly white, sat ≈ 0.08 in highlights). This is the
  strongest argument for a *content-adaptive* model: one static preset
  cannot give both.
- **Contrast.** Crisp mid-contrast on the subject's face, with shadows
  allowed to fall off to black around the frame. There's no heavy HDR or
  clarity look.
- **Skin.** True-to-life, slightly warm, no heavy smoothing.
- **Not learnable by a global colour model** (capture choices, not edits):
  - shallow depth of field with creamy bokeh from lights;
  - vertical 4:5 / 9:16 framing, the subject off-centre looking away from
    the camera;
  - the environment (signs, shop windows, blossoms) as context;
  - available light only.
  Possibly a mild vignette.

Gear notes, unverified: an Instagram post shows him announcing a new
Hasselblad. In the Korea Times photo he holds a silver compact that looks
like a Fujifilm X100-series.

## Implications for training

- This is an **unpaired style** (no before/after pairs). Collect ~100–300 of
  his published stills for private research use only and don't redistribute.
  Train with the Phase 11 losses: luminance-conditioned colour statistics
  are the key term, because the signature *is* "cool shadows / warm mids /
  neutral highs".
- Stratify the style set by **night vs. day** and check that predicted
  parameters differ between the two. That spread is the success metric.
- A static baseline to beat: per-channel curves with a zero black point,
  a blue-lifted shadow curve, and a warm mid-tone matrix.
