# Portrait looks: tutorials with numeric settings

Research on 2026-09-26. Every number below was verified on the page. Medialoot was read through
web.archive.org. Presetpedia and The Editing Studio are preset sellers.

## 1. Kodak Portra 400 (outdoor and daylight portraits, weddings)

**Medialoot, Diego Sanchez, "How to Emulate the Portra 400 Film Look"**
https://medialoot.com/blog/how-to-emulate-the-portra-400-film-look-in-lightroom/

- Point curves (input-output):
  - RGB: 0-0, 30-25, 70-75, 115-120, 190-170, 255-255
  - Red: 0-0, 40-20, 85-65, 145-170, 205-225, 255-255
  - Green: 0-0, 40-20, 80-60, 135-155, 185-210, 255-255
  - Blue: 0-0, 40-25, 120-120, 175-205, 215-235, 255-255
- The other settings are only in screenshots.

**Presetpedia, "Kodak Portra 400 Classic Film"** https://presetpedia.com/kodak-portra-lightroom-presets/

- Exposure +0.45; Contrast -15; Highlights -30; Shadows +35; Whites +15; Blacks +20.
- Temperature +200 K; Tint +8; Vibrance +12; Saturation -8.
- Red Hue +5; Orange Saturation -10; Orange Luminance +15; Yellow Hue +8; Blue Hue +10.
- Curve: Highlights -10; Shadows +15; Darks +8.
- Colour Grading: Shadows 200 / 12; Highlights 35 / 8.
- Vignette -12.

**The Editing Studio, "Kodak Portra 400"** https://theeditingstudio.co/blog/kodak-portra-400-lightroom-preset

- Exposure +0.3 to +0.5; Contrast -10 to -20; Highlights -20 to -40; Shadows +20 to +40;
  Whites -10 to -20; Blacks +15 to +25.
- Clarity -5 to -10.
- Black point lifted 12-18 units.
- Hue: Red +5 to +10; Orange +5 to +8; Yellow +5 to +8; Green +10 to +15.
- Saturation: Orange +5 to +10; Green -10 to -15; Blue -20 to -30.
- Luminance: Orange +10 to +15; Green -5; Blue -10.
- Colour Grading: Shadows hue 25-40, saturation 10-15.

## 2. Bright and airy / pastel (weddings, lifestyle)

**The Editing Studio** https://theeditingstudio.co/blog/bright-and-airy-lightroom-presets

- Exposure +0.3 to +0.6; Contrast -20 to -30; Highlights -25 to -40; Shadows +15 to +30;
  Whites -10 to -20; Blacks +15 to +25.
- Clarity -5 to -15.
- Saturation: Green -10 to -15; Blue -15 to -20; Aqua -10.
- Luminance: Orange +10 to +15; Yellow +5 to +10.

**Presetpedia, "Classic Light and Airy"** https://presetpedia.com/light-and-airy-lightroom-presets/

- Values are written as "%", so the units are ambiguous.
- Highlights -60; Shadows +40; Whites +10; Blacks -6; Vibrance +18; Saturation -6.
- Colour Grading: Highlights 45 / 10; Midtones 40 / 6; Shadows 210 / 8; Balance +10.

**Denny's Tips, "Light & Airy"** https://www.dennystips.com/light-and-airy/

- Colour Mixer: Saturation -50 and Luminance +50 for every colour except red, orange and yellow.

## 3. Dark and moody cinematic portrait

**James Young, "How to Edit Dark Witchy Portraits"** https://jamesyoungphotography.com/lightroom/cinematic-tutorial

- Contrast +6; Highlights +7; Shadows +60; Whites -30; Blacks -15.
- Temperature 7555 K; Tint +18; Vibrance -8; Saturation -2.
- Colour Mixer:
  - Reds: Hue -9, Saturation +7, Luminance -20
  - Oranges: Hue +6, Saturation +2, Luminance -6
  - Yellows: Hue +3, Saturation -27, Luminance -6
  - Greens: Hue -92, Saturation -38, Luminance -65
- Texture +12; Clarity -5.
- Directions only: the curves (gentle S-curves).

**Also usable:** Run N Gun "Dark & Moody" (see landscapes.md) and presets.io
https://presets.io/a/blog/edit-moody-portraits-lightroom (partial numbers).

## 4. High-contrast black and white (Tri-X / HP5): weakest category

No freely readable, complete Tri-X or HP5 recipe with black-and-white mixer numbers was found.

**Presetpedia** https://presetpedia.com/black-and-white-lightroom-presets/

- "High Contrast/Dramatic B&W": Contrast +50; Highlights -30; Shadows +40; Whites +30; Blacks
  -40; B&W mix Oranges +30, Greens -20; Clarity +40; Dehaze +20; Vignette -30.
- "Vintage Emulation B&W" (names Tri-X / HP5): Contrast +25; Highlights -25; Shadows +30;
  Whites +15; Blacks -15; Reds +25.

Grain only for Tri-X:

- Picfair Focus: Amount 65, Size 26, Roughness 45.
- DPS: Amount 30, Size 10, Roughness 10.
- The Editing Studio: Amount 30-40, Size 30-35, Roughness 55-65.

## 5. Warm golden-hour backlit portrait

**James Young, "How To Edit Backlit Portraits - FULL BREAKDOWN"** https://jamesyoungphotography.com/lightroom/backlit-portrait

The most complete recipe found.

- Contrast -10; Highlights +22; Shadows +15; Blacks -66; Dehaze -13; Vibrance +19; Saturation +4.
- Temperature 4669 K; Tint +8.
- Curve: (0,13) (77,57) (148,140) (255,239).
- Colour Grading: Midtones 197 / 9; Blending 89.
- HSL:

  | | Red | Orange | Yellow | Green | Aqua | Blue | Purple | Magenta |
  |---|---|---|---|---|---|---|---|---|
  | Hue | -14 | +2 | +1 | +10 | +14 | +12 | +2 | +5 |
  | Saturation | +6 | +1 | -25 | -11 | -23 | -29 | -10 | -10 |
  | Luminance | -10 | 0 | +10 | 0 | 0 | +3 | 0 | 0 |

**The Editing Studio, "Golden Hour"** https://theeditingstudio.co/blog/golden-hour-lightroom-preset-guide

- Exposure +0.2 to +0.5; Contrast -10 to -20; Highlights -20 to -35; Shadows +15 to +25;
  Whites -10 to -20; Blacks +10 to +20.
- Temperature +200 to +500; Tint +5 to +15.
- Black point lifted 10-15 units.
- Colour Grading: Highlights hue 30-50, saturation 20-30; Shadows hue 20-40, saturation 15-25.
- Saturation: Orange +10 to +15; Green -10 to -15; Blue -15 to -20.
- Luminance: Orange +10 to +15.
