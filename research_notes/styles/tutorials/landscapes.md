# Landscape looks: tutorials with numeric settings

Research on 2026-09-26. Every number below was verified on the fetched page. "Directions only" means
the page gives no number for that setting.

## 1. Moody / dark landscape (overcast forest, fields, lots of green)

**A. Run N Gun Photography, "Dark & Moody Lightroom Mobile Tutorial"**
https://therunngun.com/dark-moody-lightroom-mobile-tutorial/

- Light: Exposure about -0.5; Contrast +35 to +50; Highlights -100; Shadows +100; Whites about +25;
  Blacks -25.
- Curve: a fairly strong S-curve, shadow point lifted about 0.75 stops.
- Colour: Temperature -10; Tint -5 (toward green); Saturation about -50.
- Colour mix:

  | Colour | Hue | Saturation | Luminance |
  |---|---|---|---|
  | Red | +20 | -30 | 0 |
  | Orange | 0 | -10 | +80 |
  | Yellow | 0 | -30 | 0 |
  | Green | +50 | -80 | +40 |
  | Cyan | +40 | -30 | +30 |
  | Blue | -20 | -30 | +10 |

  Purple and Violet unchanged.
- Split tone: green highlights, purple/blue shadows, saturation at most +10.
- Directions only: Clarity down, Dehaze slightly up, light Vignette, split-tone hues.

**B. lightroomtutorials.com, "How to Create a Moody Jungle Green Look"**
https://www.lightroomtutorials.com/jungle-green/

- Green Hue +100; Yellow Luminance -100; Green Luminance -50; Yellow Saturation -100; Green
  Saturation -50; Vignette -25. The page calls these approximate starting values.
- Directions only: the curve.

## 2. Golden hour / warm sunset landscape

**A. Lucy Song, LR Presets, "How to Edit Sunrise & Sunset Photos in Lightroom"**
https://www.lightroompresets.com/blogs/pretty-presets-blog/how-to-edit-sunrise-and-sunset-photos-in-lightroom

- Highlights -100; Shadows +32; Blacks +60; Clarity +20.
- The HSL text is ambiguous: "changing Hue to +52 and Aqua to -43".
- Temperature and Tint unchanged.

**B. HueBliss, "Golden Hour Photography Settings"**
https://huebliss.com/golden-hour-photography-settings/

- Colour Grading: Highlights Hue 40, Saturation 15; Shadows Hue 220, Saturation 10.
- Shadows +20 to +30.

**C. Jason Parnell-Brookes, Fstoppers, "Create Incredible Sunset Photos" (portrait subject)**
https://fstoppers.com/originals/create-incredible-sunset-photos-editing-lightroom-547705

- Temperature 4950 -> 8113 K; Tint 8 -> 7; Exposure +1.14; Blacks -17; Vibrance +37.
- Colour Grading: Highlights Hue 46, Saturation 78.

## 3. Vivid Velvia-style landscape

**A. legendarypresets.com, "Step-by-Step Lightroom Workflow for High-Saturation Film Looks"**
https://legendarypresets.com/step-by-step-lightroom-workflow-for-high-saturation-film-looks/

- Basic: Vibrance +10 to +20; Saturation 0 to -3; Highlights -15 to -30; Whites -5 to -15.
- Curve: black point lifted +10.
- Velvia 100 HSL:

  | Colour | Hue | Saturation | Luminance |
  |---|---|---|---|
  | Yellow | -8 | +20 to +28 | 0 |
  | Green | -8 | +20 to +25 | -5 |
  | Aqua | +5 | +15 | -10 |
  | Blue | 0 | +15 to +20 | -15 |
  | Red | 0 | +10 | -5 |

- Colour grade: Shadows Magenta +8 to +12.

**B. Community preset file, `Fuji Velvia 50.xmp` (github.com/peva3/lightroom-presets)**

Low provenance: the repository cites no sources.

- Contrast +40; Highlights -65; Shadows -30; Whites +15; Blacks -30; Saturation -5.
- Tone curve: (0,0) (42,28) (128,128) (213,230) (255,255).
- HSL: Red +15/+22.5/-; Orange -10/+20/+7.5; Yellow -17.5/+12.5/+7.5; Green +22.5/+35/-15;
  Aqua +15/-15/-15; Blue -7.5/+26.3/-22.5; Purple +15/+15/-; Magenta +10/+20/-.
- Colour Grading: Highlights Hue 50, Saturation 7.5; Shadows Hue 230, Saturation 10;
  Blending 75; Balance -5.

## 4. Night sky / Milky Way landscape

**A. Mikko Lagerstedt, "4 Easy Steps to Edit Astrophotography in Lightroom"**
https://mikkolagerstedt.com/blog/how-to-edit-astrophotography-in-lightroom/

- Temperature 4200; Tint +28; Exposure -1.06; Contrast +12; Highlights +7; Whites +40;
  Texture +14; Clarity +6; Dehaze +24.
- Curve: 188 -> 180, 93 -> 75.
- Vignette -9 (Feather 100); Grain 9.
- It also uses local foreground and Milky Way filters. These are local adjustments and are not
  modelled.

**B. Dylan Giannakopoulos, Australian Photography, "How to edit the Milky Way in Lightroom"**
https://www.australianphotography.com/photo-tips/how-to-edit-the-milky-way-in-lightroom

- Temperature 4510; Tint +10; Exposure +0.40; Contrast +50; Shadows +5; Whites +25; Blacks -25;
  Dehaze +25; Vibrance +20; Saturation +10.
- Parametric curve: Highlights +10; Lights +5; Darks -15; Shadows +15.

## 5. Blue hour: no numeric tutorial found

These pages give directions only:

- Light Stalking: https://www.lightstalking.com/blue-hour-lightroom/
- Life Pixel: https://www.lifepixel.com/photo-tutorials/6-lightroom-tips-for-editing-your-blue-hour-shots
- Fstoppers: https://fstoppers.com/lightroom/make-your-blue-hour-photos-stand-out-701417
- Photzy: https://photzy.com/creating-stunning-landscape-photography-during-the-blue-hour/
- Shutter Evolve: https://www.shutterevolve.com/14-ways-to-create-stunning-blue-hour-images/

Two weaker sources with numbers:

- A preset-listing blog, not a tutorial:
  https://presetpedia.com/night-lightroom-presets/ ("Twilight Haze"). Exposure +10; Contrast
  -25; Shadows +45; Temperature -15; Vibrance +20; Purple Tint +10; Dehaze -20; Clarity -15.
- DPS, William Palfrey, "Long Exposure Photography 201": white balance almost 10000 K; everything
  else is directions only.
