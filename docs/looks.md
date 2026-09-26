# Looks and the tutorials they come from

Each look is a recipe transcribed from public grading tutorials, as Lightroom-style slider settings.
The tools are in `photostyle/develop.py`, the tables in `photostyle/recipe_library.py` and
`photostyle/recipes.py`. The recipe grades openly licensed photos, first of the subject the tutorial
was written for and then a general sample. The content-adaptive model is trained on those
before/after pairs (`photostyle style train NAME --recipe teacher --open-only`; all twenty are built
by `tools/build_library.py`).

Kinds:
- **numeric**: the tutorial states the value.
- **mid**: the middle of a range the tutorial states.
- **converted**: a stated value converted to this tool's scale. Kelvin is measured from 5500 K,
  30 K per slider unit; named colours use their hue angle.
- **direction**: the tutorial gives only the direction; the amount is chosen.

Not modelled anywhere: local adjustments (graduated/radial filters, brushes), clarity, texture, sharpening, noise reduction, grain and halation are not modelled by a global, editable renderer.

`fujifilm` is the exception. It is trained directly on 46 openly licensed Fuji film scans (the
owner's choice). Its tutorial recipe is kept below as a candidate.


## Day landscape

### Moody Forest (`moody_forest`)

Best for: misty, overcast forests with tall trees and lots of green. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | -0.5 |  | RNG | numeric | exposure down about -0.5 |
| contrast | amount | 42.5 |  | RNG | mid | contrast +35 to +50 |
| tone_regions | highlights | -100 |  | RNG | numeric | highlights -100 |
| tone_regions | shadows | 100 |  | RNG | numeric | shadows +100 |
| tone_regions | whites | 25 |  | RNG | numeric | whites +25ish |
| tone_regions | blacks | -25 |  | RNG | numeric | blacks -25 |
| point_curve | rgb | [(0, 22), (64, 50), (192, 205), (255, 255)] |  | RNG | direction | fairly strong S-curve, shadow point lifted about 0.75 stops |
| white_balance | temperature | -10 |  | RNG | numeric | temperature -10 |
| white_balance | tint | -5 |  | RNG | numeric | tint toward green -5 |
| saturation | sat | -50 |  | RNG | numeric | saturation about -50 |
| hsl | hue.red | 20 |  | RNG | numeric | red hue +20 |
| hsl | hue.green | 50 |  | RNG | numeric | green hue +50 |
| hsl | hue.aqua | 40 |  | RNG | numeric | aqua hue +40 |
| hsl | hue.blue | -20 |  | RNG | numeric | blue hue -20 |
| hsl | sat.red | -30 |  | RNG | numeric | red sat -30 |
| hsl | sat.orange | -10 |  | RNG | numeric | orange sat -10 |
| hsl | sat.yellow | -30 |  | RNG | numeric | yellow sat -30 |
| hsl | sat.green | -80 |  | RNG | numeric | green sat -80 |
| hsl | sat.aqua | -30 |  | RNG | numeric | aqua sat -30 |
| hsl | sat.blue | -30 |  | RNG | numeric | blue sat -30 |
| hsl | lum.orange | 80 |  | RNG | numeric | orange lum +80 |
| hsl | lum.green | 40 |  | RNG | numeric | green lum +40 |
| hsl | lum.aqua | 30 |  | RNG | numeric | aqua lum +30 |
| hsl | lum.blue | 10 |  | RNG | numeric | blue lum +10 |
| color_grading | highlights | (100, 10) |  | RNG | direction | green in the highlights, saturation at most +10 |
| color_grading | shadows | (250, 10) |  | RNG | direction | purple/blue in the shadows, saturation at most +10 |
| dehaze | amount | 10 |  | RNG | direction | dehaze slightly up |
| vignette | amount | -25 |  | LTJ | numeric | vignette -25 |

- **LTJ**: [lightroomtutorials.com, How to Create a Moody Jungle Green Look](https://www.lightroomtutorials.com/jungle-green/)
- **RNG**: [Run N Gun Photography, Dark & Moody Lightroom Mobile Tutorial](https://therunngun.com/dark-moody-lightroom-mobile-tutorial/)

### Golden Hour (`golden_hour`)

Best for: sunrise and sunset landscapes. Grade B: some numbers; the rest are directions.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| tone_regions | highlights | -100 |  | LS | numeric | highlights -100 |
| tone_regions | shadows | 32 |  | LS | numeric | shadows +32 |
| tone_regions | blacks | 60 |  | LS | numeric | blacks +60 |
| color_grading | highlights | (40, 15) |  | HB | numeric | highlights hue 40, saturation 15 |
| color_grading | shadows | (220, 10) |  | HB | numeric | shadows hue 220, saturation 10 |
| saturation | vibrance | 15 |  | LS | direction | vibrance up |
| dehaze | amount | 10 |  | LS | direction | dehaze up |

- **HB**: [HueBliss, Golden Hour Photography Settings](https://huebliss.com/golden-hour-photography-settings/)
- **LS**: [Lucy Song (LR Presets), How to Edit Sunrise & Sunset Photos in Lightroom](https://www.lightroompresets.com/blogs/pretty-presets-blog/how-to-edit-sunrise-and-sunset-photos-in-lightroom)

### Velvia Vivid (`velvia`)

Best for: sunny landscapes with sky, water and foliage. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| saturation | vibrance | 15 |  | LPV | mid | vibrance +10 to +20 |
| saturation | sat | -1.5 |  | LPV | mid | saturation 0 to -3 |
| tone_regions | highlights | -22.5 |  | LPV | mid | highlights -15 to -30 |
| tone_regions | whites | -10 |  | LPV | mid | whites -5 to -15 |
| point_curve | rgb | [(0, 10), (255, 255)] |  | LPV | numeric | black point lifted +10 |
| hsl | hue.yellow | -8 |  | LPV | mid | Velvia 100 colour table |
| hsl | hue.green | -8 |  | LPV | mid | Velvia 100 colour table |
| hsl | hue.aqua | 5 |  | LPV | mid | Velvia 100 colour table |
| hsl | sat.yellow | 24 |  | LPV | mid | Velvia 100 colour table |
| hsl | sat.green | 22.5 |  | LPV | mid | Velvia 100 colour table |
| hsl | sat.aqua | 15 |  | LPV | mid | Velvia 100 colour table |
| hsl | sat.blue | 17.5 |  | LPV | mid | Velvia 100 colour table |
| hsl | sat.red | 10 |  | LPV | mid | Velvia 100 colour table |
| hsl | lum.green | -5 |  | LPV | mid | Velvia 100 colour table |
| hsl | lum.aqua | -10 |  | LPV | mid | Velvia 100 colour table |
| hsl | lum.blue | -15 |  | LPV | mid | Velvia 100 colour table |
| hsl | lum.red | -5 |  | LPV | mid | Velvia 100 colour table |
| color_grading | shadows | (300, 10) |  | LPV | converted | shadows magenta +8 to +12 (magenta = hue 300) |

- **LPV**: [Legendary Presets, Step-by-Step Lightroom Workflow for High-Saturation Film Looks](https://legendarypresets.com/step-by-step-lightroom-workflow-for-high-saturation-film-looks/)


## Night landscape

### Milky Way (`milky_way`)

Best for: night skies and the Milky Way over a landscape. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | -33.0 |  | AP | converted | temperature 4510 K |
| white_balance | tint | 10 |  | AP | numeric | tint +10 |
| exposure | stops | 0.4 |  | AP | numeric | exposure +0.40 |
| contrast | amount | 50 |  | AP | numeric | contrast +50 |
| tone_regions | shadows | 5 |  | AP | numeric | shadows +5 |
| tone_regions | whites | 25 |  | AP | numeric | whites +25 |
| tone_regions | blacks | -25 |  | AP | numeric | blacks -25 |
| parametric_curve | highlights | 10 |  | AP | numeric | curve highlights +10 |
| parametric_curve | lights | 5 |  | AP | numeric | curve lights +5 |
| parametric_curve | darks | -15 |  | AP | numeric | curve darks -15 |
| parametric_curve | shadows | 15 |  | AP | numeric | curve shadows +15 |
| dehaze | amount | 25 |  | AP | numeric | dehaze +25 |
| saturation | vibrance | 20 |  | AP | numeric | vibrance +20 |
| saturation | sat | 10 |  | AP | numeric | saturation +10 |
| vignette | amount | -9 |  | ML | numeric | vignette -9 |

- **AP**: [Dylan Giannakopoulos (Australian Photography), How to edit the Milky Way in Lightroom](https://www.australianphotography.com/photo-tips/how-to-edit-the-milky-way-in-lightroom)
- **ML**: [Mikko Lagerstedt, 4 Easy Steps to Edit Astrophotography in Lightroom](https://mikkolagerstedt.com/blog/how-to-edit-astrophotography-in-lightroom/)

### Blue Hour (`blue_hour`)

Best for: twilight seascapes and landscapes after sunset. Grade C: mostly directions; the amounts were chosen.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| contrast | amount | -25 |  | PPT | numeric | contrast -25 (preset listing, weak source) |
| tone_regions | shadows | 45 |  | PPT | numeric | shadows +45 |
| white_balance | temperature | -15 |  | PPT, LPX | numeric | temperature -15; err on the cool side |
| saturation | vibrance | 20 |  | PPT | numeric | vibrance +20 |
| color_grading | midtones | (270, 10) |  | PPT | converted | purple tint +10 (purple = hue 270) |
| dehaze | amount | -20 |  | PPT | numeric | dehaze -20 |
| hsl | hue.yellow | -20 |  | FSB | direction | yellows toward orange |
| hsl | hue.blue | -20 |  | FSB | direction | blues toward cyan |
| point_curve | rgb | [(0, 0), (64, 56), (192, 200), (255, 255)] |  | LPX | direction | gentle S-curve |

- **FSB**: [Fstoppers (Alex Cooke), Make Your Blue Hour Photos Stand Out](https://fstoppers.com/lightroom/make-your-blue-hour-photos-stand-out-701417)
- **LPX**: [Life Pixel, 6 Lightroom Tips for Editing Your Blue Hour Shots](https://www.lifepixel.com/photo-tutorials/6-lightroom-tips-for-editing-your-blue-hour-shots)
- **PPT**: [Presetpedia, Twilight Haze preset (night Lightroom presets)](https://presetpedia.com/night-lightroom-presets/)


## Day city

### Teal & Orange (`teal_orange`)

Best for: city streets and architecture in daylight. Grade B: some numbers; the rest are directions.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| hsl | hue.red | 100 |  | LTT | numeric | colour mixer hue: red +100, yellow -100, green +100, blue -100 |
| hsl | hue.yellow | -100 |  | LTT | numeric | colour mixer hue: red +100, yellow -100, green +100, blue -100 |
| hsl | hue.green | 100 |  | LTT | numeric | colour mixer hue: red +100, yellow -100, green +100, blue -100 |
| hsl | hue.blue | -100 |  | LTT | numeric | colour mixer hue: red +100, yellow -100, green +100, blue -100 |
| calibration | blue_hue | -100 |  | DTO | numeric | blue primary hue -100 |
| calibration | red_hue | 50 |  | DTO | numeric | red primary hue about +50 |
| color_grading | shadows | (195, 25) |  | PPJ | numeric | shadows hue 195, saturation 25 |
| color_grading | highlights | (35, 20) |  | PPJ | numeric | highlights hue 35, saturation 20 |
| point_curve | rgb | [(0, 18), (64, 55), (192, 205), (255, 245)] |  | LTT | direction | drag the end points in for a fade, then an S-curve |

- **DTO**: [Denny's Tips, Orange and Teal Color Grading](https://www.dennystips.com/orange-teal/)
- **LTT**: [lightroomtutorials.com, How to Create a Teal and Orange Look in Lightroom](https://www.lightroomtutorials.com/teal-orange/)
- **PPJ**: [Presetpedia, Moody Japan preset](https://presetpedia.com/japan-lightroom-presets/)

### Kodachrome Street (`kodachrome`)

Best for: colourful street and travel scenes in daylight. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | 0.15 |  | ESK | mid | exposure 0 to +0.3 |
| contrast | amount | 15 |  | ESK | mid | contrast +10 to +20 |
| tone_regions | highlights | -20 |  | ESK | mid | highlights -15 to -25 |
| tone_regions | shadows | 10 |  | ESK | mid | shadows +5 to +15 |
| tone_regions | whites | -10 |  | ESK | mid | whites -5 to -15 |
| tone_regions | blacks | 5 |  | ESK | mid | blacks 0 to +10 (ESK2 says 0 to -10; ESK followed) |
| saturation | vibrance | 15 |  | ESK | mid | vibrance +10 to +20 |
| saturation | sat | 2.5 |  | ESK | mid | saturation 0 to +5 |
| hsl | hue.red | 5 |  | ESK | mid | red hue +5 |
| hsl | hue.orange | -5 |  | ESK | mid | orange hue -5 |
| hsl | hue.blue | 7.5 |  | ESK | mid | blue hue +7.5 |
| hsl | sat.red | 20 |  | ESK | mid | red sat +20 |
| hsl | sat.orange | 12.5 |  | ESK | mid | orange sat +12.5 |
| hsl | sat.yellow | 7.5 |  | ESK | mid | yellow sat +7.5 |
| hsl | sat.blue | 25 |  | ESK | mid | blue sat +25 |
| hsl | lum.orange | 12.5 |  | ESK | mid | orange lum +12.5 |
| hsl | lum.blue | -12.5 |  | ESK | mid | blue lum -12.5 |
| point_curve | rgb | [(0, 4), (64, 54), (192, 206), (255, 240)] |  | ESK2 | mid | shadow point output 0-8, highlight point 235-245, pronounced S (S shape chosen) |
| color_grading | highlights | (40, 5) |  | ESK | direction | neutral to very slight warm |

- **ESK**: [The Editing Studio, Kodachrome Film Lightroom Preset: Complete Guide](https://theeditingstudio.co/blog/kodachrome-film-lightroom-preset-guide)
- **ESK2**: [The Editing Studio, How to Get the Kodachrome Look in Lightroom](https://theeditingstudio.co/blog/how-to-get-the-kodachrome-look-in-lightroom)

### Matte Urban (`matte_urban`)

Best for: streets, walls and urban textures. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | 0.3 |  | ESV | mid | exposure +0.2 to +0.4 |
| contrast | amount | -25 |  | ESV | mid | contrast -20 to -30 |
| tone_regions | highlights | -27.5 |  | ESV | mid | highlights -20 to -35 |
| tone_regions | shadows | 27.5 |  | ESV | mid | shadows +20 to +35 |
| tone_regions | whites | -20 |  | ESV | mid | whites -15 to -25 |
| tone_regions | blacks | 27.5 |  | ESV | mid | blacks +20 to +35 |
| point_curve | rgb | [(0, 25), (255, 242)] |  | ESV | mid | black point up 20-30 units; soften the top (chosen) |
| color_grading | shadows | (35, 25) |  | ESV | converted | warm amber shadows, saturation 20-30 (amber = hue 35) |
| color_grading | highlights | (50, 8) |  | ESV | direction | slight warm yellow highlights |
| hsl | hue.yellow | 7.5 |  | ESV | mid | yellow hue +7.5 |
| hsl | hue.green | 12.5 |  | ESV | mid | green hue +12.5 |
| hsl | hue.blue | 5 |  | ESV | mid | blue hue +5 |
| hsl | sat.green | -25 |  | ESV | mid | green sat -25 |
| hsl | sat.blue | -30 |  | ESV | mid | blue sat -30 |
| hsl | sat.aqua | -17.5 |  | ESV | mid | aqua sat -17.5 |
| hsl | sat.red | -2.5 |  | ESV | mid | red sat -2.5 |
| hsl | lum.orange | 12.5 |  | ESV | mid | orange lum +12.5 |
| hsl | lum.yellow | 7.5 |  | ESV | mid | yellow lum +7.5 |
| vignette | amount | -15 |  | ESV | mid | vignette -10 to -20 |

- **ESV**: [The Editing Studio, Vintage Film Lightroom Preset: Complete Guide](https://theeditingstudio.co/blog/vintage-clean-film-lightroom-preset-guide)


## Night city

### CineStill 800T (`cinestill_800t`)

Best for: city streets at night under street and shop lights. Grade B: some numbers; the rest are directions.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | 0.35 |  | LM | mid | exposure +0.20 to +0.50 |
| contrast | amount | 15 |  | LM | numeric | contrast +15 |
| tone_regions | highlights | -25 |  | LM | numeric | highlights -25 |
| tone_regions | shadows | 15 |  | LM | mid | shadows +10 to +20 |
| tone_regions | whites | 10 |  | LM | numeric | whites +10 |
| tone_regions | blacks | -20 |  | LM | numeric | blacks -20 |
| white_balance | temperature | -10 |  | LM | direction | slightly cooler than the starting white balance |
| white_balance | tint | 5.5 |  | LM | mid | tint +3 to +8 toward magenta |
| saturation | vibrance | -5 |  | LM | numeric | vibrance -5 |
| saturation | sat | -5 |  | LM | numeric | saturation -5 |
| dehaze | amount | 5 |  | LM | numeric | dehaze +5 |
| color_grading | shadows | (185, 15) |  | LM | direction | teal/cyan shadows, low saturation |
| color_grading | highlights | (20, 15) |  | LM | direction | orange-red highlights, low saturation |
| hsl | sat.yellow | -15 |  | LM | direction | lower yellow saturation |
| hsl | sat.green | -15 |  | LM | direction | lower green saturation |
| hsl | hue.blue | -10 |  | LM | direction | blues cooler |
| hsl | hue.aqua | 10 |  | LM | direction | aquas cooler |
| vignette | amount | -7.5 |  | LM | mid | vignette -5 to -10 |

- **LM**: [Lou & Marks Presets, CineStill 800T Film Lightroom Presets](https://loumarkspresets.com/blogs/lightroom/cinestill-800t-free-download)

### Tokyo Night (`tokyo_night`)

Best for: neon-lit night streets (Masashi Wakui style). Grade B: some numbers; the rest are directions.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | 100 |  | TD | converted | temperature all the way right (50000), tint 0 |
| color_grading | highlights | (60, 50) |  | TD | numeric | split toning highlights hue ~60, saturation 50 |
| color_grading | shadows | (225, 100) |  | TD | numeric | split toning shadows hue ~225, saturation 100 |
| calibration | shadow_tint | -100 |  | TD | numeric | calibration shadows tint -100 |
| point_curve | rgb | [(0, 20), (255, 255)] |  | TD, LTW | direction | RGB curve toe up |
| point_curve | g | [(0, 10), (255, 240)] |  | TD, LTW | direction | green toe up; green top point down |
| point_curve | b | [(0, 0), (255, 225)] |  | TD, LTW | direction | blue top point down, twice the green amount |
| contrast | amount | -15 |  | TD | direction | contrast down |
| tone_regions | highlights | 15 |  | TD | direction | highlights up |
| tone_regions | whites | -15 |  | TD | direction | whites down |
| saturation | vibrance | 10 |  | TD | direction | a little vibrance |

- **LTW**: [lightroomtutorials.com, How to Achieve the Masashi Wakui Look](https://www.lightroomtutorials.com/how-to-achieve-the-masashi-wakui-look/)
- **TD**: [Tim Daniels (DIYPhotography), How to create the Wakui cinematic look in Lightroom and Photoshop](https://www.diyphotography.net/how-to-create-the-wakui-cinematic-look-in-lightroom-and-photoshop/)


## Portrait

### Portra 400 (`portra_400`)

Best for: daylight portraits outdoors. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| point_curve | rgb | [(0, 0), (30, 25), (70, 75), (115, 120), (190, 170), (255, 255)] |  | MD | numeric | RGB point curve |
| point_curve | r | [(0, 0), (40, 20), (85, 65), (145, 170), (205, 225), (255, 255)] |  | MD | numeric | red point curve |
| point_curve | g | [(0, 0), (40, 20), (80, 60), (135, 155), (185, 210), (255, 255)] |  | MD | numeric | green point curve |
| point_curve | b | [(0, 0), (40, 25), (120, 120), (175, 205), (215, 235), (255, 255)] |  | MD | numeric | blue point curve |
| exposure | stops | 0.45 |  | PPP | numeric | exposure +0.45 |
| contrast | amount | -15 |  | PPP | numeric | contrast -15 |
| tone_regions | highlights | -30 |  | PPP | numeric | highlights -30 |
| tone_regions | shadows | 35 |  | PPP | numeric | shadows +35 |
| tone_regions | whites | 15 |  | PPP | numeric | whites +15 |
| tone_regions | blacks | 20 |  | PPP | numeric | blacks +20 |
| white_balance | temperature | 6.666666666666667 |  | PPP | converted | temperature +200 K |
| white_balance | tint | 8 |  | PPP | numeric | tint +8 |
| saturation | vibrance | 12 |  | PPP | numeric | vibrance +12 |
| saturation | sat | -8 |  | PPP | numeric | saturation -8 |
| hsl | hue.red | 5 |  | PPP | numeric | red hue +5 |
| hsl | hue.yellow | 8 |  | PPP | numeric | yellow hue +8 |
| hsl | hue.blue | 10 |  | PPP | numeric | blue hue +10 |
| hsl | sat.orange | -10 |  | PPP | numeric | orange sat -10 |
| hsl | lum.orange | 15 |  | PPP | numeric | orange lum +15 |
| color_grading | shadows | (200, 12) |  | PPP | numeric | shadows hue 200, saturation 12 |
| color_grading | highlights | (35, 8) |  | PPP | numeric | highlights hue 35, saturation 8 |
| vignette | amount | -12 |  | PPP | numeric | vignette -12 |

- **MD**: [Diego Sanchez (Medialoot), How to Emulate the Portra 400 Film Look in Lightroom](https://medialoot.com/blog/how-to-emulate-the-portra-400-film-look-in-lightroom/)
- **PPP**: [Presetpedia, Kodak Portra 400 Classic Film preset](https://presetpedia.com/kodak-portra-lightroom-presets/)

### Bright & Airy (`bright_airy`)

Best for: weddings and lifestyle portraits in soft light. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | 0.45 |  | ESA | mid | exposure +0.3 to +0.6 |
| contrast | amount | -25 |  | ESA | mid | contrast -20 to -30 |
| tone_regions | highlights | -32.5 |  | ESA | mid | highlights -25 to -40 |
| tone_regions | shadows | 22.5 |  | ESA | mid | shadows +15 to +30 |
| tone_regions | whites | -15 |  | ESA | mid | whites -10 to -20 |
| tone_regions | blacks | 20 |  | ESA | mid | blacks +15 to +25 |
| saturation | vibrance | -2.5 |  | ESA | mid | vibrance -5 to 0 |
| saturation | sat | -2.5 |  | ESA | mid | saturation -5 to 0 |
| hsl | sat.green | -12.5 |  | ESA | mid | green sat -12.5 |
| hsl | sat.blue | -17.5 |  | ESA | mid | blue sat -17.5 |
| hsl | sat.aqua | -10 |  | ESA | mid | aqua sat -10 |
| hsl | lum.orange | 12.5 |  | ESA | mid | orange lum +12.5 |
| hsl | lum.yellow | 7.5 |  | ESA | mid | yellow lum +7.5 |
| point_curve | rgb | [(0, 15), (255, 245)] |  | ESA | direction | lift the black point, pull the top anchor down slightly |
| color_grading | shadows | (210, 5) |  | ESA | direction | neutral or very slight cool tint in shadows |

- **ESA**: [The Editing Studio, Bright and Airy Lightroom Presets: Complete Guide](https://theeditingstudio.co/blog/bright-and-airy-lightroom-presets)

### Moody Portrait (`moody_portrait`)

Best for: low-key and indoor portraits. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| contrast | amount | 6 |  | JYD | numeric | contrast +6 |
| tone_regions | highlights | 7 |  | JYD | numeric | highlights +7 |
| tone_regions | shadows | 60 |  | JYD | numeric | shadows +60 |
| tone_regions | whites | -30 |  | JYD | numeric | whites -30 |
| tone_regions | blacks | -15 |  | JYD | numeric | blacks -15 |
| white_balance | temperature | 68.5 |  | JYD | converted | temperature 7555 K |
| white_balance | tint | 18 |  | JYD | numeric | tint +18 |
| saturation | vibrance | -8 |  | JYD | numeric | vibrance -8 |
| saturation | sat | -2 |  | JYD | numeric | saturation -2 |
| hsl | hue.red | -9 |  | JYD | numeric | red hue -9 |
| hsl | hue.orange | 6 |  | JYD | numeric | orange hue +6 |
| hsl | hue.yellow | 3 |  | JYD | numeric | yellow hue +3 |
| hsl | hue.green | -92 |  | JYD | numeric | green hue -92 |
| hsl | sat.red | 7 |  | JYD | numeric | red sat +7 |
| hsl | sat.orange | 2 |  | JYD | numeric | orange sat +2 |
| hsl | sat.yellow | -27 |  | JYD | numeric | yellow sat -27 |
| hsl | sat.green | -38 |  | JYD | numeric | green sat -38 |
| hsl | lum.red | -20 |  | JYD | numeric | red lum -20 |
| hsl | lum.orange | -6 |  | JYD | numeric | orange lum -6 |
| hsl | lum.yellow | -6 |  | JYD | numeric | yellow lum -6 |
| hsl | lum.green | -65 |  | JYD | numeric | green lum -65 |
| point_curve | rgb | [(0, 8), (64, 58), (192, 200), (255, 250)] |  | JYD | direction | gentle S-curve |

- **JYD**: [James Young, How to Edit Dark Witchy Portraits](https://jamesyoungphotography.com/lightroom/cinematic-tutorial)

### Tri-X B&W (`tri_x`)

Best for: street and documentary portraits in black and white. Grade C: mostly directions; the amounts were chosen.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| contrast | amount | 25 |  | PPB | numeric | contrast +25 (preset listing, weak source) |
| tone_regions | highlights | -25 |  | PPB | numeric | highlights -25 |
| tone_regions | shadows | 30 |  | PPB | numeric | shadows +30 |
| tone_regions | whites | 15 |  | PPB | numeric | whites +15 |
| tone_regions | blacks | -15 |  | PPB | numeric | blacks -15 |
| black_white | mix.red | 25 |  | PPB | numeric | B&W mix reds +25 |
| dehaze | amount | 5 |  | PPB | numeric | dehaze +5 |
| parametric_curve | highlights | -10 |  | PPB | numeric | curve highlights -10 |
| parametric_curve | shadows | 10 |  | PPB | numeric | curve shadows +10 |

- **PPB**: [Presetpedia, Vintage Emulation B&W (Tri-X 400 / HP5) preset](https://presetpedia.com/black-and-white-lightroom-presets/)

### Golden Backlit (`golden_backlit`)

Best for: backlit portraits at sunset. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| contrast | amount | -10 |  | JYB | numeric | contrast -10 |
| tone_regions | highlights | 22 |  | JYB | numeric | highlights +22 |
| tone_regions | shadows | 15 |  | JYB | numeric | shadows +15 |
| tone_regions | blacks | -66 |  | JYB | numeric | blacks -66 |
| dehaze | amount | -13 |  | JYB | numeric | dehaze -13 |
| saturation | vibrance | 19 |  | JYB | numeric | vibrance +19 |
| saturation | sat | 4 |  | JYB | numeric | saturation +4 |
| white_balance | temperature | -27.7 |  | JYB | converted | temperature 4669 K |
| white_balance | tint | 8 |  | JYB | numeric | tint +8 |
| point_curve | rgb | [(0, 13), (77, 57), (148, 140), (255, 239)] |  | JYB | numeric | tone curve points |
| color_grading | midtones | (197, 9) |  | JYB | numeric | midtones hue 197, saturation 9 |
| hsl | hue.red | -14 |  | JYB | numeric | red hue -14 |
| hsl | hue.orange | 2 |  | JYB | numeric | orange hue +2 |
| hsl | hue.yellow | 1 |  | JYB | numeric | yellow hue +1 |
| hsl | hue.green | 10 |  | JYB | numeric | green hue +10 |
| hsl | hue.aqua | 14 |  | JYB | numeric | aqua hue +14 |
| hsl | hue.blue | 12 |  | JYB | numeric | blue hue +12 |
| hsl | hue.purple | 2 |  | JYB | numeric | purple hue +2 |
| hsl | hue.magenta | 5 |  | JYB | numeric | magenta hue +5 |
| hsl | sat.red | 6 |  | JYB | numeric | red sat +6 |
| hsl | sat.orange | 1 |  | JYB | numeric | orange sat +1 |
| hsl | sat.yellow | -25 |  | JYB | numeric | yellow sat -25 |
| hsl | sat.green | -11 |  | JYB | numeric | green sat -11 |
| hsl | sat.aqua | -23 |  | JYB | numeric | aqua sat -23 |
| hsl | sat.blue | -29 |  | JYB | numeric | blue sat -29 |
| hsl | sat.purple | -10 |  | JYB | numeric | purple sat -10 |
| hsl | sat.magenta | -10 |  | JYB | numeric | magenta sat -10 |
| hsl | lum.red | -10 |  | JYB | numeric | red lum -10 |
| hsl | lum.yellow | 10 |  | JYB | numeric | yellow lum +10 |
| hsl | lum.blue | 3 |  | JYB | numeric | blue lum +3 |

- **JYB**: [James Young, How To Edit Backlit Portraits: Full Breakdown](https://jamesyoungphotography.com/lightroom/backlit-portrait)


## Food & objects

### Bright Food (`bright_food`)

Best for: food on light tables, overhead and daylight. Grade B: some numbers; the rest are directions.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | 0.25 |  | RS | numeric | exposure +0.25 |
| tone_regions | highlights | -100 |  | RS | numeric | highlights all the way down |
| tone_regions | whites | 35 |  | RS | numeric | whites around 35 |
| tone_regions | shadows | 75 |  | RS | numeric | shadows around 75 |
| white_balance | temperature | 8.333333333333334 |  | FSA | converted | about 5500-6000 K |
| white_balance | tint | 7.5 |  | FSA | mid | tint toward magenta +5 to +10 |
| point_curve | rgb | [(0, 19), (64, 60), (192, 200), (255, 255)] |  | FSA | mid | black point lifted 5-10 %, gentle S-curve (S chosen) |
| hsl | hue.red | 7.5 |  | FSA | mid | red hue +7.5 |
| hsl | sat.yellow | 12.5 |  | FSA | mid | yellow sat +12.5 |
| hsl | sat.red | 7.5 |  | FSA | mid | red sat +7.5 |
| hsl | lum.yellow | -17.5 |  | FSA | mid | yellow lum -17.5 |
| hsl | lum.green | 7.5 |  | FSA | mid | green lum +7.5 |
| color_grading | shadows | (200, 6.5) |  | FSA | converted | cool blue/teal shadows, saturation 5-8 % |
| color_grading | balance | 15 |  | FSA | mid | balance +10 to +20 |

- **FSA**: [Ali Tanis (FoodShot AI), Food Color Grading: How to Make Food Photos Pop](https://foodshot.ai/blog/food-color-grading)
- **RS**: [Heather Templeton (Replica Surfaces), Light and airy food photography](https://www.replicasurfaces.com/blogs/replica-surfaces-blog/how-to-take-light-and-airy-food-photos)

### Moody Food (`moody_food`)

Best for: food, drinks and still life on dark surfaces. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| exposure | stops | -0.5 |  | FL | mid | exposure -0.3 to -0.7 |
| contrast | amount | 40 |  | FL | mid | contrast +30 to +50 |
| tone_regions | highlights | -70 |  | FL | mid | highlights -60 to -80 |
| tone_regions | shadows | 40 |  | FL | mid | shadows +30 to +50 |
| tone_regions | whites | -30 |  | FL | mid | whites -20 to -40 |
| tone_regions | blacks | -25 |  | FL | mid | blacks -20 to -30 |
| saturation | vibrance | -15 |  | FL | mid | vibrance -10 to -20 |
| saturation | sat | -20 |  | FL | mid | saturation -15 to -25 |
| hsl | sat.blue | -20 |  | FL | numeric | blue sat -20 |
| hsl | sat.green | -30 |  | FL | numeric | green sat -30 |
| hsl | sat.yellow | -15 |  | FL | numeric | yellow sat -15 |
| hsl | lum.orange | -10 |  | FL | direction | warm luminance reduced |
| color_grading | highlights | (200, 10) |  | FL | direction | cool highlights |
| color_grading | shadows | (30, 10) |  | FL | direction | warm shadows |
| color_grading | balance | -15 |  | FL | numeric | balance -15 |
| calibration | shadow_tint | 10 |  | FL | numeric | shadow tint +10 |
| calibration | blue_hue | -10 |  | FL | numeric | blue primary hue -10 |
| calibration | blue_sat | -20 |  | FL | numeric | blue primary saturation -20 |
| vignette | amount | -20 |  | FL | numeric | vignette -20 |
| point_curve | rgb | [(0, 0), (64, 50), (192, 207), (255, 255)] |  | FL | direction | S-curve |

- **FL**: [Flourish Presets, How to Edit Dark and Moody Photos With Moody Presets](https://flourishpresets.com/blogs/flourish-presets-lightroom-presets-luts/how-to-edit-dark-and-moody-photos-moody-presets)

### Ektar 100 (`ektar_100`)

Best for: colourful travel scenes and objects in daylight. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | -5.0 |  | LE | converted | 5200-5500 K |
| white_balance | tint | -6.5 |  | LE | mid | tint -5 to -8 |
| tone_regions | highlights | -25 |  | LE | mid | highlights -20 to -30 |
| tone_regions | whites | -10 |  | LPV | mid | whites -5 to -15 |
| saturation | vibrance | 15 |  | LPV | mid | vibrance +10 to +20 |
| point_curve | rgb | [(0, 10), (64, 58), (192, 200), (255, 255)] |  | LE | numeric | shadow lift +10 at the bottom-left point, gentle S (S chosen) |
| hsl | hue.yellow | -5 |  | LPV | mid | Ektar colour table |
| hsl | hue.green | -5 |  | LPV | mid | Ektar colour table |
| hsl | sat.red | 21.5 |  | LPV | mid | Ektar colour table |
| hsl | sat.orange | -6.5 |  | LPV | mid | Ektar colour table |
| hsl | sat.yellow | 5 |  | LPV | mid | Ektar colour table |
| hsl | sat.green | 5 |  | LPV | mid | Ektar colour table |
| hsl | sat.aqua | 8 |  | LPV | mid | Ektar colour table |
| hsl | sat.blue | 11 |  | LPV | mid | Ektar colour table |
| hsl | lum.red | -9 |  | LPV | mid | Ektar colour table |
| hsl | lum.orange | -5 |  | LPV | mid | Ektar colour table |
| hsl | lum.yellow | -5 |  | LPV | mid | Ektar colour table |
| hsl | lum.aqua | -8 |  | LPV | mid | Ektar colour table |
| hsl | lum.blue | -13.5 |  | LPV | mid | Ektar colour table |

- **LE**: [Legendary Presets, How to Get the Kodak Ektar 100 Look in Lightroom](https://legendarypresets.com/kodak-ektar-100-lightroom-preset-guide/)
- **LPV**: [Legendary Presets, Step-by-Step Lightroom Workflow for High-Saturation Film Looks](https://legendarypresets.com/step-by-step-lightroom-workflow-for-high-saturation-film-looks/)


## Seasons

### Autumn (`autumn`)

Best for: autumn foliage and cosy autumn scenes. Grade A: the tutorials give (nearly) all the numbers.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | 5 |  | WG | numeric | temperature +5 |
| exposure | stops | 0.3 |  | WG | numeric | exposure +0.30 |
| contrast | amount | -20 |  | WG | numeric | contrast -20 |
| tone_regions | highlights | 20 |  | WG | numeric | highlights +20 |
| tone_regions | shadows | 40 |  | WG | numeric | shadows +40 |
| tone_regions | whites | -20 |  | WG | numeric | whites -20 |
| tone_regions | blacks | -20 |  | WG | numeric | blacks -20 |
| saturation | vibrance | 30 |  | WG | numeric | vibrance +30 |
| saturation | sat | -20 |  | WG | numeric | saturation -20 |
| point_curve | rgb | [(0, 0), (65, 45), (130, 120), (195, 190), (255, 255)] |  | WG | numeric | point curve |
| hsl | hue.orange | -10 |  | WG | numeric | orange hue -10 |
| hsl | hue.yellow | -65 |  | WG | numeric | yellow hue -65 |
| hsl | hue.green | -100 |  | WG | numeric | green hue -100 |
| hsl | sat.red | -20 |  | WG | numeric | red sat -20 |
| hsl | sat.orange | 30 |  | WG | numeric | orange sat +30 |
| hsl | sat.yellow | -20 |  | WG | numeric | yellow sat -20 |
| hsl | sat.green | -40 |  | WG | numeric | green sat -40 |
| hsl | sat.aqua | -20 |  | WG | numeric | aqua sat -20 |
| hsl | sat.blue | -20 |  | WG | numeric | blue sat -20 |
| hsl | lum.red | 10 |  | WG | numeric | red lum +10 |
| hsl | lum.orange | -10 |  | WG | numeric | orange lum -10 |
| hsl | lum.yellow | -20 |  | WG | numeric | yellow lum -20 |
| hsl | lum.green | -20 |  | WG | numeric | green lum -20 |
| hsl | lum.aqua | -20 |  | WG | numeric | aqua lum -20 |
| hsl | lum.blue | -20 |  | WG | numeric | blue lum -20 |
| color_grading | shadows | (30, 20) |  | WG | numeric | shadows hue 30, saturation 20 |

- **WG**: [Diego Sanchez (WeGraphics), Enhance Autumn Colors in Lightroom with Ease](https://we.graphics/blog/enhance-autumn-colors-in-lightroom-with-ease/)

### Winter (`winter`)

Best for: snowy landscapes. Grade C: mostly directions; the amounts were chosen.

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| color_grading | shadows | (215, 12.5) |  | MLC | mid | shadows hue 210-220, saturation 10-15 |
| color_grading | highlights | (40, 7.5) |  | MLC | mid | highlights hue 35-45, saturation 5-10 |
| hsl | hue.blue | -12.5 |  | MLC | mid | blue hue toward aqua, -10 to -15 (for snow) |
| contrast | amount | 16 |  | DPSW | numeric | contrast +16 (example image) |
| vignette | amount | -20 |  | DPSW | numeric | post-crop vignette -20 |
| exposure | stops | -0.1 |  | PFW | numeric | exposure -0.10 |
| tone_regions | whites | -15 |  | PFW | mid | whites -10 to -20 |
| white_balance | temperature | 3.3333333333333335 |  | PFW | converted | temperature raised 9400 -> 9500 K |

- **DPSW**: [David Shaw (Digital Photography School), Tips for Processing Winter Landscapes in Lightroom](https://digital-photography-school.com/tips-for-processing-winter-landscapes-in-lightroom/)
- **MLC**: [Mikko Lagerstedt, Cinematic Color Grading: Creating Atmosphere in Landscape Photography](https://mikkolagerstedt.com/blog/cinematic-color-grading-landscape-photography/)
- **PFW**: [James Abbott (Picfair Focus), How to shoot and edit snowy winter landscapes](https://focus.picfair.com/articles/how-to-shoot-and-edit-snowy-landscapes)


## Cyberpunk (`cyberpunk`)

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | -25 |  | TB, SG | direction | temperature toward blue |
| white_balance | tint | 20 |  | TB, SG | direction | tint toward magenta |
| contrast | amount | 25 |  | DT, DD | direction | S-curve: darker shadows, brighter highlights |
| tone_regions | highlights | -40 |  | SG, DD | direction | decrease highlights |
| tone_regions | shadows | 30 |  | SG, DD | direction | increase shadows (HDR-style detail) |
| tone_regions | whites | 10 |  | DD | direction | boost whites slightly |
| channel_curves | b | (0.08, 0.06) |  | DT | direction | blue channel: bottom-left point up (blue shadows), top-right point down (yellow highlights) |
| hsl | hue.red | -40 |  | TB | direction | red -> magenta |
| hsl | hue.green | 80 |  | TB | direction | light green -> cyan |
| hsl | hue.blue | -40 |  | TB | direction | blue -> cyan |
| hsl | hue.purple | 40 |  | TB | direction | purple -> magenta |
| hsl | hue.magenta | -20 |  | TB | direction | magenta -> purple |
| hsl | sat.green | -100 |  | TB | numeric | green saturation -100 |
| hsl | sat.yellow | 20 |  | TB | direction | boost yellow saturation |
| hsl | sat.blue | 30 |  | TB | direction | boost blue saturation |
| hsl | sat.purple | 30 |  | TB | direction | boost purple saturation |
| hsl | lum.red | -20 |  | TB | direction | decrease red brightness |
| hsl | lum.orange | -20 |  | TB | direction | decrease orange brightness |
| hsl | lum.blue | 20 |  | TB | direction | enhance blue brightness |
| calibration | red_hue | -100 |  | DT, TB | numeric | red primary all the way left (toward magenta/pink) |
| calibration | green_hue | 50 |  | DT, TB | conflict | DT: green primary to the right (purple and cyan); TB: toward yellow. DT is followed (Lightroom-specific) |
| calibration | blue_hue | -40 |  | TB | direction | blue primary toward cyan |
| color_grading | shadows | (220, 35) |  | SG, TB, DT | direction | rich blue shadows |
| color_grading | midtones | (285, 15) |  | TB | direction | magenta and blue in the midtones |
| color_grading | highlights | (320, 30) |  | SG, TB | direction | bright pink highlights |
| dehaze | amount | 20 |  | SG, DT | direction | boost dehaze to restore contrast |
| vignette | amount | -25 |  | SG | direction | darkened edges |

- **DD**: [Drew Deltz, Editing Like a Pro: Transforming Photos into Cyberpunk Masterpieces with Photoshop](https://drewdeltz.systeme.io/blog/editing-like-a-pro-transforming-photos-into-cyberpunk-masterpieces-with-photoshop)
- **DT**: [Denny's Tips, Cyberpunk Lightroom Tutorial](https://www.dennystips.com/cyberpunk-lightroom-tutorial/)
- **SG**: [Spoon Graphics, How to Apply Cyberpunk Style Color Grading & Neon Effects to Your Photos](https://blog.spoongraphics.co.uk/tutorials/how-to-apply-cyberpunk-style-color-grading-neon-effects-to-your-photos)
- **TB**: [TourBox, Creating Cyberpunk Color Grading in Photoshop: A Step-by-Step Guide](https://www.tourboxtech.com/en/news/cyberpunk-colors.html)

Not modelled: clarity - (softer glow; SG, DT): a local operation; neon glow layers (SG, DD).


## Candidates (not built)

### Fujifilm tutorial recipe

Calibration target: statistics of 46 openly licensed Fuji film scans (Superia, C200, Pro 400H), measured on 60 open input photos

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | -8 |  | LP | numeric | white balance 5000-5200 K instead of 5500 K daylight, i.e. slightly cooler (converted to this scale) |
| contrast | amount | 10 | x4 = 40.0 | PL, JP | direction | a bit more contrast than neutral (Classic Negative) |
| tone_regions | shadows | 15 | x0 (unused) | PL | direction | softens shadows |
| tone_regions | whites | 10 | x4 = 40.0 | PL | direction | increases whites |
| tone_regions | blacks | -10 | x4 = -40.0 | PL | direction | deepens blacks |
| fade | amount | 20 | x1 = 20 | ST | direction | the 'full film look' version adds shadow fade |
| hsl | hue.green | 10 |  | LP | numeric | green hue +10 toward aqua |
| hsl | sat.yellow | -12 |  | LP | numeric | yellow saturation -10 to -15 |
| hsl | sat.blue | -10 | x4 = -40.0 | JP | direction | cooler colours desaturated more than reds and oranges |
| hsl | sat.aqua | -10 | x4 = -40.0 | JP | direction | cooler colours desaturated more than reds and oranges |
| hsl | sat.red | 5 | x0 (unused) | ST | direction | vivid reds |
| hsl | sat.green | 5 | x0 (unused) | ST | direction | green greens |
| saturation | vibrance | 10 | x0 (unused) | PL | direction | amplifies vibrance |
| color_grading | shadows | (95, 14) | x0.5 = (95, 7.0) | PL, ST | direction | lime green in the shadows / slight greens in shadows |
| color_grading | highlights | (35, 12) | x0 (unused) | PL, ST | direction | warm brown highlights / warm but not green whites |

- **JP**: [J.M. Peltier, Fujifilm Classic Chrome vs. Classic Neg](https://www.jmpeltier.com/fujifilm-classic-chrome-vs-classic-neg/)
- **LP**: [Legendary Presets, Iconic Fuji Film Looks: Stock-by-Stock Lightroom Guide (Superia 100/400/800)](https://legendarypresets.com/guide-to-iconic-fuji-film-looks/)
- **PL**: [PresetLove, Superia 400 Film Preset](https://presetlove.com/presets/superia-400/)
- **ST**: [Scott Tucker, My Fujifilm Superia X-Tra 400 Film Simulation](https://www.scotttuckerphoto.com/blog/superia400)

### Classic Chrome

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| tone_regions | highlights | -25 |  | ES | numeric | highlights -20 to -30 |
| tone_regions | shadows | 15 |  | ES | numeric | shadows +10 to +20 |
| tone_regions | blacks | 15 |  | ES | numeric | blacks +10 to +20 |
| contrast | amount | -15 |  | ES | numeric | contrast -10 to -20 |
| exposure | stops | 0.1 |  | ES | numeric | exposure 0 to +0.2 |
| saturation | vibrance | -18 |  | ES | numeric | vibrance -15 to -20 |
| saturation | sat | -10 |  | ES | numeric | saturation -10 |
| hsl | sat.green | -22 |  | ES | numeric | green saturation -20 to -25 |
| hsl | sat.blue | -18 |  | ES | numeric | blue saturation -15 to -20 |
| hsl | sat.red | -8 |  | ES | numeric | red saturation -5 to -10 |
| hsl | hue.blue | -15 |  | JP | direction | blues keep some vibrance and shift toward cyan |
| color_grading | shadows | (210, 8) |  | ES | direction | very slight cool grey in the shadows |

- **ES**: [The Editing Studio, Fujifilm Look Lightroom Preset: Complete Guide (Classic Chrome)](https://theeditingstudio.co/blog/fujifilm-look-lightroom-preset)
- **JP**: [J.M. Peltier, Fujifilm Classic Chrome vs. Classic Neg](https://www.jmpeltier.com/fujifilm-classic-chrome-vs-classic-neg/)
