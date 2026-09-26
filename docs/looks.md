# Looks and the tutorials they come from

Each built-in look is a recipe transcribed from public grading tutorials, as Lightroom-style
slider settings (`photostyle/recipes.py`, tools in `photostyle/develop.py`). The recipe grades
openly licensed photos, and the content-adaptive model is trained on those before/after pairs
(`photostyle style train NAME --recipe teacher --open-only`).

Kinds: **numeric**: the tutorial gives the number. **direction**: it gives only the direction,
and the amount is chosen. **calibrated**: a factor on a direction-only amount, fitted to a target
(`tools/calibrate_recipe.py`, stored in `photostyle/calibration/`), keeping the tutorial's sign;
x0 means the target does not use that step. **conflict**: the tutorials disagree (the note says which was followed).

## fujifilm

Calibration target: statistics of 46 openly licensed Fuji film scans (Superia, C200, Pro 400H), measured on 60 open input photos

| tool | setting | tutorial value | calibrated | source | kind | what the tutorial says |
|---|---|---|---|---|---|---|
| white_balance | temperature | -8 |  | LP | numeric | white balance 5000-5200 K instead of 5500 K daylight, i.e. slightly cooler (converted to this scale) |
| contrast | amount | 10 | x4 = 40.0 | PL, JP | direction | a bit more contrast than neutral (Classic Negative) |
| tone_regions | shadows | 15 | x0 (unused) | PL | direction | softens shadows |
| tone_regions | whites | 10 | x4 = 40.0 | PL | direction | increases whites |
| tone_regions | blacks | -10 | x4 = -40.0 | PL | direction | deepens blacks |
| fade | amount | 20 | x1 = 20.0 | ST | direction | the 'full film look' version adds shadow fade |
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

Not modelled: grain 25-30 (LP).

## cyberpunk

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

## classic_chrome (candidate, not yet a style)

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

Not modelled: grain 15-20 amount, 20-25 size, 40-50 roughness (ES).
