# Twenty proposed looks, each from tutorials

Research on 2026-09-26. Each look is backed by tutorials whose numbers were verified on the page.
The details, with every value and URL, are in the four notes in this folder.

Completeness:

- **A**: a (nearly) complete numeric recipe.
- **B**: partial numbers; some steps are given as directions only.
- **C**: mostly directions. These looks are only worth building if the owner accepts chosen
  amounts for the missing values.

| # | Look | Image type | Main tutorial(s) | Completeness |
|---|---|---|---|---|
| 1 | Moody forest | day landscape (overcast, green) | Run N Gun; lightroomtutorials "Jungle Green" | A |
| 2 | Golden hour | day landscape (sunrise/sunset) | LR Presets (Lucy Song); HueBliss grading | B |
| 3 | Velvia vivid | day landscape | Legendary Presets Velvia 100 HSL | A (ranges) |
| 4 | Milky Way | night landscape | Mikko Lagerstedt; Australian Photography | A |
| 5 | Blue hour | twilight landscape / seascape | Life Pixel, Fstoppers (directions); DPS white balance | C |
| 6 | Teal and orange | day city / street | lightroomtutorials; Denny's Tips; Presetpedia "Moody Japan" | B |
| 7 | Kodachrome street | day city / street | The Editing Studio (two guides) | A (ranges) |
| 8 | Matte urban | day city / street | The Editing Studio Vintage Film; Fstoppers faded | A (ranges) |
| 9 | CineStill 800T | night city | Lou & Marks; Joe D'Agostino (2850 K) | B |
| 10 | Wakui Tokyo night | night city (neon) | DIYPhotography (Tim Daniels); lightroomtutorials | B |
| 11 | Portra 400 | portraits, daylight | Medialoot curves; Presetpedia; The Editing Studio | A |
| 12 | Bright and airy | portraits, weddings, lifestyle | The Editing Studio; Denny's Tips | A (ranges) |
| 13 | Dark moody portrait | portraits, low key | James Young "Dark Witchy" | A |
| 14 | Tri-X black and white | documentary / street portraits | Presetpedia B&W; grain from DPS / Picfair | C |
| 15 | Golden backlit portrait | portraits at sunset | James Young "Backlit" (complete) | A |
| 16 | Bright food | food, overhead, white | Replica Surfaces; FoodShot AI | B |
| 17 | Moody food / still life | food, drinks, still life | Flourish Presets; MySpicyKitchen | A (ranges) |
| 18 | Ektar 100 | travel, objects, colour | Legendary Presets Ektar guides | A (ranges) |
| 19 | Autumn | autumn scenes, still life | WeGraphics (complete); Life Pixel | A |
| 20 | Winter snow | snowy landscapes | Mikko Lagerstedt grading; DPS; Picfair | C |

Already live: `fujifilm` (trained on Fuji scans) and `cyberpunk` (tutorial recipe).
A spare with an A-grade recipe: Classic Chrome (The Editing Studio).

## Tools the current develop() lacks

- **Point curves** with RGB and per-channel points. Needed for Portra, Autumn, Velvia and the
  backlit portrait.
- **Parametric curve regions**: Highlights / Lights / Darks / Shadows.
- **Black-and-white conversion** with a colour mixer (for Tri-X).
- **Absolute Kelvin white balance.** JPEG input has no as-shot white balance, so Kelvin values
  from raw-file tutorials must be converted to relative shifts. Each conversion would be
  documented.
- **Not modelled by a global renderer:** local adjustments (graduated or radial filters,
  brushes), clarity, texture, grain, halation and sharpening. These would be listed per look.
