"""Write docs/looks.md from the recipe tables (photostyle/recipes.py, photostyle/recipe_library.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.recipe_library import LOOKS  # noqa: E402
from photostyle.recipes import CYBERPUNK_NOT_MODELLED, describe  # noqa: E402

GRADES = {"A": "the tutorials give (nearly) all the numbers", "B": "some numbers; the rest are directions",
          "C": "mostly directions; the amounts were chosen"}

head = """# Looks and the tutorials they come from

Each tutorial look is a recipe transcribed from public grading tutorials, as Lightroom-style
slider settings (tables in `photostyle/recipe_library.py` and `photostyle/recipes.py`). Since plan
amendment A6 the editor applies a recipe exactly as those slider values in the image pipeline
([pipeline.md](pipeline.md); conversion in `photostyle/slider_looks.py`), so a look is fully
editable and needs no model. Two further groups are described at the end: [film
stocks](#film-stocks), fitted to film colour tables, and [expert styles](#expert-styles), predicted
per photo by models trained on FiveK retouchers ([data.md](data.md)).

Kinds:
- **numeric**: the tutorial states the value.
- **mid**: the middle of a range the tutorial states.
- **converted**: a stated value converted to this tool's scale. Kelvin is measured from 5500 K,
  30 K per slider unit; named colours use their hue angle.
- **direction**: the tutorial gives only the direction; the amount is chosen.

The recipes hold only whole-photo settings. Where a tutorial also uses local steps (graduated
and radial filters, brushes), those are left out; the editor's masks can add them.

`fujifilm` is the exception. It is trained directly on 46 openly licensed Fuji film scans (the
owner's choice). Its tutorial recipe is kept below as a candidate.
"""
FILM_EXPERT = """
## Film stocks

Twelve stocks from the RawTherapee Film Simulation Collection (CC BY-SA 4.0): Kodak Portra 400 and
160, Ektar 100, Kodachrome 64, Ektachrome 100 VS, Tri-X 400; Fuji Velvia 50, Provia 100F, Pro
400H, Superia 400; Agfa Vista 200; Ilford HP5 Plus 400. Each is a fixed slider setting fitted to
the stock's colour table by `tools/fit_film.py` (settings in `photostyle/film_looks.json`); the
fitted sliders are within 0.7–1.7 mean ΔE00 of the table on held-out photos
([Phase 21 report](../tests/reports/phase21_models.md#film-fit)).

## Expert styles

The editing styles of the five MIT-Adobe FiveK retouchers (A–E). Unlike the looks above, the
sliders differ per photo: a model trained on the retoucher's 3,500 training edits predicts tone,
white balance and colour-mixer settings for the photo when the look is chosen
([data.md](data.md#expert-styles)).
"""
parts = [head]
cats: dict[str, list] = {}
for name, look in LOOKS.items():
    cats.setdefault(look["category"], []).append((name, look))
for cat, looks in cats.items():
    parts.append(f"\n## {cat}\n")
    for name, look in looks:
        parts.append(f"### {look['title']} (`{name}`)\n\nBest for: {look['subject']}. Grade {look['grade']}: "
                     f"{GRADES[look['grade']]}.\n\n{describe(name)}\n")
parts.append(f"\n## Cyberpunk (`cyberpunk`)\n\n{describe('cyberpunk')}\n\nNot modelled: {'; '.join(CYBERPUNK_NOT_MODELLED)}.\n")
parts.append(FILM_EXPERT)
parts.append(f"\n## Candidates (not built)\n\n### Fujifilm tutorial recipe\n\n{describe('fujifilm')}\n\n"
             f"### Classic Chrome\n\n{describe('classic_chrome')}\n")
Path("docs/looks.md").write_text("\n".join(parts))
print("docs/looks.md written")
