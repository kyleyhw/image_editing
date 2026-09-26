"""Write docs/looks.md from the recipe tables (photostyle/recipes.py, photostyle/recipe_library.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostyle.recipe_library import LOOKS, NOT_MODELLED  # noqa: E402
from photostyle.recipes import CYBERPUNK_NOT_MODELLED, describe  # noqa: E402

GRADES = {"A": "the tutorials give (nearly) all the numbers", "B": "some numbers; the rest are directions",
          "C": "mostly directions; the amounts were chosen"}

head = """# Looks and the tutorials they come from

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

Not modelled anywhere: {nm}.

`fujifilm` is the exception. It is trained directly on 46 openly licensed Fuji film scans (the
owner's choice). Its tutorial recipe is kept below as a candidate.
"""
parts = [head.format(nm=NOT_MODELLED)]
cats: dict[str, list] = {}
for name, look in LOOKS.items():
    cats.setdefault(look["category"], []).append((name, look))
for cat, looks in cats.items():
    parts.append(f"\n## {cat}\n")
    for name, look in looks:
        parts.append(f"### {look['title']} (`{name}`)\n\nBest for: {look['subject']}. Grade {look['grade']}: "
                     f"{GRADES[look['grade']]}.\n\n{describe(name)}\n")
parts.append(f"\n## Cyberpunk (`cyberpunk`)\n\n{describe('cyberpunk')}\n\nNot modelled: {'; '.join(CYBERPUNK_NOT_MODELLED)}.\n")
parts.append(f"\n## Candidates (not built)\n\n### Fujifilm tutorial recipe\n\n{describe('fujifilm')}\n\n"
             f"### Classic Chrome\n\n{describe('classic_chrome')}\n")
Path("docs/looks.md").write_text("\n".join(parts))
print("docs/looks.md written")
