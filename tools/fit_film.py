"""Phase 21 task 20: film looks as slider settings, fitted to film colour tables.

The RawTherapee Film Simulation Collection (CC BY-SA 4.0; data/film_luts, Phase 20) gives, for
each film stock, a colour table: a fixed map of display sRGB colours. For a chosen set of stocks
this tool finds the pipeline sliders (white balance, tone, vibrance, saturation, 8-band HSL, colour
grading, calibration) and R, G, B point curves that best reproduce the table, by gradient descent
on the Oklab distance over well-exposed photos (MIT-Adobe FiveK test photos as edited by expert
C, which are close to neutral). Black-and-white stocks fit the black-and-white mix instead of
colour.

The remaining error on held-out photos (mean CIEDE2000) decides whether a look would need a
residual colour table (A6.4: proposed when above 2).

    uv run python tools/fit_film.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from photostyle import auto as A  # noqa: E402
from photostyle import pipeline as P  # noqa: E402
from photostyle.colour_diff import ciede2000, srgb_to_lab  # noqa: E402
from photostyle.haldclut import apply_haldclut, load_haldclut  # noqa: E402

STOCKS = [  # (look name, table name, title, black and white)
    ("portra_400_film", "Kodak Portra 400 2", "Kodak Portra 400", False),
    ("portra_160_film", "Kodak Portra 160 2", "Kodak Portra 160", False),
    ("ektar_film", "Kodak Ektar 100", "Kodak Ektar 100", False),
    ("kodachrome_film", "Kodak Kodachrome 64", "Kodak Kodachrome 64", False),
    ("ektachrome_film", "Kodak Ektachrome 100 VS", "Kodak Ektachrome 100 VS", False),
    ("velvia_film", "Fuji Velvia 50", "Fuji Velvia 50", False),
    ("provia_film", "Fuji Provia 100F", "Fuji Provia 100F", False),
    ("pro400h_film", "Fuji 400H 2", "Fuji Pro 400H", False),
    ("superia_film", "Fuji Superia 400 2", "Fuji Superia 400", False),
    ("vista_film", "Agfa Vista 200", "Agfa Vista 200", False),
    ("trix_film", "Kodak TRI-X 400 2", "Kodak Tri-X 400", True),
    ("hp5_film", "Ilford HP5 Plus 400", "Ilford HP5 Plus 400", True),
]
TONE = ["temp", "tint", "exposure", "contrast", "highlights", "shadows", "whites", "blacks"]
COLOUR = ["vibrance", "saturation"] + A.COLOUR + [f"grade_{r}_{k}" for r in ("shadows", "midtones", "highlights") for k in ("hue", "sat")] \
    + [f"cal_{c}_{k}" for c in ("red", "green", "blue") for k in ("hue", "sat")]
BW = [f"bw_{b}" for b in P.BAND_NAMES]
KNOTS = [i / 8 for i in range(9)]


def unit(n: str) -> float:
    return 1.0 if n == "exposure" else 360.0 if n.endswith("_hue") and n.startswith("grade_") else 100.0


def fit(xs: torch.Tensor, ys: torch.Tensor, bw: bool, steps: int = 220) -> tuple[dict, torch.Tensor]:
    names = TONE + (BW if bw else COLOUR)
    u = torch.tensor([unit(n) for n in names])
    lo = torch.tensor([float(next(s for s in P.SLIDERS if s["name"] == n)["min"]) for n in names])
    hi = torch.tensor([float(next(s for s in P.SLIDERS if s["name"] == n)["max"]) for n in names])
    z = torch.zeros(len(names))
    for i, n in enumerate(names):                     # colour-wheel hues start spread around the wheel
        if n.endswith("_hue") and n.startswith("grade_"):
            z[i] = {"shadows": 210, "midtones": 30, "highlights": 45}[n.split("_")[1]] / 360
    z.requires_grad_(True)
    cy = torch.zeros(3, 9, requires_grad=True)       # offsets of the R, G, B curve heights at KNOTS
    opt = torch.optim.Adam([z, cy], lr=0.03)
    kn = torch.tensor(KNOTS)

    def params(B: int) -> dict:
        v = torch.maximum(torch.minimum(z * u, hi), lo)
        p: dict = {n: v[i].expand(B) for i, n in enumerate(names)}
        if bw:
            p["bw"] = 1.0
        p["curves"] = {c: {"x": KNOTS, "y": P._monotone((kn + 0.3 * torch.tanh(cy[i])).clamp(0, 1).unsqueeze(0)).expand(B, -1)}
                       for i, c in enumerate("rgb")}
        return p

    for _ in range(steps):
        out = P.render(xs, params(len(xs)))
        loss = A.oklab_distance(out, ys).mean() + 1e-3 * (z.abs()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        v = torch.maximum(torch.minimum(z * u, hi), lo)
        sl: dict = {n: round(float(v[i]), 2 if n == "exposure" else 1) for i, n in enumerate(names)}
        sl = {k: x for k, x in sl.items() if abs(x) >= (0.01 if k == "exposure" else 0.5)}
        if bw:
            sl["bw"] = 1.0
        ys_c = P._monotone((kn + 0.3 * torch.tanh(cy)).clamp(0, 1))
        sl["curves"] = {c: [[round(KNOTS[j], 4), round(float(ys_c[i, j]), 4)] for j in range(9)] for i, c in enumerate("rgb")}
    return sl, loss.detach()


def de00(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(ciede2000(srgb_to_lab(a.clamp(0, 1).permute(0, 2, 3, 1).double().numpy()),
                           srgb_to_lab(b.clamp(0, 1).permute(0, 2, 3, 1).double().numpy())).mean())


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(3)
    cache = sorted((ROOT / "data/cache").glob("pairs_test_c_128_*.pt"))[0]
    photos = torch.load(cache)["y"]                   # expert C edits: well-exposed, near-neutral photos
    g = torch.Generator().manual_seed(11)
    idx = torch.randperm(len(photos), generator=g)
    train, held = photos[idx[:32]], photos[idx[32:48]]
    man = {r.split(",")[0]: r.split(",")[1] for r in (ROOT / "data/manifests/film_luts.csv").read_text().splitlines()[1:]}
    out, rows = {}, []
    for name, table, title, bw in STOCKS:
        lut = load_haldclut(ROOT / "data" / man[table])
        tgt = lambda x: torch.from_numpy(apply_haldclut(x.permute(0, 2, 3, 1).numpy(), lut)).permute(0, 3, 1, 2)  # noqa: E731
        sl, _ = fit(train, tgt(train), bw)
        with torch.no_grad():
            pred = P.render(held, sl)
        e_fit, e_id = de00(pred, tgt(held)), de00(held, tgt(held))
        out[name] = {"title": title, "table": table, "bw": bw, "sliders": sl, "heldout_de00": round(e_fit, 2),
                     "identity_de00": round(e_id, 2)}
        rows.append((title, e_id, e_fit))
        print(f"{title:26s} table vs photo dE00 {e_id:5.2f}   fitted sliders vs table {e_fit:5.2f}", flush=True)
    (ROOT / "photostyle/film_looks.json").write_text(json.dumps(out, indent=1))
    rep = {"rows": [{"stock": t, "identity_de00": round(a, 2), "fit_de00": round(b, 2)} for t, a, b in rows],
           "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "tests/reports/phase21_film_fit.json").write_text(json.dumps(rep, indent=1))
    print(f"done in {rep['runtime_s']} s")


if __name__ == "__main__":
    main()
