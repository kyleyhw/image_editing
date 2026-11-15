"""Hald colour look-up tables (HaldCLUT): loading and trilinear application.

A HaldCLUT of *level* ``L`` is a square image of side ``L**3`` pixels that
stores a 3-D lattice of ``N = L**2`` nodes per axis (``N**3 = L**6`` nodes,
one per pixel). Reading the pixels in raster order (left to right, then top
to bottom), pixel ``i`` holds the output colour for the input node

    r = i mod N,   g = (i div N) mod N,   b = i div N**2,

i.e. red varies fastest [[1]](#ref-hald). The node coordinate ``k`` stands for
the input value ``k / (N - 1)``. The RawTherapee Film Simulation Collection
uses level 12 (1728 x 1728 px, N = 144) and 8-bit sRGB code values, so a table
maps sRGB-encoded colour to sRGB-encoded colour.

``load_haldclut`` returns the lattice as ``float32`` of shape (N, N, N, 3)
indexed **[r, g, b]** (the natural raster reshape gives [b, g, r]; it is
transposed so that ``lut[r, g, b]`` reads like the colour it maps).

``apply_haldclut`` evaluates the table at arbitrary colours in [0, 1] by
trilinear interpolation: with ``x = c (N - 1)``, ``i = floor(x)`` and
``f = x - i`` per channel,

    out = sum over (dr, dg, db) in {0,1}^3 of
          w_r(dr) w_g(dg) w_b(db) * lut[i_r + dr, i_g + dg, i_b + db],
    w(0) = 1 - f,  w(1) = f.

Trilinear interpolation reproduces any affine map exactly, so the identity
table returns its input up to the table's 8-bit quantisation (half a code
value at the nodes).

References
----------
[1] Eskil Steenberg, "Hald CLUT", http://www.quelsolaar.com/technology/clut.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = ["apply_haldclut", "identity_haldclut", "load_haldclut"]


def _level_from_side(side: int) -> int:
    """HaldCLUT level L from the image side L**3; raises if not a cube."""
    level = round(side ** (1.0 / 3.0))
    if level**3 != side:
        raise ValueError(f"side {side} px is not a cube L**3; not a HaldCLUT image")
    return level


def load_haldclut(path: str | Path) -> NDArray[np.float32]:
    """Load a HaldCLUT image as a (N, N, N, 3) lattice indexed [r, g, b].

    Values are the image's code values scaled to [0, 1] (8-bit images).
    """
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA", "P", "L"):
            raise ValueError(f"unsupported image mode {im.mode!r} for {path}")
        if im.width != im.height:
            raise ValueError(f"{path}: HaldCLUT images are square, got {im.size}")
        code = np.asarray(im.convert("RGB"), dtype=np.float32)
    level = _level_from_side(code.shape[0])
    n = level * level
    bgr = code.reshape(n, n, n, 3) / np.float32(255.0)  # raster order -> [b, g, r]
    return np.ascontiguousarray(bgr.transpose(2, 1, 0, 3))


def identity_haldclut(level: int) -> NDArray[np.uint8]:
    """8-bit identity HaldCLUT image of side ``level**3`` (for tests and
    for building tables by editing the identity image)."""
    n = level * level
    k = np.arange(n**3)
    rgb = np.stack([k % n, (k // n) % n, k // (n * n)], axis=-1).astype(np.float64)
    code = np.round(rgb / (n - 1) * 255.0).astype(np.uint8)
    side = level**3
    return code.reshape(side, side, 3)


def apply_haldclut(
    img: NDArray[np.floating], lut: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Map an image (..., 3) with values in [0, 1] through ``lut`` (N, N, N, 3)
    indexed [r, g, b], by trilinear interpolation. Inputs are clipped to [0, 1]."""
    n = lut.shape[0]
    if lut.shape != (n, n, n, 3):
        raise ValueError(f"lut must have shape (N, N, N, 3), got {lut.shape}")
    x = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * np.float32(n - 1)
    # The cell index is capped at N-2 so that i+1 stays in range; at the top
    # edge (x = N-1) this gives i = N-2, f = 1, which selects the last node.
    i = np.minimum(np.floor(x), n - 2).astype(np.intp)
    f = x - i
    ir, ig, ib = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = np.zeros(x.shape, dtype=np.float32)
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dg, wg in ((0, 1.0 - fg), (1, fg)):
            for db, wb in ((0, 1.0 - fb), (1, fb)):
                out += (wr * wg * wb) * lut[ir + dr, ig + dg, ib + db]
    return out
