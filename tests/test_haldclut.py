"""Tests for photostyle/haldclut.py (Phase 20, task 12).

- An identity HaldCLUT, written to PNG and loaded back, must reproduce random
  colours to < 1/255: the identity is affine, so trilinear interpolation is
  exact and only the 8-bit quantisation of the nodes (<= 0.5/255) remains.
- A channel-permuting table (out = (g, b, r)) checks the [r, g, b] index
  order: a [b, g, r] mix-up would return a different permutation.
- Exact node inputs must return the node values (no interpolation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photostyle.haldclut import apply_haldclut, identity_haldclut, load_haldclut

RNG_SEED = 20  # any fixed seed: the properties hold for every input
TOL = 1.0 / 255.0  # one 8-bit code value, the acceptance bound of task 12


def _random_colours(n: int = 20000) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    c = rng.random((n, 3), dtype=np.float32)
    # include the cube's corners and edges, where the cell index is capped
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T
    return np.concatenate([c, corners.astype(np.float32)])


# Levels 4 (N = 16, 64 px) and 8 (N = 64, 512 px): a coarse and a fine
# lattice; level 12 (the film collection's) behaves like 8 but is 6x larger.
@pytest.mark.parametrize("level", [4, 8])
def test_identity_roundtrip(tmp_path: Path, level: int) -> None:
    path = tmp_path / f"identity_{level}.png"
    Image.fromarray(identity_haldclut(level)).save(path)
    lut = load_haldclut(path)
    n = level * level
    assert lut.shape == (n, n, n, 3) and lut.dtype == np.float32
    c = _random_colours()
    err = np.abs(apply_haldclut(c, lut) - c).max()
    assert err < TOL, f"max error {err * 255:.3f}/255"


def test_index_order_is_rgb(tmp_path: Path) -> None:
    level = 4
    perm = identity_haldclut(level)[..., [1, 2, 0]]  # node (r, g, b) -> (g, b, r)
    path = tmp_path / "perm.png"
    Image.fromarray(np.ascontiguousarray(perm)).save(path)
    lut = load_haldclut(path)
    c = _random_colours()
    err = np.abs(apply_haldclut(c, lut) - c[:, [1, 2, 0]]).max()
    assert err < TOL


def test_nodes_are_exact(tmp_path: Path) -> None:
    level = 4
    n = level * level
    rng = np.random.default_rng(RNG_SEED)
    lut = rng.random((n, n, n, 3), dtype=np.float32)
    k = rng.integers(0, n, size=(500, 3))
    out = apply_haldclut(k / (n - 1), lut)
    np.testing.assert_allclose(out, lut[k[:, 0], k[:, 1], k[:, 2]], atol=1e-5)


def test_rejects_non_cube_image(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(path)
    with pytest.raises(ValueError):
        load_haldclut(path)
