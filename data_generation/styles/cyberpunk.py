"""Cyberpunk-style data generator (Phase 4).

Cyberpunk imagery is characterised by:

  - A strongly contrasted tonal response (crushed shadows, lifted highlights).
  - A cool / teal cast in the shadows and a complementary warm push in
    the highlights, traditionally captured by a "teal-and-orange" color
    grade.
  - Mild film grain to suggest analog reproduction.

This generator implements those effects using exactly the operations the
DifferentiableGenericRenderer can reproduce: a PCHIP tone curve, an affine
3x3 + bias colour grade, and additive Gaussian grain. This is deliberate -
Phase 4 of PROJECT_PLAN.md asks for a style implemented "using the generic
primitives from Phase 3", verifying that the network architecture is style
agnostic.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..core import StyleGenerator
from .film import Grain


# --- Tonal response --------------------------------------------------------
# A symmetric S-curve through five PCHIP control points. The interior knots
# pull shadows down by 0.15 and lift highlights by 0.15, doubling the
# slope of a linear curve near mid-tones for the characteristic crushed
# look without clipping the endpoints.
_TONE_X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_TONE_Y = np.array([0.0, 0.10, 0.5, 0.90, 1.0])

# --- Colour grade ---------------------------------------------------------
# Diagonal entries: small per-channel gain. Off-diagonal entries route a
# fraction of one channel into another. The bias is a global teal lift.
#
#   c_out = M @ c_in + b
#
# with
#       [ 1.10   0.05  -0.05 ]            [ -0.02 ]
#   M = [ 0.00   0.95   0.05 ],   b =    [  0.02 ]
#       [-0.10   0.05   1.05 ]            [  0.08 ]
#
# The negative B->R entry combined with the positive B-bias gives shadows
# a clear teal cast; the positive R diagonal + slight R->G coupling keeps
# warmer highlights from going outright magenta.
_COLOR_MATRIX = np.array(
    [
        [1.10, 0.05, -0.05],
        [0.00, 0.95, 0.05],
        [-0.10, 0.05, 1.05],
    ],
    dtype=np.float32,
)
_COLOR_BIAS = np.array([-0.02, 0.02, 0.08], dtype=np.float32)

# --- Grain ----------------------------------------------------------------
_GRAIN_INTENSITY = 0.02


class CyberpunkGenerator(StyleGenerator):
    def __init__(self):
        self._tone = PchipInterpolator(_TONE_X, _TONE_Y)
        self._matrix = _COLOR_MATRIX
        self._bias = _COLOR_BIAS
        self._grain = Grain(intensity=_GRAIN_INTENSITY)

    # --- per-primitive operations -----------------------------------------

    def _apply_tone(self, image: np.ndarray) -> np.ndarray:
        return np.clip(self._tone(image), 0.0, 1.0)

    def _apply_color(self, image: np.ndarray) -> np.ndarray:
        H, W, C = image.shape
        flat = image.reshape(-1, 3)
        out = flat @ self._matrix.T + self._bias
        return np.clip(out, 0.0, 1.0).reshape(H, W, C)

    # --- StyleGenerator interface -----------------------------------------

    def generate_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        styled = image.copy()
        styled = self._apply_tone(styled)
        styled = self._apply_color(styled)
        styled = self._grain.apply(styled)
        return image, styled
