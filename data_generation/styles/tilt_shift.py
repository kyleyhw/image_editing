"""Tilt-shift data generator (Phase 5).

Tilt-shift photography simulates a miniature scene by sharply focusing a
narrow horizontal band of the image and progressively blurring everything
above and below. The effect is *spatially variant*: different pixels
receive different amounts of blur, governed by their vertical position
relative to the in-focus band.

This generator implements that effect with a two-version Gaussian blend:

  out(x, y) = m(y) * I(x, y) + (1 - m(y)) * B(x, y)

where:

  - I is the original image,
  - B is a single Gaussian-blurred copy at sigma = max_sigma,
  - m(y) = 1 - smoothstep( w/2, w/2 + f, |y - y_c| )   is the focus mask,
    with y in [0, 1], y_c the focus-band center, w the band's half-height
    (full-focus inside [y_c - w/2, y_c + w/2]), and f the feather distance
    over which the mask falls to zero.

Smoothstep s(t) = 3t^2 - 2t^3 (clamped) is C^1 continuous, so the rendered
output has no visible seams along the focus boundary.

A multi-scale (Gaussian-pyramid) blend would be strictly more accurate -
the current binary lerp produces detail at the maximum sigma everywhere
outside the focus band, when a real tilt-shift lens would produce more
blur further from the focal plane. We trade that off for simplicity and
matching the torch-side renderer, which uses the same single-blur design.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from ..core import StyleGenerator


def smoothstep(t: np.ndarray) -> np.ndarray:
    """s(t) = 3t^2 - 2t^3, clamped to [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return 3.0 * t ** 2 - 2.0 * t ** 3


def build_focus_mask(H: int, center_y: float, width: float, feather: float) -> np.ndarray:
    """(H,) mask: 1 inside the focus band, 0 outside, smooth in between."""
    y = np.linspace(0.0, 1.0, H)
    d = np.abs(y - center_y)
    t = (d - width / 2) / max(feather, 1e-6)
    return 1.0 - smoothstep(t)


class TiltShiftGenerator(StyleGenerator):
    """Spatially-variant Gaussian blur outside a horizontal focus band."""

    def __init__(
        self,
        center_y: float = 0.5,
        width: float = 0.20,
        feather: float = 0.15,
        max_sigma: float = 8.0,
    ):
        self.center_y = center_y
        self.width = width
        self.feather = feather
        self.max_sigma = max_sigma

    def generate_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        H, W, _ = image.shape

        # 1D mask along the vertical axis, broadcast across columns and channels.
        mask_y = build_focus_mask(H, self.center_y, self.width, self.feather)
        mask = mask_y[:, None, None]  # (H, 1, 1)

        # Single Gaussian-blurred copy. sigma=(sy, sx, sc): no blur across channels.
        blurred = scipy.ndimage.gaussian_filter(
            image, sigma=(self.max_sigma, self.max_sigma, 0.0)
        )

        styled = mask * image + (1.0 - mask) * blurred
        return image, np.clip(styled, 0.0, 1.0)
