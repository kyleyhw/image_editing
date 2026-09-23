"""Image-fidelity metrics against a target edit.

All inputs are float arrays (H, W, 3) in sRGB [0, 1].

* ``psnr``   - dB, higher is better (the FiveK literature's headline metric).
* ``delta_e`` - mean CIEDE2000 colour difference, lower is better; ~2.3 is a
                just-noticeable difference.
* ``ssim``   - structural similarity, higher is better.
* ``l1``     - mean absolute error in sRGB, lower is better.
"""

from __future__ import annotations

import numpy as np
import skimage as ski


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(ski.color.deltaE_ciede2000(ski.color.rgb2lab(pred), ski.color.rgb2lab(target))))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ski.metrics.structural_similarity(pred, target, channel_axis=2, data_range=1.0))


def l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(pred, target), "delta_e": delta_e(pred, target),
            "ssim": ssim(pred, target), "l1": l1(pred, target)}
