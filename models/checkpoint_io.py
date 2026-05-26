"""Checkpoint format helpers.

The Phase 3 training script saves checkpoints as a dict:

    {"state_dict": ..., "arch": "fujifilm" | "generic", "num_tone_points": ...,
     "style": ..., "recipe": ..., "history": [...]}

The Phase 1/2 training script saved checkpoints as the bare state dict.
`load_checkpoint` accepts either and returns a normalized dict.
"""

from __future__ import annotations

import os
from typing import Any

import torch


LEGACY_DEFAULT_ARCH = "fujifilm"


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint and normalize to the wrapped dict format.

    Detection rule: if the loaded object is a Python dict that contains a
    `state_dict` key, treat it as already-wrapped; otherwise treat it as a
    legacy raw state dict and fill in plausible defaults for the metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    raw = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw

    return {
        "state_dict": raw,
        "arch": LEGACY_DEFAULT_ARCH,
        "num_tone_points": None,
        "style": None,
        "recipe": None,
        "history": None,
    }


def build_model_from_checkpoint(ckpt: dict[str, Any]):
    """Instantiate a model + renderer matching the checkpoint's arch.

    Returns (model, renderer). The caller is responsible for `.to(device)`
    and `model.load_state_dict(ckpt['state_dict'])`.
    """
    # Imports here to avoid circulars with model modules importing torchvision
    # at module-load time.
    from .style_net import StyleNet
    from .generic_style_net import GenericStyleNet
    from .differentiable_renderer import DifferentiableFujifilm
    from .generic_renderer import DifferentiableGenericRenderer
    from data_generation.styles.fujifilm import FujifilmGenerator

    arch = ckpt.get("arch", LEGACY_DEFAULT_ARCH)
    if arch == "fujifilm":
        model = StyleNet()
        recipe = ckpt.get("recipe") or "classic_chrome"
        chrome_strength = FujifilmGenerator(recipe_name=recipe).chrome_strength
        renderer = DifferentiableFujifilm(chrome_strength=chrome_strength)
        return model, renderer
    if arch == "generic":
        K = ckpt.get("num_tone_points") or 9
        model = GenericStyleNet(num_tone_points=K)
        renderer = DifferentiableGenericRenderer(num_tone_points=K)
        return model, renderer
    raise ValueError(f"unknown checkpoint arch: {arch!r}")
