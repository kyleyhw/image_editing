"""Regenerate the style comparison figure used in README.md.

For each trained style (Phase 3+ models) the figure shows three panels:
  1. Original photograph.
  2. Data generator's ground truth (what training targets look like).
  3. The trained model's prediction (what the network actually produces).

Run from the project root:

    python tools/make_examples.py

Writes the PNG to ``tests/reports/assets/style_comparison.png``.
"""

from __future__ import annotations

import os
import sys

# Make the project root importable regardless of cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import torch
import torchvision.transforms as transforms

from data_generation.styles.cyberpunk import CyberpunkGenerator
from data_generation.styles.fujifilm import FujifilmGenerator
from data_generation.styles.tilt_shift import TiltShiftGenerator
from models.checkpoint_io import build_model_from_checkpoint, load_checkpoint


TEST_IMAGE = "images/test_images/climbing_test_original.jpeg"
OUT = "tests/reports/assets/style_comparison.png"


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def predict(ckpt_path: str, img: np.ndarray) -> np.ndarray:
    ckpt = load_checkpoint(ckpt_path)
    model, renderer = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    renderer.eval()
    tr_in = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    tr_full = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    inp = tr_in(img).unsqueeze(0)
    full = tr_full(img).unsqueeze(0)
    with torch.no_grad():
        params = model(inp)
        out = renderer(full, params)
    return out.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)


def main():
    img = ski.io.imread(TEST_IMAGE)
    if img.shape[-1] == 4:
        img = ski.color.rgba2rgb(img)
    src = to_float(img)

    styles = [
        (
            "Fujifilm\nclassic_chrome",
            FujifilmGenerator(recipe_name="classic_chrome"),
            "checkpoints/model_generic_fujifilm_classic_chrome.pth",
        ),
        (
            "Cyberpunk",
            CyberpunkGenerator(),
            "checkpoints/model_generic_cyberpunk.pth",
        ),
        (
            "Tilt-shift",
            TiltShiftGenerator(),
            "checkpoints/model_tilt_shift_tilt_shift.pth",
        ),
    ]

    fig, axes = plt.subplots(len(styles), 3, figsize=(12, 4 * len(styles)))
    for row, (name, gen, ckpt_path) in enumerate(styles):
        np.random.seed(0)
        _, target = gen.generate_pair(img.copy())

        if os.path.exists(ckpt_path):
            pred = predict(ckpt_path, img)
        else:
            pred = np.zeros_like(target)

        for col, (panel_img, title) in enumerate(
            [
                (src, "Original"),
                (target, "Data generator"),
                (pred, "Trained model"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(panel_img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=12)
        axes[row, 0].set_ylabel(name, fontsize=11, rotation=90, labelpad=10)

    fig.suptitle(
        "Style comparison: original vs data-generator target vs trained model",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
