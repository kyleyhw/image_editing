"""Generate the MIT-5K held-out evaluation figure.

For each held-out test pair, show three panels:
  1. Original (network input)
  2. Expert C retouch (ground truth)
  3. Trained model's prediction

The figure visualises whether the model moves the input pixels toward
the expert's edit on inputs it never saw during training.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import torch
import torchvision.transforms as transforms

from models.checkpoint_io import build_model_from_checkpoint, load_checkpoint


OUT = "tests/reports/assets/mit5k_eval.png"
CKPT = "checkpoints/model_generic_mit5k_c.pth"
TEST_DIR = "data/fivek_c_test"


def predict(model, renderer, img: np.ndarray) -> np.ndarray:
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
    ckpt = load_checkpoint(CKPT)
    model, renderer = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    renderer.eval()

    files = sorted(os.listdir(os.path.join(TEST_DIR, "original")))
    n = len(files)

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    column_titles = ["Original (input)", "Expert C (ground truth)", "Trained model"]

    for r, name in enumerate(files):
        inp_img = ski.io.imread(os.path.join(TEST_DIR, "original", name))
        exp_img = ski.io.imread(os.path.join(TEST_DIR, "expert_c", name))
        src = inp_img.astype(np.float32) / 255.0
        tgt = exp_img.astype(np.float32) / 255.0
        pred = predict(model, renderer, inp_img)

        # Sizes can differ by a few px between original and augmented webps.
        H = min(src.shape[0], tgt.shape[0], pred.shape[0])
        W = min(src.shape[1], tgt.shape[1], pred.shape[1])
        src, tgt, pred = src[:H, :W], tgt[:H, :W], pred[:H, :W]

        L_in_exp = float(np.abs(src - tgt).mean())
        L_pred_exp = float(np.abs(pred - tgt).mean())
        delta = L_in_exp - L_pred_exp
        direction = "toward expert" if delta > 0 else "away from expert"

        for c, panel in enumerate([src, tgt, pred]):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(column_titles[c], fontsize=12)

        axes[r, 0].set_ylabel(
            f"{name}\nL1(in, exp)={L_in_exp:.3f}\nL1(pred, exp)={L_pred_exp:.3f}\n({direction})",
            fontsize=9,
            rotation=0,
            labelpad=70,
            ha="right",
            va="center",
        )

    fig.suptitle(
        "MIT-5K expert C held-out evaluation (5 unseen images)",
        fontsize=14,
        y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
