"""Training entry point.

Supports two regimes:

  --arch fujifilm   (default for backwards compatibility)
        StyleNet -> 7-D Fujifilm-specific params -> DifferentiableFujifilm
        Loss: MSE (legacy Phase 1/2 setup).

  --arch generic
        GenericStyleNet -> 21-D primitive params -> DifferentiableGenericRenderer
        Loss: CompositeLoss = L1 + lambda * VGG + lambda * CDF
        This is the Phase 3 generalized pipeline; the same model + renderer
        can be retrained on any style (Fujifilm, cyberpunk, tilt-shift, ...).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import skimage as ski
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from data_generation.mit5k_loader import MIT5KDataset
from data_generation.styles.fujifilm import FujifilmGenerator
from models.composite_loss import CompositeLoss
from models.differentiable_renderer import DifferentiableFujifilm
from models.generic_renderer import DifferentiableGenericRenderer
from models.generic_style_net import GenericStyleNet
from models.style_net import StyleNet
from models.tilt_shift import DifferentiableTiltShiftComposite, TiltShiftStyleNet


class ImagePairDataset(Dataset):
    """Loads `(original, styled)` JPEG pairs produced by `generate_dataset.py`.

    Directory layout (matches `generate_dataset.py` output):

        <root_dir>/<style>/[<recipe>/]<basename>_original.jpg
        <root_dir>/<style>/[<recipe>/]<basename>_styled.jpg
    """

    def __init__(
        self,
        root_dir: str,
        style: str = "fujifilm",
        recipe: str | None = "classic_chrome",
        transform=None,
    ):
        sub = os.path.join(root_dir, style)
        if recipe:
            sub = os.path.join(sub, recipe)
        self.root_dir = sub
        self.transform = transform
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"dataset directory not found: {self.root_dir}")
        self.files = sorted(
            f for f in os.listdir(self.root_dir) if f.endswith("_original.jpg")
        )
        if not self.files:
            raise RuntimeError(f"no *_original.jpg files in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        original_name = self.files[idx]
        styled_name = original_name.replace("_original.jpg", "_styled.jpg")
        original = ski.io.imread(os.path.join(self.root_dir, original_name))
        styled = ski.io.imread(os.path.join(self.root_dir, styled_name))
        if self.transform:
            original = self.transform(original)
            styled = self.transform(styled)
        return original, styled


def _build_dataloader(args, transform) -> DataLoader:
    if args.mit5k_root:
        dataset = MIT5KDataset(
            root=args.mit5k_root, expert=args.mit5k_expert, transform=transform
        )
    else:
        # Phase 3 cyberpunk/tilt-shift use a flat <style>/ directory;
        # Fujifilm uses <style>/<recipe>/. Pass recipe=None for the flat case.
        recipe = args.recipe if args.style in {"fujifilm"} else None
        dataset = ImagePairDataset(
            root_dir=args.data_dir, style=args.style, recipe=recipe, transform=transform
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


def _build_model_and_renderer(args, device):
    """Return (model, renderer, criterion, criterion_returns_components)."""
    if args.arch == "fujifilm":
        model = StyleNet().to(device)
        chrome_strength = FujifilmGenerator(recipe_name=args.recipe).chrome_strength
        renderer = DifferentiableFujifilm(chrome_strength=chrome_strength).to(device)
        criterion = nn.MSELoss()
        return model, renderer, criterion, False

    if args.arch == "generic":
        model = GenericStyleNet(num_tone_points=args.tone_points).to(device)
        renderer = DifferentiableGenericRenderer(num_tone_points=args.tone_points).to(device)
        criterion = CompositeLoss(
            lambda_pixel=args.lambda_pixel,
            lambda_perceptual=args.lambda_perceptual,
            lambda_cdf=args.lambda_cdf,
        ).to(device)
        return model, renderer, criterion, True

    if args.arch == "tilt_shift":
        model = TiltShiftStyleNet(num_tone_points=args.tone_points).to(device)
        renderer = DifferentiableTiltShiftComposite(
            num_tone_points=args.tone_points, max_sigma=args.tilt_max_sigma
        ).to(device)
        criterion = CompositeLoss(
            lambda_pixel=args.lambda_pixel,
            lambda_perceptual=args.lambda_perceptual,
            lambda_cdf=args.lambda_cdf,
        ).to(device)
        return model, renderer, criterion, True

    raise ValueError(f"unknown --arch: {args.arch}")


def _ckpt_path(args) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    if args.checkpoint:
        return args.checkpoint
    arch_tag = args.arch
    if args.mit5k_root:
        return os.path.join("checkpoints", f"model_{arch_tag}_mit5k_{args.mit5k_expert}.pth")
    recipe_tag = f"_{args.recipe}" if args.style == "fujifilm" else ""
    return os.path.join("checkpoints", f"model_{arch_tag}_{args.style}{recipe_tag}.pth")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataloader = _build_dataloader(args, transform)
    print(f"Dataset: {len(dataloader.dataset)} pairs, {len(dataloader)} batches/epoch")

    model, renderer, criterion, returns_components = _build_model_and_renderer(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        component_sums: dict[str, float] = defaultdict(float)
        n_batches = 0
        for original, target in dataloader:
            original = original.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            params = model(original)
            rendered = renderer(original, params)

            if returns_components:
                loss, parts = criterion(rendered, target)
                for k, v in parts.items():
                    component_sums[k] += v
            else:
                loss = criterion(rendered, target)
                component_sums["total"] += loss.item()
            loss.backward()
            optimizer.step()
            n_batches += 1

        epoch_log = {k: v / max(n_batches, 1) for k, v in component_sums.items()}
        history.append(epoch_log)
        log_str = "  ".join(f"{k}={v:.4f}" for k, v in epoch_log.items())
        print(f"Epoch {epoch+1}/{args.epochs}  {log_str}")

    ckpt = _ckpt_path(args)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "num_tone_points": getattr(args, "tone_points", None),
            "tilt_max_sigma": getattr(args, "tilt_max_sigma", None),
            "style": args.style,
            "recipe": args.recipe,
            "history": history,
        },
        ckpt,
    )
    print(f"Model saved to {ckpt}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["fujifilm", "generic", "tilt_shift"], default="fujifilm",
                        help="Which model + renderer pair to train.")
    parser.add_argument("--data_dir", type=str, default="images/styled")
    parser.add_argument("--style", type=str, default="fujifilm")
    parser.add_argument("--recipe", type=str, default="classic_chrome",
                        help="Sub-folder of <data_dir>/<style>/ when --style fujifilm.")
    parser.add_argument("--mit5k_root", type=str, default=None,
                        help="Path to a MIT-Adobe FiveK root (must contain "
                             "original/ and expert_<letter>/). When set, this "
                             "overrides --style/--recipe/--data_dir.")
    parser.add_argument("--mit5k_expert", type=str, default="c",
                        help="Which MIT-5K expert (a-e) to train against.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override the default save path.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    # Generic-arch hyperparameters.
    parser.add_argument("--tone_points", type=int, default=9)
    parser.add_argument("--tilt_max_sigma", type=float, default=8.0,
                        help="Pre-baked blur sigma for the tilt_shift renderer.")
    parser.add_argument("--lambda_pixel", type=float, default=1.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.05)
    parser.add_argument("--lambda_cdf", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
