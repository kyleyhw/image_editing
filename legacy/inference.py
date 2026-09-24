"""Inference CLI.

Accepts a checkpoint produced by either the legacy Phase 1/2 trainer
(raw state dict, fujifilm arch) or the Phase 3 trainer (wrapped dict with
metadata, fujifilm or generic arch). The checkpoint format is auto-detected
via `models.checkpoint_io.load_checkpoint`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import skimage as ski
import torch
import torchvision.transforms as transforms

from models.checkpoint_io import build_model_from_checkpoint, load_checkpoint


FUJIFILM_PARAM_NAMES = [
    "Highlight",
    "Shadow",
    "Saturation",
    "WB Red",
    "WB Blue",
    "Grain",
    "Vignette",
]


def load_image(path: str) -> np.ndarray:
    image = ski.io.imread(path)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(image)
    return image


def _resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    # Backwards-compatible default: model_<style>_<recipe>.pth (Phase 1/2).
    return os.path.join("checkpoints", f"model_{args.style}_{args.recipe}.pth")


def _print_params(arch: str, params: torch.Tensor) -> None:
    vals = params[0].detach().cpu().numpy()
    if arch == "fujifilm":
        print("Predicted Parameters (Fujifilm):")
        for name, val in zip(FUJIFILM_PARAM_NAMES, vals):
            print(f"  {name:11s}: {val:+.4f}")
    else:
        print(f"Predicted Parameters (generic, {len(vals)}-D):")
        print("  ", np.array2string(vals, precision=3, suppress_small=True))


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    ckpt_path = _resolve_checkpoint(args)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model, renderer = build_model_from_checkpoint(ckpt)
    model = model.to(device)
    renderer = renderer.to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    renderer.eval()
    print(f"Loaded checkpoint: {ckpt_path}  arch={ckpt.get('arch')}")

    image = load_image(args.image_path)

    # Downscale for parameter prediction (ResNet-18 backbone is global-pooled
    # so it accepts any size, but a fixed eval size keeps the prediction
    # consistent across input resolutions and is faster).
    transform_input = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.eval_size, args.eval_size)),
            transforms.ToTensor(),
        ]
    )
    transform_full = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]
    )
    input_tensor = transform_input(image).unsqueeze(0).to(device)
    full_tensor = transform_full(image).unsqueeze(0).to(device)

    with torch.no_grad():
        params = model(input_tensor)
        _print_params(ckpt.get("arch", "fujifilm"), params)
        styled_tensor = renderer(full_tensor, params)

    styled_image = styled_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    styled_image = ski.util.img_as_ubyte(np.clip(styled_image, 0, 1))

    output_path = args.output_path
    if not output_path:
        base, ext = os.path.splitext(args.image_path)
        tag = ckpt.get("arch", "styled")
        if ckpt.get("recipe"):
            tag = f"{tag}_{ckpt['recipe']}"
        elif ckpt.get("style"):
            tag = f"{tag}_{ckpt['style']}"
        output_path = f"{base}_{tag}{ext or '.jpg'}"

    ski.io.imsave(output_path, styled_image)
    print(f"Saved styled image to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path. If omitted, derived from --style/--recipe.",
    )
    parser.add_argument("--style", type=str, default="fujifilm")
    parser.add_argument("--recipe", type=str, default="classic_chrome")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--eval_size", type=int, default=256)
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
