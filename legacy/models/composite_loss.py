"""
Composite loss for style-imitation training.

    L  =  lambda_pixel * L1_pixel
       +  lambda_perceptual * L1_VGG
       +  lambda_cdf * L1_CDF

Each component answers a different question about the rendered image:

  - L1_pixel    : "Are the pixel values numerically close?"
                  Direct, low-frequency, but blind to perceptual structure.

  - L1_VGG      : "Are the rendered and target images structurally similar
                  at multiple scales?" L1 distance between activations of
                  a frozen ImageNet-pre-trained VGG-16 at four common
                  feature taps (relu1_2, relu2_2, relu3_3, relu4_3). The
                  network "perceives" colour, texture, and gross structure;
                  matching VGG features penalises perceptually salient
                  errors that L1 ignores.

  - L1_CDF      : "Do the rendered and target images agree on the global
                  intensity distribution per channel?" Uses the existing
                  DifferentiableCDF (Gaussian-kernel soft binning) so the
                  loss is end-to-end differentiable. This is the term that
                  directly couples training to the project's foundational
                  CDF analysis.

Default weights are tuned so each component lands in a similar order of
magnitude at the start of training; they can be overridden per run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .feature_extractor import DifferentiableCDF


class _VGGFeatures(nn.Module):
    """Frozen VGG-16 feature taps at four canonical layers.

    Layers (slice indices into vgg16.features):
        relu1_2 -> indices 0:4
        relu2_2 -> indices 4:9
        relu3_3 -> indices 9:16
        relu4_3 -> indices 16:23

    Inputs are renormalised from [0, 1] to the ImageNet mean/std the
    pre-trained weights expect.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT
        except AttributeError:  # very old torchvision
            weights = None
        vgg = models.vgg16(weights=weights).features
        # Slice into the four standard perceptual taps.
        self.slices = nn.ModuleList(
            [vgg[0:4], vgg[4:9], vgg[9:16], vgg[16:23]]
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer(
            "mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        feats: list[torch.Tensor] = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


class CompositeLoss(nn.Module):
    """L = lambda_pixel * L1 + lambda_perceptual * L_VGG + lambda_cdf * L_CDF."""

    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 0.05,
        lambda_cdf: float = 1.0,
        cdf_bins: int = 256,
        use_vgg: bool = True,
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_cdf = lambda_cdf
        self.use_vgg = use_vgg and lambda_perceptual > 0.0

        self.cdf = DifferentiableCDF(bins=cdf_bins)
        self.vgg: _VGGFeatures | None = _VGGFeatures() if self.use_vgg else None

    def _pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    def _vgg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.zeros((), device=pred.device)
        feats_pred = self.vgg(pred)
        feats_tgt = self.vgg(target)
        return sum(F.l1_loss(p, t) for p, t in zip(feats_pred, feats_tgt)) / len(feats_pred)

    def _cdf_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cdf_pred = self.cdf(pred)
        cdf_tgt = self.cdf(target)
        return F.l1_loss(cdf_pred, cdf_tgt)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pixel = self._pixel_loss(pred, target)
        perceptual = self._vgg_loss(pred, target)
        cdf = self._cdf_loss(pred, target)
        total = (
            self.lambda_pixel * pixel
            + self.lambda_perceptual * perceptual
            + self.lambda_cdf * cdf
        )
        return total, {
            "pixel": float(pixel.detach().item()),
            "perceptual": float(perceptual.detach().item()),
            "cdf": float(cdf.detach().item()),
            "total": float(total.detach().item()),
        }
