"""Generic style network: FeatureExtractor + GenericTransformationHead."""

from __future__ import annotations

import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .generic_head import GenericTransformationHead


class GenericStyleNet(nn.Module):
    """End-to-end network: image -> 21-D generic-renderer parameter vector.

    The encoder is shared with the Fujifilm-specific StyleNet, so the only
    architectural change between the two networks is the head's output
    dimension. This isolates the effect of the parameter space (Fujifilm-
    specific vs. style-agnostic primitives) for Phase 3 comparison.
    """

    def __init__(self, num_tone_points: int = 9):
        super().__init__()
        self.encoder = FeatureExtractor()
        # Encoder produces 1280-D features = 3 channels * 256 CDF bins (768)
        # concatenated with 512-D ResNet-18 spatial features.
        self.head = GenericTransformationHead(
            input_dim=1280, num_tone_points=num_tone_points
        )

    @property
    def num_params(self) -> int:
        return self.head.num_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
