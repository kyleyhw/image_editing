"""Generic (style-agnostic) parameter regression head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .generic_renderer import parameter_count


class GenericTransformationHead(nn.Module):
    """MLP that maps the FeatureExtractor output to renderer parameters.

    The output dimension is fixed by the generic renderer's parameter layout
    (default 21 = K-2 + 14 with K = 9 control points). The final linear
    layer is initialised to zero so the renderer starts at the identity map;
    the network must learn each primitive's nonzero offset from data.

    Architecture: 1280 -> 512 (ReLU, BN, Dropout) -> 256 (ReLU, BN) -> num_params.
    Matches the existing TransformationHead's depth so that comparisons
    between the Fujifilm-specific and generic models isolate the *parameter
    space* as the experimental variable rather than the regressor capacity.
    """

    def __init__(self, input_dim: int = 1280, num_tone_points: int = 9):
        super().__init__()
        self.num_tone_points = num_tone_points
        self.num_params = parameter_count(num_tone_points)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.num_params),
        )

        # Zero-initialise the final layer so the rendered output equals the
        # input at the start of training (see DifferentiableGenericRenderer
        # docstring for the identity-at-zero property).
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
