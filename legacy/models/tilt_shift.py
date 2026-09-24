"""Differentiable tilt-shift renderer and the matching style network.

The renderer is spatially variant: a horizontal focus band stays sharp
while pixels above and below are progressively blended into a Gaussian-
blurred copy of the image. The blend is controlled by three predicted
*scalar* parameters, not by a per-pixel parameter map.

Why scalar parameters, not parameter maps?
------------------------------------------

A truly per-pixel blur control (the project plan's "parameter maps"
direction) requires the head to emit a tensor of shape (B, H, W) and
therefore a decoder architecture matching the encoder's resolution
(e.g. a U-Net). That is a substantial architectural change.

For the canonical tilt-shift effect, however, the *spatial* structure is
fixed: blur grows monotonically with distance from a horizontal focus
band. Only three numbers (band center, band half-width, peak blur)
determine the entire spatial profile, and the renderer derives the
per-pixel blur amount from those numbers. This factorisation gives us
genuine spatial variance in the *output* while keeping the parameter
regression head identical in shape to the generic case.

Composition with the generic renderer
-------------------------------------

`DifferentiableTiltShiftComposite` chains
`DifferentiableGenericRenderer` (21 generic parameters) with a
tilt-shift stage (3 extra parameters). The full parameter vector is
[generic | tilt_shift] of length 24.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from .generic_renderer import DifferentiableGenericRenderer, parameter_count
from .generic_head import GenericTransformationHead


def _gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    """Return a normalised 1-D Gaussian kernel covering 3 sigma each side."""
    half = int(round(3.0 * sigma))
    ksize = 2 * half + 1
    x = torch.arange(ksize, dtype=torch.float32) - half
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """3t^2 - 2t^3 on a clamped input - C^1 continuous and bounded in [0, 1]."""
    t = t.clamp(0.0, 1.0)
    return 3.0 * t ** 2 - 2.0 * t ** 3


class DifferentiableTiltShift(nn.Module):
    """Spatially-variant Gaussian blur controlled by 3 scalar parameters.

    Parameters
    ----------
    max_sigma : float
        Standard deviation of the pre-baked Gaussian blur kernel. The
        learned `blur_strength` parameter linearly interpolates between
        the sharp original and this maximally-blurred version; sharper
        levels of blur are *not* representable, only weaker (matching the
        numpy generator).

    Input parameter layout (shape (B, 3)):
        [:, 0] = center_y       in [0, 1] (vertical focus position)
        [:, 1] = width          in [0, 1] (full-focus half-band)
        [:, 2] = blur_strength  in [0, 1] (overall blur amount)
    """

    DEFAULT_FEATHER = 0.15

    def __init__(self, max_sigma: float = 8.0):
        super().__init__()
        self.max_sigma = float(max_sigma)
        # 1-D separable Gaussian (registered as a buffer so it travels to GPU).
        kernel = _gaussian_kernel_1d(max_sigma)
        self.register_buffer("kernel_1d", kernel)

    def _blur(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        k = self.kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)  # depthwise
        pad_w = k.shape[-1] // 2
        # Horizontal pass.
        out = F.conv2d(image, k, padding=(0, pad_w), groups=C)
        # Vertical pass.
        out = F.conv2d(out, k.transpose(2, 3), padding=(pad_w, 0), groups=C)
        return out

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        center_y = params[:, 0:1].clamp(0.0, 1.0)
        width = params[:, 1:2].clamp(0.0, 1.0)
        strength = params[:, 2:3].clamp(0.0, 1.0)

        # 1-D vertical mask (B, 1, H, 1).
        y = torch.linspace(0.0, 1.0, H, device=image.device).view(1, 1, H, 1)
        d = (y - center_y.view(B, 1, 1, 1)).abs()
        t = (d - width.view(B, 1, 1, 1) / 2.0) / self.DEFAULT_FEATHER
        focus_mask = 1.0 - smoothstep(t)  # 1 in focus, 0 fully blurred

        blurred = self._blur(image)
        alpha = (1.0 - focus_mask) * strength.view(B, 1, 1, 1)
        return torch.clamp(image * (1.0 - alpha) + blurred * alpha, 0.0, 1.0)


class DifferentiableTiltShiftComposite(nn.Module):
    """Generic colour pipeline followed by tilt-shift spatial blur.

    Total parameter vector length = parameter_count(num_tone_points) + 3.
    """

    def __init__(self, num_tone_points: int = 9, max_sigma: float = 8.0):
        super().__init__()
        self.num_tone_points = num_tone_points
        self.generic = DifferentiableGenericRenderer(num_tone_points=num_tone_points)
        self.tilt_shift = DifferentiableTiltShift(max_sigma=max_sigma)
        self.max_sigma = max_sigma

    @property
    def num_params(self) -> int:
        return self.generic.num_params + 3

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        n = self.generic.num_params
        x = self.generic(image, params[:, :n])
        x = self.tilt_shift(x, params[:, n : n + 3])
        return x


class TiltShiftStyleNet(nn.Module):
    """End-to-end network for the generic+tilt-shift composite.

    Reuses the shared FeatureExtractor; the head emits
    parameter_count(num_tone_points) + 3 values.

    The final layer's *weight* is zeroed (identity-at-init for the generic
    primitives) but its *bias* is initialised to put the three tilt-shift
    outputs at a sensible centred focus band: (center=0.5, width=0.2,
    strength=0.5). Without this warm start the rendered focus band lives
    at the image corner at init (center=0, width=0), the gradient w.r.t.
    `strength` is dominated by an inconsistent signal across regions, and
    optimisation gets stuck at strength->0.
    """

    TILT_SHIFT_BIAS_INIT = (0.5, 0.2, 0.5)

    def __init__(self, num_tone_points: int = 9):
        super().__init__()
        self.encoder = FeatureExtractor()
        extra = 3
        total = parameter_count(num_tone_points) + extra
        self.head = _TiltShiftHead(
            input_dim=1280,
            num_outputs=total,
            tail_bias_init=self.TILT_SHIFT_BIAS_INIT,
        )
        self.num_tone_points = num_tone_points

    @property
    def num_params(self) -> int:
        return self.head.num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class _TiltShiftHead(nn.Module):
    """Same shape as GenericTransformationHead but with a configurable output width.

    `tail_bias_init` (if provided) is applied to the last len(tail_bias_init)
    output dimensions of the final layer's bias. The remaining outputs and
    all weights are zero-initialised.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_outputs: int = 24,
        tail_bias_init: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_outputs),
        )
        final = self.net[-1]
        nn.init.constant_(final.weight, 0.0)
        nn.init.constant_(final.bias, 0.0)
        if tail_bias_init is not None:
            n = len(tail_bias_init)
            with torch.no_grad():
                final.bias[-n:] = torch.tensor(tail_bias_init, dtype=final.bias.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
