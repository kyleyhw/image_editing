"""
Style-agnostic differentiable renderer for Phase 3 of the project.

The renderer composes four generic primitives, each chosen to be a faithful,
differentiable counterpart of an operation common across photographic styles:

  1. ToneCurve     - Piecewise-linear 1D function applied per pixel value.
                     Models the global tonal response of a film stock,
                     digital curve, or display LUT.

  2. ColorMatrix   - Affine map  c_out = M @ c_in + b  on the RGB channel.
                     Models white-balance shifts, saturation, hue rotation,
                     and arbitrary linear color grading.

  3. Grain         - Additive zero-mean Gaussian noise, intensity-scaled.

  4. Vignette      - Multiplicative radial mask  m = 1 - s * (d / d_max)^2.

Total predicted parameter count
-------------------------------

    tone_deltas    : K - 2 (interior knot offsets; endpoints anchored)
    color_matrix_d : 9     (offset from identity)
    color_bias     : 3
    grain          : 1
    vignette       : 1
    --------------------
    total          : K - 2 + 14

With the default K = 9 this is 7 + 14 = 21 parameters.

Identity at init
----------------

Calling sites must initialise the head's final linear layer to zero
(`nn.init.constant_(..., 0)`). The renderer is then the identity map at
the start of training:

    delta = 0        =>  y_k = x_k                (tone curve = identity)
    M_d   = 0, b = 0 =>  c_out = c_in             (no color shift)
    grain = 0        =>  no noise added
    vign  = 0        =>  mask is unity everywhere

This neutral start lets the loss gradients drive each primitive away from
identity in whichever direction reduces the reconstruction error.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class ToneCurve(nn.Module):
    """Piecewise-linear tone curve through K control points.

    Knot positions are fixed on a uniform grid x_k = k/(K-1), k = 0..K-1, with
    endpoints anchored at y_0 = 0 and y_{K-1} = 1. The network supplies the
    K-2 interior y-values as offsets from identity (so that zero output is
    the identity curve).

    For an input pixel value x in [0, 1]:
        i = floor(x * (K - 1))         (clamped to [0, K-2])
        u = x * (K - 1) - i
        f(x) = y_i + u * (y_{i+1} - y_i)

    Differentiable: df/dy_k is piecewise-linear in x with finite support;
    df/dx is the segment slope (y_{i+1} - y_i) * (K - 1), which autograd
    derives from the explicit formula.

    Shapes:
        image : (B, C, H, W) in [0, 1]
        delta : (B, K - 2)
        out   : (B, C, H, W) in [0, 1]
    """

    def __init__(self, num_points: int = 9):
        super().__init__()
        if num_points < 3:
            raise ValueError("num_points must be >= 3 (need at least one interior knot)")
        self.K = num_points
        # Anchored identity y-values; the network only edits y[1 : K-1].
        identity = torch.linspace(0.0, 1.0, num_points)
        self.register_buffer("identity", identity)

    def forward(self, image: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        K = self.K

        # Build per-batch y vector with endpoints anchored to 0 and 1.
        y = self.identity.unsqueeze(0).expand(B, -1).clone()  # (B, K)
        y[:, 1 : K - 1] = y[:, 1 : K - 1] + delta             # apply interior offsets

        # Flatten image to (B, N) so each pixel is gathered against y.
        flat = image.reshape(B, -1)                           # (B, N)
        scaled = flat.clamp(0.0, 1.0) * (K - 1)               # (B, N) in [0, K-1]
        idx_lo = torch.floor(scaled).long().clamp(0, K - 2)   # (B, N)
        u = scaled - idx_lo.float()                           # (B, N) in [0, 1]

        y_lo = torch.gather(y, 1, idx_lo)                     # (B, N)
        y_hi = torch.gather(y, 1, idx_lo + 1)                 # (B, N)

        out = y_lo + u * (y_hi - y_lo)                        # (B, N)
        return out.reshape_as(image).clamp(0.0, 1.0)


class ColorMatrix(nn.Module):
    """Affine map on RGB:  c_out = (I + dM) @ c_in + b.

    The network supplies the 9 dM elements (offset from identity) and the
    3 bias components. At dM = 0, b = 0 the layer is the identity map.

    Shapes:
        image : (B, 3, H, W)
        dM    : (B, 9)
        bias  : (B, 3)
        out   : (B, 3, H, W)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(3))

    def forward(self, image: torch.Tensor, dM: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        if C != 3:
            raise ValueError(f"ColorMatrix expects 3 channels, got C={C}")
        # Reshape (B, 3, H, W) -> (B, 3, H*W) for batched matrix multiply.
        x = image.reshape(B, 3, -1)
        M = self.eye + dM.reshape(B, 3, 3)                    # (B, 3, 3)
        out = torch.bmm(M, x) + bias.reshape(B, 3, 1)         # (B, 3, H*W)
        return out.reshape(B, 3, H, W).clamp(0.0, 1.0)


class Grain(nn.Module):
    """Additive zero-mean Gaussian noise, intensity-scaled.

    out = clamp(image + epsilon * intensity), epsilon ~ N(0, 1).

    Applied unconditionally so that train and inference share the same
    statistical signature as the data generator.
    """

    def forward(self, image: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image)
        return torch.clamp(image + noise * intensity.view(-1, 1, 1, 1), 0.0, 1.0)


class Vignette(nn.Module):
    """Radial multiplicative darkening.

    mask = clamp(1 - strength * (d / d_max)^2)
    d    = sqrt(X^2 + Y^2), with (X, Y) ranging over [-1, 1].
    d_max = sqrt(2) (corner of the normalized square).
    """

    def forward(self, image: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        Y, X = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=image.device),
            torch.linspace(-1.0, 1.0, W, device=image.device),
            indexing="ij",
        )
        dist = torch.sqrt(X ** 2 + Y ** 2) / 1.4142135623730951
        mask = 1.0 - strength.view(-1, 1, 1, 1) * (dist ** 2)
        return image * mask.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


def parameter_count(num_tone_points: int = 9) -> int:
    """Total length of the parameter vector consumed by the renderer."""
    tone = num_tone_points - 2
    return tone + 9 + 3 + 1 + 1


def unpack_params(params: torch.Tensor, num_tone_points: int = 9) -> dict[str, torch.Tensor]:
    """Slice a flat parameter tensor into the per-primitive named tensors.

    Layout:
        [ tone_deltas (K-2) | dM (9) | bias (3) | grain (1) | vignette (1) ]
    """
    tone = num_tone_points - 2
    return {
        "tone_delta": params[:, 0:tone],
        "dM": params[:, tone : tone + 9],
        "bias": params[:, tone + 9 : tone + 12],
        "grain": params[:, tone + 12 : tone + 13],
        "vignette": params[:, tone + 13 : tone + 14],
    }


# ---------------------------------------------------------------------------
# Composed renderer
# ---------------------------------------------------------------------------


class DifferentiableGenericRenderer(nn.Module):
    """Style-agnostic renderer composing the four generic primitives.

    Pipeline (applied in order):

        x' = ToneCurve(x;  delta)
        x'' = ColorMatrix(x'; dM, b)
        x''' = Grain(x''; intensity)
        out  = Vignette(x'''; strength)

    The ordering mirrors a conventional photographic pipeline: tonal shaping
    -> color grading -> noise/texture -> optical falloff.
    """

    def __init__(self, num_tone_points: int = 9):
        super().__init__()
        self.num_tone_points = num_tone_points
        self.tone = ToneCurve(num_points=num_tone_points)
        self.color = ColorMatrix()
        self.grain = Grain()
        self.vignette = Vignette()

    @property
    def num_params(self) -> int:
        return parameter_count(self.num_tone_points)

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        p = unpack_params(params, num_tone_points=self.num_tone_points)
        x = self.tone(image, p["tone_delta"])
        x = self.color(x, p["dM"], p["bias"])
        x = self.grain(x, p["grain"])
        x = self.vignette(x, p["vignette"])
        return x
