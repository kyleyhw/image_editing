"""Editable global renderers (identity at zero parameters).

Two variants, so the Phase 7 learning curve can compare them directly:

``shared``       the Phase 3 renderer minus grain: one tone curve shared by all
                 three channels (K-2 interior knots, endpoints pinned), a 3x3
                 colour matrix + bias, and a vignette.
``per_channel``  the Phase 9 upgrade (brought forward by amendment A2 because
                 split toning needs it): an independent **monotone** curve
                 per channel with a free black point, plus the same matrix,
                 bias and vignette.

Grain is left out: it is a random texture that cannot be learned from a
single before/after pair, and it only adds noise to the loss. It returns
later as a user slider.

Monotone curve parameterisation (per channel, K knots on x_k = k/(K-1)):

    y_0 = b                    (black point; b < 0 crushes, b > 0 lifts)
    y_k = y_0 + sum_{j<=k} exp(r_j) / (K-1)

At b = 0, r = 0 this is exactly the identity. Every choice of (b, r) gives a
non-decreasing curve, so the renderer can never produce tone reversals. The
output is clamped to [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _interp(img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear lookup. img: (B, C, H, W) in [0,1]; y: (B, C, K) knot values."""
    B, C, H, W = img.shape
    K = y.shape[-1]
    t = img.clamp(0, 1).reshape(B, C, -1) * (K - 1)
    lo = t.floor().long().clamp(0, K - 2)
    u = t - lo
    y_lo = torch.gather(y, 2, lo)
    y_hi = torch.gather(y, 2, lo + 1)
    return (y_lo + u * (y_hi - y_lo)).reshape(B, C, H, W)


def _vignette(img: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    B, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    r2 = (xx**2 + yy**2) / 2.0
    return img * (1 - strength.view(B, 1, 1, 1) * r2).clamp(0, 1)


def _matrix(img: torch.Tensor, dM: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    B, C, H, W = img.shape
    M = torch.eye(3).expand(B, 3, 3) + dM.view(B, 3, 3)
    return (torch.bmm(M, img.reshape(B, 3, -1)) + bias.view(B, 3, 1)).reshape(B, C, H, W)


class GlobalRenderer(nn.Module):
    def __init__(self, kind: str = "per_channel", knots: int = 9):
        super().__init__()
        if kind not in ("shared", "per_channel"):
            raise ValueError(kind)
        self.kind, self.K = kind, knots
        curve = (knots - 2) if kind == "shared" else 3 * knots
        self.layout = {"curve": curve, "dM": 9, "bias": 3, "vignette": 1}
        self.num_params = sum(self.layout.values())

    def unpack(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for k, n in self.layout.items():
            out[k] = p[:, i:i + n]
            i += n
        return out

    def curves(self, curve_params: torch.Tensor) -> torch.Tensor:
        """Knot values (B, 3, K) for the given curve parameters."""
        B, K = curve_params.shape[0], self.K
        x = torch.linspace(0, 1, K)
        if self.kind == "shared":
            y = x.expand(B, K).clone()
            y[:, 1:-1] = y[:, 1:-1] + curve_params
            return y[:, None, :].expand(B, 3, K)
        p = curve_params.view(B, 3, K)
        black, r = p[..., :1], p[..., 1:]
        steps = torch.exp(r) / (K - 1)
        return torch.cat([black, black + torch.cumsum(steps, -1)], -1)

    def forward(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q = self.unpack(params)
        x = _interp(img, self.curves(q["curve"]))
        x = _matrix(x, q["dM"], q["bias"])
        x = _vignette(x, q["vignette"])
        return x.clamp(0, 1)
