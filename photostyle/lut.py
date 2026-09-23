"""Residual basis-LUT stage (PROJECT_PLAN Phase 9; Zeng et al. 2020 / SepLUT).

    out = G(x) + sum_n w_n(x) * B_n(G(x))

G is the editable global renderer (per-channel curves + colour matrix).
B_1..B_N are shared, learned *residual* 3D LUTs on a small (default 9^3)
lattice, applied by trilinear interpolation. w_n are per-image weights
predicted by the head. With B_n initialised to zero, the stage starts as the
identity.

Regularisers (Zeng et al.): total-variation smoothness of each basis, and
monotonicity of identity + weighted-mean residual along each colour axis. These
are exposed via ``regularizer()``, which the Phase 7 trainer adds to the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from photostyle.render import GlobalRenderer


def apply_residual_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """img (B, 3, H, W); lut (B, 3, D, D, D) indexed [c, b, g, r]."""
    B, _, H, W = img.shape
    grid = (img.clamp(0, 1).permute(0, 2, 3, 1) * 2 - 1).reshape(B, 1, H, W, 3)
    return F.grid_sample(lut, grid, mode="bilinear", align_corners=True).reshape(B, 3, H, W)


class LUTRenderer(nn.Module):
    def __init__(self, kind: str = "per_channel", knots: int = 9, n_basis: int = 4, size: int = 9,
                 lam_tv: float = 1e-4, lam_mono: float = 10.0):
        super().__init__()
        self.base = GlobalRenderer(kind, knots)
        self.K = knots
        self.n_basis, self.size = n_basis, size
        self.bases = nn.Parameter(torch.zeros(n_basis, 3, size, size, size))
        self.lam_tv, self.lam_mono = lam_tv, lam_mono
        self.layout = {**self.base.layout, "lut_w": n_basis}
        self.num_params = sum(self.layout.values())

    def unpack(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for k, n in self.layout.items():
            out[k] = p[:, i:i + n]
            i += n
        return out

    def forward(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        pg = params[:, : self.base.num_params]
        w = params[:, self.base.num_params:]
        x = self.base(img, pg)
        lut = torch.einsum("bn,ncdhw->bcdhw", w + 1.0 / self.n_basis, self.bases)
        return (x + apply_residual_lut(x, lut)).clamp(0, 1)

    def regularizer(self) -> torch.Tensor:
        b = self.bases
        tv = sum(((b.diff(dim=d)) ** 2).mean() for d in (2, 3, 4))
        g = torch.linspace(0, 1, self.size)
        bb, gg, rr = torch.meshgrid(g, g, g, indexing="ij")
        ident = torch.stack([rr, gg, bb])                     # (3, D, D, D) channel c along its own axis
        full = ident[None] + b                                # each basis applied at unit weight
        mono = (F.relu(-full[:, 0].diff(dim=3)).mean() + F.relu(-full[:, 1].diff(dim=2)).mean()
                + F.relu(-full[:, 2].diff(dim=1)).mean())
        return self.lam_tv * tv + self.lam_mono * mono
