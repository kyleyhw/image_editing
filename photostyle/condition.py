"""Style-conditioned head and style encoder (PROJECT_PLAN Phase 10).

``StyleHead``    one model for many looks: the image descriptor goes through
                 an MLP whose hidden layer is FiLM-modulated by a style code
                 (Perez et al. 2018; StarEnhancer's conditioning idea).
``StyleEncoder`` maps descriptors of example *after* images to a code; a new
                 style's code is the normalised mean over its examples
                 (StarEnhancer / PieNet recipe), so a look can be added from
                 a handful of photos without pairs.

Known styles use a learned embedding table; a new style can also get an
embedding fitted on a few pairs with the rest of the model frozen (CSRNet's
condition-only fine-tuning).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_styles: int, code_dim: int = 32, hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))
        self.embed = nn.Embedding(n_styles, code_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.l1 = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(code_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out_dim)
        self.code_bias = nn.Linear(code_dim, out_dim)       # style-specific offset of the edit
        for m in (self.out, self.code_bias):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def set_norm(self, feats: torch.Tensor) -> None:
        self.mu.copy_(feats.mean(0))
        self.sigma.copy_(feats.std(0).clamp_min(1e-4))

    def forward(self, f: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.l1((f - self.mu) / self.sigma))
        g, b = self.film(code).chunk(2, dim=-1)
        h = self.drop(F.gelu(h * (1 + g) + b))
        return self.out(h) + self.code_bias(code)


class StyleEncoder(nn.Module):
    """Descriptors of in-style images -> style code (mean-pooled, L2-normalised x scale)."""

    def __init__(self, in_dim: int, code_dim: int = 32, hidden: int = 128, scale: float = 1.0):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, code_dim))
        self.scale = scale

    def set_norm(self, feats: torch.Tensor) -> None:
        self.mu.copy_(feats.mean(0))
        self.sigma.copy_(feats.std(0).clamp_min(1e-4))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(n, D) examples of one style -> (code_dim,) code."""
        z = self.net((feats - self.mu) / self.sigma).mean(0)
        return self.scale * F.normalize(z, dim=0) * (z.shape[0] ** 0.5) / 4
