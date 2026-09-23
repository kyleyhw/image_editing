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


BASE_PATH = "checkpoints/base_stylehead.pt"


class CodedHead(nn.Module):
    """A StyleHead with one fixed style code: behaves like ``Head`` (features -> parameters).

    This is how a style made on the shared base is stored in a style pack: the
    base weights plus the style's code (fitted on pairs, or from the encoder).
    """

    def __init__(self, head: StyleHead, code: torch.Tensor):
        super().__init__()
        self.head = head
        self.register_buffer("code", code.detach().clone())

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f, self.code.expand(f.shape[0], -1))


def load_base(path: str = BASE_PATH):
    """-> (StyleHead, StyleEncoder, info) or None if no base has been built."""
    import os

    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = StyleHead(ck["in_dim"], ck["out_dim"], n_styles=ck["n_styles"])
    head.load_state_dict(ck["head"])
    enc = StyleEncoder(ck["in_dim"])
    enc.load_state_dict(ck["encoder"])
    head.eval()
    enc.eval()
    return head, enc, ck["info"]


def fit_code(head: StyleHead, renderer, items: list[dict], steps: int = 400, seed: int = 0) -> torch.Tensor:
    """Condition-only fine-tuning (Phase 10 protocol (b)): fit a new code on pairs, base frozen.

    items: dicts with "feat", "src_t", "tgt_t"."""
    import random

    torch.manual_seed(seed)
    rng = random.Random(seed)
    for prm in head.parameters():
        prm.requires_grad_(False)
    code = nn.Parameter(head.embed.weight.mean(0).clone())
    opt = torch.optim.Adam([code], lr=1e-2)
    for _ in range(steps):
        b = rng.sample(items, min(8, len(items)))
        th = head(torch.stack([it["feat"] for it in b]), code.expand(len(b), -1))
        loss = sum((renderer(it["src_t"][None], t[None])[0] - it["tgt_t"]).abs().mean()
                   for it, t in zip(b, th)) / len(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for prm in head.parameters():
        prm.requires_grad_(True)
    return code.detach()
