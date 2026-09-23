"""Parameter-regression head (no BatchNorm; zero-initialised output)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Head(nn.Module):
    """Standardised features -> renderer parameters.

    LayerNorm instead of BatchNorm: the Phase 6 head's BatchNorm at batch
    size 4 was one of the confounds in the 80 -> 500 pair regression. The
    last layer starts at zero, so an untrained model is the identity edit.
    Input standardisation statistics are stored as buffers so a saved head
    is self-contained.
    """

    # Defaults chosen by a small-data check on FiveK landscapes (n = 10 / 50):
    # hidden 64 + dropout 0.3 beat hidden 256 + dropout 0.1 by +1.1 / +0.3 dB
    # and ~3 / ~1.4 dE, which the wider head lost to overfitting.
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Shrinkage toward the mean training prediction (alpha = 1: off). Set by
        # photostyle.train.calibrate_shrinkage; stabilises small-data heads.
        self.register_buffer("theta_mean", torch.zeros(out_dim))
        self.register_buffer("alpha", torch.ones(()))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before shrinkage existed load with shrinkage off.
        state_dict.setdefault(prefix + "theta_mean", torch.zeros_like(self.theta_mean))
        state_dict.setdefault(prefix + "alpha", torch.ones_like(self.alpha))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def set_norm(self, feats: torch.Tensor) -> None:
        self.mu.copy_(feats.mean(0))
        self.sigma.copy_(feats.std(0).clamp_min(1e-4))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        raw = self.net((f - self.mu) / self.sigma)
        return self.theta_mean + self.alpha * (raw - self.theta_mean)


class Preset(nn.Module):
    """Input-independent parameters: the 'static preset' baseline.

    One learned parameter vector applied to every image. The gap between this
    and ``Head`` is exactly the value of content adaptivity.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(out_dim))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.theta.expand(f.shape[0], -1)
