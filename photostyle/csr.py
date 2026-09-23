"""CSRNet-style baseline renderer (He et al. 2020), for the Phase 9 ablation.

A tiny per-pixel MLP (3 -> 32 -> 32 -> 3, residual) whose hidden layers are
modulated per image by FiLM parameters predicted by the head (the
"condition network"). Global (pixel-independent), so it can be baked into a
3D LUT, but its parameters are *not* human-readable sliders. It is the
small-model yardstick the editable renderers are compared against.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CSRLite(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Conv2d(3, hidden, 1)
        self.l2 = nn.Conv2d(hidden, hidden, 1)
        self.l3 = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.l3.weight)
        nn.init.zeros_(self.l3.bias)
        self.layout = {"film": 4 * hidden}
        self.num_params = 4 * hidden
        self.K = 0

    def forward(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        B, h = img.shape[0], self.hidden
        g1, b1, g2, b2 = params.view(B, 4, h, 1, 1).unbind(1)
        x = torch.relu(self.l1(img) * (1 + g1) + b1)
        x = torch.relu(self.l2(x) * (1 + g2) + b2)
        return (img + self.l3(x)).clamp(0, 1)
