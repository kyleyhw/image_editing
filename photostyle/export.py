"""Export predicted edits as standard 3D LUTs (``.cube``).

The global part of an edit (curves -> colour matrix -> bias) is a pure
per-pixel colour map, so it can be baked into a 3D LUT that any editor
(Resolve, Premiere, Photoshop, Lightroom profiles, OBS) can apply. The
vignette is spatial and cannot live in a LUT; it is left out, and exporters
report it separately.

``.cube`` format (Adobe / Resolve): ``LUT_3D_SIZE N`` followed by N^3 lines
"r g b", with the **red index varying fastest**.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from photostyle.render import GlobalRenderer


@torch.no_grad()
def bake_lut(renderer: GlobalRenderer, theta: torch.Tensor, size: int = 33) -> torch.Tensor:
    """Return a (size, size, size, 3) LUT indexed [b, g, r] -> rgb out.

    theta: (P,) or (1, P) renderer parameters; the vignette is zeroed.
    """
    theta = theta.reshape(1, -1).clone()
    q = renderer.unpack(theta)
    q["vignette"].zero_()
    g = torch.linspace(0, 1, size)
    b_, g_, r_ = torch.meshgrid(g, g, g, indexing="ij")  # red fastest in memory
    lattice = torch.stack([r_, g_, b_]).reshape(1, 3, size * size, size)
    out = renderer(lattice, theta)
    return out.reshape(3, size, size, size).permute(1, 2, 3, 0).contiguous()


def write_cube(lut: torch.Tensor, path: Path, title: str = "photostyle") -> None:
    size = lut.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0",
             "DOMAIN_MAX 1.0 1.0 1.0"]
    flat = lut.reshape(-1, 3).clamp(0, 1)
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in flat.tolist()]
    Path(path).write_text("\n".join(lines) + "\n")


def read_cube(path: Path) -> torch.Tensor:
    size, rows = None, []
    for line in Path(path).read_text().splitlines():
        t = line.strip().split()
        if not t or t[0].startswith("#") or t[0] in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX"):
            continue
        if t[0] == "LUT_3D_SIZE":
            size = int(t[1])
            continue
        rows.append([float(v) for v in t])
    return torch.tensor(rows).reshape(size, size, size, 3)


def apply_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Trilinear LUT application. img: (B, 3, H, W) in [0, 1]."""
    B, _, H, W = img.shape
    vol = lut.permute(3, 0, 1, 2)[None].expand(B, -1, -1, -1, -1)  # (B, 3, D=b, H=g, W=r)
    # grid_sample's last dim is (x, y, z) = (r, g, b), scaled to [-1, 1].
    grid = (img.permute(0, 2, 3, 1) * 2 - 1).reshape(B, 1, H, W, 3)
    out = F.grid_sample(vol, grid, mode="bilinear", align_corners=True)  # trilinear in 3-D
    return out.reshape(B, 3, H, W)
