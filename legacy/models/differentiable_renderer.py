import torch
import torch.nn as nn


class DifferentiableFujifilm(nn.Module):
    """
    Differentiable, HSV-faithful counterpart of FujifilmGenerator.

    Saturation (S-scaling) and the Color Chrome Effect (V-attenuation by S)
    are HSV operations. A full differentiable RGB<->HSV round-trip has
    branch discontinuities at the hue boundaries. We avoid those by noting
    that when only S or V is rescaled (H preserved), the inverse HSV->RGB
    map collapses to closed-form expressions in RGB:

        V = max_c c,  m = min_c c,  S = (V - m) / V

        S -> beta * S, H,V fixed:   c' = V - beta * (V - c)
        V -> gamma * V, H,S fixed:  c' = gamma * c

    The Color Chrome Effect sets gamma = 1 - alpha * S with S unchanged,
    which by the second identity gives c' = c * (1 - alpha * S). The
    parameter alpha is fixed per recipe (see CHROME_STRENGTHS in
    data_generation.styles.fujifilm) and supplied at construction.
    """

    def __init__(self, chrome_strength: float = 0.2):
        super().__init__()
        # Default 0.2 matches CHROME_STRENGTHS["strong"], i.e. the
        # "classic_chrome" recipe used by the Phase 1/2 prototype.
        self.chrome_strength = float(chrome_strength)

    def apply_tone_curve(
        self, image: torch.Tensor, h_tone: torch.Tensor, s_tone: torch.Tensor
    ) -> torch.Tensor:
        """
        Piecewise-linear tone curve through control points
            x = [0, 0.25, 0.5, 0.75, 1]
            y = [0, 0.25 - 0.05*s_tone, 0.5, 0.75 + 0.05*h_tone, 1]
        matching FujifilmGenerator._apply_tone_curve. We use piecewise-linear
        rather than the numpy generator's PCHIP spline because (a) it is
        trivially differentiable and (b) the spline's third-order corrections
        are visually small compared to the parameter-induced shifts the
        network is trying to learn.
        """
        y0 = torch.zeros_like(h_tone)
        y1 = 0.25 - (s_tone * 0.05)
        y2 = torch.full_like(h_tone, 0.5)
        y3 = 0.75 + (h_tone * 0.05)
        y4 = torch.ones_like(h_tone)

        B, C, H, W = image.shape
        img_flat = image.view(B, -1)

        mask0 = (img_flat < 0.25).float()
        mask1 = ((img_flat >= 0.25) & (img_flat < 0.5)).float()
        mask2 = ((img_flat >= 0.5) & (img_flat < 0.75)).float()
        mask3 = (img_flat >= 0.75).float()

        out0 = y0 + (y1 - y0) * (img_flat / 0.25)
        out1 = y1 + (y2 - y1) * ((img_flat - 0.25) / 0.25)
        out2 = y2 + (y3 - y2) * ((img_flat - 0.5) / 0.25)
        out3 = y3 + (y4 - y3) * ((img_flat - 0.75) / 0.25)

        out = out0 * mask0 + out1 * mask1 + out2 * mask2 + out3 * mask3
        return out.view(B, C, H, W)

    def apply_chrome_effect(self, image: torch.Tensor) -> torch.Tensor:
        """
        c' = c * (1 - alpha * S), with S = (V - m) / V per pixel.

        Uniformly attenuates all channels of a pixel by the same factor
        (1 - alpha*S). Hue is preserved because all three channels scale
        identically; S is preserved because chroma and V scale together.
        """
        alpha = self.chrome_strength
        if alpha == 0.0:
            return image
        v = torch.amax(image, dim=1, keepdim=True)
        m = torch.amin(image, dim=1, keepdim=True)
        s = (v - m) / (v + 1e-6)  # epsilon guards V=0 (pure black).
        return torch.clamp(image * (1.0 - alpha * s), 0, 1)

    def apply_saturation(self, image: torch.Tensor, sat_param: torch.Tensor) -> torch.Tensor:
        """
        HSV-faithful saturation scaling via c' = V - beta * (V - c).

        beta = 1 + 0.1 * sat_param matches FujifilmGenerator._apply_saturation,
        whose `color` setting ranges over [-4, +4] -> beta in [0.6, 1.4].
        """
        beta = (1.0 + sat_param * 0.1).view(-1, 1, 1, 1)
        v = torch.amax(image, dim=1, keepdim=True)
        return torch.clamp(v - beta * (v - image), 0, 1)

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        image:  (B, 3, H, W) in [0, 1]
        params: (B, 7) = [highlight, shadow, color, wb_r, wb_b, grain, vignette]
        """
        h_tone = params[:, 0:1]
        s_tone = params[:, 1:2]
        color = params[:, 2:3]
        wb_r = params[:, 3:4]
        wb_b = params[:, 4:5]
        grain = params[:, 5:6]
        vignette = params[:, 6:7]

        x = image

        # 1. WB shift (per FujifilmGenerator: r_scale = 1 + 0.02*wb_r, etc.)
        r_scale = (1.0 + wb_r * 0.02).view(-1, 1, 1, 1)
        b_scale = (1.0 + wb_b * 0.02).view(-1, 1, 1, 1)
        r = x[:, 0:1] * r_scale
        g = x[:, 1:2]
        b = x[:, 2:3] * b_scale
        x = torch.clamp(torch.cat([r, g, b], dim=1), 0, 1)

        # 2. Tone curve
        x = self.apply_tone_curve(x, h_tone, s_tone)

        # 3. Chrome effect (BEFORE saturation; mirrors the numpy generator's
        #    ordering in FujifilmGenerator.generate_pair).
        x = self.apply_chrome_effect(x)

        # 4. Saturation
        x = self.apply_saturation(x, color)

        # 5. Grain. Applied unconditionally (the numpy generator always adds
        #    noise during data generation, so inference must too if it is to
        #    reproduce the training distribution).
        noise = torch.randn_like(x)
        x = torch.clamp(x + noise * grain.view(-1, 1, 1, 1), 0, 1)

        # 6. Vignette: mask = 1 - vignette * (d/d_max)^2 where d is the
        #    Euclidean distance from the image center and d_max = sqrt(2)
        #    after normalizing coordinates to [-1, 1].
        B, C, H, W = x.shape
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        dist = torch.sqrt(X**2 + Y**2) / 1.414
        mask = torch.clamp(1.0 - vignette.view(-1, 1, 1, 1) * (dist**2), 0, 1)

        return x * mask
