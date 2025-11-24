import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableFujifilm(nn.Module):
    """
    Differentiable version of the FujifilmGenerator.
    Applies parameters predicted by the network to the input image.
    """
    def __init__(self):
        super().__init__()

    def rgb_to_hsv(self, image):
        """
        Convert RGB to HSV.
        image: (B, 3, H, W) in [0, 1]
        Returns: (B, 3, H, W)
        """
        # Simplified implementation or use kornia if available.
        # Implementing a robust differentiable RGB2HSV is non-trivial due to conditionals.
        # For now, let's use a simplified approximation or just operate in RGB for saturation 
        # (mixing with grayscale).
        
        # Saturation approximation: Lerp between image and its grayscale version.
        return image # Placeholder, we will use mix-with-gray for saturation

    def apply_saturation_approx(self, image, saturation_param):
        """
        Adjust saturation by mixing with grayscale.
        saturation_param: (B, 1) -4 to +4
        """
        # Map param to scale: 0 -> 1.0, +4 -> 1.4, -4 -> 0.6
        scale = 1.0 + (saturation_param * 0.1)
        scale = scale.view(-1, 1, 1, 1)
        
        # Grayscale
        gray = image.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        
        # Lerp: out = gray + (image - gray) * scale
        out = gray + (image - gray) * scale
        return torch.clamp(out, 0, 1)

    def apply_tone_curve(self, image, h_tone, s_tone):
        """
        Applies a piecewise linear tone curve defined by Highlight and Shadow params.
        h_tone, s_tone: (B, 1)
        """
        # Base points
        # x = [0.0, 0.25, 0.5, 0.75, 1.0]
        # y = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Adjust y
        y0 = torch.zeros_like(h_tone)
        y1 = 0.25 - (s_tone * 0.05)
        y2 = torch.full_like(h_tone, 0.5)
        y3 = 0.75 + (h_tone * 0.05)
        y4 = torch.ones_like(h_tone)
        
        # We need to apply this curve to 'image'.
        # Since we can't easily do spline interpolation, we'll use a 
        # differentiable polynomial approximation or piecewise linear.
        # Let's use a simple polynomial: f(x) = x + a*x*(1-x) ? No, that's too simple.
        # Let's use Piecewise Linear Interpolation.
        
        # Flatten image
        B, C, H, W = image.shape
        img_flat = image.view(B, -1) # (B, N)
        
        # Find bins
        # 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
        mask0 = (img_flat < 0.25).float()
        mask1 = ((img_flat >= 0.25) & (img_flat < 0.5)).float()
        mask2 = ((img_flat >= 0.5) & (img_flat < 0.75)).float()
        mask3 = (img_flat >= 0.75).float()
        
        # Interpolate
        # Region 0: 0 -> 0.25
        # y = y0 + (y1-y0) * (x - 0) / 0.25
        out0 = y0 + (y1 - y0) * (img_flat / 0.25)
        
        # Region 1: 0.25 -> 0.5
        out1 = y1 + (y2 - y1) * ((img_flat - 0.25) / 0.25)
        
        # Region 2: 0.5 -> 0.75
        out2 = y2 + (y3 - y2) * ((img_flat - 0.5) / 0.25)
        
        # Region 3: 0.75 -> 1.0
        out3 = y3 + (y4 - y3) * ((img_flat - 0.75) / 0.25)
        
        out = out0*mask0 + out1*mask1 + out2*mask2 + out3*mask3
        return out.view(B, C, H, W)

    def forward(self, image, params):
        """
        image: (B, 3, H, W)
        params: (B, 7)
        """
        # Unpack parameters
        # 0: Highlight Tone
        # 1: Shadow Tone
        # 2: Color (Saturation)
        # 3: WB Red
        # 4: WB Blue
        # 5: Grain
        # 6: Vignette
        
        h_tone = params[:, 0:1]
        s_tone = params[:, 1:2]
        color = params[:, 2:3]
        wb_r = params[:, 3:4]
        wb_b = params[:, 4:5]
        grain = params[:, 5:6]
        vignette = params[:, 6:7]
        
        x = image
        
        # 1. WB Shift
        # Scale: 1 + param * 0.02
        r_scale = 1.0 + (wb_r * 0.02)
        b_scale = 1.0 + (wb_b * 0.02)
        
        # Apply to channels
        # x is (B, 3, H, W)
        r = x[:, 0:1] * r_scale.view(-1, 1, 1, 1)
        g = x[:, 1:2]
        b = x[:, 2:3] * b_scale.view(-1, 1, 1, 1)
        x = torch.cat([r, g, b], dim=1)
        x = torch.clamp(x, 0, 1)
        
        # 2. Tone Curve
        x = self.apply_tone_curve(x, h_tone, s_tone)
        
        # 3. Saturation
        x = self.apply_saturation_approx(x, color)
        
        # 4. Grain (Additive Gaussian)
        # We need to generate noise on the fly
        # Noise should be detached? No, we want to learn the INTENSITY.
        # So noise is random, but scaled by 'grain' param.
        if self.training:
            noise = torch.randn_like(x)
            x = x + noise * grain.view(-1, 1, 1, 1)
            x = torch.clamp(x, 0, 1)
            
        # 5. Vignette
        # Generate vignette mask
        B, C, H, W = x.shape
        Y, X = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        dist = torch.sqrt(X**2 + Y**2).to(x.device) # 0 at center, sqrt(2) at corner
        dist = dist.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Mask = 1 - strength * dist^2
        # Normalize dist so corner is 1?
        dist = dist / 1.414
        
        mask = 1.0 - (vignette.view(-1, 1, 1, 1) * (dist ** 2))
        mask = torch.clamp(mask, 0, 1)
        
        x = x * mask
        
        return x
