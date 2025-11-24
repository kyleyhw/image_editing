import torch
import torch.nn as nn
import torchvision.models as models

class DifferentiableCDF(nn.Module):
    """
    Computes a differentiable Cumulative Distribution Function (CDF) for an image.
    Uses soft binning (Gaussian kernels) to make the histogram calculation differentiable.
    """
    def __init__(self, bins=256, sigma=0.01):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        # Register bin centers as buffer (not a learnable parameter)
        self.register_buffer('centers', torch.linspace(0, 1, bins))

    def forward(self, x):
        """
        x: (B, C, H, W) input image in range [0, 1]
        Returns: (B, C, bins) CDF vectors
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).unsqueeze(-1) # (B, C, N, 1)
        centers = self.centers.view(1, 1, 1, -1) # (1, 1, 1, bins)
        
        # Soft assignment: Gaussian kernel
        # (x - center)^2 / sigma^2
        dist = torch.pow(x_flat - centers, 2)
        weights = torch.exp(-dist / (self.sigma ** 2)) # (B, C, N, bins)
        
        # Soft Histogram
        hist = torch.sum(weights, dim=2) # (B, C, bins)
        
        # Normalize to get PDF
        pdf = hist / (hist.sum(dim=2, keepdim=True) + 1e-6)
        
        # Cumulative Sum to get CDF
        cdf = torch.cumsum(pdf, dim=2)
        
        return cdf

class SpatialEncoder(nn.Module):
    """
    Extracts spatial features using a CNN backbone (ResNet-18).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet18, remove fully connected layer
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, 512) feature vector
        """
        x = self.features(x)
        return x.view(x.size(0), -1)

class FeatureExtractor(nn.Module):
    """
    Combines Global Statistical Features (CDF) and Local Spatial Features (CNN).
    """
    def __init__(self, cdf_bins=256):
        super().__init__()
        self.cdf_extractor = DifferentiableCDF(bins=cdf_bins)
        self.spatial_encoder = SpatialEncoder()
        
    def forward(self, x):
        # 1. CDF Features: (B, 3, 256) -> Flatten to (B, 768)
        cdf = self.cdf_extractor(x)
        cdf_flat = cdf.view(cdf.size(0), -1)
        
        # 2. Spatial Features: (B, 512)
        spatial = self.spatial_encoder(x)
        
        # 3. Concatenate: (B, 768 + 512) = (B, 1280)
        combined = torch.cat([cdf_flat, spatial], dim=1)
        return combined
