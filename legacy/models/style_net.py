import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .transformation_head import TransformationHead

class StyleNet(nn.Module):
    """
    End-to-End Network for Image Stylization.
    Input: Image (B, 3, H, W)
    Output: Transformation Parameters (B, 7)
    """
    def __init__(self):
        super().__init__()
        self.encoder = FeatureExtractor()
        # Input dim is 1280 (768 from CDF + 512 from ResNet)
        self.head = TransformationHead(input_dim=1280)
        
    def forward(self, x):
        features = self.encoder(x)
        params = self.head(features)
        return params
