import torch
import torch.nn as nn

class TransformationHead(nn.Module):
    """
    Predicts the parameters for the image transformation.
    Based on the Fujifilm Generator, we need to predict:
    1. Highlight Tone (approx -2 to +4)
    2. Shadow Tone (approx -2 to +4)
    3. Color (Saturation) (approx -4 to +4)
    4. WB Shift Red (approx -9 to +9)
    5. WB Shift Blue (approx -9 to +9)
    6. Grain Intensity (0 to 1)
    7. Vignette Strength (0 to 1)
    
    Total: 7 parameters.
    """
    def __init__(self, input_dim=1280): # 768 (CDF) + 512 (Spatial)
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 7) # Output 7 parameters
        )
        
        # Initialize last layer to output near 0 (neutral)
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        """
        x: (B, input_dim)
        Returns: (B, 7) parameters
        """
        return self.net(x)
