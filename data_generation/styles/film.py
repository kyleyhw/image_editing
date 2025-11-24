import numpy as np
import skimage as ski
from scipy.interpolate import PchipInterpolator
from ..core import Transformation, StyleGenerator

class ToneCurve(Transformation):
    def __init__(self, contrast=1.0):
        self.contrast = contrast

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies an S-curve to the image to simulate film contrast.
        """
        # Simple S-curve using smooth step or similar
        # Here we use a PchipInterpolator for control
        x = np.array([0, 0.25, 0.5, 0.75, 1])
        # Steepen the curve based on contrast
        y = np.array([0, 0.25 - 0.05 * self.contrast, 0.5, 0.75 + 0.05 * self.contrast, 1])
        
        interpolator = PchipInterpolator(x, y)
        
        # Apply per channel or globally? Film usually affects channels differently,
        # but for a start, let's apply globally to L or RGB.
        # Let's apply to RGB for now.
        return np.clip(interpolator(image), 0, 1)

class Grain(Transformation):
    def __init__(self, intensity=0.05):
        self.intensity = intensity

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Adds Gaussian noise to simulate film grain.
        """
        noise = np.random.normal(0, self.intensity, image.shape)
        return np.clip(image + noise, 0, 1)

class Vignette(Transformation):
    def __init__(self, strength=0.5):
        self.strength = strength

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Adds a vignette effect.
        """
        H, W, C = image.shape
        Y, X = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        
        # Distance from center, normalized
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        # Vignette mask (1 at center, 1-strength at corners)
        mask = 1 - self.strength * (normalized_dist ** 2)
        mask = np.clip(mask, 0, 1)
        
        # Expand mask to 3 channels
        mask = np.dstack([mask] * C)
        
        return image * mask

class FilmGenerator(StyleGenerator):
    def __init__(self):
        self.tone = ToneCurve(contrast=1.2)
        self.grain = Grain(intensity=0.03)
        self.vignette = Vignette(strength=0.4)

    def generate_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Ensure float [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        styled = image.copy()
        styled = self.tone.apply(styled)
        styled = self.grain.apply(styled)
        styled = self.vignette.apply(styled)
        
        return image, styled
