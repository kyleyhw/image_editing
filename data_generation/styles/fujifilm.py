import numpy as np
import skimage as ski
from scipy.interpolate import PchipInterpolator
from ..core import StyleGenerator
from .film import Grain, Vignette

class FujifilmGenerator(StyleGenerator):
    """
    Simulates Fujifilm film recipes.
    """
    def __init__(self, recipe_name="classic_chrome"):
        self.recipe_name = recipe_name
        self.recipe = self._get_recipe(recipe_name)
        
        # Initialize components
        self.grain = Grain(intensity=self.recipe.get("grain_intensity", 0.0))
        self.vignette = Vignette(strength=self.recipe.get("vignette_strength", 0.0))

    def _get_recipe(self, name):
        """
        Returns the configuration dict for a given recipe name.
        Values are based on Fujifilm settings:
        - Highlight/Shadow Tone: -2 (Soft) to +4 (Hard)
        - Color: -4 (Low) to +4 (High)
        - WB Shift: (Red, Blue) integers
        """
        recipes = {
            "classic_chrome": {
                "highlight_tone": -1,  # Soft highlights
                "shadow_tone": 1,      # Harder shadows
                "color": -1,           # Low saturation
                "wb_shift": (1, -2),   # R+1, B-2 (Warm)
                "color_chrome": "strong",
                "grain_intensity": 0.04,
                "vignette_strength": 0.3
            },
            "velvia": {
                "highlight_tone": 1,
                "shadow_tone": 2,
                "color": 4,            # High saturation
                "wb_shift": (-1, 0),
                "color_chrome": "weak",
                "grain_intensity": 0.02,
                "vignette_strength": 0.1
            }
        }
        return recipes.get(name, recipes["classic_chrome"])

    def _apply_tone_curve(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates Highlight and Shadow tone settings.
        """
        h_tone = self.recipe["highlight_tone"]
        s_tone = self.recipe["shadow_tone"]
        
        # Base points (0, 0.5, 1)
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Adjust Shadows (Toe)
        # +Shadow means darker shadows (pull down 0.25)
        # -Shadow means lighter shadows (push up 0.25)
        # Scale: 1 unit approx 0.05 shift
        y[1] -= s_tone * 0.05
        
        # Adjust Highlights (Shoulder)
        # +Highlight means brighter highlights (push up 0.75)
        # -Highlight means softer highlights (pull down 0.75)
        y[3] += h_tone * 0.05
        
        # Clamp control points
        y = np.clip(y, 0, 1)
        
        interpolator = PchipInterpolator(x, y)
        return np.clip(interpolator(image), 0, 1)

    def _apply_color_shift(self, image: np.ndarray) -> np.ndarray:
        """
        Applies White Balance Shift (Red, Blue).
        """
        r_shift, b_shift = self.recipe["wb_shift"]
        
        # Scale: 1 unit approx 2% shift
        r_scale = 1.0 + (r_shift * 0.02)
        b_scale = 1.0 + (b_shift * 0.02)
        
        image[:, :, 0] *= r_scale # Red
        image[:, :, 2] *= b_scale # Blue
        
        return np.clip(image, 0, 1)

    def _apply_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        Adjusts saturation based on 'Color' setting.
        """
        color_setting = self.recipe["color"]
        if color_setting == 0:
            return image
            
        hsv = ski.color.rgb2hsv(image)
        # Scale: 1 unit approx 10% saturation change
        sat_scale = 1.0 + (color_setting * 0.1)
        hsv[:, :, 1] *= sat_scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        
        return ski.color.hsv2rgb(hsv)

    def _apply_chrome_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates Color Chrome Effect: Deepens high-saturation colors.
        """
        effect = self.recipe["color_chrome"]
        if effect == "off":
            return image
            
        strength = 0.2 if effect == "strong" else 0.1
        
        hsv = ski.color.rgb2hsv(image)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Reduce Value where Saturation is high
        # V_new = V * (1 - strength * S)
        value_mod = value * (1.0 - strength * saturation)
        
        hsv[:, :, 2] = value_mod
        return ski.color.hsv2rgb(hsv)

    def generate_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Ensure float [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        styled = image.copy()
        
        # 1. WB Shift (Pre-tone)
        styled = self._apply_color_shift(styled)
        
        # 2. Tone Curve
        styled = self._apply_tone_curve(styled)
        
        # 3. Chrome Effect & Saturation
        styled = self._apply_chrome_effect(styled)
        styled = self._apply_saturation(styled)
        
        # 4. Texture & Optics
        styled = self.grain.apply(styled)
        styled = self.vignette.apply(styled)
        
        return image, styled
