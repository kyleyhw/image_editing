from abc import ABC, abstractmethod
import numpy as np

class Transformation(ABC):
    """
    Abstract base class for a single image transformation (e.g., ToneCurve, Grain).
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the image.
        :param image: Input image (H, W, C) in range [0, 1] or [0, 255].
        :return: Transformed image.
        """
        pass

class StyleGenerator(ABC):
    """
    Abstract base class for a complete style generator (e.g., FilmGenerator, CyberpunkGenerator).
    Composes multiple Transformations.
    """
    @abstractmethod
    def generate_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes an original image and returns a pair (original, styled).
        The 'original' might be slightly modified (e.g. resized) if needed,
        but usually it's just passed through.
        
        :param image: Input image.
        :return: Tuple (original, styled).
        """
        pass
