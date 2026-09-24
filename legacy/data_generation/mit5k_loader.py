"""MIT-Adobe FiveK data loader (Phase 6b).

The MIT-Adobe FiveK Dataset [Bychkovsky et al. 2011] consists of 5,000
raw photographs together with five expert-retouched versions per photo
(experts A-E). Training on this dataset lets the generic model learn a
real-world "professional retouch" style rather than a programmatically
generated effect - the one regime in which the network does something
no closed-form data generator can.

How to obtain a usable subset
-----------------------------

Run the bundled helper:

    python tools/download_fivek_subset.py --expert c --count 200 --out_dir data/fivek_c_200

It streams the ``logasja/mit-adobe-fivek`` HuggingFace dataset (one
config per expert, ``a`` ... ``e``), downsizes each image to a 512 px
long edge, and saves paired JPEGs in exactly the layout this loader
expects. Streaming avoids downloading the full ~120 GB per expert.

Expected directory layout
-------------------------

    <root>/
        original/img_0000.jpg
        original/img_0001.jpg
        ...
        expert_<letter>/img_0000.jpg
        expert_<letter>/img_0001.jpg

If you assemble the dataset by other means (e.g. from the official MIT
release at https://data.csail.mit.edu/graphics/fivek/), ensure the
filenames in ``original/`` and ``expert_<letter>/`` match exactly; the
loader pairs by filename intersection.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import skimage as ski
from torch.utils.data import Dataset


class MIT5KDataset(Dataset):
    """Paired (original, expert-retouched) loader for the MIT-Adobe FiveK set.

    Parameters
    ----------
    root : str
        Path containing the `original/` and `expert_<x>/` subdirectories.
    expert : str
        Which expert's retouches to use (e.g. 'a', 'c').
    transform : callable, optional
        Applied separately to original and styled images. Typically a
        torchvision Compose([ToPILImage, Resize, ToTensor]).
    suffixes : tuple of str
        Image file extensions to consider.
    """

    def __init__(
        self,
        root: str,
        expert: str = "c",
        transform: Callable | None = None,
        suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ):
        self.root = root
        self.expert = expert.lower()
        self.transform = transform
        self.suffixes = tuple(s.lower() for s in suffixes)

        original_dir = os.path.join(root, "original")
        expert_dir = os.path.join(root, f"expert_{self.expert}")
        if not os.path.isdir(original_dir):
            raise FileNotFoundError(
                f"missing originals directory: {original_dir}. See MIT5KDataset "
                "docstring for the expected layout."
            )
        if not os.path.isdir(expert_dir):
            raise FileNotFoundError(
                f"missing expert directory: {expert_dir}. The MIT-5K release "
                "ships five experts (a-e); pass --expert <letter>."
            )
        self.original_dir = original_dir
        self.expert_dir = expert_dir

        # Build the paired filename list (intersection of the two dirs).
        originals = {f for f in os.listdir(original_dir) if f.lower().endswith(self.suffixes)}
        experts = {f for f in os.listdir(expert_dir) if f.lower().endswith(self.suffixes)}
        common = sorted(originals & experts)
        if not common:
            raise RuntimeError(
                "no matching filenames between original/ and expert_"
                f"{self.expert}/ in {root}"
            )
        self.files = common

    def __len__(self) -> int:
        return len(self.files)

    def _load_uint8(self, path: str) -> np.ndarray:
        img = ski.io.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = ski.color.rgba2rgb(img)
            img = (img * 255.0).round().astype(np.uint8)
        if img.dtype != np.uint8:
            img = ski.util.img_as_ubyte(img)
        return img

    def __getitem__(self, idx: int):
        name = self.files[idx]
        original = self._load_uint8(os.path.join(self.original_dir, name))
        styled = self._load_uint8(os.path.join(self.expert_dir, name))
        if self.transform is not None:
            original = self.transform(original)
            styled = self.transform(styled)
        return original, styled
