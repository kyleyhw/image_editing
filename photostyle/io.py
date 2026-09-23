"""Colour-managed image I/O.

* JPEG / PNG / TIFF / HEIC (if pillow-heif is installed): EXIF orientation is
  applied, and the embedded ICC profile (iPhones use Display P3) is converted
  to the sRGB working space.
* Camera RAW (.dng .cr2 .cr3 .nef .arw .raf .orf .rw2): a *neutral* develop
  via rawpy (camera white balance, no auto-brightening, sRGB output). This
  neutral develop is the "before" for RAW+JPEG camera pairs.
* Saving keeps the source EXIF (minus orientation, which is baked in) and
  tags the output as sRGB.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}
_SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(_SRGB).tobytes()


def load_image(path: str | Path) -> tuple[Image.Image, dict]:
    """Load any supported file as an sRGB PIL image. Returns (image, info)."""
    path = Path(path)
    info: dict = {"source": str(path)}
    if path.suffix.lower() in RAW_EXT:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8,
                                  output_color=rawpy.ColorSpace.sRGB)
        info["icc"] = "raw (neutral develop, sRGB)"
        return Image.fromarray(rgb), info
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif  # type: ignore[import-not-found]

            pillow_heif.register_heif_opener()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("HEIC needs `pip install pillow-heif`") from exc
    img = Image.open(path)
    info["exif"] = img.info.get("exif")
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        info["icc"] = ImageCms.getProfileDescription(src).strip()
        img = ImageCms.profileToProfile(img.convert("RGB"), src, _SRGB,
                                        renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
    else:
        info["icc"] = "none (assumed sRGB)"
        img = img.convert("RGB")
    return img, info


def save_image(img: Image.Image, path: str | Path, info: dict | None = None, quality: int = 95) -> None:
    path = Path(path)
    kw: dict = {"icc_profile": SRGB_ICC}
    exif = (info or {}).get("exif")
    if exif:
        e = Image.Exif()
        e.load(exif)
        e[0x0112] = 1  # orientation already applied
        kw["exif"] = e.tobytes()
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        kw["quality"] = quality
    img.save(path, **kw)


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
