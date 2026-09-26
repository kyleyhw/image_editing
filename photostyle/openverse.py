"""Openverse access and image-set hygiene, shared by the style tools and the new-style pipeline.

Only licences that allow reuse and adaptation are requested (CC0, Public
Domain Mark, CC BY, CC BY-SA). Search responses and downloaded images are
cached, so re-filtering never repeats API calls. Anonymous limits: 20
requests/min, 200/day.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import skimage as ski
from PIL import Image

API = "https://api.openverse.org/v1/images/"
USER_AGENT = "image_editing-style-research/0.1 (github.com/kyleyhw/image_editing)"
ALLOWED_LICENSES = "cc0,pdm,by,by-sa"


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def search(query: str, page: int, session: requests.Session, cache: Path) -> list[dict]:
    key = cache / "search" / (hashlib.sha1(f"{query}|{page}|{ALLOWED_LICENSES}".encode()).hexdigest() + ".json")
    if key.exists():
        return json.loads(key.read_text())
    params = {
        "q": query,
        "license": ALLOWED_LICENSES,
        "page_size": 20,
        "page": page,
        "mature": "false",
    }
    for wait in (0, 20, 60, 180):            # 403/429 are throttling: back off before giving up
        time.sleep(wait)
        r = session.get(API, params=params, timeout=30)
        if r.status_code not in (403, 429):
            break
    if r.status_code in (403, 429):
        raise RuntimeError(f"Openverse refused the request ({r.status_code}): rate limit; rerun later (200 requests/day).")
    r.raise_for_status()
    results = r.json().get("results", [])
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(results))
    time.sleep(3.2)  # stay under 20 requests/min (only for real API calls)
    return results


def download(item: dict, session: requests.Session, cache: Path) -> Image.Image | None:
    path = cache / "images" / f"{item['id']}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    try:
        r = session.get(item["url"], timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    img.thumbnail((1024, 1024))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    return img


def crop_letterbox(img: Image.Image, thresh: float = 0.03) -> Image.Image:
    """Remove near-black bars on any edge (letterbox/pillarbox).

    A row/column counts as bar if >= 90 % of its pixels are near-black, which
    tolerates the caption text often printed inside the bar. A bar is only
    cropped if it ends in a sharp edge (the first photo row/column is mostly
    not black), so genuinely dark night skies are left alone.
    """
    g = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    dark = g < thresh
    row_frac, col_frac = dark.mean(1), dark.mean(0)

    def edge(frac: np.ndarray, from_start: bool) -> int:
        n = len(frac)
        idx = range(n) if from_start else range(n - 1, -1, -1)
        k = 0
        for i in idx:
            if frac[i] < 0.9:
                break
            k += 1
        if k < 3 or k >= n // 2:
            return 0
        first_photo = frac[k] if from_start else frac[n - 1 - k]
        return k if first_photo < 0.6 else 0

    top, bottom = edge(row_frac, True), edge(row_frac, False)
    left, right = edge(col_frac, True), edge(col_frac, False)
    if top == bottom == left == right == 0:
        return img
    return img.crop((left, top, img.width - right, img.height - bottom))


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class PersonDetector:
    """Largest-person area fraction via torchvision Faster R-CNN (COCO)."""

    def __init__(self) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        self.torch = torch
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()
        self.preprocess = weights.transforms()

    def people(self, img: Image.Image) -> list[tuple[float, tuple[float, float, float, float]]]:
        """Return [(area_fraction, (x0, y0, x1, y1)), ...] sorted largest first."""
        import torchvision.transforms.functional as F

        x = self.preprocess(F.pil_to_tensor(img))
        with self.torch.no_grad():
            out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] > 0.7)
        res = []
        for box in out["boxes"][keep].tolist():
            area = (box[2] - box[0]) * (box[3] - box[1]) / (img.width * img.height)
            res.append((area, tuple(box)))
        return sorted(res, reverse=True)


def dof_log_ratio(img: Image.Image, box: tuple[float, float, float, float]) -> float:
    """log(sharpness inside the subject box / sharpness outside it).

    Sharpness = mean absolute Laplacian on a 512 px greyscale copy. Large
    values mean a sharp subject on a blurred background (shallow depth of
    field); ~0 means everything is equally sharp.
    """
    scale = 512 / max(img.size)
    g = np.asarray(img.convert("L").resize((round(img.width * scale), round(img.height * scale))),
                   dtype=np.float64) / 255.0
    lap = np.abs(ski.filters.laplace(g))
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    inside = np.zeros_like(lap, dtype=bool)
    inside[y0:y1, x0:x1] = True
    if inside.all() or not inside.any():
        return 0.0
    return float(np.log((lap[inside].mean() + 1e-4) / (lap[~inside].mean() + 1e-4)))


def score(stats: dict[str, float], profile: dict[str, tuple]) -> float:
    terms = []
    for k, spec in profile.items():
        t, tol = spec[0], spec[1]
        if len(spec) > 2 and spec[2] == ">=" and stats[k] >= t:
            terms.append(1.0)
        else:
            terms.append(np.exp(-(((stats[k] - t) / tol) ** 2)))
    return float(np.mean(terms))


def attribution(item: dict) -> str:
    lic = item.get("license", "").upper()
    ver = item.get("license_version") or ""
    lic_txt = "CC0 1.0" if lic == "CC0" else ("Public Domain Mark" if lic == "PDM" else f"CC {lic} {ver}")
    return f'"{item.get("title") or "Untitled"}" by {item.get("creator") or "unknown"}, {lic_txt}, via {item.get("source")}'
