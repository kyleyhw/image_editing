"""Photo-vs-digital-art filter for reference searches (owner decision A5.6 #5).

Zero-shot CLIP (LAION ViT-B/32, MIT licence): each image is scored against
"a photograph" prompts vs. "digital art / 3D render / video game screenshot /
illustration / AI-generated concept art / map / collage" prompts, and the
photo probability is the softmax mass on the photo prompts.

Measured on 93 hand-labelled "cyberpunk" candidates (PROJECT_PLAN A5.6): at
the default threshold 0.05 it caught 30/32 renders and flagged 7/61 real
photos (graffiti, an installation, glossy tuned cars). So flagged images are
*greyed out, not deleted*: one click restores them. About 0.16 s per image on
CPU, plus a one-time 605 MB download.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
THRESHOLD = 0.05
PHOTO_PROMPTS = ["a photograph taken with a camera", "a real photo"]
ART_PROMPTS = ["digital art", "a 3D render", "a video game screenshot", "an illustration or drawing",
               "AI-generated concept art", "a map or a diagram", "a collage"]
_model = _proc = None
_failed = False


def _load(cache_dir: str = "data/cache/hf_models"):
    global _model, _proc, _failed
    if _model is None and not _failed:
        try:
            from transformers import AutoModel, AutoProcessor

            _model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache_dir).eval()
            _proc = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        except Exception:  # offline / not cached: no filtering (never blocks a search)
            _failed = True
    return _model, _proc


@torch.no_grad()
def photo_score(img: Image.Image | str | Path) -> float | None:
    """Probability that ``img`` is a photograph (None if the model is unavailable)."""
    model, proc = _load()
    if model is None:
        return None
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    texts = PHOTO_PROMPTS + ART_PROMPTS
    inp = proc(text=texts, images=img.convert("RGB"), padding=True, return_tensors="pt")
    p = model(**inp).logits_per_image[0].softmax(0)
    return float(p[: len(PHOTO_PROMPTS)].sum())


def is_likely_art(score: float | None, threshold: float = THRESHOLD) -> bool:
    return score is not None and score < threshold


@torch.no_grad()
def subject_scores(images: list, text: str) -> list[float] | None:
    """CLIP similarity of each image to ``text`` (e.g. "a photo of a misty forest"); None if unavailable."""
    model, proc = _load()
    if model is None:
        return None
    out = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        inp = proc(text=[text], images=img.convert("RGB"), padding=True, return_tensors="pt")
        out.append(float(model(**inp).logits_per_image[0, 0]))
    return out
