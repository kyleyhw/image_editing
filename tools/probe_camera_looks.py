"""Camera-look feasibility probe (PROJECT_PLAN Phase 20, task 13).

Question: can a camera maker's rendering ("look") be learned from pairs of
(our raw decode, the camera's embedded JPEG)?

For every raw file under ``--raw_dir``:

1. **Preview.** ``rawpy`` ``extract_thumb()`` returns the embedded JPEG
   LibRaw selects (the largest it finds). Its pixel size is recorded, and its
   IJG quality is estimated by encoding a test image at every quality 1-100
   with Pillow and picking the quality whose luminance quantisation table is
   closest (L1) to the camera's, so no table is typed in by hand.
2. **Pair.** The raw is decoded to linear sRGB with the camera white balance
   and no auto-brightening; it is rotated to the preview's orientation
   (chosen by luminance correlation, as for FiveK), centre-cropped to the
   preview's aspect ratio and resized to the preview's size in linear light.
3. **Global-map fit.** A 3D table of 24^3 cells over the sRGB-encoded decode
   is fitted by averaging the JPEG colour of the pixels that fall in each
   cell, on a random half of the pixels; the other half measures how well a
   global (per-colour, position-independent) map explains the JPEG, as the
   mean CIEDE2000 difference. A small held-out error means the look is a
   global colour transform, the class the slider + residual-table model of
   A6.4 represents. Brightness is matched first by one exposure scale (the
   camera's exposure and our decode differ by a constant factor), which the
   table would otherwise have to absorb.

Writes ``data/manifests/camera_looks.csv`` and prints a summary.

Usage::

    uv run python tools/probe_camera_looks.py --raw_dir data/camera_looks/raw
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import rawpy
from numpy.typing import NDArray
from PIL import Image, JpegImagePlugin
from skimage.color import deltaE_ciede2000, rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fetch_fivek import resize_linear, srgb_decode, srgb_encode  # noqa: E402

RNG_SEED = 13  # any fixed seed: only splits pixels into fit and test halves
CELLS = 24  # table resolution: ~10 code values per cell, fine enough for a smooth look
THUMB = 64  # px, long edge of the luminance thumbnails used for orientation
# Box-downsampling factor of the colour-only evaluation: a few pixels of
# residual misregistration become a fraction of a pixel.
CENTRE_DOWN = 8
EVAL_EDGE = 2048  # px, long edge of the evaluation pair

FloatImage = NDArray[np.float32]


def _luma_table(f: io.BytesIO) -> list[int]:
    """Luminance quantisation table (table 0) of a JPEG stream."""
    im = Image.open(f)
    if not isinstance(im, JpegImagePlugin.JpegImageFile):
        raise ValueError("not a JPEG")
    return list(im.quantization[0])


def estimate_jpeg_quality(jpeg: bytes) -> tuple[int, float]:
    """(IJG quality whose luma table best matches, mean luma quantiser)."""
    q_cam = np.array(_luma_table(io.BytesIO(jpeg)), dtype=np.float64)
    probe = Image.fromarray(np.full((16, 16), 128, np.uint8))
    best = (0, np.inf)
    for q in range(1, 101):
        buf = io.BytesIO()
        probe.save(buf, format="JPEG", quality=q)
        t = np.array(_luma_table(buf), dtype=np.float64)
        d = float(np.abs(t - q_cam).sum())
        if d < best[1]:
            best = (q, d)
    return best[0], float(q_cam.mean())


def _luma(lin: FloatImage, size: tuple[int, int]) -> NDArray[np.float32]:
    y = np.ascontiguousarray(lin @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    t = np.asarray(
        Image.fromarray(y).resize(size, Image.Resampling.BOX), dtype=np.float32
    )
    return (t - t.mean()) / (t.std() + 1e-12)


def crop_to_aspect(img: FloatImage, aspect: float) -> FloatImage:
    """Centre-crop (H, W, 3) to width/height = aspect."""
    h, w = img.shape[:2]
    if w / h > aspect:
        nw = round(h * aspect)
        x0 = (w - nw) // 2
        return img[:, x0 : x0 + nw]
    nh = round(w / aspect)
    y0 = (h - nh) // 2
    return img[y0 : y0 + nh]


def pair(raw_lin: FloatImage, jpeg_lin: FloatImage) -> tuple[FloatImage, int, float]:
    """Rotate, crop and resize the raw decode onto the JPEG; returns (decode, k, r)."""
    jh, jw = jpeg_lin.shape[:2]
    scale = THUMB / max(jh, jw)
    size = (max(2, round(jw * scale)), max(2, round(jh * scale)))
    ref = _luma(jpeg_lin, size).ravel()
    best: tuple[FloatImage, int, float] = (raw_lin, 0, -2.0)
    for k in range(4):
        o = crop_to_aspect(np.rot90(raw_lin[::4, ::4], k), jw / jh)
        r = float(np.mean(ref * _luma(np.ascontiguousarray(o), size).ravel()))
        if r > best[2]:
            best = (raw_lin, k, r)
    k = best[1]
    full = crop_to_aspect(np.rot90(raw_lin, k), jw / jh)
    out = resize_linear(np.ascontiguousarray(full), max(jw, jh))
    if out.shape[:2] != (jh, jw):  # rounding of the aspect crop: resample exactly
        chans = [
            np.asarray(
                Image.fromarray(np.ascontiguousarray(out[..., c])).resize((jw, jh))
            )
            for c in range(3)
        ]
        out = np.stack(chans, -1).astype(np.float32)
    return out, k, best[2]


def fit_global_map(x: FloatImage, y: FloatImage) -> tuple[float, float, float]:
    """Held-out mean and p95 Delta E 2000 of a binned 3D map x -> y, and the
    fraction of test pixels whose cell was seen in the fit half.

    x, y: (N, 3) sRGB-encoded values in [0, 1].
    """
    rng = np.random.default_rng(RNG_SEED)
    fit = rng.random(len(x)) < 0.5
    cell = np.minimum((x * CELLS).astype(np.int64), CELLS - 1)
    key = (cell[:, 0] * CELLS + cell[:, 1]) * CELLS + cell[:, 2]
    n = CELLS**3
    cnt = np.bincount(key[fit], minlength=n).astype(np.float64)
    table = np.stack(
        [np.bincount(key[fit], y[fit, c], minlength=n) for c in range(3)], -1
    ).astype(np.float64)
    seen = cnt > 0
    table[seen] /= cnt[seen, None]
    test = ~fit & seen[key]
    pred = table[key[test]]
    de = deltaE_ciede2000(rgb2lab(pred[None]), rgb2lab(y[test][None]))[0]
    return (
        float(de.mean()),
        float(np.percentile(de, 95)),
        float(test.sum() / (~fit).sum()),
    )


def probe(path: Path) -> dict[str, object]:
    with rawpy.imread(str(path)) as raw:
        thumb = raw.extract_thumb()
        raw_size = (raw.sizes.width, raw.sizes.height)  # before half_size decoding
        # half_size: 2x2 photosite binning instead of demosaicing; quarter
        # the memory, and still above the 2048 px evaluation size below
        lin16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
            half_size=True,
        )
    row: dict[str, object] = {
        "file": path.name,
        "raw_w": raw_size[0],
        "raw_h": raw_size[1],
    }
    if thumb.format != rawpy.ThumbFormat.JPEG:
        row["preview"] = "bitmap"
        return row
    jpeg = bytes(thumb.data)
    im = Image.open(io.BytesIO(jpeg))
    q, qmean = estimate_jpeg_quality(jpeg)
    row.update(
        preview_w=im.width,
        preview_h=im.height,
        preview_kb=round(len(jpeg) / 1024),
        jpeg_quality=q,
        luma_quant_mean=round(qmean, 2),
    )
    row["subsampling"] = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(
        JpegImagePlugin.get_sampling(im), "?"
    )
    # Evaluate at <= EVAL_EDGE px: a 24 MP preview would need ~5 GB of
    # float copies, and the conclusions concern 512 px training proxies.
    jpeg_lin = resize_linear(
        srgb_decode(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0), EVAL_EDGE
    )
    jpeg_code = srgb_encode(jpeg_lin)
    dec, k, r = pair(lin16.astype(np.float32) / 65535.0, jpeg_lin)
    # one exposure scale: match median luminance (the camera's metering and
    # our unscaled decode differ by a constant factor)
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    gain = float(np.median(jpeg_lin @ w) / max(np.median(dec @ w), 1e-6))
    x = srgb_encode(np.clip(dec * gain, 0, 1)).reshape(-1, 3)
    y = jpeg_code.reshape(-1, 3)
    de_id = deltaE_ciede2000(rgb2lab(x[None]), rgb2lab(y[None]))[0]
    m, p95, cov = fit_global_map(x, y)
    # Colour-only variant: the camera corrects lens distortion and vignetting
    # in its JPEG, our decode does not, so pixels drift apart towards the
    # edges. The central half of the frame, box-downsampled by CENTRE_DOWN,
    # keeps registration error below a pixel and vignetting small.
    cx, cy = _centre_lowres(dec * gain), _centre_lowres(jpeg_lin)
    xc, yc = (
        srgb_encode(np.clip(cx, 0, 1)).reshape(-1, 3),
        srgb_encode(cy).reshape(-1, 3),
    )
    mc, p95c, _ = fit_global_map(xc, yc)
    row.update(
        rot90=k,
        align_r=round(r, 3),
        exposure_gain=round(gain, 3),
        de00_decode_vs_jpeg=round(float(de_id.mean()), 2),
        de00_global_map_heldout=round(m, 2),
        de00_global_map_p95=round(p95, 2),
        heldout_coverage=round(cov, 3),
        de00_centre_lowres_heldout=round(mc, 2),
        de00_centre_lowres_p95=round(p95c, 2),
    )
    return row


def _centre_lowres(lin: FloatImage) -> FloatImage:
    """Central half (each axis) of a linear image, box-downsampled."""
    h, w = lin.shape[:2]
    c = lin[h // 4 : h - h // 4, w // 4 : w - w // 4]
    hh, ww = (
        (c.shape[0] // CENTRE_DOWN) * CENTRE_DOWN,
        (c.shape[1] // CENTRE_DOWN) * CENTRE_DOWN,
    )
    c = c[:hh, :ww].reshape(
        hh // CENTRE_DOWN, CENTRE_DOWN, ww // CENTRE_DOWN, CENTRE_DOWN, 3
    )
    return c.mean(axis=(1, 3)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw_dir", default="data/camera_looks/raw")
    ap.add_argument("--manifest", default="data/manifests/camera_looks.csv")
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.raw_dir).iterdir()):
        if p.is_file() and p.stat().st_size > 0:
            row = probe(p)
            print(row, flush=True)
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.manifest, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
