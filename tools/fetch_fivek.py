"""Fetch the whole MIT-Adobe FiveK dataset (all splits, experts A-E) as
512 px sRGB proxies, plus each expert's Lightroom settings (XMP).

Source
------
HuggingFace ``logasja/mit-adobe-fivek``: five configurations ``a`` ... ``e``
(one per expert), splits ``train`` (3500), ``test`` (1000), ``validation``
(500), stored as parquet shards of lossless WebP images. Each row holds

- ``original``: the raw file rendered by LibRaw with the camera white
  balance (``rawpy.postprocess(use_camera_wb=True)``: sRGB primaries,
  BT.709 tone curve, auto-brightness, 8 bit, the full sensor area without the
  DNG default crop). This was established by decoding the MIT DNG of one row
  (``a4340``) and comparing: mean absolute difference 0.22 code values.
- ``augmented``: the expert's Lightroom export, 8 bit, **ProPhoto RGB**
  (embedded ICC profile, gamma 1.8), cropped to the DNG default crop. Its
  XMP packet holds the expert's slider settings (``crs:*``).

Processing per row
------------------
1. Decode to linear light in the image's own primaries. Original (no ICC
   profile, sRGB): inverse sRGB transfer function. Expert (ProPhoto RGB):
   ``v ** gamma`` with the gamma read from the profile's ``rTRC`` tag.
2. Original only: centre-crop to the expert's pixel size, so the pair is
   aligned (the DNG crop margins of the probed file were 14 px and 10 px,
   i.e. centred to within 1 px).
3. Resize to ``--max_edge`` on the long side in **linear light** (resampling
   averages radiance, which is what a smaller sensor would record).
4. Re-encode with the same transfer function and primaries, quantise to
   8 bit and save as lossless PNG; the expert keeps its ICC profile. The
   expert is *not* converted to sRGB: 7-43 % of its pixels lie outside the
   sRGB gamut, and clipping them shifts colour by a mean of up to 3.5
   Delta E 2000 per image. ``decode_linear_srgb`` converts on load, with
   the 3x3 matrix ``M_sRGB^-1 @ M_ProPhoto`` built from the profiles' own
   D50-adapted colorant tags (the ICC matrix-shaper path, relative
   colorimetric intent); negative values mark out-of-gamut colours.
5. Record the out-of-sRGB-gamut pixel fraction and save the expert's XMP
   packet (its Lightroom settings) as ``img_NNNN.xmp``.

Rows are keyed by their position in config ``c`` (the canonical order); the
other configs are matched by FiveK id (``a0001`` ...), so the layout is
correct whether or not the configs share a row order. Originals are saved
once, from config ``c``.

Output::

    data/fivek/{split}/original/img_NNNN.png     (sRGB)
    data/fivek/{split}/expert_X/img_NNNN.png     (ProPhoto RGB, ICC embedded)
    data/fivek/{split}/expert_X/img_NNNN.xmp
    data/fivek/.state/...            (index and per-shard records)
    data/manifests/fivek.csv          (one row per photo; no images)

Resumable and idempotent: outputs are written atomically and finished shards
are marked; re-running the same command continues where it stopped.

Usage::

    uv run python tools/fetch_fivek.py --experts c,a,b,d,e --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mmap
import os
import struct
import sys
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem
from numpy.typing import NDArray
from PIL import Image, ImageCms

REPO = "datasets/logasja/mit-adobe-fivek"
RESOLVE = "https://huggingface.co/datasets/logasja/mit-adobe-fivek/resolve/main"
SPLITS: tuple[str, ...] = ("train", "test", "validation")
CANONICAL = "c"  # expert C is the standard target in the literature
LABEL_COLUMNS: tuple[str, ...] = ("license", "location", "time", "light", "subject")
# Range-request piece size and parallel connections per shard. Measured in
# this container: one connection gives 14-23 MB/s, four give ~80 MB/s in
# total; 32 MiB pieces keep the per-request overhead (a redirect plus TLS)
# below ~5 % of the transfer time.
PIECE = 32 * 2**20
CONNECTIONS = 4
# Pillow's reducing_gap: box-reduce by an integer factor first, then Lanczos
# over the last factor <= 3. Pillow documents 3.0 as indistinguishable from
# a full Lanczos resample, at a fraction of the cost.
REDUCING_GAP = 3.0
# Tolerance for the out-of-gamut count: 8-bit quantisation alone moves a
# linear value by up to ~1e-3 near white, which is not a gamut excursion.
GAMUT_EPS = 1e-3
# The gamut statistic is taken on every 4th pixel in each axis: ~0.6 M of a
# 10 M pixel image, a sampling error far below the reported precision, at 1/16
# of the cost.
OOG_STRIDE = 4
# Aspect-ratio tolerance (1 %) for originals smaller than the expert frame:
# DNG default crops change the aspect by < 0.5 %, a wrong rotation by ~50 %.
ASPECT_TOL = 0.01
# Orientation search: work on every 4th pixel and compare 64 px thumbnails;
# 64 px resolves the rotation of any photo with structure, and the cost is
# negligible next to decoding.
ALIGN_STRIDE = 4
ALIGN_THUMB = 64

FloatImage = NDArray[np.float32]


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def srgb_decode(v: FloatImage) -> FloatImage:
    """sRGB code values in [0, 1] -> linear light (IEC 61966-2-1)."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def srgb_encode(lin: FloatImage) -> FloatImage:
    """Linear light in [0, 1] -> sRGB code values in [0, 1]."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(
        lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055
    ).astype(np.float32)


def _colorant_matrix(profile: Any) -> NDArray[np.float64]:
    """Columns = D50-adapted XYZ of the R, G, B primaries (ICC rXYZ/gXYZ/bXYZ)."""
    cols = [
        profile.red_colorant[0],
        profile.green_colorant[0],
        profile.blue_colorant[0],
    ]
    return np.array(cols, dtype=np.float64).T


def _icc_gamma(icc: bytes) -> float:
    """Gamma of a matrix-shaper profile whose TRC is a single-value ``curv``."""
    (n_tags,) = struct.unpack(">I", icc[128:132])
    for i in range(n_tags):
        sig, off, _size = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        if sig == b"rTRC":
            kind, count = (
                icc[off : off + 4],
                struct.unpack(">I", icc[off + 8 : off + 12])[0],
            )
            if kind != b"curv" or count != 1:
                raise ValueError(f"unsupported TRC {kind!r} with {count} entries")
            # u8Fixed8Number: gamma = raw / 256
            return struct.unpack(">H", icc[off + 12 : off + 14])[0] / 256.0
    raise ValueError("profile has no rTRC tag")


_SRGB_M: NDArray[np.float64] = _colorant_matrix(ImageCms.createProfile("sRGB"))
_MATRIX_CACHE: dict[str, tuple[float, NDArray[np.float32]]] = {}


def profile_to_srgb_linear(icc: bytes) -> tuple[float, NDArray[np.float32]]:
    """(gamma, 3x3 matrix) taking the profile's linear RGB to linear sRGB."""
    key = hashlib.sha1(icc).hexdigest()
    if key not in _MATRIX_CACHE:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile
        m = np.linalg.solve(_SRGB_M, _colorant_matrix(prof)).astype(np.float32)
        _MATRIX_CACHE[key] = (_icc_gamma(icc), m)
    return _MATRIX_CACHE[key]


@dataclass
class Decoded:
    """Linear-light image in its own primaries, with its encoding."""

    lin: FloatImage  # (H, W, 3) float32
    icc: bytes  # empty for sRGB
    gamma: float  # 0.0 for the sRGB transfer function
    xmp: bytes


def decode_native(data: bytes) -> Decoded:
    """Decode to linear light without changing primaries (no gamut loss)."""
    im = Image.open(io.BytesIO(data))
    icc: bytes = im.info.get("icc_profile") or b""
    xmp = im.info.get("xmp", b"")
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    code = np.asarray(im.convert("RGB"))
    # 8-bit input has 256 possible values, so a lookup table gives the exact
    # transfer function at a fraction of the cost of evaluating it per pixel.
    levels = np.arange(256, dtype=np.float32) / 255.0
    if not icc:
        return Decoded(srgb_decode(levels)[code], b"", 0.0, xmp)
    gamma = _icc_gamma(icc)
    return Decoded(np.power(levels, gamma, dtype=np.float32)[code], icc, gamma, xmp)


def to_linear_srgb(d: Decoded) -> FloatImage:
    """Linear sRGB primaries, unclipped (negative or > 1 = out of gamut)."""
    if not d.icc:
        return d.lin
    _, m = profile_to_srgb_linear(d.icc)
    return (d.lin @ m.T).astype(np.float32)


def decode_linear_srgb(data: bytes, clip: bool = True) -> FloatImage:
    """Loader for the saved proxies: any file -> linear sRGB (H, W, 3)."""
    lin = to_linear_srgb(decode_native(data))
    return np.clip(lin, 0.0, 1.0) if clip else lin


def oog_fraction(d: Decoded, stride: int = OOG_STRIDE) -> float:
    """Fraction of pixels with a linear sRGB channel outside [0, 1],
    estimated on every ``stride``-th full-resolution pixel in each axis."""
    lin = to_linear_srgb(Decoded(d.lin[::stride, ::stride], d.icc, d.gamma, d.xmp))
    bad = (lin.min(-1) < -GAMUT_EPS) | (lin.max(-1) > 1 + GAMUT_EPS)
    return float(np.mean(bad))


def centre_crop(img: FloatImage, w: int, h: int) -> FloatImage:
    """Centre-crop (H, W, 3) to (h, w); no-op if it is already smaller."""
    hh, ww = img.shape[:2]
    if ww < w or hh < h:
        return img
    top, left = (hh - h) // 2, (ww - w) // 2
    return img[top : top + h, left : left + w]


def fit_frame(img: FloatImage, w: int, h: int) -> FloatImage | None:
    """Bring (H, W, 3) onto a w x h frame by a centred crop.

    If the image contains the frame, crop to it exactly. Otherwise (a few
    originals are stored at lower resolution than the expert's export) it
    is used whole if its aspect ratio matches the frame's within ASPECT_TOL,
    and the resize does the scaling. Returns None if neither holds.
    """
    hh, ww = img.shape[:2]
    if ww >= w and hh >= h:
        return centre_crop(img, w, h)
    if abs((ww / hh) / (w / h) - 1.0) > ASPECT_TOL:
        return None
    return img


def _luma_thumb(lin: FloatImage, size: tuple[int, int]) -> NDArray[np.float32]:
    """Luminance thumbnail (Rec. 709 weights; exact primaries do not matter
    for the correlation it is used in) resized to ``size`` = (w, h)."""
    y = np.ascontiguousarray(lin @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    return np.asarray(
        Image.fromarray(y).resize(size, Image.Resampling.BOX), dtype=np.float32
    )


def _thumb_size(ew: int, eh: int) -> tuple[int, int]:
    scale = ALIGN_THUMB / max(eh, ew)
    return max(2, round(ew * scale)), max(2, round(eh * scale))


def expert_thumb(exp: FloatImage) -> NDArray[np.float32]:
    """Standardised luminance thumbnail of the expert, for orientation."""
    eh, ew = exp.shape[:2]
    s = ALIGN_STRIDE
    ref = _luma_thumb(np.ascontiguousarray(exp[::s, ::s]), _thumb_size(ew, eh)).ravel()
    return (ref - ref.mean()) / (ref.std() + 1e-12)


def best_orientation(
    org: FloatImage, ref: NDArray[np.float32], ew: int, eh: int
) -> tuple[int, float]:
    """Rotation k (``np.rot90(org, k)``) that best matches the expert.

    The HF originals ignore the camera's orientation tag while the expert
    exports apply it. For every k whose rotated original can be brought onto
    the expert's frame (``fit_frame``), the original and the expert are
    reduced to 64 px luminance thumbnails and compared by Pearson
    correlation (invariant to the expert's affine tone changes). Returns
    (k, r); a low r flags a misaligned pair. ``ref`` is ``expert_thumb``
    of the expert (``ew`` x ``eh`` px).
    """
    s = ALIGN_STRIDE
    size = _thumb_size(ew, eh)
    small = org[::s, ::s]
    best = (0, -2.0)
    for k in range(4):
        o = fit_frame(np.rot90(small, k), ew // s, eh // s)
        if o is None:
            continue  # this rotation cannot hold the expert's frame
        t = _luma_thumb(np.ascontiguousarray(o), size).ravel()
        r = float(np.mean(ref * (t - t.mean()) / (t.std() + 1e-12)))
        if r > best[1]:
            best = (k, r)
    return best


def proxy_size(w: int, h: int, max_edge: int) -> tuple[int, int]:
    """(w, h) of the proxy of a w x h image: long edge <= max_edge."""
    scale = min(1.0, max_edge / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def resize_linear(
    lin: FloatImage, max_edge: int, size: tuple[int, int] | None = None
) -> FloatImage:
    """Resize linear-light (H, W, 3) so the long edge is <= max_edge, or to
    exactly ``size`` = (w, h) if given."""
    h, w = lin.shape[:2]
    size = size or proxy_size(w, h, max_edge)
    if size == (w, h):
        return lin
    chans = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(lin[..., k])).resize(
                size, Image.Resampling.LANCZOS, reducing_gap=REDUCING_GAP
            ),
            dtype=np.float32,
        )
        for k in range(3)
    ]
    return np.clip(np.stack(chans, -1), 0.0, 1.0)


def save_native(lin: FloatImage, enc: Decoded, path: Path) -> None:
    """Encode linear light with ``enc``'s transfer function; write a PNG
    (with ``enc``'s ICC profile, if any) atomically."""
    lin = np.clip(lin, 0.0, 1.0)
    v = srgb_encode(lin) if not enc.icc else np.power(lin, 1.0 / enc.gamma)
    im = Image.fromarray(np.round(v * 255.0).astype(np.uint8), "RGB")
    tmp = path.with_suffix(path.suffix + ".tmp")
    kw: dict[str, Any] = {"icc_profile": enc.icc} if enc.icc else {}
    # compress_level 6: zlib's default; level 9 saves ~2 % for ~3x the time.
    im.save(tmp, format="PNG", compress_level=6, **kw)
    os.replace(tmp, path)


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass
class ShardInfo:
    config: str
    split: str
    path: str
    ids: list[str]
    labels: dict[str, list[int]]


def _fivek_id(webp_name: str) -> str:
    """'a3660-jmac_MG_8044.webp' -> 'a3660'."""
    return webp_name.split("-", 1)[0]


def build_index(config: str, state: Path) -> list[ShardInfo]:
    """Ids and labels for every shard of a config (cached as JSON)."""
    cache = state / f"index_{config}.json"
    if cache.exists():
        return [ShardInfo(**d) for d in json.loads(cache.read_text())]
    fs = HfFileSystem()
    shards: list[ShardInfo] = []
    for split in SPLITS:
        paths = sorted(
            p for p in fs.ls(f"{REPO}/{config}", detail=False) if f"/{split}-" in p
        )
        for p in paths:
            with fs.open(p, block_size=2**20) as f:
                pf = pq.ParquetFile(f)
                t = pf.read(columns=["original.path", *LABEL_COLUMNS])
            ids = [_fivek_id(d["path"]) for d in t.column("original").to_pylist()]
            labels = {
                c: [int(x) for x in t.column(c).to_pylist()] for c in LABEL_COLUMNS
            }
            shards.append(ShardInfo(config, split, p, ids, labels))
            print(
                f"  index {config}/{split} {Path(p).name}: {len(ids)} rows", flush=True
            )
    cache.write_text(json.dumps([asdict(s) for s in shards]))
    return shards


def label_names() -> dict[str, list[str]]:
    """Class-label names from the dataset's HF feature metadata."""
    fs = HfFileSystem()
    p = sorted(fs.ls(f"{REPO}/{CANONICAL}", detail=False))[0]
    with fs.open(p, block_size=2**20) as f:
        meta = pq.ParquetFile(f).schema_arrow.metadata[b"huggingface"]
    feats = json.loads(meta)["info"]["features"]
    return {c: list(feats[c]["names"]) for c in LABEL_COLUMNS}


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------


def _get_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """GET bytes [start, end) with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(
                url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120
            )
            r.raise_for_status()
            if len(r.content) != end - start:
                raise OSError(f"short read {len(r.content)} != {end - start}")
            return r.content
        except (requests.RequestException, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_sparse(repo_path: str, local: Path, columns: list[str]) -> None:
    """Download only the footer and the named columns' chunks of a parquet
    file into a sparse local file at their original offsets."""
    url = f"{RESOLVE}/{repo_path.split('/', 3)[3]}"
    size = HfFileSystem().info(repo_path)["size"]
    tail = _get_range(url, size - 8, size)
    footer_len = struct.unpack("<I", tail[:4])[0]
    fstart = size - 8 - footer_len
    with local.open("wb") as f:
        f.truncate(size)
        f.seek(fstart)
        f.write(_get_range(url, fstart, size))
    md = pq.read_metadata(local)
    ranges: list[tuple[int, int]] = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema.split(".")[0] in columns:
                start = c.dictionary_page_offset or c.data_page_offset
                ranges.append((start, start + c.total_compressed_size))
    pieces = [(a, min(a + PIECE, b)) for lo, b in ranges for a in range(lo, b, PIECE)]
    fd = os.open(local, os.O_WRONLY)

    def fetch_piece(piece: tuple[int, int]) -> None:
        # each thread writes its own byte range, so at most CONNECTIONS
        # pieces are held in memory
        os.pwrite(fd, _get_range(url, *piece), piece[0])

    try:
        with ThreadPoolExecutor(CONNECTIONS) as ex:
            list(ex.map(fetch_piece, pieces))
    finally:
        os.close(fd)


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    """Unsigned LEB128 varint (Thrift compact protocol)."""
    out = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, pos
        shift += 7


def _thrift_value(buf: memoryview, pos: int, t: int) -> tuple[Any, int]:
    """One Thrift compact-protocol value of type ``t`` starting at ``pos``."""
    if t in (1, 2):  # bool, carried in the field header
        return t == 1, pos
    if t == 3:  # byte
        return buf[pos], pos + 1
    if t in (4, 5, 6):  # i16, i32, i64: zigzag varint
        v, pos = _varint(buf, pos)
        return (v >> 1) ^ -(v & 1), pos
    if t == 7:  # double
        return None, pos + 8
    if t == 8:  # binary
        n, pos = _varint(buf, pos)
        return bytes(buf[pos : pos + n]), pos + n
    if t in (9, 10):  # list, set
        head = buf[pos]
        pos += 1
        size, et = head >> 4, head & 0x0F
        if size == 15:
            size, pos = _varint(buf, pos)
        items = []
        for _ in range(size):
            if et in (1, 2):  # bools inside a list take one byte each
                items.append(buf[pos] == 1)
                pos += 1
            else:
                v, pos = _thrift_value(buf, pos, et)
                items.append(v)
        return items, pos
    if t == 11:  # map
        size, pos = _varint(buf, pos)
        if size == 0:
            return {}, pos
        kv = buf[pos]
        pos += 1
        out: dict[Any, Any] = {}
        for _ in range(size):
            k, pos = _thrift_value(buf, pos, kv >> 4)
            out[k], pos = _thrift_value(buf, pos, kv & 0x0F)
        return out, pos
    if t == 12:  # struct
        return _thrift_struct(buf, pos)
    raise ValueError(f"unknown Thrift compact type {t}")


def _thrift_struct(buf: memoryview, pos: int) -> tuple[dict[int, Any], int]:
    """A Thrift compact-protocol struct as {field id: value}."""
    fields: dict[int, Any] = {}
    last = 0
    while True:
        head = buf[pos]
        pos += 1
        if head == 0:  # STOP
            return fields, pos
        delta, t = head >> 4, head & 0x0F
        if delta:
            fid = last + delta
        else:
            v, pos = _varint(buf, pos)
            fid = (v >> 1) ^ -(v & 1)
        last = fid
        fields[fid], pos = _thrift_value(buf, pos, t)


# parquet.thrift: PageType.DICTIONARY_PAGE; PageHeader fields 2, 3, 7
_DICTIONARY_PAGE = 2


def _iter_dictionary_page(local: Path, column: str) -> Any | None:
    """Fast path: when a column chunk is one dictionary page holding one
    distinct value per row (how these shards were written), Snappy-
    decompress that page from a memory map and walk its PLAIN values
    (``[uint32 length][bytes]``). Dictionary order is first-occurrence
    order, which for distinct values is row order. Peak memory is one
    decompressed page (~1 GB) instead of pyarrow's ~3 GB. Returns None when
    the layout differs, so the caller falls back to pyarrow."""
    md = pq.read_metadata(local)
    if md.num_row_groups != 1:
        return None
    rg = md.row_group(0)
    col = next(
        (
            rg.column(j)
            for j in range(rg.num_columns)
            if rg.column(j).path_in_schema == f"{column}.bytes"
        ),
        None,
    )
    if (
        col is None
        or col.dictionary_page_offset is None
        or col.compression not in ("SNAPPY", "UNCOMPRESSED")
    ):
        return None
    with local.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    view = memoryview(mm)
    header, start = _thrift_struct(view, col.dictionary_page_offset)
    if header.get(1) != _DICTIONARY_PAGE or header.get(7, {}).get(1) != rg.num_rows:
        view.release()
        mm.close()
        return None
    comp = pa.py_buffer(view[start : start + header[3]])
    page = (
        comp
        if col.compression == "UNCOMPRESSED"
        else pa.decompress(comp, header[2], codec="snappy")
    )
    del comp
    view.release()
    mm.close()

    def gen() -> Any:
        data = memoryview(page)
        pos = 0
        for _ in range(rg.num_rows):
            n = int.from_bytes(data[pos : pos + 4], "little")
            yield bytes(data[pos + 4 : pos + 4 + n])
            pos += 4 + n
        if pos != len(data):
            raise RuntimeError(
                f"{local}: dictionary page has {len(data) - pos} trailing bytes"
            )

    return gen()


def _iter_column(local: Path, column: str) -> Any:
    """Yield the encoded image bytes of one column, row by row. Each column
    chunk is a single ~1 GB dictionary page; the fast path reads it directly,
    the fallback (pyarrow) loads it whole, so columns are read one at a
    time."""
    fast = _iter_dictionary_page(local, column)
    if fast is not None:
        yield from fast
        pa.default_memory_pool().release_unused()
        return
    pf = pq.ParquetFile(local, pre_buffer=False, buffer_size=16 * 2**20)
    for batch in pf.iter_batches(batch_size=1, columns=[column]):
        for i in range(batch.num_rows):
            yield batch.column(column)[i].as_py()["bytes"]
    del pf
    pa.default_memory_pool().release_unused()


def process_shard(
    shard: dict[str, Any],
    canon_idx: list[int],
    out_root: str,
    max_edge: int,
    done_path: str,
) -> tuple[str, int, int]:
    """Download one shard and write its proxies. Returns (shard, saved, skipped).

    Pass 1 reads the expert column; pass 2 (config c only) reads the
    original column and crops each original to its expert's size.
    """
    config, split = shard["config"], shard["split"]
    with_original = config == CANONICAL
    root = Path(out_root) / split
    odir, edir = root / "original", root / f"expert_{config}"
    odir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)

    def paths(idx: int) -> tuple[Path, Path, Path, Path]:
        name = f"img_{idx:04d}"
        return (
            odir / f"{name}.png",
            edir / f"{name}.png",
            edir / f"{name}.xmp",
            edir / f"{name}.json",
        )

    def finished(idx: int) -> bool:
        # the .json record is written last, so its presence marks a complete row
        o, e, x, r = paths(idx)
        return (
            r.exists()
            and e.exists()
            and x.exists()
            and (not with_original or o.exists())
        )

    saved = 0
    if not all(finished(i) for i in canon_idx):
        tmp_dir = Path(out_root) / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local = tmp_dir / f"{config}_{Path(shard['path']).name}"
        columns = ["augmented", *(["original"] if with_original else [])]
        todo = {i for i in canon_idx if not finished(i)}
        recs: dict[int, dict[str, Any]] = {}
        exp_thumbs: dict[int, NDArray[np.float32]] = {}  # for orientation
        try:
            fetch_sparse(shard["path"], local, columns)
            n = 0
            for row, data in enumerate(_iter_column(local, "augmented")):
                n += 1
                idx = canon_idx[row]
                if idx not in todo:
                    continue
                exp = decode_native(data)
                eh, ew = exp.lin.shape[:2]
                _, e_path, x_path, _ = paths(idx)
                save_native(resize_linear(exp.lin, max_edge), exp, e_path)
                write_atomic(x_path, exp.xmp)
                if with_original:
                    exp_thumbs[idx] = expert_thumb(exp.lin)
                recs[idx] = {
                    "split": split,
                    "idx": idx,
                    "fivek_id": shard["ids"][row],
                    "expert": config,
                    "w": ew,
                    "h": eh,
                    "oog": round(oog_fraction(exp), 6),
                    "has_crop": b'crs:HasCrop="True"' in exp.xmp
                    or b"<crs:HasCrop>True" in exp.xmp,
                }
            if n != len(canon_idx):
                raise RuntimeError(
                    f"{shard['path']}: {n} rows, index has {len(canon_idx)}"
                )
            if with_original:
                for row, data in enumerate(_iter_column(local, "original")):
                    idx = canon_idx[row]
                    if idx not in todo:
                        continue
                    org = decode_native(data)
                    rec = recs[idx]
                    rec["orig_w"], rec["orig_h"] = org.lin.shape[1], org.lin.shape[0]
                    k, r = best_orientation(
                        org.lin, exp_thumbs.pop(idx), rec["w"], rec["h"]
                    )
                    framed = fit_frame(np.rot90(org.lin, k), rec["w"], rec["h"])
                    lin = np.rot90(org.lin, k) if framed is None else framed
                    rec["rot90"], rec["align_r"] = k, round(r, 4)
                    rec["aligned"] = framed is not None
                    rec["orig_smaller"] = (
                        framed is not None and framed.shape[0] < rec["h"]
                    )
                    # resized to exactly the expert proxy's size, so pairs match
                    size = proxy_size(rec["w"], rec["h"], max_edge)
                    out = resize_linear(np.ascontiguousarray(lin), max_edge, size)
                    save_native(out, org, paths(idx)[0])
            for idx, rec in recs.items():
                write_atomic(paths(idx)[3], json.dumps(rec).encode())
            saved = len(recs)
        finally:
            local.unlink(missing_ok=True)
    records = [json.loads(paths(i)[3].read_text()) for i in canon_idx]
    write_atomic(Path(done_path), json.dumps(records).encode())
    return shard["path"], saved, len(records) - saved


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def alignment_report(
    indexes: dict[str, list[ShardInfo]],
) -> dict[str, dict[str, float]]:
    """Per config and split: fraction of rows whose id matches config c's row."""
    out: dict[str, dict[str, float]] = {}
    for cfg, shards in indexes.items():
        out[cfg] = {}
        for split in SPLITS:
            a = [i for s in shards if s.split == split for i in s.ids]
            c = [i for s in indexes[CANONICAL] if s.split == split for i in s.ids]
            same_set = sorted(a) == sorted(c)
            match = sum(x == y for x, y in zip(a, c, strict=False)) / max(1, len(c))
            out[cfg][split] = match if same_set else -1.0
    return out


def write_manifest(
    state: Path, indexes: dict[str, list[ShardInfo]], manifest: Path
) -> int:
    """One CSV row per photo with labels, sizes and per-expert records."""
    names = label_names()
    recs: dict[tuple[str, int], dict[str, Any]] = {}
    for done in sorted(state.glob("done_*.json")):
        for r in json.loads(done.read_text()):
            recs.setdefault((r["split"], r["idx"]), {})[r["expert"]] = r
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        idx = 0
        for s in indexes[CANONICAL]:
            if s.split != split:
                continue
            for j, fid in enumerate(s.ids):
                r = recs.get((split, idx), {})
                c = r.get(CANONICAL, {})
                row: dict[str, Any] = {
                    "split": split,
                    "idx": idx,
                    "fivek_id": fid,
                    "original": f"{split}/original/img_{idx:04d}.png" if c else "",
                    "orig_w": c.get("orig_w", ""),
                    "orig_h": c.get("orig_h", ""),
                }
                for col in LABEL_COLUMNS:
                    row[col] = names[col][s.labels[col][j]]
                for x in "abcde":
                    e = r.get(x, {})
                    row[f"{x}_w"], row[f"{x}_h"] = e.get("w", ""), e.get("h", "")
                    row[f"{x}_crop"] = int(e["has_crop"]) if e else ""
                    row[f"{x}_oog"] = e.get("oog", "")
                rows.append(row)
                idx += 1
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--experts", default="c,a,b,d,e", help="order of configs; c must come first"
    )
    ap.add_argument("--out", default="data/fivek")
    ap.add_argument("--manifest", default="data/manifests/fivek.csv")
    ap.add_argument("--max_edge", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument(
        "--limit_shards", type=int, default=0, help="debug: shards per config"
    )
    ap.add_argument("--manifest_only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    state = out / ".state"
    state.mkdir(parents=True, exist_ok=True)
    for stale in (out / ".tmp").glob("*.parquet"):  # partial downloads of a killed run
        stale.unlink()
    experts = args.experts.split(",")
    t0 = time.time()

    indexes = {
        cfg: build_index(cfg, state) for cfg in dict.fromkeys([CANONICAL, *experts])
    }
    align = alignment_report(indexes)
    (state / "alignment.json").write_text(json.dumps(align, indent=1))
    print(
        "row alignment with config c (1.0 = same order; -1 = different id set):",
        align,
        flush=True,
    )

    # canonical index per split: fivek id -> position in config c
    canon: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        ids = [i for s in indexes[CANONICAL] if s.split == split for i in s.ids]
        canon[split] = {fid: k for k, fid in enumerate(ids)}

    if not args.manifest_only:
        for cfg in experts:
            tasks = (
                indexes[cfg][: args.limit_shards] if args.limit_shards else indexes[cfg]
            )
            for attempt in range(1, args.retries + 1):
                todo = [
                    s
                    for s in tasks
                    if not (state / f"done_{cfg}_{Path(s.path).stem}.json").exists()
                ]
                if not todo:
                    break
                print(
                    f"[{cfg}] attempt {attempt}: {len(todo)} shard(s) to do", flush=True
                )
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs: dict[Future[tuple[str, int, int]], str] = {}
                    for s in todo:
                        cidx = [canon[s.split][fid] for fid in s.ids]
                        done = str(state / f"done_{cfg}_{Path(s.path).stem}.json")
                        futs[
                            ex.submit(
                                process_shard,
                                asdict(s),
                                cidx,
                                str(out),
                                args.max_edge,
                                done,
                            )
                        ] = s.path
                    for fut in as_completed(futs):
                        try:
                            p, saved, skipped = fut.result()
                            print(
                                f"  [{cfg}] {Path(p).name}: saved {saved}, skipped {skipped} "
                                f"({time.time() - t0:.0f} s)",
                                flush=True,
                            )
                        except Exception as exc:  # network errors: retried next attempt
                            print(
                                f"  [{cfg}] {Path(futs[fut]).name} FAILED: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
            n = write_manifest(state, indexes, Path(args.manifest))
            print(
                f"[{cfg}] finished; manifest rows {n} ({time.time() - t0:.0f} s)",
                flush=True,
            )
    n = write_manifest(state, indexes, Path(args.manifest))
    print(
        f"done: manifest {args.manifest} with {n} rows; {time.time() - t0:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
