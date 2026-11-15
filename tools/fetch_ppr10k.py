"""Phase 20 task 11: PPR10K 360p sources and the three experts' retouches, as small proxies.

The four official zips (Google Drive; sources 12.1 GB, each expert 4.6 GB) do not fit the disk
allowance together, so they are never downloaded whole. Each zip is read in place over HTTP range
requests (``RangeFile``: a seekable file object with a read-ahead buffer, which is all
``zipfile`` needs), and each member is decoded to linear light (every file checked carries the standard sRGB IEC61966-2.1
profile: sources 16-bit, retouches 8-bit; any other profile raises an error rather than being
misread), resized in linear light to EDGE px on the long side and saved as an 8-bit sRGB PNG, like the FiveK
proxies. A file list per zip records progress, so a rerun resumes.

`--split` then lays the proxies out like data/fivek (split/original, split/expert_X, as symbolic
links) using the official split: the first 8,875 files for training and the last 2,286 for
testing, in file order (group, then index within the group; README of csjliang/PPR10K).

    uv run python tools/fetch_ppr10k.py [--only source a b c] [--list] [--split]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import tifffile
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.fetch_fivek import srgb_decode, srgb_encode  # noqa: E402

OUT = ROOT / "data/ppr10k"
EDGE = 270          # half the 540 px of the 360p release: above the 128-256 px used for training,
                    # and 44,000 proxies fit the disk allowance (about 4.5 GB as PNG)
BLOCK = 16 << 20    # read-ahead per range request; members are read in order, so most reads hit it
ZIPS = {"source": "source.zip", "a": "target_a.zip", "b": "target_b.zip", "c": "target_c.zip"}


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched in BLOCK-sized range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.s = url, 0, requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.buf_start, self.buf = 0, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start: int, n: int) -> bytes:
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                raise OSError(f"HTTP {r.status_code}")
            except (OSError, requests.RequestException) as e:
                print(f"  retry {attempt + 1} at {start}: {e}", flush=True)
                time.sleep(2 ** attempt)
        raise OSError(f"range {start}+{n} failed")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            self.buf_start = self.pos
            self.buf = self._fetch(self.pos, min(max(n, BLOCK), self.size - self.pos))
        out = self.buf[self.pos - self.buf_start:end - self.buf_start]
        self.pos = end
        return out


def urls() -> dict[str, str]:
    rows = list(csv.DictReader((ROOT / "data/manifests/ppr10k_sources.csv").open()))
    return {r["file"]: r["download_url"] for r in rows}


def decode(data: bytes) -> np.ndarray:
    """A PPR10K TIFF -> (H, W, 3) linear sRGB, float32."""
    with tifffile.TiffFile(io.BytesIO(data)) as t:
        page = t.pages.first
        a = page.asarray()
        icc = page.tags.get(34675)                       # InterColorProfile
    if icc is not None:
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(bytes(icc.value)))).strip()
        if desc != "sRGB IEC61966-2.1":
            raise ValueError(f"unexpected colour profile {desc!r}")
    scale = float(np.iinfo(a.dtype).max)
    return srgb_decode((a[..., :3].astype(np.float32) / scale)).astype(np.float32)


def proxy(data: bytes) -> Image.Image:
    lin = decode(data)
    h, w = lin.shape[:2]
    s = EDGE / max(h, w)
    size = (round(w * s), round(h * s))
    # resize each channel in linear light (a float image per channel: PIL has no float RGB mode)
    ch = [np.asarray(Image.fromarray(lin[..., k].astype(np.float32), "F").resize(size, Image.Resampling.LANCZOS)) for k in range(3)]
    enc = srgb_encode(np.clip(np.stack(ch, -1), 0, 1))
    return Image.fromarray((enc * 255 + 0.5).astype(np.uint8))


def fetch(name: str, url: str, list_only: bool) -> None:
    t0 = time.time()
    z = zipfile.ZipFile(RangeFile(url))
    members = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]
    print(f"{name}: {len(members)} images, {sum(m.compress_size for m in members) / 1e9:.1f} GB compressed", flush=True)
    if list_only:
        for m in members[:5]:
            print("  ", m.filename, m.file_size)
        return
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out.glob("*.png")}
    members.sort(key=lambda m: m.header_offset)          # sequential reads hit the read-ahead buffer
    for i, m in enumerate(members):
        stem = Path(m.filename).stem
        if stem in done:
            continue
        proxy(z.read(m)).save(out / f"{stem}.png", optimize=False)
        if i % 500 == 0:
            print(f"  {name} {i}/{len(members)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)


N_TRAIN = 8875      # official split (README of csjliang/PPR10K): first 8,875 files train, last 2,286 test


def split() -> None:
    def order(p: Path) -> tuple[int, int]:
        g, i = p.stem.split("_")[:2]
        return int(g), int(i)

    src = sorted((OUT / "source").glob("*.png"), key=order)
    assert len(src) == 11161, f"{len(src)} sources: run the download first"
    base = ROOT / "data/ppr10k_split"
    for k, f in enumerate(src):
        sp = "train" if k < N_TRAIN else "test"
        for d, name in (("source", "original"), ("a", "expert_a"), ("b", "expert_b"), ("c", "expert_c")):
            tgt = OUT / d / f.name
            link = base / sp / name / f.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if tgt.exists() and not link.exists():
                link.symlink_to(tgt)
    print({sp: len(list((base / sp / "original").iterdir())) for sp in ("train", "test")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(ZIPS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--split", action="store_true", help="lay out the train/test split (after the download)")
    args = ap.parse_args()
    if args.split:
        return split()
    u = urls()
    for k in args.only:
        fetch(k, u[ZIPS[k]], args.list)


if __name__ == "__main__":
    main()
