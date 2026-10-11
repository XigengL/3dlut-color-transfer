"""Image loading and validity masking.

No colour management is applied: JPEG code values are used verbatim, which is the
correct behaviour for data with no trustworthy embedded ICC profile (see DESIGN.md
§2.4).  EXIF orientation *is* applied, because several targets in this dataset are
stored rotated with a matching orientation tag.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_code(path: str, target_max: int = 1800) -> np.ndarray:
    """Load a JPEG as float32 code values in [0,1], EXIF orientation applied.

    ``target_max`` caps the longest edge; Pillow's DCT-domain ``draft`` is used so
    large JPEGs are decoded cheaply.  This affects only the sampling density, not
    the colour statistics.
    """
    im = Image.open(path)
    w, h = im.size
    scale = 1
    while max(w, h) // (scale * 2) >= target_max:
        scale *= 2
    if scale > 1:
        im.draft("RGB", (w // scale, h // scale))
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    im.close()
    return np.ascontiguousarray(arr)


def luma(x: np.ndarray) -> np.ndarray:
    return x @ REC709


def gradient_magnitude(x: np.ndarray) -> np.ndarray:
    l = luma(x)
    dx = np.diff(l, axis=1, prepend=l[:, :1])
    dy = np.diff(l, axis=0, prepend=l[:1, :])
    return np.hypot(dx, dy)


def _dilate(mask: np.ndarray, times: int = 1) -> np.ndarray:
    out = mask
    for _ in range(times):
        acc = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = acc
    return out


def build_mask(a: np.ndarray, b: np.ndarray, *,
               hi: float = 0.98, lo: float = 0.015,
               grad_pct: float = 90.0, dilate: int = 1) -> np.ndarray:
    """Pixels safe to use as colour-correspondence samples.

    Excluded: near-clipped pixels in either image (censored data — A's code value
    no longer carries scene information once it is near 0 or 1), very dark pixels
    (JPEG noise dominates), and high-gradient pixels in *either* image (chroma
    subsampling and DCT ringing; B has been re-encoded so its ringing is worse).
    """
    m = (a.max(axis=2) < hi) & (b.max(axis=2) < hi)
    m &= (a.min(axis=2) > lo) & (b.min(axis=2) > lo)
    g = np.maximum(gradient_magnitude(a), gradient_magnitude(b))
    thr = np.percentile(g, grad_pct)
    m &= ~_dilate(g > thr, dilate)
    return m


def find_pairs(input_dir: str, target_dir: str):
    """Match by case-insensitive *stem*, so extensions may differ.

    ``IMG_1.heic`` in A matches ``IMG_1.jpg`` in B.  Datasets mix JPEG with HEIC and
    camera RAW, and the target has usually been converted, so matching on the full
    filename would silently drop those pairs instead of reporting them.
    Returns ``[(name, path_a, path_b)]``.
    """
    import os
    a_files = sorted(f for f in os.listdir(input_dir) if not f.startswith("."))
    b_index = {}
    for f in os.listdir(target_dir):
        if not f.startswith("."):
            b_index.setdefault(os.path.splitext(f)[0].lower(), f)
    pairs = []
    for f in a_files:
        g = b_index.get(os.path.splitext(f)[0].lower())
        if g is None:
            continue
        pairs.append((f, os.path.join(input_dir, f), os.path.join(target_dir, g)))
    return pairs
