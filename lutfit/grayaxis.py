"""Empirical neutral-axis curve measured from near-neutral pixels."""
from __future__ import annotations

import numpy as np

from .project import pava_1d


def gray_curve(stats: dict, min_count: int | None = None, smooth: int = 5):
    """Return ``(t, v, info)``: the measured neutral response of the edit.

    ``t`` is input luma in code values and ``v`` is the (R,G,B) response, extended
    smoothly to t=0 and t=1 so the black and white ends are not left undefined.

    The tint is kept *per luminance*, which is what lets a legitimate split-tone
    (warm shadows, cool highlights, or the reverse) survive while still pinning the
    neutral axis strongly.
    """
    if min_count is None:
        min_count = max(50, int(200 * stats["gray_cnt"].sum() / 1.54e8))

    cnt = stats["gray_cnt"]
    sb = stats["gray_sb"]
    levels = stats["gray_levels"]
    t = (np.arange(levels) + 0.5) / levels
    ok = cnt >= min_count
    if ok.sum() < 4:
        return None, None, None
    tt, vv = t[ok], sb[ok] / cnt[ok][:, None]

    # edge-safe moving average: replicate the ends instead of zero-padding, which
    # would drag the endpoints toward zero (the white point once read 0.57 instead
    # of 0.94 because of this)
    if smooth > 1 and len(tt) >= smooth:
        pad = smooth // 2
        ker = np.ones(smooth) / smooth
        vv = np.stack([
            np.convolve(np.pad(vv[:, c], pad, mode="edge"), ker, mode="valid")
            for c in range(3)], axis=1)

    # linear extension of the end chords to t = 0 and t = 1
    s_lo = (vv[1] - vv[0]) / max(tt[1] - tt[0], 1e-9)
    s_hi = (vv[-1] - vv[-2]) / max(tt[-1] - tt[-2], 1e-9)
    lo = np.clip(vv[0] - s_lo * tt[0], 0.0, None)
    hi = np.clip(vv[-1] + s_hi * (1.0 - tt[-1]), None, 1.0)

    t = np.concatenate([[0.0], tt, [1.0]])
    v = np.concatenate([lo[None], vv, hi[None]], axis=0)
    v = np.clip(v, 0.0, 1.0)
    v = np.stack([pava_1d(v[:, c]) for c in range(3)], axis=1)

    info = dict(black_lift=float(vv[0].mean()), white_end=float(vv[-1].mean()),
                white_extrap=float(hi.mean()),
                t_valid_lo=float(tt[0]), t_valid_hi=float(tt[-1]),
                tint_rg=float((vv[:, 0] - vv[:, 1]).mean()),
                tint_gb=float((vv[:, 1] - vv[:, 2]).mean()))
    return t, v, info
