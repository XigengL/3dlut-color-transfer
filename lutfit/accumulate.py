"""Streaming accumulation of the A->B colour correspondence statistics.

One pass over the pairs builds everything downstream needs:

* a ``grid^3`` conditional histogram of B given A (count, sum, sum of squares)
* the empirical neutral-axis curve, from near-neutral pixels only
* per-pair summary statistics, which drive the pair-level robustness weights

Memory is O(grid^3) and independent of the number of pixels.
"""
from __future__ import annotations

import time
import numpy as np

from .errors import Cancelled
from .loader import build_mask, find_pairs, load_code, luma


def _bin_index(a: np.ndarray, grid: int) -> np.ndarray:
    """Nearest-node bin index; bin i is centred on code value i/(grid-1)."""
    q = np.rint(a * (grid - 1)).astype(np.int64)
    np.clip(q, 0, grid - 1, out=q)
    return q[..., 0] + q[..., 1] * grid + q[..., 2] * grid * grid


def build_statistics(pairs, *, grid: int = 65, target_max: int = 1800,
                     gray_chroma: float = 0.02, gray_levels: int = 64,
                     verbose: bool = True, progress=None, should_cancel=None) -> dict:
    M = grid ** 3
    cnt = np.zeros(M, dtype=np.float64)
    sa = np.zeros((M, 3), dtype=np.float64)
    sb = np.zeros((M, 3), dtype=np.float64)
    sb2 = np.zeros((M, 3), dtype=np.float64)

    gray_cnt = np.zeros(gray_levels, dtype=np.float64)
    gray_sb = np.zeros((gray_levels, 3), dtype=np.float64)
    gray_sb2 = np.zeros((gray_levels, 3), dtype=np.float64)

    per_pair = []
    skipped = []
    for pi, (name, pa, pb) in enumerate(pairs):
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        if progress is not None:
            progress(pi / max(len(pairs), 1), f"统计颜色分布  {name}")
        t0 = time.time()
        try:
            a = load_code(pa, target_max)
            b = load_code(pb, target_max)
        except Exception as exc:                       # 无法解码的格式（RAW / HEIC …）
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  !! {name}: 无法读取（{type(exc).__name__}）— 跳过")
            continue
        if a.shape != b.shape:
            skipped.append((name, f"尺寸不一致 {a.shape} vs {b.shape}"))
            if verbose:
                print(f"  !! {name}: shape {a.shape} vs {b.shape} — skipped")
            continue
        m = build_mask(a, b)
        npx = int(m.sum())
        av = a[m]
        bv = b[m]
        if npx < 1000:
            skipped.append((name, f"有效像素仅 {npx}"))
            if verbose:
                print(f"  !! {name}: only {npx} valid pixels — skipped")
            continue

        idx = _bin_index(av, grid)
        cnt += np.bincount(idx, minlength=M)
        for c in range(3):
            sa[:, c] += np.bincount(idx, weights=av[:, c], minlength=M)
            sb[:, c] += np.bincount(idx, weights=bv[:, c], minlength=M)
            sb2[:, c] += np.bincount(idx, weights=bv[:, c] ** 2, minlength=M)

        # neutral axis: near-neutral input pixels, keyed by input luma
        chroma = av.max(axis=1) - av.min(axis=1)
        ng = chroma < gray_chroma
        if ng.sum() > 100:
            la = luma(av[ng])
            lvl = np.clip((la * gray_levels).astype(np.int64), 0, gray_levels - 1)
            bn = bv[ng]
            gray_cnt += np.bincount(lvl, minlength=gray_levels)
            for c in range(3):
                gray_sb[:, c] += np.bincount(lvl, weights=bn[:, c], minlength=gray_levels)
                gray_sb2[:, c] += np.bincount(lvl, weights=bn[:, c] ** 2, minlength=gray_levels)

        la_all, lb_all = luma(av), luma(bv)
        per_pair.append(dict(
            name=name, pixels=npx,
            mean_abs=float(np.mean(np.abs(lb_all - la_all))),
            edit_rms=float(np.std(lb_all - la_all)),
            luma_a_mean=float(la_all.mean()),
            luma_b_mean=float(lb_all.mean()),
            frac_clipped_out=float(1.0 - npx / (a.shape[0] * a.shape[1])),
        ))
        if verbose:
            print(f"  {name:<32} {a.shape[1]}x{a.shape[0]}  valid={npx/1e6:.2f}M  "
                  f"mean|dL|={per_pair[-1]['mean_abs']:.4f}  [{time.time()-t0:.1f}s]")

    return dict(grid=grid, cnt=cnt, sa=sa, sb=sb, sb2=sb2,
                gray_cnt=gray_cnt, gray_sb=gray_sb, gray_sb2=gray_sb2,
                per_pair=per_pair, gray_levels=gray_levels,
                gray_chroma=gray_chroma, skipped=skipped)


def summarize(stats: dict) -> dict:
    """Derive the occupied-bin samples, the irreducible error floor and coverage."""
    grid = stats["grid"]
    cnt = stats["cnt"]
    occ = cnt > 0
    n_occ = int(occ.sum())
    total_nodes = grid ** 3

    a_bar = np.zeros((n_occ, 3))
    b_bar = np.zeros((n_occ, 3))
    b_var = np.zeros((n_occ, 3))
    w = cnt[occ]
    sa, sb, sb2 = stats["sa"][occ], stats["sb"][occ], stats["sb2"][occ]
    for c in range(3):
        a_bar[:, c] = sa[:, c] / w
        b_bar[:, c] = sb[:, c] / w
        b_var[:, c] = np.maximum(sb2[:, c] / w - b_bar[:, c] ** 2, 0.0)

    # per-channel RMS of the within-bin spread, weighted by bin population
    floor = np.sqrt((b_var * w[:, None]).sum(0) / w.sum())

    # coverage restricted to bins with a meaningful number of samples
    strong = cnt >= 50
    return dict(
        n_occupied=n_occ, total_nodes=total_nodes,
        coverage=float(n_occ / total_nodes),
        coverage_strong=float(strong.sum() / total_nodes),
        mass_in_strong=float(cnt[strong].sum() / cnt.sum()),
        a_bar=a_bar, b_bar=b_bar, b_var=b_var, w=w, occ_idx=np.flatnonzero(occ),
        irreducible_rms=floor,
        median_count=float(np.median(cnt[occ])),
    )
