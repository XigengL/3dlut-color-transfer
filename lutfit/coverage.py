"""Pre-flight coverage diagnostics: how much of the colour cube actually has data.

Natural images live on a thin surface in RGB space, so a large part of any 33^3
grid is extrapolation.  That is not a defect of the fit, but the user should know
how much of the cube is guesswork *before* spending time on a fit — and the
thresholds that decide it must scale with the amount of data available.
"""
from __future__ import annotations

import numpy as np

from .tetra import GridOperator, nodes_flat


def scaled_min_count(total_pixels: float, reference_pixels: float = 1.54e8,
                     reference_count: int = 5, floor: int = 3) -> int:
    """Bin-count threshold scaled to the amount of data.

    ``w >= 5`` was tuned on 154M valid pixels where the median occupied 65^3 bin
    holds 20 samples.  With a tenth of that data the median falls to ~2, so a fixed
    threshold silently discards most of the histogram.  Scaling keeps the
    *fraction of mass retained* roughly constant across datasets.
    """
    n = int(round(reference_count * total_pixels / reference_pixels))
    return max(floor, n)


def node_statistics(a_bar: np.ndarray, w: np.ndarray, n: int = 33):
    """Per-node sample-bin count and pixel-mass for an ``n^3`` grid."""
    M = n ** 3
    op = GridOperator(a_bar, n, weights=w)
    cnt = np.zeros(M)
    mass = np.zeros(M)
    for t in range(4):
        cnt += np.bincount(op.idx[:, t], minlength=M)
        mass += np.bincount(op.idx[:, t], weights=op.weights, minlength=M)
    return cnt, mass


def preflight(a_bar: np.ndarray, w: np.ndarray, n: int = 33, min_support: int = 4) -> dict:
    """Quantify coverage before fitting.

    ``node_coverage`` is the share of grid nodes with any real support;
    ``mass_coverage`` is the share of *pixels* that land in well-supported nodes.
    The second number is the one that predicts how the LUT will behave on real
    photographs; the first predicts how it behaves on synthetic or extreme colours.
    """
    cnt, mass = node_statistics(a_bar, w, n)
    supported = cnt >= min_support
    total = float(mass.sum()) + 1e-300
    return dict(
        n=n, n_nodes=int(n ** 3),
        n_supported=int(supported.sum()),
        n_bare=int((~supported).sum()),
        node_coverage=float(supported.mean()),
        mass_coverage=float(mass[supported].sum() / total),
        mass_uncovered=float(mass[~supported].sum() / total),
        median_bin_count=float(np.median(cnt[cnt > 0])) if np.any(cnt > 0) else 0.0,
        cnt=cnt, mass=mass, supported=supported,
    )


def extrapolation_risk(x: np.ndarray, pf: dict, n: int = 33) -> dict:
    """How strongly the LUT acts where there is no data.

    There is no way to measure the *error* of an extrapolation — that is what having
    no data means.  What can be measured is how much the LUT changes colours there:
    if the uncovered regions were nearly a no-op, a wrong extrapolation could not
    hurt.  This reports that magnitude so the risk is at least bounded and visible.
    """
    nodes = nodes_flat(n)
    bare = ~pf["supported"]
    sup = pf["supported"]
    dev = np.abs(x - nodes)
    out = dict(
        bare_mean_abs_shift=float(dev[bare].mean()) if bare.any() else 0.0,
        bare_max_abs_shift=float(dev[bare].max()) if bare.any() else 0.0,
        supported_mean_abs_shift=float(dev[sup].mean()) if sup.any() else 0.0,
    )
    # output chroma introduced where the input is neutral but unsupported
    neutch = (np.abs(x.max(axis=1) - x.min(axis=1)))
    out["bare_mean_output_chroma"] = float(neutch[bare].mean()) if bare.any() else 0.0
    out["supported_mean_output_chroma"] = float(neutch[sup].mean()) if sup.any() else 0.0
    return out


def coverage_map_image(pf: dict, n: int = 33, axis: int = 2, slice_index: int | None = None):
    """Render one slice of the grid: supported nodes in green, bare nodes in red."""
    import numpy as np
    cnt = pf["cnt"].reshape(n, n, n)            # [k, j, i]
    if slice_index is None:
        slice_index = n // 2
    sl = [slice(None)] * 3
    sl[axis] = slice_index
    c = cnt[tuple(sl)]
    img = np.zeros(c.shape + (3,), dtype=np.uint8)
    # log-scaled support -> brightness
    v = np.log1p(np.maximum(c, 0))
    v = v / max(v.max(), 1e-9)
    img[..., 0] = (255 * (1 - v) * (c < 4)).astype(np.uint8)
    img[..., 1] = (255 * v).astype(np.uint8)
    img[..., 2] = (60 * v).astype(np.uint8)
    return np.transpose(img, (1, 0, 2) if axis == 2 else ((0, 1, 2) if axis == 0 else (1, 0, 2)))


def verdict(pf: dict, risk: dict | None = None) -> str:
    """Plain-language assessment, for a GUI to show before the fit is run."""
    mc = pf["mass_coverage"]
    nc = pf["node_coverage"]
    lines = []
    if mc > 0.995:
        lines.append(f"像素质量覆盖 {mc*100:.2f}% —— 训练数据本身完全落在有支撑区域，"
                     f"LUT 在真实照片上的行为可靠。")
    elif mc > 0.98:
        lines.append(f"像素质量覆盖 {mc*100:.2f}% —— 可接受，极少数像素落在外推区。")
    else:
        lines.append(f"像素质量覆盖仅 {mc*100:.2f}% —— 有较多像素落在无数据区，"
                     f"建议增加或更换训练图。")
    lines.append(f"色彩网格覆盖 {nc*100:.1f}%（{pf['n_bare']} / {pf['n_nodes']} 个节点无支撑）。"
                 f"这些区域是平滑外推，不会有色带或反转，但颜色是推测值。")
    if risk:
        lines.append(f"无支撑区域的平均色偏强度 {risk['bare_mean_abs_shift']:.4f} / "
                     f"有支撑区域 {risk['supported_mean_abs_shift']:.4f}（码值）。"
                     f"两者接近说明外推区的改动幅度与整体一致。")
    return "\n".join(lines)
