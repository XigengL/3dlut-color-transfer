"""Structural and perceptual verification of a fitted LUT."""
from __future__ import annotations

import numpy as np

from .color import code_to_lab, de00
from .loader import REC709 as LUMA
from .project import fold_report
from .tetra import evaluate


# --------------------------------------------------------------- structural --
def monotone_excursion(x: np.ndarray, n: int, n_dirs: int = 400, n_steps: int = 512,
                       seed: int = 0):
    """Largest *backward* movement of the interpolated LUT along straight lines.

    Cell-count metrics are hard to interpret; this reports the artefact in the
    units that matter — code values.  A backward excursion of one code level
    (1/255 = 0.0039) is the perceptual yardstick: below that a fold cannot be seen.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_steps)
    worst = 0.0
    worst_info = None

    def scan(p0, d):
        nonlocal worst, worst_info
        pts = p0[None, :] + t[:, None] * d[None, :]
        inside = np.all((pts >= 0.0) & (pts <= 1.0), axis=1)
        pts = pts[inside]
        if len(pts) < 8:
            return
        y = evaluate(x, pts, n)
        dd = np.diff(y, axis=0)
        m = dd.min()
        if m < worst:
            worst = float(m)
            worst_info = dict(start=p0.tolist(), dir=d.tolist())

    for ax in range(3):
        for u in np.linspace(0.05, 0.95, 7):
            for v in np.linspace(0.05, 0.95, 7):
                p0 = np.zeros(3)
                p0[ax] = 0.0
                p0[(ax + 1) % 3] = u
                p0[(ax + 2) % 3] = v
                d = np.zeros(3)
                d[ax] = 1.0
                scan(p0, d)
    scan(np.zeros(3), np.ones(3))
    for _ in range(n_dirs):
        p0 = rng.random(3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        scan(p0, d)

    return dict(worst_backward=worst, code_levels=abs(worst) * 255.0, where=worst_info)


def luma_backward_excursion(x: np.ndarray, n: int, ndir: int = 400,
                            nsteps: int = 800, seed: int = 0) -> float:
    """Largest **drawdown** of output luma along straight lines, in code levels.

    ``max over t of (running_max(L) - L(t))`` — how far luma falls below its own
    previous maximum along the line.  This is the quantity that corresponds to what
    a viewer can see, and unlike a per-step difference it is independent of the
    step size: with a finer step the per-step delta shrinks but the drawdown does
    not.

    The fold *count* is not the right acceptance test.  On the second dataset a LUT
    with 155 folded cells and one with zero folds both measured well under a code
    level; legitimate off-diagonal crosstalk moves luma backwards too.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, nsteps)
    worst = 0.0

    def scan(pts):
        nonlocal worst
        pts = pts[np.all((pts >= 0.0) & (pts <= 1.0), axis=1)]
        if len(pts) < 32:
            return
        # Orient the line so *input* luma increases.  A line heading towards darker
        # colours makes output luma drop, and that is the mapping working correctly,
        # not a fold -- scoring it as one gave absurd numbers (168 code levels).
        if float((pts[-1] - pts[0]) @ LUMA) < 0.0:
            pts = pts[::-1]
        lum = evaluate(x, pts, n) @ LUMA
        peak = np.maximum.accumulate(lum)
        worst = max(worst, float((peak - lum).max()))

    for ax in range(3):
        for u in np.linspace(0.05, 0.95, 5):
            for v in np.linspace(0.05, 0.95, 5):
                pts = np.empty((nsteps, 3))
                pts[:, ax] = t
                pts[:, (ax + 1) % 3] = u
                pts[:, (ax + 2) % 3] = v
                scan(pts)
    for _ in range(ndir):
        p0 = rng.random(3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        scan(p0[None, :] + t[:, None] * d[None, :])
    return worst * 255.0


def artifact_metrics(x: np.ndarray, n: int) -> dict:
    """Cheap, deterministic artefact metrics, used to drive the adaptive search.

    The criterion is the fold count: ``det(J) <= 0`` is the formal definition of a
    local colour inversion, it needs no sampling, and it is achievable (dataset 1
    reaches zero).  ``max_second`` guards banding.

    ``backward`` is reported as a *diagnostic only* and must not be thresholded:
    it measures luma drawdown along arbitrary lines, and a line crossing from one
    hue to another legitimately moves luma backwards by tens of code levels.
    """
    from .project import monotonicity_report
    folds = fold_report(x, n)
    mono = monotonicity_report(x, n)
    sw = axis_sweeps(x, n, n_lines=5, n_steps=400)
    return dict(
        folds=folds["n_folded_cells"],
        n_tetra=folds["n_folded_tetra"],
        min_det=folds["min_det"],
        luma_viol=mono["luma_n_negative"],
        mono_viol=mono["n_negative"],
        max_second=max(q["max_second_diff"] for q in sw.values()),
        backward=luma_backward_excursion(x, n, ndir=150),
    )


def axis_sweeps(x: np.ndarray, n: int, n_lines: int = 9, n_steps: int = 1024):
    """Dense sweeps along R, G, B at a grid of fixed values on the other two axes.

    Only the *matching* output channel is required to be non-decreasing: a
    legitimate grade routinely makes output-R fall as input-G rises, so checking
    every channel against every axis would be wrong.
    """
    t = np.linspace(0.0, 1.0, n_steps)
    fixed = np.linspace(0.05, 0.95, n_lines)
    out = {}
    for name, ax in (("R", 0), ("G", 1), ("B", 2)):
        worst_mono = 0.0
        worst_second = 0.0
        for u in fixed:
            for v in fixed:
                pts = np.empty((n_steps, 3))
                pts[:, ax] = t
                pts[:, (ax + 1) % 3] = u
                pts[:, (ax + 2) % 3] = v
                y = evaluate(x, pts, n)
                d1 = np.diff(y[:, ax])
                worst_mono = min(worst_mono, float(d1.min()))
                worst_second = max(worst_second,
                                   float(np.abs(np.diff(y[:, ax], n=2)).max()))
        out[name] = dict(min_first_diff=worst_mono, max_second_diff=worst_second)
    return out


def neutral_scan(x: np.ndarray, n: int, n_steps: int = 2048, eps: float = 0.01,
                 curve_t=None, curve_v=None):
    """Neutral axis: monotonicity, smoothness and deviation from the intended look.

    The absolute channel spread along the neutral axis is *not* an artefact metric:
    a split-tone grade legitimately maps grey to a tinted ramp (warm shadows, cool
    highlights, or vice versa).  What matters is the deviation from the neutral
    response measured from the data.
    """
    t = np.linspace(0.0, 1.0, n_steps)
    pts = np.stack([t, t, t], axis=1)
    y = evaluate(x, pts, n)
    d = np.diff(y, axis=0)
    out = dict(min_first_diff=float(d.min()),
               max_second_diff=float(np.abs(np.diff(y, n=2, axis=0)).max()),
               neutral_spread=float(np.abs(y.max(axis=1) - y.min(axis=1)).max()),
               black_point=float(y[0].mean()), white_point=float(y[-1].mean()))

    if curve_t is not None and curve_v is not None:
        ref = np.stack([np.interp(t, curve_t, curve_v[:, c]) for c in range(3)], axis=1)
        dev = np.abs(y - ref)
        out["dev_from_curve"] = float(dev.max())
        out["dev_from_curve_rms"] = float(np.sqrt((dev ** 2).mean()))

    # chroma gain near neutral: a bounded, smooth response is what we want
    gain = 0.0
    for t0 in np.linspace(0.05, 0.95, 19):
        base = evaluate(x, np.array([[t0, t0, t0]]), n)[0]
        probe = np.array([[t0 + eps, t0, t0], [t0 - eps, t0, t0],
                          [t0, t0 + eps, t0], [t0, t0 - eps, t0],
                          [t0, t0, t0 + eps], [t0, t0, t0 - eps]])
        yy = evaluate(x, probe, n)
        base_c = float(base.max() - base.min())
        for row in yy:
            gain = max(gain, abs(float(row.max() - row.min()) - base_c) / eps)
    out["near_neutral_chroma_gain"] = float(gain)
    return out


def jacobian_check(x: np.ndarray, n: int, frac=(0.62, 0.56, 0.51), h_cell: float = 0.02):
    """Determinant of the local 3x3 Jacobian over every cell.

    ``det(J) <= 0`` means the local mapping folds — the formal definition of a
    colour inversion.  This is a *verification* metric, not a fitting penalty
    (DESIGN.md §6.4c).

    The sample point is placed strictly inside one tetrahedron (``fr > fg > fb``),
    not at the cell centre: the centre lies on a tetrahedron boundary where the
    interpolant is continuous but not differentiable, so a finite-difference
    Jacobian taken there is meaningless.
    """
    m = n - 1
    h = h_cell / m
    idx = np.arange(m)
    comps = [(idx + frac[a]) / m for a in range(3)]
    k, j, i = np.meshgrid(*comps, indexing="ij")
    c = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    f0 = evaluate(x, c, n)
    cols = []
    for ax in range(3):
        p = c.copy()
        p[:, ax] += h
        cols.append((evaluate(x, p, n) - f0) / h)
    J = np.stack(cols, axis=2)                       # (N,3,3): J[:, out, in]
    det = np.linalg.det(J)
    sv = np.linalg.svd(J, compute_uv=False)
    return dict(min_det=float(det.min()), n_nonpositive=int((det <= 0).sum()),
                cells=int(det.size), min_singular=float(sv.min()),
                max_singular=float(sv.max()),
                max_condition=float((sv[:, 0] / np.maximum(sv[:, -1], 1e-12)).max()))


def range_check(x: np.ndarray, n: int):
    return dict(min=float(x.min()), max=float(x.max()),
                n_below=int((x < 0).any(axis=1).sum()),
                n_above=int((x > 1).any(axis=1).sum()))


def identity_roundtrip(n: int = 33, seed: int = 0):
    """Feed the pipeline a known LUT and check it is recovered."""
    from .fit import fit_multiscale
    from .tetra import nodes_flat
    rng = np.random.default_rng(seed)
    n_true = 9
    truth = nodes_flat(n_true).copy()
    truth *= 1.0
    truth[:, 0] = np.clip(truth[:, 0] ** 0.95, 0, 1)
    truth[:, 1] = np.clip(truth[:, 1] ** 1.05, 0, 1)
    truth[:, 2] = np.clip(0.02 + 0.96 * truth[:, 2], 0, 1)

    pts = rng.random((60000, 3))
    y = evaluate(truth, pts, n_true)
    x, _ = fit_multiscale(pts, y, np.ones(len(pts)), levels=(5, 9, 17, 33),
                          lam_s=1e-4, beta=1e-4, gamma=0.0, maxiter=300, verbose=False)
    err = np.abs(x - nodes_flat(n)).max()
    return dict(max_abs_error=float(err))
