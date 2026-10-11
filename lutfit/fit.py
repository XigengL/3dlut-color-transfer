"""Multi-scale regularised least-squares fit of a 33^3 LUT.

The objective at each scale ``n`` is

    J(x) = (1/W) sum_b w_b ||T_n(u_b) x - y_b||^2          data
         + s   * ||L x||^2 / 42                            smoothness
         + beta  * sum_i wp_i ||x_i - x_ref_i||^2 / M      coarse-to-fine prior
         + gamma * sum_i wg_i ||x_i - g_i||^2 / M          neutral-axis prior
         + mu    * sum_c sum_{d_c x < floor} (d_c x)^2 / M monotonicity (IRLS)

subject to ``0 <= x <= 1``.  All of ``s``, ``beta``, ``gamma`` and ``mu`` are
dimensionless and expressed *relative to the typical data coefficient*
``d0 / W``, so a value of 1 means "as strong as the data at a median-supported
node".  Without that normalisation the knobs are uninterpretable.

``x_ref`` is the previous (coarser) scale's solution resampled onto this scale's
nodes, so where the data runs out the solution falls back to a smooth,
data-supported transform rather than to identity (DESIGN.md §4).

Solved with preconditioned conjugate gradients plus a primal active set for the
box constraint; no external linear-algebra dependency.
"""
from __future__ import annotations

import numpy as np

from .project import JacobianPenalty
from .tetra import GridOperator, Laplacian, diagonal_indices, evaluate, nodes_flat

LAP_DIAG = 42.0          # diagonal of L^T L for the clamped 6-neighbour Laplacian


# ------------------------------------------------------------------- CG ------
def pcg(matvec, b, diag, x0=None, maxiter=400, tol=1e-8, verbose=False):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = r / diag
    p = z.copy()
    rz = float((r * z).sum())
    bnorm = float(np.sqrt((b * b).sum())) + 1e-300
    k = 0
    for k in range(maxiter):
        Ap = matvec(p)
        denom = float((p * Ap).sum())
        if denom <= 0:
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        rnorm = float(np.sqrt((r * r).sum()))
        if verbose and (k % 50 == 0 or k == maxiter - 1):
            print(f"      cg {k:4d}  rel_res={rnorm/bnorm:.3e}")
        if rnorm <= tol * bnorm:
            break
        z = r / diag
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, k + 1, rnorm / bnorm


def _pcg_free(matvec, b, diag, x0, free, maxiter, tol):
    x = x0.copy()
    r = np.where(free, b - matvec(x), 0.0)
    z = np.where(free, r / diag, 0.0)
    p = z.copy()
    rz = float((r * z).sum())
    bnorm = float(np.sqrt((b * b).sum())) + 1e-300
    k = 0
    for k in range(maxiter):
        Ap = np.where(free, matvec(p), 0.0)
        denom = float((p * Ap).sum())
        if denom <= 0:
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        if float(np.sqrt((r * r).sum())) <= tol * bnorm:
            break
        z = np.where(free, r / diag, 0.0)
        rz_new = float((r * z).sum())
        if rz <= 0:
            break
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, k + 1


def pcg_box(matvec, b, diag, lo, hi, maxiter=400, outer=12, tol=1e-8):
    """Box-constrained QP  min 0.5 x'Ax - b'x  s.t. lo <= x <= hi.

    Primal active set: solve on the free variables, clip, re-identify, repeat.
    Keeping the LUT inside the code-value range *during* the solve lets the rest
    of the grid compensate, which post-hoc clipping cannot do.
    """
    x = np.clip(np.zeros_like(b), lo, hi)
    prev_active = None
    for _ in range(outer):
        grad = matvec(x) - b
        active = ((x <= lo + 1e-12) & (grad >= 0.0)) | ((x >= hi - 1e-12) & (grad <= 0.0))
        if prev_active is not None and np.array_equal(active, prev_active):
            break
        prev_active = active
        free = ~active
        if not free.any():
            break
        x, _ = _pcg_free(matvec, b, diag, x, free, maxiter, tol)
        x = np.clip(x, lo, hi)
    return np.clip(x, lo, hi)


# --------------------------------------------------------------- priors ------
def neutral_targets(n: int, gray_curve_t, gray_curve_v, chroma_max: float = 0.10):
    """``(g, wg)``: neutral-axis target per node and its weight.

    The neutral axis is the highest-consequence part of a LUT — the eye is far more
    sensitive to a cast on grey than to an equal ΔE on a saturated colour — so the
    measured curve is imposed there much more strongly than elsewhere.  The tint is
    taken *per luminance*, so a legitimate split-tone survives while a
    chroma-dependent hue twist near neutral is suppressed.
    """
    coords = nodes_flat(n)
    luma = coords @ np.array([0.2126, 0.7152, 0.0722])
    g = np.stack([np.interp(luma, gray_curve_t, gray_curve_v[:, c]) for c in range(3)], axis=1)
    chroma = coords.max(axis=1) - coords.min(axis=1)
    wg = np.clip(1.0 - chroma / chroma_max, 0.0, 1.0)
    diag = diagonal_indices(n)
    wg[diag] = 1.0
    g[diag] = np.stack([np.interp(coords[diag, 0], gray_curve_t, gray_curve_v[:, c])
                        for c in range(3)], axis=1)
    return g, wg


def support_prior(n: int, a_bar, b_bar, w) -> np.ndarray:
    """1 where a node is unsupported, -> 0 where the data is solid."""
    op = GridOperator(a_bar, n, weights=w)
    d = op.diag().mean(axis=1)
    d0 = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return 1.0 / (1.0 + d / d0)


# ---------------------------------------------------------------- level ------
def fit_level(n: int, a_bar, b_bar, w, *, x_ref, wp, xr, g, wg,
              lam_s: float = 0.0, beta: float = 2.0, gamma: float = 0.0,
              mono_mu: float = 0.0, mono_floor: float = 0.0, mono_outer: int = 6,
              maxiter: int = 400, box=(0.0, 1.0), verbose: bool = False):
    M = n ** 3
    W = w.sum()
    op = GridOperator(a_bar, n, weights=w)
    lap = Laplacian(n)
    mp = JacobianPenalty(n)

    d = op.diag()
    d0 = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    unit = d0 / W                                   # typical data coefficient per node

    wp_eff = (wp * beta * unit)[:, None]
    wg_eff = (wg * gamma * unit)[:, None]
    lam_eff = lam_s * unit / LAP_DIAG
    mu_eff = mono_mu * unit

    def base(x):
        return (op.normal(x) / W + lam_eff * lap.normal(x) + wp_eff * x + wg_eff * x)

    b_rhs = op.rhs(b_bar) / W + wp_eff * xr + wg_eff * g
    pre = op.diag() / W + lam_eff * LAP_DIAG + wp_eff + wg_eff + 1e-12

    x = np.clip(np.zeros_like(b_rhs), box[0], box[1])
    viol_hist = []
    for it in range(max(1, mono_outer)):
        pre_it = pre
        if mono_mu > 0 and it > 0:
            weights = prev_weights

            def matvec(y, weights=weights):
                return base(y) + mu_eff * mp.apply(y, weights)

            pre_it = pre + mu_eff * mp.diag(weights)
        else:
            matvec = base
        x = pcg_box(matvec, b_rhs, pre_it, box[0], box[1], maxiter=maxiter)
        if mono_mu <= 0:
            break
        prev_weights, n_viol = mp.active_weights(x, mono_floor)
        viol_hist.append(n_viol)
        if n_viol == 0:
            break

    info = dict(d0=d0, unit=unit, lam_eff=lam_eff, mu_eff=mu_eff,
                mono_violations=viol_hist, range=(float(x.min()), float(x.max())))
    if verbose:
        print(f"    n={n:>2}  range=[{x.min():+.4f},{x.max():+.4f}]  "
              f"mono_viol={viol_hist}  lam_eff={lam_eff:.3e} mu_eff={mu_eff:.3e}")
    return x, info


def fit_multiscale(a_bar, b_bar, w, *, levels=(5, 9, 17, 25, 33),
                   lam_s: float = 0.0, beta: float = 2.0, gamma: float = 0.0,
                   mono_mu: float = 0.0, mono_floor: float = 0.0, mono_outer: int = 6,
                   gray_curve_t=None, gray_curve_v=None, chroma_max: float = 0.10,
                   maxiter: int = 400, box=(0.0, 1.0), verbose: bool = True,
                   progress=None, should_cancel=None,
                   project=None, adaptive: bool = True, max_trials: int = 3,
                   fold_target: int = None, model_tol: float = 0.25,
                   fold_tol_frac: float = 1e-4):
    """Fit coarse-to-fine, then adapt the regularisation on the final level.

    ``project(x, n) -> (y, info)`` runs the post-processing (gamut compression,
    isotonic projection, fold repair) and reports the model error.  The final level
    is then retried with doubled ``lam_s`` / ``mono_mu`` while folds remain, keeping
    the best attempt.  This costs nothing when the default already reaches the
    target (dataset 1: zero folds on the first trial) and only spends extra time
    when the transform is strong enough to need it (dataset 2).
    """
    from .verify import artifact_metrics

    hist = []
    x = None
    levels = tuple(levels)
    final_n = levels[-1]

    # ---------------- coarse levels: fitted once ----------------
    for li, n in enumerate(levels[:-1]):
        if should_cancel is not None and should_cancel():
            from .errors import Cancelled
            raise Cancelled()
        if progress is not None:
            progress(li / len(levels), f"拟合 LUT  {n}\u00b3 网格  （第 {li+1}/{len(levels)} 层）")
        M = n ** 3
        x_ref = nodes_flat(n) if x is None else evaluate(x, nodes_flat(n), levels[li - 1])
        wp = support_prior(n, a_bar, b_bar, w)
        if gray_curve_t is not None and gray_curve_v is not None:
            g, wg = neutral_targets(n, gray_curve_t, gray_curve_v, chroma_max)
        else:
            g, wg = np.zeros((M, 3)), np.zeros(M)
        x, info = fit_level(n, a_bar, b_bar, w, x_ref=x_ref, wp=wp, xr=x_ref,
                            g=g, wg=wg, lam_s=lam_s, beta=beta, gamma=gamma,
                            mono_mu=mono_mu, mono_floor=mono_floor,
                            mono_outer=mono_outer, maxiter=maxiter, box=box,
                            verbose=verbose)
        info["n"] = n
        hist.append(info)

    # ---------------- final level: adaptive regularisation ----------------
    n = final_n
    M = n ** 3
    x_ref = nodes_flat(n) if x is None else evaluate(x, nodes_flat(n), levels[-2])
    wp = support_prior(n, a_bar, b_bar, w)
    if gray_curve_t is not None and gray_curve_v is not None:
        g, wg = neutral_targets(n, gray_curve_t, gray_curve_v, chroma_max)
    else:
        g, wg = np.zeros((M, 3)), np.zeros(M)

    # A handful of folded cells out of 32768 is far below anything visible; demanding
    # exactly zero made the search burn trials on an unreachable target.
    if fold_target is None:
        fold_target = int(fold_tol_frac * (n - 1) ** 3)
    ls, mu = lam_s, mono_mu
    trials, best = [], None
    n_trials = max(1, max_trials) if adaptive else 1
    for trial in range(n_trials):
        if should_cancel is not None and should_cancel():
            from .errors import Cancelled
            raise Cancelled()
        if progress is not None:
            note = "" if trial == 0 else f"（正则化加强，第 {trial+1} 次）"
            progress((len(levels) - 1) / len(levels),
                     f"拟合 LUT  {n}\u00b3 网格  （第 {len(levels)}/{len(levels)} 层）{note}")
        xf, info = fit_level(n, a_bar, b_bar, w, x_ref=x_ref, wp=wp, xr=x_ref,
                             g=g, wg=wg, lam_s=ls, beta=beta, gamma=gamma,
                             mono_mu=mu, mono_floor=mono_floor,
                             mono_outer=mono_outer, maxiter=maxiter, box=box,
                             verbose=verbose)
        if project is not None:
            y, pinfo = project(xf, n)
        else:
            y, pinfo = xf, {}
        m = artifact_metrics(y, n)
        m["model"] = float(pinfo.get("model", float("nan")))
        rec = dict(trial=trial, lam_s=ls, mono_mu=mu, **m)
        trials.append(rec)
        if verbose:
            print(f"    trial {trial}: lam_s={ls:g} mu={mu:g} -> folds={m['folds']} "
                  f"min_det={m['min_det']:+.4f} max2nd={m['max_second']:.5f} "
                  f"model={m['model']:.5f}")

        better = (best is None
                  or (m["folds"], m["luma_viol"]) < (best["m"]["folds"], best["m"]["luma_viol"])
                  or ((m["folds"], m["luma_viol"]) == (best["m"]["folds"], best["m"]["luma_viol"])
                      and m["max_second"] < best["m"]["max_second"]))
        if better:
            best = dict(y=y, m=m, trial=trial, lam_s=ls, mono_mu=mu)

        if m["folds"] <= fold_target:
            break
        # Stronger smoothing buying nothing is the normal case: stop rather than
        # spending another expensive trial on the final level.
        if trial > 0 and m["folds"] >= trials[trial - 1]["folds"]:
            if verbose:
                print(f"    加强正则化没有改善折叠（{trials[trial-1]['folds']} -> {m['folds']}），停止")
            break
        base_model = trials[0]["model"]
        if base_model == base_model and m["model"] > base_model * (1.0 + model_tol):
            if verbose:
                print("    模型误差已明显变差，停止加强")
            break
        ls *= 2.0
        mu *= 2.0

    hist.append(dict(n=n, trials=trials, chosen_trial=best["trial"],
                     chosen_lam_s=best["lam_s"], chosen_mono_mu=best["mono_mu"]))
    if verbose:
        print(f"    采用 trial {best['trial']}  (lam_s={best['lam_s']:g})，"
              f"折叠 {best['m']['folds']}")
    return best["y"], hist
