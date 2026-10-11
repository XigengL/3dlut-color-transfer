"""Constraints and projections: Jacobian-diagonal / luma monotonicity, gamut compression."""
from __future__ import annotations

import numpy as np

from .color import code_to_oklab, in_gamut, oklab_to_code

LUMA = np.array([0.2126, 0.7152, 0.0722])

# (grid array axis, output channel) for the pair (in_c, out_c).
# Flat index is f = i + j*n + k*n*n, so a reshaped array is A[k, j, i]:
# input R maps to array axis 2, G to 1, B to 0; output channels are 0,1,2.
JACOBIAN_DIAG_PAIRS = ((2, 0), (1, 1), (0, 2))


def _diff_adjoint(d: np.ndarray, ga: int, n: int) -> np.ndarray:
    """Adjoint of the forward difference along grid axis ``ga`` (signed: +hi, -lo)."""
    adj = np.zeros((n, n, n))
    hi = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi[ga] = slice(1, n)
    lo[ga] = slice(0, n - 1)
    adj[tuple(hi)] += d
    adj[tuple(lo)] -= d
    return adj


def _diff_diag(w: np.ndarray, ga: int, n: int) -> np.ndarray:
    """Diagonal of ``D^T diag(w) D`` for the forward difference along ``ga``.

    Both endpoints of a difference get ``+w`` (the entry is squared), unlike the
    signed adjoint.  Using the signed version here yields a preconditioner with
    negative entries, which makes CG diverge.
    """
    adj = np.zeros((n, n, n))
    hi = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi[ga] = slice(1, n)
    lo[ga] = slice(0, n - 1)
    adj[tuple(hi)] += w
    adj[tuple(lo)] += w
    return adj


class JacobianPenalty:
    """Keep the local mapping well behaved, fixing only what is violated.

    Two constraint families, applied through IRLS so the correction stays local
    rather than redistributing whole chains of nodes:

    * **per-channel diagonal** — ``out_R`` must not decrease as ``in_R`` rises, and
      likewise for G and B.  This is the real definition of "no colour reversal".
      Requiring *every* output channel to increase along *every* input axis, as a
      global isotonic projection does, is wrong: a legitimate grade routinely makes
      output-R fall as input-G rises, and enforcing it anyway destroyed the fit
      (core error 0.0079 -> 0.058).
    * **output luma along each input axis** — the constraint that actually prevents
      solarisation, i.e. the picture getting darker as a channel is raised.  It is a
      linear functional of the node values, so it costs nothing extra.

    A crosstalk-magnitude term was tried and removed: scaled against the local
    diagonal it degenerates wherever the transform legitimately flattens, and scaled
    against the identity slope it made both the fold count and the measured artefact
    worse.
    """

    def __init__(self, n: int):
        self.n = n
        self.M = n ** 3

    def _fields(self, x: np.ndarray):
        n = self.n
        d = {(ga, ch): np.diff(x[:, ch].reshape(n, n, n), axis=ga)
             for ga in (0, 1, 2) for ch in (0, 1, 2)}
        luma = x @ LUMA
        dl = {ga: np.diff(luma.reshape(n, n, n), axis=ga) for ga in (0, 1, 2)}
        return d, dl

    def active_weights(self, x: np.ndarray, floor: float = 0.0, luma_floor=None):
        """0/1 masks of the currently violated constraints, plus the violation count."""
        if luma_floor is None:
            luma_floor = floor
        d, dl = self._fields(x)
        w, n_viol = {}, 0
        for ga, ch in ((2, 0), (1, 1), (0, 2)):
            act = d[(ga, ch)] < floor
            n_viol += int(act.sum())
            w[(ga, ch)] = act.astype(np.float64)
        for ga in (0, 1, 2):
            act = dl[ga] < luma_floor
            n_viol += int(act.sum())
            w[("luma", ga)] = act.astype(np.float64)
        return w, n_viol

    def apply(self, x: np.ndarray, weights: dict) -> np.ndarray:
        n = self.n
        g = np.zeros_like(x)
        for ga, ch in ((2, 0), (1, 1), (0, 2)):
            w = weights.get((ga, ch))
            if w is not None and w.any():
                d = np.diff(x[:, ch].reshape(n, n, n), axis=ga) * w
                g[:, ch] += _diff_adjoint(d, ga, n).reshape(-1)
        luma = x @ LUMA
        for ga in (0, 1, 2):
            w = weights.get(("luma", ga))
            if w is not None and w.any():
                d = np.diff(luma.reshape(n, n, n), axis=ga) * w
                adj = _diff_adjoint(d, ga, n).reshape(-1)
                for c in range(3):
                    g[:, c] += LUMA[c] * adj
        return g

    def diag(self, weights: dict) -> np.ndarray:
        """Diagonal of the penalty's normal matrix (M,3), for preconditioning."""
        n = self.n
        g = np.zeros((self.M, 3))
        for ga, ch in ((2, 0), (1, 1), (0, 2)):
            w = weights.get((ga, ch))
            if w is not None and w.any():
                g[:, ch] += _diff_diag(w, ga, n).reshape(-1)
        for ga in (0, 1, 2):
            w = weights.get(("luma", ga))
            if w is not None and w.any():
                adj = _diff_diag(w, ga, n).reshape(-1)
                for c in range(3):
                    g[:, c] += (LUMA[c] ** 2) * adj
        return g

    def report(self, x: np.ndarray) -> dict:
        d, dl = self._fields(x)
        per = []
        for ga, ch in ((2, 0), (1, 1), (0, 2)):
            dd = d[(ga, ch)]
            per.append(("RGB"[ch], dict(min_slope=float(dd.min()),
                                        n_negative=int((dd < 0).sum()),
                                        n_total=int(dd.size))))
        out = dict(per)
        out["n_negative"] = sum(v["n_negative"] for _, v in per)
        out["min_slope"] = min(v["min_slope"] for _, v in per)
        out["luma_min_slope"] = min(float(dl[ga].min()) for ga in (0, 1, 2))
        out["luma_n_negative"] = sum(int((dl[ga] < 0).sum()) for ga in (0, 1, 2))
        return out


# backwards-compatible alias
MonotonePenalty = JacobianPenalty


# ------------------------------------------------------------------- PAVA ----
def pava_1d(y: np.ndarray) -> np.ndarray:
    """L2-optimal non-decreasing (isotonic) projection of a 1-D sequence."""
    n = len(y)
    val = [0.0] * n
    wt = [0] * n
    m = 0
    for i in range(n):
        val[m] = float(y[i])
        wt[m] = 1
        while m > 0 and val[m - 1] > val[m]:
            nw = wt[m - 1] + wt[m]
            val[m - 1] = (val[m - 1] * wt[m - 1] + val[m] * wt[m]) / nw
            wt[m - 1] = nw
            m -= 1
        m += 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for j in range(m):
        out[pos:pos + wt[j]] = val[j]
        pos += wt[j]
    return out


def _pava_cols(Y: np.ndarray) -> np.ndarray:
    out = np.empty_like(Y)
    for c in range(Y.shape[1]):
        out[:, c] = pava_1d(Y[:, c])
    return out


def enforce_monotone(x: np.ndarray, n: int, iters: int = 8) -> np.ndarray:
    """Safety net: isotonic projection on the Jacobian diagonal and on luma.

    Applied per (input axis, matching output channel) pair plus the luma field,
    alternating until no violations remain.  Exists to guarantee the invariant if
    the IRLS penalty has not fully converged.
    """
    y = x.copy()
    luma_sq = float(LUMA @ LUMA)

    def pava_along(v: np.ndarray, ga: int):
        A = np.moveaxis(v.reshape(n, n, n), ga, 0)
        flat = A.reshape(n, -1)
        if not (np.diff(flat, axis=0) < 0).any():
            return None
        return np.moveaxis(_pava_cols(flat).reshape(A.shape), 0, ga).reshape(-1)

    for _ in range(iters):
        changed = False
        for ga, ch in JACOBIAN_DIAG_PAIRS:
            v = pava_along(y[:, ch], ga)
            if v is not None:
                y[:, ch] = v
                changed = True
        # luma: distribute the correction proportionally to the luma weights so the
        # change in luma equals the projected delta
        for ga in (0, 1, 2):
            cur = y @ LUMA
            v = pava_along(cur, ga)
            if v is None:
                continue
            delta = v - cur
            for c in range(3):
                y[:, c] += LUMA[c] * delta / luma_sq
            changed = True
        if not changed:
            break
    return y


def monotonicity_report(x: np.ndarray, n: int) -> dict:
    return JacobianPenalty(n).report(x)


# ------------------------------------------------------------------ folds ----
# one sample strictly inside each of the six tetrahedra of a cell
_TETRA_PROBES = np.array([
    [0.75, 0.50, 0.20],   # fr > fg > fb
    [0.75, 0.20, 0.50],   # fr > fb > fg
    [0.50, 0.20, 0.75],   # fb > fr > fg
    [0.50, 0.75, 0.20],   # fg > fr > fb
    [0.20, 0.75, 0.50],   # fg > fb > fr
    [0.20, 0.50, 0.75],   # fb > fg > fr
])


def cell_determinants(x: np.ndarray, n: int, h_cell: float = 0.01):
    """Jacobian determinant sampled inside every tetrahedron of every cell.

    Returns an array of shape (n-1)^3 x 6.  Sampling one point per *tetrahedron*
    matters: the interpolant is piecewise linear, so a cell can fold in one
    tetrahedron and be perfectly fine in the other five.
    """
    from .tetra import evaluate
    m = n - 1
    h = h_cell / m
    idx = np.arange(m)
    dets = []
    f0s = []
    for probe in _TETRA_PROBES:
        comps = [(idx + probe[a]) / m for a in range(3)]
        k, j, i = np.meshgrid(*comps, indexing="ij")
        c = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
        f0 = evaluate(x, c, n)
        cols = []
        for ax in range(3):
            p = c.copy()
            p[:, ax] += h
            cols.append((evaluate(x, p, n) - f0) / h)
        J = np.stack(cols, axis=2)
        dets.append(np.linalg.det(J))
        f0s.append(c)
    return np.stack(dets, axis=1), m


def fold_report(x: np.ndarray, n: int) -> dict:
    det, m = cell_determinants(x, n)
    worst_per_cell = det.min(axis=1)
    return dict(n_cells=int(det.shape[0]),
                n_folded_cells=int((worst_per_cell <= 0).sum()),
                n_folded_tetra=int((det <= 0).sum()),
                n_tetra=int(det.size),
                min_det=float(det.min()),
                p1_det=float(np.percentile(det, 1)))


def repair_folds(x: np.ndarray, n: int, iters: int = 40, target: float = 0.05,
                 relax: float = 0.6, verbose: bool = False) -> np.ndarray:
    """Locally smooth the regions whose Jacobian folds, **keeping the best state seen**.

    A fold (det J <= 0) is the formal definition of a local colour inversion.
    Rather than globally over-smoothing the LUT to remove a handful of bad cells,
    this blends the nodes of the offending cells toward their neighbourhood mean,
    leaving the rest of the grid untouched.

    The relaxation is a whack-a-mole process: smoothing one cell can fold another,
    so the fold count oscillates as iterations proceed (measured: 37 -> 68 -> 88 on
    the same LUT at 40/100/300 iterations).  Returning the *final* state therefore
    made the fold count depend chaotically on the iteration budget, which is useless
    as an acceptance criterion and as a signal for the adaptive search.  The best
    state encountered is kept instead, and the per-iteration fold count comes for
    free from the determinant sweep already being computed.
    """
    from .tetra import Laplacian

    lap = Laplacian(n)
    y = x.copy()
    probes = len(_TETRA_PROBES)

    def fold_count_from(det_cells):
        return int((det_cells.reshape(-1, probes).min(axis=1) <= 0).sum())

    best_y = y.copy()
    det, _ = cell_determinants(y, n)
    best_folds = fold_count_from(det)
    if verbose:
        print(f"    repair: iter 0 -> folds {best_folds}")

    for it in range(iters):
        worst = det.reshape(-1, probes).min(axis=1)
        bad = (worst <= target).reshape(n - 1, n - 1, n - 1)
        if not bad.any():
            break
        node_mask = np.zeros((n, n, n), dtype=bool)
        kk, jj, ii = np.nonzero(bad)
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    node_mask[kk + dk, jj + dj, ii + di] = True
        mask = node_mask.reshape(-1)
        smoothed = y - lap.apply(y) / 6.0
        for c in range(3):
            y[mask, c] += relax * (smoothed[mask, c] - y[mask, c])
        y = np.clip(y, 0.0, 1.0)

        det, _ = cell_determinants(y, n)
        fc = fold_count_from(det)
        if fc < best_folds:
            best_folds, best_y = fc, y.copy()
        if verbose and (it % 5 == 0 or fc == 0):
            print(f"    repair: iter {it+1} -> folds {fc} (best {best_folds})")
        if best_folds == 0:
            break

    if verbose:
        print(f"    repair: 最终采用折叠数 {best_folds}")
    return best_y


def polish_lut(x: np.ndarray, n: int, rounds: int = 4, repair_iters: int = 40,
               target: float = 0.02, space: str = "srgb", verbose: bool = False):
    """Alternate fold repair and isotonic projection, keeping the best state seen.

    The two operations work against each other: repair removes folds but breaks
    monotonicity, and the isotonic projection restores monotonicity but re-introduces
    a few folds.  Run once, the pipeline reported 155 folds on dataset 2; run from an
    already-polished LUT, the same cycle reported 2.  The interaction is chaotic, so
    the cycle is iterated and the best joint state is kept rather than the last.

    Returns ``(lut, info)``.  ``info`` records every candidate evaluated.
    """
    from .verify import artifact_metrics

    def key(m):
        return (m["folds"], m["luma_viol"], m["max_second"])

    y, gp = gamut_compress(x, space)
    history = []
    best = None
    for r in range(max(1, rounds)):
        y = enforce_monotone(y, n)
        y = repair_folds(y, n, iters=repair_iters, target=target)
        y = enforce_monotone(y, n)
        m = artifact_metrics(y, n)
        m["round"] = r
        history.append({k: v for k, v in m.items() if k != "backward"})
        if verbose:
            print(f"    polish {r}: folds={m['folds']} luma_viol={m['luma_viol']} "
                  f"max2nd={m['max_second']:.5f}")
        if best is None or key(m) < key(best[1]):
            best = (y.copy(), m)
        if m["folds"] == 0 and m["luma_viol"] == 0:
            break
    return best[0], dict(gamut=gp, history=history, chosen=best[1])


def gamut_compress(x: np.ndarray, space: str = "srgb", steps: int = 20):
    """Bring out-of-range node values back into [0,1] by reducing chroma at constant
    Oklab hue and lightness.

    Hard clipping is deliberately avoided: it produces hue-shifted edges and, in a
    LUT, visible contours.
    """
    out = x.copy()
    bad = (x < 0.0).any(axis=1) | (x > 1.0).any(axis=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return out, dict(n_bad=0, max_excursion=0.0)

    exc = float(max(-x[bad].min(), x[bad].max() - 1.0))
    lab = code_to_oklab(x[bad], space)
    lab[:, 0] = np.clip(lab[:, 0], 0.0, 1.0)     # chroma compression cannot fix lightness

    lo = np.zeros(n_bad)
    hi = np.ones(n_bad)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        trial = lab.copy()
        trial[:, 1] *= mid
        trial[:, 2] *= mid
        ok = in_gamut(oklab_to_code(trial, space))
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    trial = lab.copy()
    trial[:, 1] *= lo
    trial[:, 2] *= lo
    out[bad] = np.clip(oklab_to_code(trial, space), 0.0, 1.0)
    return out, dict(n_bad=n_bad, max_excursion=exc)
