"""Tetrahedral interpolation and its sparse linear operator.

The LUT grid is indexed ``(i, j, k)`` = (R index, G index, B index) with flat
index ``f = i + j*n + k*n*n``.  This is exactly the ordering required by the
``.cube`` format (R varies fastest), so no reshuffling is ever needed.  All LUT
data is carried as a flat ``(M, 3)`` array with ``M = n**3``.

``T(u)`` is linear in the LUT node values: which tetrahedron a query point falls
into depends only on the input coordinate, never on the node values.  All fitting
is therefore a sparse linear least-squares problem.
"""
from __future__ import annotations

import numpy as np

# permutation code -> case id, for the six tetrahedra
_PERM_TO_CASE = np.full(27, -1, dtype=np.int8)
for _perm, _case in (((0, 1, 2), 0), ((0, 2, 1), 1), ((2, 0, 1), 2),
                     ((1, 0, 2), 3), ((1, 2, 0), 4), ((2, 1, 0), 5)):
    _PERM_TO_CASE[_perm[0] * 9 + _perm[1] * 3 + _perm[2]] = _case


def _offsets(n: int) -> np.ndarray:
    """Corner flat-index offsets (relative to the base node) for each of the 6 tetrahedra."""
    n2 = n * n
    return np.array([
        [0, 1,  1 + n,   1 + n + n2],   # 0: fr >= fg >= fb
        [0, 1,  1 + n2,  1 + n + n2],   # 1: fr >= fb >= fg
        [0, n2, 1 + n2,  1 + n + n2],   # 2: fb >= fr >= fg
        [0, n,  1 + n,   1 + n + n2],   # 3: fg >= fr >= fb
        [0, n,  n + n2,  1 + n + n2],   # 4: fg >= fb >= fr
        [0, n2, n + n2,  1 + n + n2],   # 5: fb >= fg >= fr
    ], dtype=np.int64)


def tetra_scheme(points: np.ndarray, n: int):
    """Return ``(idx, w)`` of shape (N,4): node flat indices and barycentric weights."""
    pts = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0)
    q = pts * (n - 1)
    i0 = np.clip(np.floor(q), 0, n - 2).astype(np.int64)
    f = q - i0
    i, j, k = i0[:, 0], i0[:, 1], i0[:, 2]
    base = i + j * n + k * n * n

    order = np.argsort(-f, axis=1, kind="stable")
    case = _PERM_TO_CASE[order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]]
    if np.any(case < 0):                                    # pragma: no cover
        raise RuntimeError("tetrahedral case lookup failed")

    idx = base[:, None] + _offsets(n)[case]

    fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
    table = np.stack([
        np.stack([1 - fr, fr - fg, fg - fb, fb], axis=1),
        np.stack([1 - fr, fr - fb, fb - fg, fg], axis=1),
        np.stack([1 - fb, fb - fr, fr - fg, fg], axis=1),
        np.stack([1 - fg, fg - fr, fr - fb, fb], axis=1),
        np.stack([1 - fg, fg - fb, fb - fr, fr], axis=1),
        np.stack([1 - fb, fb - fg, fg - fr, fr], axis=1),
    ], axis=1)                                              # (N,6,4)
    w = np.take_along_axis(table, case[:, None, None], axis=1)[:, 0, :]
    return idx, w


def evaluate(x: np.ndarray, points: np.ndarray, n: int) -> np.ndarray:
    """Tetrahedral interpolation of flat LUT ``x`` (M,3) at ``points`` (N,3) -> (N,3)."""
    idx, w = tetra_scheme(points, n)
    out = np.zeros((points.shape[0], 3), dtype=np.float64)
    for t in range(4):
        out += w[:, t, None] * x[idx[:, t]]
    return out


def nodes_flat(n: int) -> np.ndarray:
    """Grid node coordinates (M,3) in [0,1] for a size-n LUT (R fastest)."""
    a = np.linspace(0.0, 1.0, n)
    k, j, i = np.meshgrid(a, a, a, indexing="ij")           # -> A[k, j, i]
    return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)


def diagonal_indices(n: int) -> np.ndarray:
    """Flat indices of the neutral (R=G=B) nodes."""
    return np.arange(n) * (1 + n + n * n)


class GridOperator:
    """Weighted tetrahedral sampling operator over a set of sample points.

    ``forward(x) = T x`` and ``adjoint(r) = T^T (weights * r)``, so the
    normal-equation matrix is ``T^T W T``.
    """

    def __init__(self, points: np.ndarray, n: int, weights: np.ndarray | None = None):
        self.n = n
        self.M = n ** 3
        self.idx, self.w = tetra_scheme(points, n)
        self.nsamp = self.idx.shape[0]
        self.weights = (np.ones(self.nsamp, dtype=np.float64) if weights is None
                        else np.asarray(weights, dtype=np.float64))

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((self.nsamp, 3), dtype=np.float64)
        for t in range(4):
            out += self.w[:, t, None] * x[self.idx[:, t]]
        return out

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        rw = r * self.weights[:, None]
        g = np.zeros((self.M, 3), dtype=np.float64)
        for t in range(4):
            wt = self.w[:, t]
            ii = self.idx[:, t]
            for c in range(3):
                g[:, c] += np.bincount(ii, weights=wt * rw[:, c], minlength=self.M)
        return g

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.adjoint(self.forward(x))

    def diag(self) -> np.ndarray:
        """Diagonal of ``T^T W T`` (M,3), for a Jacobi preconditioner."""
        g = np.zeros((self.M, 3), dtype=np.float64)
        for t in range(4):
            wt2 = (self.w[:, t] ** 2) * self.weights
            ii = self.idx[:, t]
            for c in range(3):
                g[:, c] += np.bincount(ii, weights=wt2, minlength=self.M)
        return g

    def rhs(self, y: np.ndarray) -> np.ndarray:
        return self.adjoint(y)


class Laplacian:
    """Second-difference regulariser on the LUT grid.

    Neighbours are clamped at the boundary so a constant field has exactly zero
    penalty everywhere, including on faces and edges.
    """

    def __init__(self, n: int):
        self.n = n
        self.M = n ** 3

    def _lap(self, v: np.ndarray) -> np.ndarray:
        n = self.n
        A = v.reshape(n, n, n)                              # axes (k, j, i)

        def sl(ax, a, b):
            s = [slice(None)] * 3
            s[ax] = slice(a, b)
            return tuple(s)

        out = 6.0 * A
        for ax in range(3):
            lo = np.concatenate([A[sl(ax, 0, 1)], A[sl(ax, 0, n - 1)]], axis=ax)
            hi = np.concatenate([A[sl(ax, 1, n)], A[sl(ax, n - 1, n)]], axis=ax)
            out = out - lo - hi
        return out.reshape(-1)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self._lap(x[:, c]) for c in range(3)], axis=1)

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.apply(self.apply(x))
