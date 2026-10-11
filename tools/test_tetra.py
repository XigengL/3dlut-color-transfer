import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lutfit.tetra import tetra_scheme, evaluate, nodes_flat, GridOperator, Laplacian

rng = np.random.default_rng(0)
n = 7
M = n**3
for trial in range(3):
    pts = rng.random((500, 3))
    idx, w = tetra_scheme(pts, n)
    # weights must be non-negative and sum to 1
    assert np.allclose(w.sum(1), 1.0), w.sum(1).max()
    assert w.min() > -1e-12, w.min()
    # identity LUT -> interpolation of a linear field must be exact
    x = nodes_flat(n)
    out = evaluate(x, pts, n)
    err = np.abs(out - pts).max()
    print(f"  linear-exactness max err = {err:.2e}")
    assert err < 1e-12
    # corners reproduce the input exactly at grid nodes
    gn = nodes_flat(n)
    assert np.abs(evaluate(x, gn, n) - gn).max() < 1e-12

# adjoint test
op = GridOperator(rng.random((400, 3)), n, weights=rng.random(400) + 0.1)
a = rng.normal(size=(M, 3)); b = rng.normal(size=(400, 3))
lhs = (op.forward(a) * op.weights[:, None] * b).sum()
rhs = (a * op.adjoint(b)).sum()
print(f"adjoint test: {lhs:.10f} vs {rhs:.10f}  diff={abs(lhs-rhs):.2e}")
assert abs(lhs - rhs) < 1e-8

# Laplacian: constant field -> zero penalty
lap = Laplacian(n)
c = np.ones((M, 3)) * 3.7
print("laplacian(constant) max =", np.abs(lap.apply(c)).max())
assert np.abs(lap.apply(c)).max() < 1e-12
# linear ramp -> zero second difference in the interior
g = nodes_flat(n)
r = lap.apply(g)
interior = ((g > 0.05) & (g < 0.95)).all(1)
print("laplacian(ramp) interior max =", np.abs(r[interior]).max())
assert np.abs(r[interior]).max() < 1e-12

# normal-equation diag must be positive and match an explicit quadratic form
d = op.diag()
for node in (3, 17, 100):
    v = np.zeros((M, 3)); v[node] = 1.0
    q = (op.forward(v) ** 2 * op.weights[:, None]).sum()
    print(f"diag consistency node {node}: {q:.10f} vs {d[node].sum():.10f}")
    assert abs(q - d[node].sum()) < 1e-10
print("ALL TETRA TESTS PASSED")
