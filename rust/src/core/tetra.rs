//! Tetrahedral interpolation and its sparse linear operator.
//!
//! Flat node index is `f = i + j*n + k*n*n` (R varies fastest), which is exactly the
//! `.cube` ordering.  Ported from `lut3d/tetra.py`.
//!
//! `T(u)` is linear in the node values — which tetrahedron a query falls into depends
//! only on the input coordinate — so fitting is a sparse linear least-squares problem.

/// Corner offsets for each of the six tetrahedra, as a function of grid size.
fn offsets(n: usize) -> [[usize; 4]; 6] {
    let n2 = n * n;
    [
        [0, 1, 1 + n, 1 + n + n2],
        [0, 1, 1 + n2, 1 + n + n2],
        [0, n2, 1 + n2, 1 + n + n2],
        [0, n, 1 + n, 1 + n + n2],
        [0, n, n + n2, 1 + n + n2],
        [0, n2, n + n2, 1 + n + n2],
    ]
}

/// Ordering of (r, g, b) fractions, descending, with stable tie-breaking.
fn case_of(f: [f64; 3]) -> usize {
    let mut o = [0usize, 1, 2];
    // stable descending sort of 3 elements
    if f[o[1]] > f[o[0]] {
        o.swap(0, 1);
    }
    if f[o[2]] > f[o[1]] {
        o.swap(1, 2);
        if f[o[1]] > f[o[0]] {
            o.swap(0, 1);
        }
    }
    match (o[0], o[1], o[2]) {
        (0, 1, 2) => 0, // fr >= fg >= fb
        (0, 2, 1) => 1, // fr >= fb >= fg
        (2, 0, 1) => 2, // fb >= fr >= fg
        (1, 0, 2) => 3, // fg >= fr >= fb
        (1, 2, 0) => 4, // fg >= fb >= fr
        (2, 1, 0) => 5, // fb >= fg >= fr
        _ => unreachable!(),
    }
}

/// Returns `(idx, w)` of shape `(npoints, 4)`: node flat indices and barycentric weights.
pub fn tetra_scheme(points: &[[f64; 3]], n: usize) -> (Vec<[usize; 4]>, Vec<[f64; 4]>) {
    let off = offsets(n);
    let nf = (n - 1) as f64;
    let mut idx = Vec::with_capacity(points.len());
    let mut wts = Vec::with_capacity(points.len());
    for p in points {
        let q = [
            p[0].clamp(0.0, 1.0) * nf,
            p[1].clamp(0.0, 1.0) * nf,
            p[2].clamp(0.0, 1.0) * nf,
        ];
        let mut i0 = [0usize; 3];
        let mut f = [0.0f64; 3];
        for a in 0..3 {
            let fl = q[a].floor();
            let i = if fl < 0.0 { 0 } else { fl as usize };
            let i = i.min(n - 2);
            i0[a] = i;
            f[a] = q[a] - i as f64;
        }
        let base = i0[0] + i0[1] * n + i0[2] * n * n;
        let c = case_of(f);
        let o = off[c];
        idx.push([base + o[0], base + o[1], base + o[2], base + o[3]]);
        let (fr, fg, fb) = (f[0], f[1], f[2]);
        let w = match c {
            0 => [1.0 - fr, fr - fg, fg - fb, fb],
            1 => [1.0 - fr, fr - fb, fb - fg, fg],
            2 => [1.0 - fb, fb - fr, fr - fg, fg],
            3 => [1.0 - fg, fg - fr, fr - fb, fb],
            4 => [1.0 - fg, fg - fb, fb - fr, fr],
            _ => [1.0 - fb, fb - fg, fg - fr, fr],
        };
        wts.push(w);
    }
    (idx, wts)
}

/// Tetrahedral interpolation of a flat LUT (`m*3` values) at `points`.
pub fn evaluate(x: &[f64], points: &[[f64; 3]], n: usize) -> Vec<f64> {
    let (idx, w) = tetra_scheme(points, n);
    let mut out = vec![0.0; points.len() * 3];
    for (s, (ix, ws)) in idx.iter().zip(&w).enumerate() {
        for t in 0..4 {
            let node = ix[t] * 3;
            let ww = ws[t];
            out[s * 3] += ww * x[node];
            out[s * 3 + 1] += ww * x[node + 1];
            out[s * 3 + 2] += ww * x[node + 2];
        }
    }
    out
}

/// Grid node coordinates (M,3) in [0,1], R fastest.
pub fn nodes_flat(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n * n * n * 3);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                v.push(i as f64 / (n - 1) as f64);
                v.push(j as f64 / (n - 1) as f64);
                v.push(k as f64 / (n - 1) as f64);
            }
        }
    }
    v
}

pub fn diagonal_indices(n: usize) -> Vec<usize> {
    (0..n).map(|i| i * (1 + n + n * n)).collect()
}

/// Weighted tetrahedral sampling operator over a fixed set of sample points.
pub struct GridOperator {
    pub n: usize,
    pub m: usize,
    pub idx: Vec<[usize; 4]>,
    pub w: Vec<[f64; 4]>,
    pub weights: Vec<f64>,
}

impl GridOperator {
    pub fn new(points: &[[f64; 3]], n: usize, weights: Option<&[f64]>) -> Self {
        let (idx, w) = tetra_scheme(points, n);
        let nsamp = idx.len();
        Self {
            n,
            m: n * n * n,
            idx,
            w,
            weights: weights
                .map(|x| x.to_vec())
                .unwrap_or_else(|| vec![1.0; nsamp]),
        }
    }

    /// `y = T x`
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.idx.len() * 3];
        for s in 0..self.idx.len() {
            for t in 0..4 {
                let node = self.idx[s][t] * 3;
                let ww = self.w[s][t];
                out[s * 3] += ww * x[node];
                out[s * 3 + 1] += ww * x[node + 1];
                out[s * 3 + 2] += ww * x[node + 2];
            }
        }
        out
    }

    /// `g = T^T (weights * r)`
    pub fn adjoint(&self, r: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.m * 3];
        for s in 0..self.idx.len() {
            let ws = self.weights[s];
            for t in 0..4 {
                let node = self.idx[s][t] * 3;
                let ww = self.w[s][t] * ws;
                g[node] += ww * r[s * 3];
                g[node + 1] += ww * r[s * 3 + 1];
                g[node + 2] += ww * r[s * 3 + 2];
            }
        }
        g
    }

    pub fn normal(&self, x: &[f64]) -> Vec<f64> {
        self.adjoint(&self.forward(x))
    }

    /// Diagonal of `T^T W T`, for a Jacobi preconditioner.
    pub fn diag(&self) -> Vec<f64> {
        let mut g = vec![0.0; self.m * 3];
        for s in 0..self.idx.len() {
            let ws = self.weights[s];
            for t in 0..4 {
                let node = self.idx[s][t] * 3;
                let ww = self.w[s][t] * self.w[s][t] * ws;
                g[node] += ww;
                g[node + 1] += ww;
                g[node + 2] += ww;
            }
        }
        g
    }

    pub fn rhs(&self, y: &[f64]) -> Vec<f64> {
        self.adjoint(y)
    }

    /// Number of sample bins touching each node (a support measure).
    pub fn support(&self) -> Vec<f64> {
        let mut c = vec![0.0; self.m];
        for s in 0..self.idx.len() {
            for t in 0..4 {
                c[self.idx[s][t]] += 1.0;
            }
        }
        c
    }
}

/// Second-difference regulariser with clamped boundaries, so a constant field has
/// exactly zero penalty everywhere.  Array layout is `[k, j, i]`.
pub struct Laplacian {
    pub n: usize,
    pub m: usize,
}

impl Laplacian {
    pub fn new(n: usize) -> Self {
        Self { n, m: n * n * n }
    }

    fn lap_channel(&self, v: &[f64], out: &mut [f64]) {
        let n = self.n;
        let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let km = if k == 0 { 0 } else { k - 1 };
                    let kp = if k + 1 == n { n - 1 } else { k + 1 };
                    let jm = if j == 0 { 0 } else { j - 1 };
                    let jp = if j + 1 == n { n - 1 } else { j + 1 };
                    let im = if i == 0 { 0 } else { i - 1 };
                    let ip = if i + 1 == n { n - 1 } else { i + 1 };
                    let s = v[idx(km, j, i)]
                        + v[idx(kp, j, i)]
                        + v[idx(k, jm, i)]
                        + v[idx(k, jp, i)]
                        + v[idx(k, j, im)]
                        + v[idx(k, j, ip)];
                    out[idx(k, j, i)] = 6.0 * v[idx(k, j, i)] - s;
                }
            }
        }
    }

    /// `L x` (applied per channel)
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.m * 3];
        for c in 0..3 {
            let ch: Vec<f64> = (0..self.m).map(|i| x[i * 3 + c]).collect();
            let mut o = vec![0.0; self.m];
            self.lap_channel(&ch, &mut o);
            for i in 0..self.m {
                out[i * 3 + c] = o[i];
            }
        }
        out
    }

    pub fn normal(&self, x: &[f64]) -> Vec<f64> {
        self.apply(&self.apply(x))
    }
}
