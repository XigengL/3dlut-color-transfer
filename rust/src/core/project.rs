//! Constraints and projections: monotonicity, fold repair, gamut compression.
//! Ported from `lut3d/project.py` and `lut3d/verify.py`.

use crate::core::color::{code_to_oklab, in_gamut, oklab_to_code, LUMA};
use crate::core::tetra::{evaluate, nodes_flat, Laplacian};

pub type Px = [f64; 3];

/// One sample strictly inside each of the six tetrahedra of a cell.
const TETRA_PROBES: [[f64; 3]; 6] = [
    [0.75, 0.50, 0.20],
    [0.75, 0.20, 0.50],
    [0.50, 0.20, 0.75],
    [0.50, 0.75, 0.20],
    [0.20, 0.75, 0.50],
    [0.20, 0.50, 0.75],
];

/// `(input grid array axis, output channel)` for out_c vs in_c.
/// Array layout is `[k, j, i]`, so input R is axis 2, G is 1, B is 0.
const DIAG_PAIRS: [(usize, usize); 3] = [(2, 0), (1, 1), (0, 2)];

fn node(x: &[f64], idx: usize, c: usize) -> f64 {
    x[idx * 3 + c]
}

// ---------------------------------------------------------------- penalty ---
/// Keeps the Jacobian well behaved: positive diagonal and non-decreasing luma,
/// applied only where violated (IRLS) so the correction stays local.
pub struct JacobianPenalty {
    pub n: usize,
}

#[derive(Clone)]
pub struct PenaltyWeights {
    /// One mask per `DIAG_PAIRS` entry (input axis, matching output channel).
    pub diag: [Vec<f64>; 3],
    /// Output-luma masks, one per input axis.
    pub luma: [Vec<f64>; 3],
    pub n_viol: usize,
}

impl JacobianPenalty {
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    fn diff_field(&self, x: &[f64], ga: usize, ch: usize) -> Vec<f64> {
        // differences along grid axis `ga` (which is array axis ga in [k,j,i])
        let n = self.n;
        let mut out = Vec::new();
        let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
        // iterate over all lines along `ga`
        let (dims_n, dims_rest): (usize, [usize; 2]) = match ga {
            0 => (n, [n, n]),
            1 => (n, [n, n]),
            _ => (n, [n, n]),
        };
        let _ = dims_n;
        let _ = dims_rest;
        for a in 0..n {
            for b in 0..n {
                let mut prev: Option<f64> = None;
                for t in 0..n {
                    let (k, j, i) = match ga {
                        0 => (t, a, b),
                        1 => (a, t, b),
                        _ => (a, b, t),
                    };
                    let v = node(x, idx(k, j, i), ch);
                    if let Some(p) = prev {
                        out.push(v - p);
                    }
                    prev = Some(v);
                }
            }
        }
        out
    }

    pub fn active_weights(&self, x: &[f64], floor: f64) -> PenaltyWeights {
        let n = self.n;
        let mut w = PenaltyWeights {
            diag: [Vec::new(), Vec::new(), Vec::new()],
            luma: [Vec::new(), Vec::new(), Vec::new()],
            n_viol: 0,
        };
        for (pi, &(ga, ch)) in DIAG_PAIRS.iter().enumerate() {
            let f = self.diff_field(x, ga, ch);
            w.diag[pi] = f
                .iter()
                .map(|&d| if d < floor { 1.0 } else { 0.0 })
                .collect();
            w.n_viol += w.diag[pi].iter().filter(|&&v| v > 0.0).count();
        }
        let luma = self.luma_field(x);
        for (ga, lm) in w.luma.iter_mut().enumerate() {
            for a in 0..n {
                for b in 0..n {
                    let mut prev = f64::NAN;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let v = luma[k * n * n + j * n + i];
                        if !prev.is_nan() {
                            lm.push(if v - prev < floor { 1.0 } else { 0.0 });
                        }
                        prev = v;
                    }
                }
            }
            w.n_viol += lm.iter().filter(|&&v| v > 0.0).count();
        }
        w
    }

    fn luma_field(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n * self.n * self.n)
            .map(|i| node(x, i, 0) * LUMA[0] + node(x, i, 1) * LUMA[1] + node(x, i, 2) * LUMA[2])
            .collect()
    }

    /// Gradient of the penalty: `D^T W D x` summed over active constraints.
    pub fn apply(&self, x: &[f64], w: &PenaltyWeights) -> Vec<f64> {
        let n = self.n;
        let mut g = vec![0.0; n * n * n * 3];
        let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
        // diagonal pairs
        for (pi, &(ga, ch)) in DIAG_PAIRS.iter().enumerate() {
            let mask = &w.diag[pi];
            if mask.is_empty() {
                continue;
            }
            let mut row = 0usize;
            for a in 0..n {
                for b in 0..n {
                    let mut prev: Option<usize> = None;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let cur = idx(k, j, i);
                        if let Some(p) = prev {
                            let d = (node(x, cur, ch) - node(x, p, ch)) * mask[row];
                            g[cur * 3 + ch] += d;
                            g[p * 3 + ch] -= d;
                            row += 1;
                        }
                        prev = Some(cur);
                    }
                }
            }
        }
        // luma
        let luma = self.luma_field(x);
        for ga in 0..3 {
            let mask = &w.luma[ga];
            if mask.is_empty() {
                continue;
            }
            let mut row = 0usize;
            for a in 0..n {
                for b in 0..n {
                    let mut prev: Option<usize> = None;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let cur = idx(k, j, i);
                        if let Some(p) = prev {
                            let d = (luma[cur] - luma[p]) * mask[row];
                            for c in 0..3 {
                                g[cur * 3 + c] += LUMA[c] * d;
                                g[p * 3 + c] -= LUMA[c] * d;
                            }
                            row += 1;
                        }
                        prev = Some(cur);
                    }
                }
            }
        }
        g
    }

    /// Diagonal of the penalty normal matrix, for preconditioning.
    pub fn diag(&self, w: &PenaltyWeights) -> Vec<f64> {
        let n = self.n;
        let mut g = vec![0.0; n * n * n * 3];
        let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
        for (pi, &(ga, ch)) in DIAG_PAIRS.iter().enumerate() {
            let mask = &w.diag[pi];
            let mut row = 0usize;
            for a in 0..n {
                for b in 0..n {
                    let mut prev: Option<usize> = None;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let cur = idx(k, j, i);
                        if let Some(p) = prev {
                            let m = mask[row];
                            g[cur * 3 + ch] += m;
                            g[p * 3 + ch] += m;
                            row += 1;
                        }
                        prev = Some(cur);
                    }
                }
            }
        }
        for ga in 0..3 {
            let mask = &w.luma[ga];
            let mut row = 0usize;
            for a in 0..n {
                for b in 0..n {
                    let mut prev: Option<usize> = None;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let cur = idx(k, j, i);
                        if let Some(p) = prev {
                            let m = mask[row];
                            for c in 0..3 {
                                g[cur * 3 + c] += LUMA[c] * LUMA[c] * m;
                                g[p * 3 + c] += LUMA[c] * LUMA[c] * m;
                            }
                            row += 1;
                        }
                        prev = Some(cur);
                    }
                }
            }
        }
        g
    }
}

// ------------------------------------------------------------------- PAVA ---
/// L2-optimal non-decreasing (isotonic) projection of a sequence.
pub fn pava_1d(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut val = vec![0.0f64; n];
    let mut wt = vec![0usize; n];
    let mut m = 0usize;
    for i in 0..n {
        val[m] = y[i];
        wt[m] = 1;
        while m > 0 && val[m - 1] > val[m] {
            let nw = wt[m - 1] + wt[m];
            val[m - 1] = (val[m - 1] * wt[m - 1] as f64 + val[m] * wt[m] as f64) / nw as f64;
            wt[m - 1] = nw;
            m -= 1;
        }
        m += 1;
    }
    let mut out = vec![0.0f64; n];
    let mut pos = 0usize;
    for j in 0..m {
        for k in 0..wt[j] {
            out[pos + k] = val[j];
        }
        pos += wt[j];
    }
    out
}

fn pava_along(x: &[f64], n: usize, c: usize, ga: usize) -> Option<Vec<f64>> {
    let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
    let mut out: Vec<f64> = x.to_vec();
    let mut changed = false;
    for a in 0..n {
        for b in 0..n {
            let mut line = Vec::with_capacity(n);
            for t in 0..n {
                let (k, j, i) = match ga {
                    0 => (t, a, b),
                    1 => (a, t, b),
                    _ => (a, b, t),
                };
                line.push(node(x, idx(k, j, i), c));
            }
            if line.windows(2).any(|w| w[1] < w[0]) {
                changed = true;
                let p = pava_1d(&line);
                for t in 0..n {
                    let (k, j, i) = match ga {
                        0 => (t, a, b),
                        1 => (a, t, b),
                        _ => (a, b, t),
                    };
                    out[idx(k, j, i) * 3 + c] = p[t];
                }
            }
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

/// Isotonic projection on the Jacobian diagonal and on luma, alternating until no
/// violations remain.  Safety net behind the IRLS penalty.
pub fn enforce_monotone(x: &[f64], n: usize, iters: usize) -> Vec<f64> {
    let mut y = x.to_vec();
    for _ in 0..iters {
        let mut changed = false;
        for &(ga, ch) in &DIAG_PAIRS {
            if let Some(v) = pava_along(&y, n, ch, ga) {
                y = v;
                changed = true;
            }
        }
        // luma: distribute the correction proportionally to the luma weights
        for ga in 0..3 {
            let luma: Vec<f64> = (0..n * n * n)
                .map(|i| {
                    node(&y, i, 0) * LUMA[0] + node(&y, i, 1) * LUMA[1] + node(&y, i, 2) * LUMA[2]
                })
                .collect();
            let mut line_ok = true;
            let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
            'outer: for a in 0..n {
                for b in 0..n {
                    let mut prev = f64::NAN;
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let v = luma[idx(k, j, i)];
                        if !prev.is_nan() && v < prev {
                            line_ok = false;
                            break 'outer;
                        }
                        prev = v;
                    }
                }
            }
            if line_ok {
                continue;
            }
            // project luma then push the delta back into the channels
            let mut newy = y.clone();
            let luma_sq: f64 = LUMA.iter().map(|c| c * c).sum();
            for a in 0..n {
                for b in 0..n {
                    let mut line = Vec::with_capacity(n);
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        line.push(luma[idx(k, j, i)]);
                    }
                    let p = pava_1d(&line);
                    for t in 0..n {
                        let (k, j, i) = match ga {
                            0 => (t, a, b),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        let ni = idx(k, j, i);
                        let delta = p[t] - luma[ni];
                        for c in 0..3 {
                            newy[ni * 3 + c] += LUMA[c] * delta / luma_sq;
                        }
                    }
                }
            }
            y = newy;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    y
}

// ------------------------------------------------------------------ folds ---
/// Jacobian determinant per cell per tetrahedron probe.  Returns (n-1)^3 * 6 values.
pub fn cell_determinants(x: &[f64], n: usize, h_cell: f64) -> Vec<f64> {
    let m = n - 1;
    let h = h_cell / m as f64;
    let mut dets = vec![0.0f64; m * m * m * 6];
    for (pi, probe) in TETRA_PROBES.iter().enumerate() {
        let mut pts: Vec<Px> = Vec::with_capacity(m * m * m);
        for k in 0..m {
            for j in 0..m {
                for i in 0..m {
                    pts.push([
                        (i as f64 + probe[0]) / m as f64,
                        (j as f64 + probe[1]) / m as f64,
                        (k as f64 + probe[2]) / m as f64,
                    ]);
                }
            }
        }
        let f0 = evaluate(x, &pts, n);
        let mut cols: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for ax in 0..3 {
            let shifted: Vec<Px> = pts
                .iter()
                .map(|p| {
                    let mut q = *p;
                    q[ax] += h;
                    q
                })
                .collect();
            let f1 = evaluate(x, &shifted, n);
            cols[ax] = (0..pts.len() * 3).map(|i| (f1[i] - f0[i]) / h).collect();
        }
        for c in 0..m * m * m {
            // J[out][in]
            let mut j = [[0.0f64; 3]; 3];
            for a in 0..3 {
                for o in 0..3 {
                    j[o][a] = cols[a][c * 3 + o];
                }
            }
            let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
            dets[c * 6 + pi] = det;
        }
    }
    dets
}

pub struct FoldReport {
    pub n_cells: usize,
    pub n_folded_cells: usize,
    pub n_folded_tetra: usize,
    pub n_tetra: usize,
    pub min_det: f64,
}

pub fn fold_report(x: &[f64], n: usize) -> FoldReport {
    let dets = cell_determinants(x, n, 0.01);
    let nc = dets.len() / 6;
    let mut folded_cells = 0;
    let mut folded_tetra = 0;
    let mut min_det = f64::INFINITY;
    for c in 0..nc {
        let mut worst = f64::INFINITY;
        for t in 0..6 {
            let d = dets[c * 6 + t];
            if d <= 0.0 {
                folded_tetra += 1;
            }
            if d < worst {
                worst = d;
            }
            if d < min_det {
                min_det = d;
            }
        }
        if worst <= 0.0 {
            folded_cells += 1;
        }
    }
    FoldReport {
        n_cells: nc,
        n_folded_cells: folded_cells,
        n_folded_tetra: folded_tetra,
        n_tetra: dets.len(),
        min_det,
    }
}

/// Locally smooth the folding regions, **keeping the best state seen**.
///
/// The relaxation is a whack-a-mole process — smoothing one cell can fold another —
/// so the fold count oscillates with iteration count.  Returning the final state made
/// the count depend chaotically on the budget and broke the adaptive search; the best
/// state encountered is kept instead.
pub fn repair_folds(x: &[f64], n: usize, iters: usize, target: f64, relax: f64) -> Vec<f64> {
    let lap = Laplacian::new(n);
    let mut y = x.to_vec();
    let mut best_y = y.clone();
    let mut dets = cell_determinants(&y, n, 0.01);
    let count = |d: &Vec<f64>| -> usize {
        (0..d.len() / 6)
            .filter(|&c| (0..6).any(|t| d[c * 6 + t] <= 0.0))
            .count()
    };
    let mut best_folds = count(&dets);

    let m = n - 1;
    for _ in 0..iters {
        if best_folds == 0 {
            break;
        }
        let mut bad = vec![false; m * m * m];
        let mut any = false;
        for c in 0..m * m * m {
            let worst = (0..6)
                .map(|t| dets[c * 6 + t])
                .fold(f64::INFINITY, f64::min);
            if worst <= target {
                bad[c] = true;
                any = true;
            }
        }
        if !any {
            break;
        }
        let mut node_mask = vec![false; n * n * n];
        for k in 0..m {
            for j in 0..m {
                for i in 0..m {
                    if !bad[k * m * m + j * m + i] {
                        continue;
                    }
                    for dk in 0..2 {
                        for dj in 0..2 {
                            for di in 0..2 {
                                node_mask[(k + dk) * n * n + (j + dj) * n + i + di] = true;
                            }
                        }
                    }
                }
            }
        }
        let smoothed = {
            let s = lap.apply(&y);
            (0..n * n * n * 3)
                .map(|i| y[i] - s[i] / 6.0)
                .collect::<Vec<f64>>()
        };
        for nd in 0..n * n * n {
            if !node_mask[nd] {
                continue;
            }
            for c in 0..3 {
                let t = y[nd * 3 + c] + relax * (smoothed[nd * 3 + c] - y[nd * 3 + c]);
                y[nd * 3 + c] = t.clamp(0.0, 1.0);
            }
        }
        dets = cell_determinants(&y, n, 0.01);
        let fc = count(&dets);
        if fc < best_folds {
            best_folds = fc;
            best_y = y.clone();
        }
    }
    best_y
}

pub struct ArtifactMetrics {
    pub folds: usize,
    pub n_tetra: usize,
    pub min_det: f64,
    pub luma_viol: usize,
    pub mono_viol: usize,
    pub max_second: f64,
}

pub fn monotonicity_report(x: &[f64], n: usize) -> (usize, f64, usize) {
    // returns (n_negative_diag, min_slope_diag, luma_n_negative)
    let p = JacobianPenalty::new(n);
    let mut n_neg = 0usize;
    let mut min_slope = f64::INFINITY;
    for &(ga, ch) in &DIAG_PAIRS {
        let f = p.diff_field_pub(x, ga, ch);
        for d in &f {
            if *d < 0.0 {
                n_neg += 1;
            }
            if *d < min_slope {
                min_slope = *d;
            }
        }
    }
    let luma = p.luma_field_pub(x);
    let idx = |k: usize, j: usize, i: usize| k * n * n + j * n + i;
    let mut luma_neg = 0usize;
    for ga in 0..3 {
        for a in 0..n {
            for b in 0..n {
                let mut prev = f64::NAN;
                for t in 0..n {
                    let (k, j, i) = match ga {
                        0 => (t, a, b),
                        1 => (a, t, b),
                        _ => (a, b, t),
                    };
                    let v = luma[idx(k, j, i)];
                    if !prev.is_nan() && v < prev {
                        luma_neg += 1;
                    }
                    prev = v;
                }
            }
        }
    }
    (n_neg, min_slope, luma_neg)
}

impl JacobianPenalty {
    pub fn diff_field_pub(&self, x: &[f64], ga: usize, ch: usize) -> Vec<f64> {
        self.diff_field(x, ga, ch)
    }
    pub fn luma_field_pub(&self, x: &[f64]) -> Vec<f64> {
        self.luma_field(x)
    }
}

/// Largest second difference of the *interpolated* LUT along an axis (banding proxy).
///
/// Sampled the same way as the Python reference (512 points along the axis at a grid
/// of fixed values on the other two axes) so the numbers are comparable.  Computing it
/// on the raw node values instead gives a value ~30x larger, because the step is 16x
/// coarser and a second difference scales with the square of the step.
pub fn max_second_difference(x: &[f64], n: usize) -> f64 {
    let steps = 512usize;
    let lines = 5usize;
    let t: Vec<f64> = (0..steps).map(|i| i as f64 / (steps - 1) as f64).collect();
    let fixed: Vec<f64> = (0..lines)
        .map(|i| 0.05 + 0.9 * i as f64 / (lines - 1) as f64)
        .collect();
    let mut worst = 0.0f64;
    for &(ax, ch) in &[(0usize, 0usize), (1, 1), (2, 2)] {
        for &u in &fixed {
            for &v in &fixed {
                let pts: Vec<[f64; 3]> = t
                    .iter()
                    .map(|&tt| {
                        let mut p = [0.0f64; 3];
                        p[ax] = tt;
                        p[(ax + 1) % 3] = u;
                        p[(ax + 2) % 3] = v;
                        p
                    })
                    .collect();
                let y = evaluate(x, &pts, n);
                let mut d1 = Vec::with_capacity(steps - 1);
                for i in 0..steps - 1 {
                    d1.push(y[(i + 1) * 3 + ch] - y[i * 3 + ch]);
                }
                for w in d1.windows(2) {
                    let s = (w[1] - w[0]).abs();
                    if s > worst {
                        worst = s;
                    }
                }
            }
        }
    }
    worst
}

pub fn artifact_metrics(x: &[f64], n: usize) -> ArtifactMetrics {
    let f = fold_report(x, n);
    let (mono, _min_slope, luma_viol) = monotonicity_report(x, n);
    let max_second = max_second_difference(x, n);
    ArtifactMetrics {
        folds: f.n_folded_cells,
        n_tetra: f.n_folded_tetra,
        min_det: f.min_det,
        luma_viol,
        mono_viol: mono,
        max_second,
    }
}

// ------------------------------------------------------------------ gamut ---
/// Pull out-of-range node values back into [0,1] by reducing chroma at constant
/// Oklab hue and lightness — hard clipping would produce hue-shifted edges.
pub fn gamut_compress(x: &[f64], p3: bool, steps: usize) -> (Vec<f64>, usize, f64) {
    let m = x.len() / 3;
    let mut out = x.to_vec();
    let mut n_bad = 0usize;
    let mut max_exc = 0.0f64;
    for i in 0..m {
        let p: Px = [x[i * 3], x[i * 3 + 1], x[i * 3 + 2]];
        if in_gamut(p, 0.0) {
            continue;
        }
        n_bad += 1;
        let exc = p
            .iter()
            .fold(0.0f64, |a, &v| a.max(-v))
            .max(p.iter().fold(0.0f64, |a, &v| a.max(v - 1.0)));
        if exc > max_exc {
            max_exc = exc;
        }
        let mut lab = code_to_oklab(p, p3);
        lab[0] = lab[0].clamp(0.0, 1.0);
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..steps {
            let mid = 0.5 * (lo + hi);
            let trial = [lab[0], lab[1] * mid, lab[2] * mid];
            if in_gamut(oklab_to_code(trial, p3), 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let fixed = oklab_to_code([lab[0], lab[1] * lo, lab[2] * lo], p3);
        for c in 0..3 {
            out[i * 3 + c] = fixed[c].clamp(0.0, 1.0);
        }
    }
    (out, n_bad, max_exc)
}

/// Alternate fold repair and isotonic projection, keeping the best joint state.
/// The two operations fight each other: repair removes folds but breaks monotonicity,
/// and the isotonic projection restores monotonicity but re-introduces folds.
pub fn polish_lut(
    x: &[f64],
    n: usize,
    rounds: usize,
    repair_iters: usize,
    target: f64,
    p3: bool,
    _nodes: &[f64],
) -> (Vec<f64>, Vec<ArtifactMetrics>) {
    let (mut y, _nb, _me) = gamut_compress(x, p3, 20);
    let mut history = Vec::new();
    let mut best: Option<(Vec<f64>, (usize, usize, f64))> = None;
    for _ in 0..rounds.max(1) {
        y = enforce_monotone(&y, n, 8);
        y = repair_folds(&y, n, repair_iters, target, 0.6);
        y = enforce_monotone(&y, n, 8);
        let m = artifact_metrics(&y, n);
        let key = (m.folds, m.luma_viol, m.max_second);
        history.push(ArtifactMetrics {
            folds: m.folds,
            n_tetra: m.n_tetra,
            min_det: m.min_det,
            luma_viol: m.luma_viol,
            mono_viol: m.mono_viol,
            max_second: m.max_second,
        });
        if best.as_ref().is_none_or(|b| key < b.1) {
            best = Some((y.clone(), key));
        }
        if m.folds == 0 && m.luma_viol == 0 {
            break;
        }
    }
    (best.unwrap().0, history)
}

/// Blend the fitted LUT toward identity (`alpha = 1` is the full look).
pub fn with_strength(x: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let ident = nodes_flat(n);
    x.iter()
        .zip(ident.iter())
        .map(|(a, b)| b + alpha * (a - b))
        .collect()
}

// --------------------------------------------------------------- smoothing ---
/// Separable Gaussian low-pass on the 3D lattice, edges clamped.
///
/// `strength` 0..1 maps to a standard deviation of 0..0.9 grid steps.  Clamping the
/// ends (rather than zero-padding) keeps black and white from being dragged toward
/// grey; a LUT that no longer maps 0 to 0 is a bug, not a smoothing artefact.
pub fn smooth_lut(x: &[f64], n: usize, strength: f64) -> Vec<f64> {
    if strength <= 0.0 {
        return x.to_vec();
    }
    let sigma = (strength * 0.9).max(1e-6);
    let r = (2.0 * sigma).ceil() as usize;
    let mut k = vec![0.0f64; 2 * r + 1];
    for (i, kv) in k.iter_mut().enumerate() {
        let d = i as f64 - r as f64;
        *kv = (-0.5 * (d / sigma).powi(2)).exp();
    }
    let s: f64 = k.iter().sum();
    for kv in k.iter_mut() {
        *kv /= s;
    }

    let idx = |a: usize, b: usize, c: usize| (a * n * n + b * n + c) * 3;
    let mut cur = x.to_vec();
    // axis 0 = array axis k (B), 1 = j (G), 2 = i (R)
    for axis in 0..3 {
        let mut next = cur.clone();
        for a in 0..n {
            for b in 0..n {
                for t in 0..n {
                    let (mut kk, mut jj, mut ii) = (a, b, t);
                    match axis {
                        0 => kk = t,
                        1 => jj = t,
                        _ => ii = t,
                    }
                    let _ = (kk, jj, ii);
                    for c in 0..3 {
                        let mut acc = 0.0;
                        for (ki, &kv) in k.iter().enumerate() {
                            let d = ki as i64 - r as i64;
                            let tt = (t as i64 + d).clamp(0, n as i64 - 1) as usize;
                            let (k2, j2, i2) = match axis {
                                0 => (tt, b, a),
                                1 => (a, tt, b),
                                _ => (a, b, tt),
                            };
                            acc += kv * cur[idx(k2, j2, i2) + c];
                        }
                        let (k2, j2, i2) = match axis {
                            0 => (t, b, a),
                            1 => (a, t, b),
                            _ => (a, b, t),
                        };
                        next[idx(k2, j2, i2) + c] = acc;
                    }
                }
            }
        }
        cur = next;
    }
    cur
}

pub struct Banding {
    /// Largest second difference of the interpolated LUT, in normalised output units.
    pub max_second: f64,
    /// Smallest Jacobian diagonal — how close the LUT comes to a flat spot.
    pub min_slope: f64,
    pub folds: usize,
    pub luma_viol: usize,
}

impl Banding {
    /// The same curvature expressed in output code levels.  This is the number that
    /// actually decides whether a delivery bands: the identical LUT is fine in 10-bit
    /// and marginal in 8-bit, so an unqualified number says nothing.
    pub fn levels_8bit(&self) -> f64 {
        self.max_second * 255.0
    }
    pub fn levels_10bit(&self) -> f64 {
        self.max_second * 1023.0
    }
}

pub fn banding(x: &[f64], n: usize) -> Banding {
    let m = artifact_metrics(x, n);
    let (_, min_slope, _) = monotonicity_report(x, n);
    Banding {
        max_second: m.max_second,
        min_slope,
        folds: m.folds,
        luma_viol: m.luma_viol,
    }
}

/// High-pass view: `output - blur(output)` on **luma**, signed grey, auto-gained.
///
/// A fixed multiplier cannot work here.  The residual left by the blur is of order the
/// LUT's second difference (~1e-3), while the display has 255 levels, so any constant
/// gain is either invisible or blown out — the first version used 24x and rendered a
/// uniformly grey field.  The gain is derived from the residual's own high percentile
/// instead, and returned so the UI can state it.
///
/// Working on luma (not per channel) matters too: channel residuals partly cancel, and
/// banding is a lightness phenomenon.
pub fn amplify_steps(img: &[f64], w: usize, h: usize, target: f64) -> (Vec<u8>, f64, f64) {
    let lu: Vec<f64> = (0..w * h)
        .map(|i| 0.2126 * img[i * 3] + 0.7152 * img[i * 3 + 1] + 0.0722 * img[i * 3 + 2])
        .collect();
    // separable 5-tap box blur
    let r = 2usize;
    let mut t = vec![0.0f64; lu.len()];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for d in -(r as i64)..=r as i64 {
                let xx = (x as i64 + d).clamp(0, w as i64 - 1) as usize;
                a += lu[y * w + xx];
            }
            t[y * w + x] = a / (2 * r + 1) as f64;
        }
    }
    let mut res = vec![0.0f64; lu.len()];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for d in -(r as i64)..=r as i64 {
                let yy = (y as i64 + d).clamp(0, h as i64 - 1) as usize;
                a += t[yy * w + x];
            }
            res[y * w + x] = lu[y * w + x] - a / (2 * r + 1) as f64;
        }
    }
    let mut mag: Vec<f64> = res.iter().map(|v| v.abs()).collect();
    mag.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let peak = *mag.last().unwrap_or(&0.0);
    let p999 = mag[((mag.len() as f64 - 1.0) * 0.999) as usize];
    let scale = if p999 > 0.0 { p999 } else { peak.max(1e-12) };
    let gain = target / scale;
    let mut out = vec![0u8; lu.len() * 3];
    for i in 0..lu.len() {
        let v = ((0.5 + res[i] * gain).clamp(0.0, 1.0) * 255.0) as u8;
        out[i * 3] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    (out, gain, peak)
}
