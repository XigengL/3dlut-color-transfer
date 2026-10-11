//! Multi-scale regularised least-squares fit of an n^3 LUT, with adaptive
//! regularisation on the final level.  Ported from `lutfit/fit.py`.

use crate::core::project::{
    artifact_metrics, enforce_monotone, fold_report, polish_lut, repair_folds,
    JacobianPenalty, PenaltyWeights,
};
use crate::core::tetra::{evaluate, nodes_flat, GridOperator, Laplacian};

/// Diagonal of `L^T L` for the clamped 6-neighbour Laplacian.
const LAP_DIAG: f64 = 42.0;

/// Counts matvec applications, to make the GPU question answerable with data rather
/// than intuition: the solver is a sequence of *tiny* matrix-vector products, so what
/// matters is how many of them there are and how big each one is.
pub static MATVEC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct FitOptions {
    pub lut_size: usize,
    pub levels: Vec<usize>,
    pub lam_s: f64,
    pub beta: f64,
    pub gamma: f64,
    pub mono_mu: f64,
    pub mono_floor: f64,
    pub mono_outer: usize,
    pub maxiter: usize,
    pub preserve_neutral: bool,
    pub repair_folds: usize,
    pub adaptive: bool,
    pub max_trials: usize,
    pub fold_tol_frac: f64,
    pub p3: bool,
}

impl FitOptions {
    pub fn new(lut_size: usize) -> Self {
        let mut levels = vec![5usize, 9, 17, 25, 33, 49, 65];
        levels.retain(|&l| l < lut_size);
        levels.push(lut_size);
        Self {
            lut_size,
            levels,
            lam_s: 6.0,
            beta: 2.0,
            gamma: 20.0,
            mono_mu: 1000.0,
            mono_floor: 0.0,
            mono_outer: 8,
            maxiter: 400,
            preserve_neutral: true,
            repair_folds: 40,
            adaptive: true,
            max_trials: 3,
            fold_tol_frac: 1e-4,
            p3: false,
        }
    }
}

pub struct TrialRecord {
    pub trial: usize,
    pub lam_s: f64,
    pub mono_mu: f64,
    pub folds: usize,
    pub min_det: f64,
    pub luma_viol: usize,
    pub max_second: f64,
    pub model: f64,
}

pub struct FitResultCore {
    pub lut: Vec<f64>,
    pub trials: Vec<TrialRecord>,
    pub chosen_trial: usize,
    pub chosen_lam_s: f64,
    pub model: f64,
}

// ------------------------------------------------------------------- PCG ----
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn pcg_free<F: Fn(&[f64]) -> Vec<f64>>(
    matvec: &F,
    b: &[f64],
    diag: &[f64],
    x0: &[f64],
    free: &[bool],
    maxiter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = b.len();
    let mut x = x0.to_vec();
    let bx = matvec(&x);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = if free[i] { b[i] - bx[i] } else { 0.0 };
    }
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = if free[i] { r[i] / diag[i] } else { 0.0 };
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let bnorm = dot(b, b).sqrt() + 1e-300;
    for _ in 0..maxiter {
        let ap_raw = matvec(&p);
        let mut ap = vec![0.0; n];
        for i in 0..n {
            ap[i] = if free[i] { ap_raw[i] } else { 0.0 };
        }
        let denom = dot(&p, &ap);
        if denom <= 0.0 { break; }
        let alpha = rz / denom;
        for i in 0..n { x[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * ap[i]; }
        if dot(&r, &r).sqrt() <= tol * bnorm { break; }
        for i in 0..n {
            z[i] = if free[i] { r[i] / diag[i] } else { 0.0 };
        }
        let rz_new = dot(&r, &z);
        if rz <= 0.0 { break; }
        let beta = rz_new / rz;
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
        rz = rz_new;
    }
    x
}

/// Box-constrained QP `min 0.5 x'Ax - b'x` s.t. `lo <= x <= hi`, primal active set.
/// Keeping the LUT in range *during* the solve lets the rest of the grid compensate,
/// which post-hoc clipping cannot do.
pub fn pcg_box<F: Fn(&[f64]) -> Vec<f64>>(
    matvec: &F,
    b: &[f64],
    diag: &[f64],
    lo: f64,
    hi: f64,
    maxiter: usize,
    outer: usize,
) -> Vec<f64> {
    pcg_box_from(matvec, b, diag, lo, hi, maxiter, outer, None)
}

/// Warm-startable variant.  Restarting every IRLS sweep from zero threw away the
/// previous solution and re-converged from scratch each time.
#[allow(clippy::too_many_arguments)]
pub fn pcg_box_from<F: Fn(&[f64]) -> Vec<f64>>(
    matvec: &F,
    b: &[f64],
    diag: &[f64],
    lo: f64,
    hi: f64,
    maxiter: usize,
    outer: usize,
    x0: Option<&[f64]>,
) -> Vec<f64> {
    let n = b.len();
    let mut x: Vec<f64> = match x0 {
        Some(v) => v.to_vec(),
        None => vec![0.0; n],
    };
    for v in x.iter_mut() { *v = v.clamp(lo, hi); }
    let mut prev_active: Option<Vec<bool>> = None;
    for _ in 0..outer {
        let g = matvec(&x);
        let mut active = vec![false; n];
        for i in 0..n {
            let gi = g[i] - b[i];
            if (x[i] <= lo + 1e-12 && gi >= 0.0) || (x[i] >= hi - 1e-12 && gi <= 0.0) {
                active[i] = true;
            }
        }
        if let Some(pa) = &prev_active {
            if *pa == active { break; }
        }
        prev_active = Some(active.clone());
        let free: Vec<bool> = active.iter().map(|a| !a).collect();
        if !free.iter().any(|&f| f) { break; }
        x = pcg_free(matvec, b, diag, &x, &free, maxiter, 1e-8);
        for v in x.iter_mut() { *v = v.clamp(lo, hi); }
    }
    for v in x.iter_mut() { *v = v.clamp(lo, hi); }
    x
}

// ----------------------------------------------------------------- priors ---
pub fn interp(t: &[f64], v: &[f64], q: f64) -> f64 {
    if q <= t[0] { return v[0]; }
    if q >= t[t.len() - 1] { return v[v.len() - 1]; }
    let mut lo = 0usize;
    while lo + 1 < t.len() && t[lo + 1] < q { lo += 1; }
    let t0 = t[lo];
    let t1 = t[lo + 1];
    let w = if t1 > t0 { (q - t0) / (t1 - t0) } else { 0.0 };
    v[lo] * (1.0 - w) + v[lo + 1] * w
}

pub struct GrayCurve {
    pub t: Vec<f64>,
    pub v: Vec<f64>, // length t.len()*3
}

/// `(g, wg)`: neutral-axis target per node and its weight, flat `m*3` / `m`.
pub fn neutral_targets(n: usize, gc: &GrayCurve, chroma_max: f64) -> (Vec<f64>, Vec<f64>) {
    let coords = nodes_flat(n);
    let m = n * n * n;
    let mut g = vec![0.0; m * 3];
    let mut wg = vec![0.0; m];
    let tv: Vec<f64> = (0..gc.t.len()).collect::<Vec<_>>().iter().map(|&i| gc.t[i]).collect();
    for nd in 0..m {
        let luma = coords[nd * 3] * 0.2126 + coords[nd * 3 + 1] * 0.7152 + coords[nd * 3 + 2] * 0.0722;
        for c in 0..3 {
            let col: Vec<f64> = (0..gc.t.len()).map(|i| gc.v[i * 3 + c]).collect();
            g[nd * 3 + c] = interp(&tv, &col, luma);
        }
        let chroma = coords[nd * 3].max(coords[nd * 3 + 1]).max(coords[nd * 3 + 2])
            - coords[nd * 3].min(coords[nd * 3 + 1]).min(coords[nd * 3 + 2]);
        wg[nd] = (1.0 - chroma / chroma_max).clamp(0.0, 1.0);
    }
    for i in 0..n {
        let nd = i * (1 + n + n * n);
        wg[nd] = 1.0;
        for c in 0..3 {
            let col: Vec<f64> = (0..gc.t.len()).map(|k| gc.v[k * 3 + c]).collect();
            g[nd * 3 + c] = interp(&tv, &col, coords[nd * 3]);
        }
    }
    (g, wg)
}

/// 1 where a node is unsupported, -> 0 where the data is solid.
pub fn support_prior(op: &GridOperator) -> Vec<f64> {
    let d = op.diag();
    let mut positives: Vec<f64> = (0..op.m)
        .map(|i| (d[i * 3] + d[i * 3 + 1] + d[i * 3 + 2]) / 3.0)
        .filter(|&v| v > 0.0)
        .collect();
    positives.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d0 = if positives.is_empty() { 1.0 } else { positives[positives.len() / 2] };
    (0..op.m)
        .map(|i| {
            let di = (d[i * 3] + d[i * 3 + 1] + d[i * 3 + 2]) / 3.0;
            1.0 / (1.0 + di / d0)
        })
        .collect()
}

// ------------------------------------------------------------------ level ---
struct Problem<'a> {
    op: &'a GridOperator,
    lap: &'a Laplacian,
    pen: &'a JacobianPenalty,
    w_sum: f64,
    lam_eff: f64,
    wp_eff: Vec<f64>,
    wg_eff: Vec<f64>,
    mu_eff: f64,
    b_rhs: Vec<f64>,
    pre: Vec<f64>,
    weights: Option<PenaltyWeights>,
}

impl<'a> Problem<'a> {
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        MATVEC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n = x.len();
        let norm = self.op.normal(x);
        let l = if self.lam_eff != 0.0 { self.lap.normal(x) } else { vec![0.0; n] };
        let pen = match &self.weights {
            Some(w) => self.pen.apply(x, w),
            None => vec![0.0; n],
        };
        (0..n)
            .map(|i| {
                norm[i] / self.w_sum
                    + self.lam_eff * l[i]
                    + self.wp_eff[i] * x[i]
                    + self.wg_eff[i] * x[i]
                    + self.mu_eff * pen[i]
            })
            .collect()
    }
}

pub struct LevelSolution {
    pub x: Vec<f64>,
    pub d0: f64,
    pub unit: f64,
}

#[allow(clippy::too_many_arguments)]
fn fit_level(
    op: &GridOperator,
    lap: &Laplacian,
    pen: &JacobianPenalty,
    b_bar: &[f64],
    w: &[f64],
    xr: &[f64],
    wp: &[f64],
    g: &[f64],
    wg: &[f64],
    lam_s: f64,
    beta: f64,
    gamma: f64,
    mono_mu: f64,
    mono_floor: f64,
    mono_outer: usize,
    maxiter: usize,
) -> LevelSolution {
    let m = op.m;
    let w_sum: f64 = w.iter().sum();
    let d = op.diag();
    let mut positives: Vec<f64> = d.iter().cloned().filter(|&v| v > 0.0).collect();
    positives.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d0 = if positives.is_empty() { 1.0 } else { positives[positives.len() / 2] };
    let unit = d0 / w_sum;

    let wp_eff: Vec<f64> = (0..m * 3).map(|i| wp[i / 3] * beta * unit).collect();
    let wg_eff: Vec<f64> = (0..m * 3).map(|i| wg[i / 3] * gamma * unit).collect();
    let lam_eff = lam_s * unit / LAP_DIAG;
    let mu_eff = mono_mu * unit;

    let rhs0 = op.rhs(b_bar);
    let b_rhs: Vec<f64> = (0..m * 3)
        .map(|i| rhs0[i] / w_sum + wp_eff[i] * xr[i] + wg_eff[i] * g[i])
        .collect();
    let pre0: Vec<f64> = (0..m * 3)
        .map(|i| d[i] / w_sum + lam_eff * LAP_DIAG + wp_eff[i] + wg_eff[i] + 1e-12)
        .collect();

    let mut x = vec![0.0f64; m * 3];
    let mut weights: Option<PenaltyWeights> = None;
    for it in 0..mono_outer.max(1) {
        let prob = Problem {
            op, lap, pen, w_sum, lam_eff,
            wp_eff: wp_eff.clone(), wg_eff: wg_eff.clone(),
            mu_eff,
            b_rhs: b_rhs.clone(),
            pre: pre0.clone(),
            weights: weights.clone(),
        };
        let mv = |v: &[f64]| prob.matvec(v);
        let mut pre_it = pre0.clone();
        if mu_eff != 0.0 {
            if let Some(wt) = &weights {
                let pd = pen.diag(wt);
                for i in 0..m * 3 { pre_it[i] += mu_eff * pd[i]; }
            }
        }
        x = pcg_box_from(&mv, &b_rhs, &pre_it, 0.0, 1.0, maxiter, 12,
                         if it == 0 { None } else { Some(&x) });
        if mono_mu <= 0.0 { break; }
        let wnew = pen.active_weights(&x, mono_floor);
        let viol = wnew.n_viol;
        weights = Some(wnew);
        if viol == 0 { break; }
        let _ = it;
    }
    LevelSolution { x, d0, unit }
}

// -------------------------------------------------------------- multiscale --
pub struct FitProgress<'a> {
    pub on_level: &'a mut dyn FnMut(usize, usize, usize), // level idx, total, n
    pub on_trial: &'a mut dyn FnMut(usize, f64, usize),
    pub cancel: &'a dyn Fn() -> bool,
}

/// Fit coarse-to-fine, then adapt the regularisation on the final level.
///
/// `project(x, n) -> model_error` runs the post-processing (gamut compression,
/// isotonic projection, fold repair) and returns the model error.  The final level
/// is retried with doubled `lam_s` / `mono_mu` while folds remain.
pub fn fit_multiscale(
    a_bar: &[[f64; 3]],
    b_bar: &[f64],
    w: &[f64],
    opts: &FitOptions,
    gc: Option<&GrayCurve>,
    mut progress: Option<&mut FitProgress>,
) -> FitResultCore {
    let levels = &opts.levels;
    let final_n = *levels.last().unwrap();
    let pen = JacobianPenalty::new(final_n);
    let mut x: Option<Vec<f64>> = None;
    let mut trials: Vec<TrialRecord> = Vec::new();
    let mut chosen = 0usize;
    let mut chosen_ls = opts.lam_s;
    let mut chosen_model = f64::NAN;

    for (li, &n) in levels.iter().enumerate() {
        if let Some(p) = progress.as_mut() {
            if (p.cancel)() { break; }
            (p.on_level)(li, levels.len(), n);
        }
        let m = n * n * n;
        let op = GridOperator::new(a_bar, n, Some(w));
        let lap = Laplacian::new(n);
        let pn = JacobianPenalty::new(n);
        let x_ref = match &x {
            None => nodes_flat(n),
            Some(prev) => {
                let coords = nodes_flat(n);
                let pts: Vec<[f64; 3]> = (0..m)
                    .map(|i| [coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]])
                    .collect();
                evaluate(prev, &pts, levels[li - 1])
            }
        };
        let wp = support_prior(&op);
        let (g, wg) = match (gc, opts.preserve_neutral) {
            (Some(gc), true) => neutral_targets(n, gc, 0.10),
            _ => (vec![0.0; m * 3], vec![0.0; m]),
        };
        let is_final = li == levels.len() - 1;
        if !is_final {
            let sol = fit_level(&op, &lap, &pn, b_bar, w, &x_ref, &wp, &g, &wg,
                                opts.lam_s, opts.beta, opts.gamma, opts.mono_mu,
                                opts.mono_floor, opts.mono_outer, opts.maxiter);
            x = Some(sol.x);
            continue;
        }

        // ---- final level: adaptive regularisation ----
        let mut ls = opts.lam_s;
        let mut mu = opts.mono_mu;
        let n_trials = if opts.adaptive { opts.max_trials.max(1) } else { 1 };
        let fold_target = (opts.fold_tol_frac * ((final_n - 1).pow(3)) as f64) as usize;
        let mut best: Option<(Vec<f64>, usize, usize, f64)> = None; // x, folds, luma_viol, max_second
        for trial in 0..n_trials {
            if let Some(p) = progress.as_mut() {
                if (p.cancel)() { break; }
                (p.on_trial)(trial, ls, 0);
            }
            let sol = fit_level(&op, &lap, &pn, b_bar, w, &x_ref, &wp, &g, &wg,
                                ls, opts.beta, opts.gamma, mu, opts.mono_floor,
                                opts.mono_outer, opts.maxiter);
            let (y, _hist) = polish_lut(&sol.x, final_n, 4, opts.repair_folds, 0.02,
                                        opts.p3, &[]);
            let am = artifact_metrics(&y, final_n);
            let r = evaluate(&y, a_bar, final_n);
            let rr: Vec<f64> = (0..r.len()).map(|i| r[i] - b_bar[i]).collect();
            let ws: f64 = w.iter().sum();
            let model = (0..a_bar.len())
                .map(|i| w[i] * (0..3).map(|c| rr[i * 3 + c].powi(2)).sum::<f64>())
                .sum::<f64>()
                / ws;
            let model = model.sqrt();
            trials.push(TrialRecord {
                trial, lam_s: ls, mono_mu: mu, folds: am.folds, min_det: am.min_det,
                luma_viol: am.luma_viol, max_second: am.max_second, model,
            });
            let key = (am.folds, am.luma_viol, am.max_second);
            let better = best.as_ref().map_or(true, |b| {
                key < (b.1, b.2, b.3)
            });
            if better {
                best = Some((y.clone(), am.folds, am.luma_viol, am.max_second));
                chosen = trial;
                chosen_ls = ls;
                chosen_model = model;
            }
            if am.folds <= fold_target { break; }
            if trial > 0 && am.folds >= trials[trial - 1].folds { break; }
            ls *= 2.0;
            mu *= 2.0;
        }
        x = Some(best.map(|b| b.0).unwrap_or_else(|| x_ref.clone()));
    }

    FitResultCore {
        lut: x.unwrap(),
        trials,
        chosen_trial: chosen,
        chosen_lam_s: chosen_ls,
        model: chosen_model,
    }
}

/// Convenience wrapper doing the post-processing and returning the finished LUT.
pub fn finish_lut(lut: &[f64], n: usize, repair_folds_iters: usize, p3: bool) -> Vec<f64> {
    let (y, _h) = polish_lut(lut, n, 4, repair_folds_iters, 0.02, p3, &[]);
    let _ = (fold_report(&y, n).n_folded_cells, enforce_monotone(&y, n, 8),
             repair_folds(&y, n, repair_folds_iters, 0.02, 0.6));
    y
}
