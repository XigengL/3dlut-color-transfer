//! End-to-end pipeline: two folders of JPEGs in, a fitted LUT plus a report out.

use std::time::Instant;

use crate::core::accumulate::{
    build_statistics, irreducible_rms, scaled_min_count, summarize, Stats,
};
use crate::core::color::{code_to_lab, de00, luma, Px};
use crate::core::coverage::{extrapolation_risk, preflight, verdict, Preflight};
use crate::core::cube::write_cube;
use crate::core::fit::{
    fit_multiscale, FitOptions as CoreFitOptions, FitProgress, GrayCurve, TrialRecord,
};
use crate::core::loader::{find_pairs, load_code, REC709};
use crate::core::project::{pava_1d, with_strength};
use crate::core::tetra::{evaluate, nodes_flat, GridOperator};
use rayon::prelude::*;

/// `par_iter` closures cannot reach back into the caller's mutable state.  A
/// thread-local would be wrong here: rayon work-stealing means the closure runs on
/// whichever worker is free, so per-thread storage silently drops most of the entries.
static CACHED: std::sync::Mutex<Vec<CachedPair>> = std::sync::Mutex::new(Vec::new());

pub struct PipelineOptions {
    pub core: CoreFitOptions,
    pub grid: usize,
    pub target_max: usize,
    pub eval_max: usize,
    pub preview_count: usize,
    pub preview_max: usize,
    pub max_pairs: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            core: CoreFitOptions::new(33),
            grid: 65,
            target_max: 1800,
            eval_max: 1200,
            preview_count: 5,
            preview_max: 900,
            max_pairs: 0,
        }
    }
}

pub struct PairDe {
    pub name: String,
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
    /// Mean input luma and chroma of this pair, used to pick a *diverse* preview set
    /// rather than the five that happen to sort first.
    pub luma: f64,
    pub chroma: f64,
}

impl Preview {
    /// Every preview is emitted at exactly this size (see `make_preview`).
    pub const W: usize = PREVIEW_W;
    pub const H: usize = PREVIEW_H;
}

pub struct Preview {
    pub name: String,
    pub w: usize,
    pub h: usize,
    pub a: Vec<u8>,
    pub pred: Vec<u8>,
    pub b: Vec<u8>,
    pub de: f64,
    pub label: String,
    pub luma: f64,
}

/// A few real pairs kept in memory so the cost of smoothing can be measured without
/// re-decoding the whole dataset (which takes ~20 s).
pub struct CachedPair {
    pub name: String,
    pub w: usize,
    pub h: usize,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub mask: Vec<bool>,
}

#[derive(Default)]
pub struct EvalCache {
    pub pairs: Vec<CachedPair>,
}

/// Mean ΔE00 of `lut` over the cached pairs.  Used to price the smoothing slider.
pub fn mean_de(lut: &[f64], cache: &EvalCache, p3: bool) -> f64 {
    let n = (lut.len() as f64 / 3.0).cbrt().round() as usize;
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for p in &cache.pairs {
        let f: Vec<f32> = p.a.iter().map(|&v| v as f32 / 255.0).collect();
        let pred = apply_lut(lut, &f, n);
        for k in 0..p.w * p.h {
            if !p.mask[k] {
                continue;
            }
            let cp: Px = [pred[k * 3], pred[k * 3 + 1], pred[k * 3 + 2]];
            let cb: Px = [
                p.b[k * 3] as f64 / 255.0,
                p.b[k * 3 + 1] as f64 / 255.0,
                p.b[k * 3 + 2] as f64 / 255.0,
            ];
            sum += de00(code_to_lab(cp, p3), code_to_lab(cb, p3));
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / cnt as f64
    }
}

/// Splits the irreducible floor into the part that a *set* of LUTs could remove
/// (photos edited differently from each other) and the part that no global LUT can
/// (local adjustments inside a single photo).
///
/// Uses the identity `Var(r) = E_j[Var_j(r)] + Var_j[E_j(r)]` with `r = B - b̄(bin(A))`:
/// the cross term vanishes because each image's residuals average to its own mean.
/// So one extra accumulator per image is enough — no per-(bin, image) storage.
#[derive(Default)]
pub struct FloorDecomp {
    pub total: f64,
    pub within: f64,
    pub between: f64,
    pub per_image: Vec<(String, f64)>,
    pub pixels: f64,
}

impl FloorDecomp {
    pub fn within_share(&self) -> f64 {
        if self.total > 0.0 {
            self.within / self.total
        } else {
            0.0
        }
    }
    pub fn between_share(&self) -> f64 {
        if self.total > 0.0 {
            self.between / self.total
        } else {
            0.0
        }
    }
}

pub struct FitResult {
    pub lut: Vec<f64>,
    pub n: usize,
    pub trials: Vec<TrialRecord>,
    pub chosen_lam_s: f64,
    pub preflight: Preflight,
    pub risk: (f64, f64, f64, f64),
    pub residual_model: f64,
    pub residual_floor: f64,
    pub per_pair: Vec<PairDe>,
    pub agg_mean: f64,
    pub agg_p95: f64,
    pub previews: Vec<Preview>,
    pub eval_cache: EvalCache,
    pub floor_decomp: FloorDecomp,
    pub p3: bool,
    pub skipped: Vec<(String, String)>,
    pub n_pairs: usize,
    pub pixels: f64,
    pub elapsed_s: f64,
    pub t_stats: f64,
    pub t_fit: f64,
    pub t_eval: f64,
    pub folds_before: usize,
    pub folds_after: usize,
    pub luma_viol: usize,
    pub max_second: f64,
    pub verdict: String,
}

pub fn gray_curve(st: &Stats) -> Option<GrayCurve> {
    let levels = st.gray_levels;
    let total: f64 = st.gray_cnt.iter().sum();
    let min_count = (200.0 * total / 1.54e8).max(50.0);
    let mut tt = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    for i in 0..levels {
        if st.gray_cnt[i] >= min_count {
            tt.push((i as f64 + 0.5) / levels as f64);
            for c in 0..3 {
                vv.push(st.gray_sb[i * 3 + c] / st.gray_cnt[i]);
            }
        }
    }
    if tt.len() < 4 {
        return None;
    }
    // edge-safe moving average
    let k = 5usize;
    if tt.len() >= k {
        let pad = k / 2;
        let mut sm = vec![0.0; vv.len()];
        for c in 0..3 {
            let col: Vec<f64> = (0..tt.len()).map(|i| vv[i * 3 + c]).collect();
            for i in 0..tt.len() {
                let mut s = 0.0;
                for d in 0..k {
                    let j = (i + d).saturating_sub(pad).min(tt.len() - 1);
                    s += col[j];
                }
                sm[i * 3 + c] = s / k as f64;
            }
        }
        vv = sm;
    }
    // linear extension of the end chords to t=0 and t=1
    let s_lo = (0..3)
        .map(|c| (vv[3 + c] - vv[c]) / (tt[1] - tt[0]).max(1e-9))
        .collect::<Vec<_>>();
    let n1 = tt.len() - 1;
    let s_hi = (0..3)
        .map(|c| (vv[n1 * 3 + c] - vv[(n1 - 1) * 3 + c]) / (tt[n1] - tt[n1 - 1]).max(1e-9))
        .collect::<Vec<_>>();
    let mut t = vec![0.0];
    let mut v = (0..3)
        .map(|c| (vv[c] - s_lo[c] * tt[0]).max(0.0))
        .collect::<Vec<_>>();
    t.extend_from_slice(&tt);
    v.extend_from_slice(&vv);
    t.push(1.0);
    for c in 0..3 {
        v.push((vv[n1 * 3 + c] + s_hi[c] * (1.0 - tt[n1])).min(1.0));
    }
    // monotone per channel
    let mut out = vec![0.0; v.len()];
    for c in 0..3 {
        let col: Vec<f64> = (0..t.len()).map(|i| v[i * 3 + c]).collect();
        let p = pava_1d(&col);
        for i in 0..t.len() {
            out[i * 3 + c] = p[i];
        }
    }
    Some(GrayCurve { t, v: out })
}

/// Apply a LUT to an image (flat `w*h*3` code values).
pub fn apply_lut(x: &[f64], img: &[f32], n: usize) -> Vec<f64> {
    let np = img.len() / 3;
    let pts: Vec<Px> = (0..np)
        .map(|i| {
            [
                img[i * 3] as f64,
                img[i * 3 + 1] as f64,
                img[i * 3 + 2] as f64,
            ]
        })
        .collect();
    let mut out = evaluate(x, &pts, n);
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}

pub struct Progress<'a> {
    pub on_stage: &'a mut dyn FnMut(&str, f64),
    pub cancel: &'a dyn Fn() -> bool,
}

pub fn run_fit(
    input_dir: &str,
    target_dir: &str,
    opts: &PipelineOptions,
    progress: &mut Progress,
) -> Result<FitResult, String> {
    let t0 = Instant::now();
    let mut pairs = find_pairs(input_dir, target_dir);
    if pairs.is_empty() {
        return Err("两个文件夹里没有找到同名图片对".into());
    }
    if opts.max_pairs > 0 {
        pairs.truncate(opts.max_pairs);
    }
    let n_pairs_total = pairs.len();

    // ---- accumulate ----
    (progress.on_stage)("读取图片、统计颜色分布", 0.0);
    let mut cb = |done: usize, total: usize, name: &str| {
        (progress.on_stage)(
            &format!("读取图片、统计颜色分布  {name}"),
            0.35 * done as f64 / total as f64,
        );
    };
    let t_a = Instant::now();
    let st = build_statistics(
        &pairs,
        opts.grid,
        opts.target_max,
        0.02,
        64,
        Some(&mut cb),
        progress.cancel,
    );
    let t_stats = t_a.elapsed().as_secs_f64();
    if (progress.cancel)() {
        return Err("已取消".into());
    }

    // ---- preflight ----
    (progress.on_stage)("预检覆盖度", 0.36);
    let sum = summarize(&st);
    let keep: Vec<usize> = (0..sum.w.len())
        .filter(|&i| sum.w[i] >= sum.min_count as f64)
        .collect();
    let a_bar: Vec<Px> = keep.iter().map(|&i| sum.a_bar[i]).collect();
    let mut b_bar: Vec<f64> = Vec::with_capacity(keep.len() * 3);
    let mut w: Vec<f64> = Vec::with_capacity(keep.len());
    let mut b_var: Vec<f64> = Vec::with_capacity(keep.len() * 3);
    for &i in &keep {
        for c in 0..3 {
            b_bar.push(sum.b_bar[i * 3 + c]);
        }
        w.push(sum.w[i]);
        for c in 0..3 {
            b_var.push(sum.b_var[i * 3 + c]);
        }
    }
    let floor = {
        let num: f64 = (0..w.len())
            .map(|i| (0..3).map(|c| b_var[i * 3 + c] * w[i]).sum::<f64>())
            .sum();
        // Same convention as the model error: sum over channels / sum over weights.
        // (Dividing by 3 here made the floor sqrt(3) too small and inconsistent.)
        let den: f64 = w.iter().sum::<f64>();
        (num / den.max(1e-30)).sqrt()
    };
    // Flat per-bin mean of B, for the floor decomposition in the eval pass.
    let mut bmean = vec![[0.0f64; 3]; st.grid * st.grid * st.grid];
    for i in 0..bmean.len() {
        if st.cnt[i] > 0.0 {
            for c in 0..3 {
                bmean[i][c] = st.sb[i * 3 + c] / st.cnt[i];
            }
        }
    }
    let pf = preflight(&a_bar, &w, opts.core.lut_size, 4.0);
    let gc = gray_curve(&st);
    let total_px: f64 = w.iter().sum();

    // ---- fit ----
    (progress.on_stage)("拟合 LUT", 0.36);
    // Both hooks need the same sink; RefCell gives interior mutability without
    // handing out two &mut borrows.
    let sink = std::cell::RefCell::new(&mut *progress.on_stage);
    // Weight each level by n^3 — the 33^3 solve is ~60% of the work, so an unweighted
    // level counter sits at 80% while most of the time is still ahead.
    let weights: Vec<f64> = opts
        .core
        .levels
        .iter()
        .map(|&n| (n * n * n) as f64)
        .collect();
    let w_total: f64 = weights.iter().sum::<f64>();
    let mut done_w = 0.0f64;
    let mut on_level = |li: usize, tot: usize, n: usize| {
        if li > 0 {
            done_w += weights.get(li - 1).copied().unwrap_or(0.0);
        }
        let f = 0.36 + 0.56 * (done_w / w_total);
        let msg = format!("拟合 LUT  {n}³ 网格（第 {}/{tot} 层）", li + 1);
        (sink.borrow_mut())(&msg, f);
    };
    let coarse_w: f64 = weights.iter().rev().skip(1).sum();
    let mut on_trial = |trial: usize, ls: f64, _f: usize| {
        let note = if trial == 0 {
            String::new()
        } else {
            format!("（正则化加强 第 {} 次）", trial + 1)
        };
        let frac = 0.36
            + 0.56 * (coarse_w / w_total)
            + 0.56 * (1.0 - coarse_w / w_total) * (trial as f64 / 3.0);
        let msg = format!("末层求解  33³ 网格  lam_s={ls:.4} {note}");
        (sink.borrow_mut())(&msg, frac.min(0.92));
    };
    let mut fp = FitProgress {
        on_level: &mut on_level,
        on_trial: &mut on_trial,
        cancel: progress.cancel,
    };
    let t_b = Instant::now();
    let fit = fit_multiscale(&a_bar, &b_bar, &w, &opts.core, gc.as_ref(), Some(&mut fp));
    let t_fit = t_b.elapsed().as_secs_f64();

    // ---- structural metrics ----
    (progress.on_stage)("结构验证", 0.93);
    let n = opts.core.lut_size;
    let raw_mono = crate::core::project::artifact_metrics(&enforce_mono_probe(&fit.lut, n), n);
    let am = crate::core::project::artifact_metrics(&fit.lut, n);
    let risk = extrapolation_risk(&fit.lut, &pf, n);

    // ---- ΔE on every pair (no previews yet) ----
    let t_c = Instant::now();
    let n_eval = pairs.len();
    let grid = opts.grid;
    let bin_of = |p: [f32; 3]| -> usize {
        let g = (grid - 1) as f32;
        let q = |v: f32| ((v * g).round() as i64).clamp(0, grid as i64 - 1) as usize;
        q(p[0]) + q(p[1]) * grid + q(p[2]) * grid * grid
    };
    let evals: Vec<Option<(PairDe, [f64; 4])>> = pairs
        .par_iter()
        .enumerate()
        .map(|(i, (name, pa, pb))| {
            let a = load_code(pa, opts.eval_max).ok()?;
            let b = load_code(pb, opts.eval_max).ok()?;
            if a.w != b.w || a.h != b.h {
                return None;
            }
            let mask = crate::core::loader::build_mask(&a, &b, 0.98, 0.015, 90.0, 1);
            let pred = apply_lut(&fit.lut, &a.data, n);
            let mut de: Vec<f64> = Vec::new();
            let (mut lu, mut ch) = (0.0f64, 0.0f64);
            let mut cnt = 0.0f64;
            // [n, Σr, Σr², Σ(Σr per channel)^2/n] accumulated per image for the decomposition
            let mut s1 = [0.0f64; 3];
            let mut s2 = [0.0f64; 3];
            let mut npx = 0.0f64;
            for k in 0..a.w * a.h {
                if !mask[k] {
                    continue;
                }
                let cp: Px = [pred[k * 3], pred[k * 3 + 1], pred[k * 3 + 2]];
                let cb: Px = [
                    b.data[k * 3] as f64,
                    b.data[k * 3 + 1] as f64,
                    b.data[k * 3 + 2] as f64,
                ];
                de.push(de00(
                    code_to_lab(cp, opts.core.p3),
                    code_to_lab(cb, opts.core.p3),
                ));
                let pa_ = a.px(k);
                lu += luma([pa_[0] as f64, pa_[1] as f64, pa_[2] as f64]);
                ch += (pa_[0].max(pa_[1]).max(pa_[2]) - pa_[0].min(pa_[1]).min(pa_[2])) as f64;
                cnt += 1.0;
                let bi = bin_of(pa_);
                if st.cnt[bi] > 0.0 {
                    for c in 0..3 {
                        let r = cb[c] - bmean[bi][c];
                        s1[c] += r;
                        s2[c] += r * r;
                    }
                    npx += 1.0;
                }
            }
            if de.is_empty() {
                return None;
            }
            let mean = de.iter().sum::<f64>() / de.len() as f64;
            let mut s = de.clone();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let p95 = s[((s.len() as f64 * 0.95) as usize).min(s.len() - 1)];
            let mx = *s.last().unwrap();
            let _ = i;
            let within: f64 = (0..3)
                .map(|c| (s2[c] - s1[c] * s1[c] / npx.max(1.0)).max(0.0))
                .sum();
            let between: f64 = (0..3).map(|c| s1[c] * s1[c] / npx.max(1.0)).sum();
            Some((
                PairDe {
                    name: name.clone(),
                    mean,
                    p95,
                    max: mx,
                    luma: lu / cnt,
                    chroma: ch / cnt,
                },
                [npx, within, between, 0.0],
            ))
        })
        .collect();
    let mut floor_decomp = FloorDecomp::default();
    let mut per_pair: Vec<PairDe> = Vec::new();
    for (pd, st_) in evals.into_iter().flatten() {
        if st_[0] > 0.0 {
            floor_decomp.total += st_[1] + st_[2];
            floor_decomp.within += st_[1];
            floor_decomp.between += st_[2];
            floor_decomp.pixels += st_[0];
            floor_decomp
                .per_image
                .push((pd.name.clone(), st_[1] + st_[2]));
        }
        per_pair.push(pd);
    }
    floor_decomp
        .per_image
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    per_pair.sort_by(|a, b| a.name.cmp(&b.name));

    // ---- pick a diverse preview set, then load just those ----
    (progress.on_stage)("生成预览", 0.99);
    let picks = select_previews(&per_pair, opts.preview_count);
    CACHED.lock().unwrap().clear();
    let mut previews: Vec<Preview> = picks
        .par_iter()
        .filter_map(|&i| {
            let p = &per_pair[i];
            let (_, pa, pb) = pairs.iter().find(|(nm, _, _)| *nm == p.name)?;
            let a = load_code(pa, opts.preview_max).ok()?;
            let b = load_code(pb, opts.preview_max).ok()?;
            if a.w != b.w || a.h != b.h {
                return None;
            }
            let pred = apply_lut(&fit.lut, &a.data, n);
            let mut pv = make_preview(&p.name, &a, &b, &pred, p);
            pv.label = grade_label(p.mean);
            // stash the raw pair so smoothing can be priced later without re-decoding
            let to8 = |v: &[f32]| {
                v.iter()
                    .map(|&x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                    .collect()
            };
            let mask = crate::core::loader::build_mask(&a, &b, 0.98, 0.015, 90.0, 1);
            CACHED.lock().unwrap().push(CachedPair {
                name: p.name.clone(),
                w: a.w,
                h: a.h,
                a: to8(&a.data),
                b: to8(&b.data),
                mask,
            });
            Some(pv)
        })
        .collect();
    previews.sort_by(|a, b| a.de.partial_cmp(&b.de).unwrap());

    let eval_cache = EvalCache {
        pairs: CACHED.lock().unwrap().drain(..).collect(),
    };
    let agg_mean = if per_pair.is_empty() {
        f64::NAN
    } else {
        per_pair.iter().map(|p| p.mean).sum::<f64>() / per_pair.len() as f64
    };
    let agg_p95 = if per_pair.is_empty() {
        f64::NAN
    } else {
        per_pair.iter().map(|p| p.p95).sum::<f64>() / per_pair.len() as f64
    };
    let _ = n_eval;

    let v = verdict(&pf);
    let n_ok = n_pairs_total - st.skipped.len();
    let skipped = st.skipped.clone();
    Ok(FitResult {
        n,
        folds_before: raw_mono.folds,
        folds_after: am.folds,
        luma_viol: am.luma_viol,
        max_second: am.max_second,
        lut: fit.lut,
        trials: fit.trials,
        chosen_lam_s: fit.chosen_lam_s,
        preflight: pf,
        risk,
        residual_model: fit.model,
        residual_floor: floor,
        per_pair,
        agg_mean,
        agg_p95,
        previews,
        eval_cache,
        floor_decomp,
        p3: opts.core.p3,
        skipped,
        n_pairs: n_ok,
        pixels: total_px,
        elapsed_s: t0.elapsed().as_secs_f64(),
        t_stats,
        t_fit,
        t_eval: t_c.elapsed().as_secs_f64(),
        verdict: v,
    })
}

/// Greedy farthest-point selection in (luma, chroma, ΔE) space.
///
/// Picking five names at random gave five pictures of the same kind of scene.  Starting
/// from the median-ΔE pair and repeatedly taking whatever is furthest from everything
/// chosen so far spreads the selection across brightness, saturation and difficulty.
fn select_previews(pairs: &[PairDe], k: usize) -> Vec<usize> {
    let n = pairs.len();
    if n <= k {
        return (0..n).collect();
    }
    let norm = |v: f64, lo: f64, hi: f64| if hi > lo { (v - lo) / (hi - lo) } else { 0.0 };
    let lo = |f: fn(&PairDe) -> f64| pairs.iter().map(f).fold(f64::INFINITY, f64::min);
    let hi = |f: fn(&PairDe) -> f64| pairs.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
    let (ll, hl) = (lo(|p| p.luma), hi(|p| p.luma));
    let (lc, hc) = (lo(|p| p.chroma), hi(|p| p.chroma));
    let (ld, hd) = (lo(|p| p.mean), hi(|p| p.mean));
    let feat: Vec<[f64; 3]> = pairs
        .iter()
        .map(|p| {
            [
                norm(p.luma, ll, hl) * 1.0,
                norm(p.chroma, lc, hc) * 1.0,
                norm(p.mean, ld, hd) * 1.4,
            ]
        })
        .collect();

    // start at the median ΔE — the "typical" picture on the target's own scale
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pairs[a].mean.partial_cmp(&pairs[b].mean).unwrap());
    let mut sel = vec![order[n / 2]];
    let d2 = |a: &[f64; 3], b: &[f64; 3]| {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    };
    while sel.len() < k {
        let mut best = 0usize;
        let mut best_d = -1.0f64;
        for i in 0..n {
            if sel.contains(&i) {
                continue;
            }
            let m = sel
                .iter()
                .map(|&j| d2(&feat[i], &feat[j]))
                .fold(f64::INFINITY, f64::min);
            if m > best_d {
                best_d = m;
                best = i;
            }
        }
        sel.push(best);
    }
    sel.sort_by(|&a, &b| pairs[a].mean.partial_cmp(&pairs[b].mean).unwrap());
    sel
}

fn grade_label(de: f64) -> String {
    if de < 1.0 {
        "很接近".into()
    } else if de < 2.0 {
        "接近".into()
    } else if de < 4.0 {
        "一般".into()
    } else if de < 7.0 {
        "偏差".into()
    } else {
        "偏差较大".into()
    }
}

fn enforce_mono_probe(x: &[f64], n: usize) -> Vec<f64> {
    crate::core::project::enforce_monotone(x, n, 8)
}

/// All previews are emitted at exactly this size.  Letting each one keep its own
/// aspect ratio meant the UI indexed them with the first one's dimensions, which is
/// what turned every column after the first into garbage.
pub const PREVIEW_W: usize = 420;
pub const PREVIEW_H: usize = 315;

/// Centre-crop to 4:3, then average down to exactly `PREVIEW_W x PREVIEW_H`.
fn crop_scale(src: &[f32], sw: usize, sh: usize, out: &mut [u8]) {
    let aspect = PREVIEW_W as f64 / PREVIEW_H as f64;
    let (mut cw, mut ch) = (sw as f64, sh as f64);
    if cw / ch > aspect {
        cw = ch * aspect;
    } else {
        ch = cw / aspect;
    }
    let ox = (sw as f64 - cw) / 2.0;
    let oy = (sh as f64 - ch) / 2.0;
    let sx = cw / PREVIEW_W as f64;
    let sy = ch / PREVIEW_H as f64;
    for y in 0..PREVIEW_H {
        for x in 0..PREVIEW_W {
            let x0 = (ox + x as f64 * sx) as usize;
            let y0 = (oy + y as f64 * sy) as usize;
            let x1 = ((ox + (x + 1) as f64 * sx) as usize).max(x0 + 1).min(sw);
            let y1 = ((oy + (y + 1) as f64 * sy) as usize).max(y0 + 1).min(sh);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0f32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let i = (yy * sw + xx) * 3;
                    acc[0] += src[i];
                    acc[1] += src[i + 1];
                    acc[2] += src[i + 2];
                    n += 1.0;
                }
            }
            let di = (y * PREVIEW_W + x) * 3;
            for c in 0..3 {
                out[di + c] = ((acc[c] / n).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
}

fn make_preview(
    name: &str,
    a: &crate::core::loader::Img,
    b: &crate::core::loader::Img,
    pred: &[f64],
    d: &PairDe,
) -> Preview {
    let mut av = vec![0u8; PREVIEW_W * PREVIEW_H * 3];
    let mut pv = vec![0u8; PREVIEW_W * PREVIEW_H * 3];
    let mut bv = vec![0u8; PREVIEW_W * PREVIEW_H * 3];
    crop_scale(&a.data, a.w, a.h, &mut av);
    crop_scale(&b.data, b.w, b.h, &mut bv);
    let pf: Vec<f32> = pred.iter().map(|&v| v as f32).collect();
    crop_scale(&pf, a.w, a.h, &mut pv);
    Preview {
        name: name.to_string(),
        w: PREVIEW_W,
        h: PREVIEW_H,
        a: av,
        pred: pv,
        b: bv,
        de: d.mean,
        label: String::new(),
        luma: d.luma,
    }
}

/// The LUT as it will actually be delivered: strength scaling, then smoothing.
///
/// Smoothing is a post-process on the grid, so it has to be applied *after* the strength
/// blend and it has to be applied everywhere the LUT is consumed.  An earlier version
/// only fed it to the test chart, which meant dragging the slider changed nothing in the
/// photo preview and — worse — nothing in the exported file.
pub fn final_lut(result: &FitResult, strength: f64, smooth: f64) -> Vec<f64> {
    let x = with_strength(&result.lut, result.n, strength);
    crate::core::project::smooth_lut(&x, result.n, smooth)
}

pub fn export(result: &FitResult, path: &str, strength: f64, smooth: f64) -> std::io::Result<()> {
    let x = final_lut(result, strength, smooth);
    write_cube(path, &x, result.n, "3dlut A->B grade")
}

pub fn strength_preview(result: &FitResult, strength: f64) -> Vec<f64> {
    with_strength(&result.lut, result.n, strength)
}

#[allow(dead_code)]
fn unused(_: &GridOperator, _: f64, _: f64, _: f64, _: f64) {
    let _ = (
        luma([0.0; 3]),
        nodes_flat(2),
        irreducible_rms,
        scaled_min_count,
        REC709,
    );
}
