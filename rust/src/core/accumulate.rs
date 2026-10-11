//! Streaming accumulation of the A->B colour correspondence statistics.

use crate::core::loader::{build_mask, Img, REC709};
use rayon::prelude::*;

pub struct PairStat {
    pub name: String,
    pub pixels: usize,
    pub mean_abs: f64,
    pub luma_a: f64,
    pub luma_b: f64,
}

pub struct Stats {
    pub grid: usize,
    pub cnt: Vec<f64>,
    pub sa: Vec<f64>,
    pub sb: Vec<f64>,
    pub sb2: Vec<f64>,
    pub gray_cnt: Vec<f64>,
    pub gray_sb: Vec<f64>,
    pub gray_levels: usize,
    pub per_pair: Vec<PairStat>,
    pub skipped: Vec<(String, String)>,
    pub pixels: f64,
}

#[inline]
fn bin_index(p: [f32; 3], grid: usize) -> usize {
    let g = (grid - 1) as f32;
    let q = |v: f32| ((v * g).round() as i64).clamp(0, grid as i64 - 1) as usize;
    q(p[0]) + q(p[1]) * grid + q(p[2]) * grid * grid
}

/// One pass over the pairs builds everything downstream needs.
///
/// Images are decoded in parallel (decode dominates the cost) but accumulated
/// sequentially into a single shared histogram.
pub fn build_statistics(
    pairs: &[(String, String, String)],
    grid: usize,
    target_max: usize,
    gray_chroma: f32,
    gray_levels: usize,
    mut progress: Option<&mut dyn FnMut(usize, usize, &str)>,
    cancel: &dyn Fn() -> bool,
) -> Stats {
    let m = grid * grid * grid;
    let mut st = Stats {
        grid,
        cnt: vec![0.0; m],
        sa: vec![0.0; m * 3],
        sb: vec![0.0; m * 3],
        sb2: vec![0.0; m * 3],
        gray_cnt: vec![0.0; gray_levels],
        gray_sb: vec![0.0; gray_levels * 3],
        gray_levels,
        per_pair: Vec::new(),
        skipped: Vec::new(),
        pixels: 0.0,
    };
    const CHUNK: usize = 4;
    let mut done = 0usize;
    for chunk in pairs.chunks(CHUNK) {
        if cancel() {
            break;
        }
        let decoded: Vec<Result<(Img, Img), String>> = chunk
            .par_iter()
            .map(|(_, pa, pb)| {
                let a = crate::core::loader::load_code(pa, target_max)?;
                let b = crate::core::loader::load_code(pb, target_max)?;
                if a.w != b.w || a.h != b.h {
                    return Err(format!("尺寸不一致 {}x{} vs {}x{}", a.w, a.h, b.w, b.h));
                }
                Ok((a, b))
            })
            .collect();

        for ((name, _, _), d) in chunk.iter().zip(decoded) {
            done += 1;
            if let Some(p) = progress.as_mut() {
                p(done, pairs.len(), name);
            }
            let (a, b) = match d {
                Ok(v) => v,
                Err(e) => {
                    st.skipped.push((name.clone(), e));
                    continue;
                }
            };
            let mask = build_mask(&a, &b, 0.98, 0.015, 90.0, 1);
            let n = a.w * a.h;
            let mut valid = 0usize;
            let mut la_sum = 0.0f64;
            let mut lb_sum = 0.0f64;
            let mut abs_sum = 0.0f64;
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                valid += 1;
                let pa = a.px(i);
                let pb = b.px(i);
                let bi = bin_index(pa, grid);
                st.cnt[bi] += 1.0;
                for c in 0..3 {
                    st.sa[bi * 3 + c] += pa[c] as f64;
                    st.sb[bi * 3 + c] += pb[c] as f64;
                    st.sb2[bi * 3 + c] += (pb[c] as f64) * (pb[c] as f64);
                }
                let chroma = pa[0].max(pa[1]).max(pa[2]) - pa[0].min(pa[1]).min(pa[2]);
                if chroma < gray_chroma {
                    let l = pa[0] * REC709[0] + pa[1] * REC709[1] + pa[2] * REC709[2];
                    let lvl = ((l * gray_levels as f32) as usize).min(gray_levels - 1);
                    st.gray_cnt[lvl] += 1.0;
                    for c in 0..3 {
                        st.gray_sb[lvl * 3 + c] += pb[c] as f64;
                    }
                }
                let la = pa[0] * REC709[0] + pa[1] * REC709[1] + pa[2] * REC709[2];
                let lb = pb[0] * REC709[0] + pb[1] * REC709[1] + pb[2] * REC709[2];
                la_sum += la as f64;
                lb_sum += lb as f64;
                abs_sum += (lb - la).abs() as f64;
            }
            if valid < 1000 {
                st.skipped
                    .push((name.clone(), format!("有效像素仅 {valid}")));
                continue;
            }
            st.pixels += valid as f64;
            st.per_pair.push(PairStat {
                name: name.clone(),
                pixels: valid,
                mean_abs: abs_sum / valid as f64,
                luma_a: la_sum / valid as f64,
                luma_b: lb_sum / valid as f64,
            });
        }
    }
    st
}

pub struct Summary {
    pub a_bar: Vec<[f64; 3]>,
    pub b_bar: Vec<f64>,
    pub w: Vec<f64>,
    pub b_var: Vec<f64>,
    pub min_count: usize,
    pub n_occ: usize,
    pub total_nodes: usize,
    pub coverage: f64,
    pub mass_coverage: f64,
}

pub fn scaled_min_count(total_pixels: f64) -> usize {
    ((5.0 * total_pixels / 1.54e8).round() as usize).max(3)
}

pub fn summarize(st: &Stats) -> Summary {
    let grid = st.grid;
    let mut a_bar = Vec::new();
    let mut b_bar = Vec::new();
    let mut w = Vec::new();
    let mut b_var = Vec::new();
    let mut total_px = 0.0f64;
    for i in 0..grid * grid * grid {
        if st.cnt[i] <= 0.0 {
            continue;
        }
        total_px += st.cnt[i];
        a_bar.push([
            st.sa[i * 3] / st.cnt[i],
            st.sa[i * 3 + 1] / st.cnt[i],
            st.sa[i * 3 + 2] / st.cnt[i],
        ]);
        let bb = [
            st.sb[i * 3] / st.cnt[i],
            st.sb[i * 3 + 1] / st.cnt[i],
            st.sb[i * 3 + 2] / st.cnt[i],
        ];
        b_bar.extend_from_slice(&bb);
        for c in 0..3 {
            b_var.push((st.sb2[i * 3 + c] / st.cnt[i] - bb[c] * bb[c]).max(0.0));
        }
        w.push(st.cnt[i]);
    }
    let total_nodes = grid * grid * grid;
    let mc = scaled_min_count(total_px);
    let mut mass = 0.0f64;
    for i in 0..w.len() {
        if w[i] >= 50.0 {
            mass += w[i];
        }
    }
    let n_occ = w.len();
    let coverage = n_occ as f64 / total_nodes as f64;
    Summary {
        a_bar,
        b_bar,
        w,
        b_var,
        min_count: mc,
        n_occ,
        total_nodes,
        coverage,
        mass_coverage: mass / total_px.max(1e-9),
    }
}

/// Irreducible error floor: within-bin spread of B (RMS per channel).
pub fn irreducible_rms(s: &Summary) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..s.w.len() {
        for c in 0..3 {
            num += s.b_var[i * 3 + c] * s.w[i];
            den += s.w[i];
        }
    }
    (num / den.max(1e-30)).sqrt()
}
