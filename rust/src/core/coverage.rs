//! Pre-flight coverage: how much of the colour cube actually has data.

use crate::core::tetra::GridOperator;
pub type Px = [f64; 3];

pub struct Preflight {
    pub n: usize,
    pub n_nodes: usize,
    pub n_supported: usize,
    pub n_bare: usize,
    pub node_coverage: f64,
    pub mass_coverage: f64,
    pub median_bin_count: f64,
    pub supported: Vec<bool>,
    pub mass: Vec<f64>,
}

pub fn preflight(a_bar: &[Px], w: &[f64], n: usize, min_support: f64) -> Preflight {
    let m = n * n * n;
    let op = GridOperator::new(a_bar, n, Some(w));
    let mut cnt = vec![0.0f64; m];
    let mut mass = vec![0.0f64; m];
    for s in 0..op.idx.len() {
        for t in 0..4 {
            let nd = op.idx[s][t];
            cnt[nd] += 1.0;
            mass[nd] += op.weights[s];
        }
    }
    let total: f64 = mass.iter().sum::<f64>().max(1e-300);
    let supported: Vec<bool> = cnt.iter().map(|&c| c >= min_support).collect();
    let n_supported = supported.iter().filter(|&&s| s).count();
    let mass_cov: f64 = (0..m)
        .filter(|&i| supported[i])
        .map(|i| mass[i])
        .sum::<f64>()
        / total;
    let mut occ: Vec<f64> = cnt.iter().cloned().filter(|&c| c > 0.0).collect();
    occ.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if occ.is_empty() {
        0.0
    } else {
        occ[occ.len() / 2]
    };
    Preflight {
        n,
        n_nodes: m,
        n_supported,
        n_bare: m - n_supported,
        node_coverage: n_supported as f64 / m as f64,
        mass_coverage: mass_cov,
        median_bin_count: med,
        supported,
        mass,
    }
}

/// How strongly the LUT acts where there is no data.  There is no way to measure the
/// *error* of an extrapolation — that is what having no data means — but the magnitude
/// of the change can be bounded and reported.
pub fn extrapolation_risk(x: &[f64], pf: &Preflight, n: usize) -> (f64, f64, f64, f64) {
    let coords = crate::core::tetra::nodes_flat(n);
    let m = n * n * n;
    let mut bare_sum = 0.0;
    let mut bare_n = 0usize;
    let mut bare_max: f64 = 0.0;
    let mut sup_sum = 0.0;
    let mut sup_n = 0usize;
    for i in 0..m {
        let mut dev: f64 = 0.0;
        for c in 0..3 {
            dev = dev.max((x[i * 3 + c] - coords[i * 3 + c]).abs());
        }
        if pf.supported[i] {
            sup_sum += dev;
            sup_n += 1;
        } else {
            bare_sum += dev;
            bare_n += 1;
            bare_max = bare_max.max(dev);
        }
    }
    (
        if bare_n > 0 {
            bare_sum / bare_n as f64
        } else {
            0.0
        },
        bare_max,
        if sup_n > 0 {
            sup_sum / sup_n as f64
        } else {
            0.0
        },
        pf.mass_coverage,
    )
}

pub fn verdict(pf: &Preflight) -> String {
    let mc = pf.mass_coverage;
    let mut s = String::new();
    if mc > 0.995 {
        s.push_str(&format!(
            "像素质量覆盖 {:.3}% —— 训练数据完全落在有支撑区域，LUT 在真实照片上可靠。\n",
            mc * 100.0
        ));
    } else if mc > 0.98 {
        s.push_str(&format!(
            "像素质量覆盖 {:.3}% —— 可接受，极少数像素落在外推区。\n",
            mc * 100.0
        ));
    } else {
        s.push_str(&format!(
            "像素质量覆盖仅 {:.3}% —— 较多像素落在无数据区，建议增加或更换训练图。\n",
            mc * 100.0
        ));
    }
    s.push_str(&format!(
        "色彩网格覆盖 {:.1}%（{} / {} 个节点无支撑）。这些区域是平滑外推：不会出色带或反转，但颜色是推测值。",
        pf.node_coverage * 100.0, pf.n_bare, pf.n_nodes));
    s
}
