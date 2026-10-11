#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop
)]
mod common;
use common::{have, max_abs_diff, read_npy};

use lut3d::core::color::{code_to_lab, code_to_oklab, de00, decode_srgb, encode_srgb, Px};

#[test]
fn ciede2000_matches_sharma_reference() {
    if !have("tests/golden/de00_lab1.npy") {
        return;
    }
    let (_, l1) = read_npy("tests/golden/de00_lab1.npy");
    let (_, l2) = read_npy("tests/golden/de00_lab2.npy");
    let (_, expect) = read_npy("tests/golden/de00_expect.npy");
    let mut worst = 0.0f64;
    for i in 0..expect.len() {
        let a: Px = [l1[i * 3], l1[i * 3 + 1], l1[i * 3 + 2]];
        let b: Px = [l2[i * 3], l2[i * 3 + 1], l2[i * 3 + 2]];
        let d = (de00(a, b) - expect[i]).abs();
        if d > worst {
            worst = d;
        }
    }
    println!("CIEDE2000 与 Sharma 参考值最大偏差 = {worst:.6}");
    assert!(worst < 1e-4, "de00 mismatch {worst}");
}

#[test]
fn transfer_functions_match_python() {
    if !have("tests/golden/color_code.npy") {
        return;
    }
    let (_, code) = read_npy("tests/golden/color_code.npy");
    let (_, lin) = read_npy("tests/golden/color_lin.npy");
    let (_, enc) = read_npy("tests/golden/color_enc.npy");
    let got_lin: Vec<f64> = code.iter().map(|&c| decode_srgb(c)).collect();
    let got_enc: Vec<f64> = code
        .iter()
        .map(|&c| encode_srgb(c.clamp(0.0, 1.0)))
        .collect();
    let d1 = max_abs_diff(&got_lin, &lin);
    let d2 = max_abs_diff(&got_enc, &enc);
    println!("decode_srgb 偏差 {d1:.3e}   encode_srgb 偏差 {d2:.3e}");
    assert!(d1 < 1e-15 && d2 < 1e-15);
}

#[test]
fn oklab_and_lab_match_python() {
    if !have("tests/golden/color_code.npy") {
        return;
    }
    let (_, code) = read_npy("tests/golden/color_code.npy");
    let (_, ok_ref) = read_npy("tests/golden/color_oklab.npy");
    let (_, lab_ref) = read_npy("tests/golden/color_lab.npy");
    let n = code.len() / 3;
    let mut ok = Vec::with_capacity(n * 3);
    let mut lab = Vec::with_capacity(n * 3);
    for i in 0..n {
        let p: Px = [code[i * 3], code[i * 3 + 1], code[i * 3 + 2]];
        ok.extend_from_slice(&code_to_oklab(p, false));
        lab.extend_from_slice(&code_to_lab(p, false));
    }
    let d1 = max_abs_diff(&ok, &ok_ref);
    let d2 = max_abs_diff(&lab, &lab_ref);
    println!("Oklab 偏差 {d1:.3e}   CIELAB 偏差 {d2:.3e}");
    assert!(d1 < 1e-12 && d2 < 1e-12);
}

#[test]
fn tetra_scheme_matches_python() {
    if !have("tests/golden/tetra_pts.npy") {
        return;
    }
    use lut3d::core::tetra::tetra_scheme;
    let (shape, pts) = read_npy("tests/golden/tetra_pts.npy");
    let (_, idx_ref) = read_npy("tests/golden/tetra_idx.npy");
    let (_, w_ref) = read_npy("tests/golden/tetra_w.npy");
    let n = 5usize;
    let np = shape[0];
    let points: Vec<Px> = (0..np)
        .map(|i| [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]])
        .collect();
    let (idx, w) = tetra_scheme(&points, n);
    let mut idx_bad = 0;
    let mut w_worst = 0.0f64;
    let mut wsum_worst = 0.0f64;
    for s in 0..np {
        for t in 0..4 {
            if idx[s][t] as f64 != idx_ref[s * 4 + t] {
                idx_bad += 1;
            }
            w_worst = w_worst.max((w[s][t] - w_ref[s * 4 + t]).abs());
        }
        let sum: f64 = w[s].iter().sum();
        wsum_worst = wsum_worst.max((sum - 1.0).abs());
    }
    println!(
        "四面体：索引不匹配 {idx_bad} 个，权重最大偏差 {w_worst:.3e}，权重和偏差 {wsum_worst:.3e}"
    );
    assert_eq!(idx_bad, 0);
    assert!(w_worst < 1e-15 && wsum_worst < 1e-15);
}

#[test]
fn tetra_identity_reproduces_input_exactly() {
    if !have("tests/golden/tetra_pts.npy") {
        return;
    }
    // Tetrahedral interpolation is exact for linear fields, so an identity LUT must
    // map every point to itself.
    use lut3d::core::tetra::{evaluate, nodes_flat};
    let (_, pts) = read_npy("tests/golden/tetra_pts.npy");
    let np = pts.len() / 3;
    let points: Vec<Px> = (0..np)
        .map(|i| [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]])
        .collect();
    for &n in &[2usize, 5, 17, 33] {
        let x = nodes_flat(n);
        let y = evaluate(&x, &points, n);
        let worst = (0..np * 3)
            .map(|i| (y[i] - pts[i]).abs())
            .fold(0.0f64, f64::max);
        println!("  n={n:2} 恒等 LUT 重构最大误差 {worst:.3e}");
        assert!(worst < 1e-11, "n={n} err {worst}");
    }
}

#[test]
fn laplacian_annihilates_constant_and_linear_fields() {
    use lut3d::core::tetra::{nodes_flat, Laplacian};
    for &n in &[3usize, 9, 33] {
        let lap = Laplacian::new(n);
        let c = vec![3.7f64; n * n * n * 3];
        let lc = lap.apply(&c);
        assert!(lc.iter().all(|v| v.abs() < 1e-12), "constant n={n}");
        // linear ramp: zero second difference in the interior
        let g = nodes_flat(n);
        let lg = lap.apply(&g);
        let mut worst = 0.0f64;
        for node in 0..n * n * n {
            let interior = (0..3).all(|c| g[node * 3 + c] > 0.05 && g[node * 3 + c] < 0.95);
            if interior {
                for c in 0..3 {
                    worst = worst.max(lg[node * 3 + c].abs());
                }
            }
        }
        println!("  n={n:2} 线性场内部最大二阶差分 {worst:.3e}");
        assert!(worst < 1e-12);
    }
}

/// End-to-end: fit the real dataset and compare against the Python snapshot.
/// Slow (about a minute) so it is ignored by default:
///   cargo test --release -- --ignored end_to_end
#[test]
#[ignore]
fn end_to_end_matches_python_snapshot() {
    use lut3d::core::fit::FitOptions;
    use lut3d::core::pipeline::{run_fit, PipelineOptions, Progress};

    // Compare statistics, not the maximum.  Fold repair is a whack-a-mole process, so
    // a handful of nodes always differ; what matters is whether the two LUTs *do* the
    // same thing.  Measured on dataset 1: mean node deviation 0.0007, only 0.13% of
    // nodes above 0.01, and the achieved ΔE00 is 1.1994 (Python) vs 1.1993 (Rust).
    for (dir_in, dir_tg, snap) in [
        ("../input", "../target", "tests/golden/lut_ds1.npy"),
        ("../input2", "../target2", "tests/golden/lut_ds2.npy"),
    ] {
        if std::fs::metadata(dir_in).is_err() {
            eprintln!("跳过：没有找到 {dir_in}（需要私有的图对素材）");
            continue;
        }
        if !have(snap) {
            continue;
        }
        let (_, expect) = read_npy(snap);
        let mut opts = PipelineOptions::default();
        opts.core = FitOptions::new(33);
        let cancel = || false;
        let mut stage = |_: &str, _: f64| {};
        let mut pr = Progress {
            on_stage: &mut stage,
            cancel: &cancel,
        };
        let r = run_fit(dir_in, dir_tg, &opts, &mut pr).expect("fit");
        let mx = max_abs_diff(&r.lut, &expect);
        let mean = r
            .lut
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / r.lut.len() as f64;
        let mut dev: Vec<f64> = r
            .lut
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p999 = dev[(dev.len() as f64 * 0.999) as usize];
        println!(
            "{dir_in}: 平均偏差 {mean:.5}  99.9分位 {p999:.5}  最大 {mx:.5}  ΔE {:.2}",
            r.agg_mean
        );
        // The preview sheet is drawn as one texture indexed by a single (w, h).  If any
        // preview keeps its own aspect ratio the columns after the first turn to
        // garbage — which is exactly what happened before this invariant was added.
        let dims: Vec<(usize, usize)> = r.previews.iter().map(|p| (p.w, p.h)).collect();
        for d in &dims {
            assert_eq!(*d, dims[0], "{dir_in} 预览尺寸不一致: {dims:?}");
        }
        assert!(r.previews.len() >= 3, "{dir_in} 预览数量不足");
        println!("  预览 {} 张，尺寸统一 {:?}", dims.len(), dims[0]);

        // The smoothing slider must be able to price itself without re-decoding the
        // dataset, which requires the cached eval pairs to have survived the parallel
        // preview pass (a thread-local there silently dropped most of them).
        assert_eq!(
            r.eval_cache.pairs.len(),
            r.previews.len(),
            "{dir_in} 缓存的图对数与预览数不一致"
        );
        let base = lut3d::core::pipeline::mean_de(
            &lut3d::core::pipeline::final_lut(&r, 1.0, 0.0),
            &r.eval_cache,
            r.p3,
        );
        let sm = lut3d::core::pipeline::mean_de(
            &lut3d::core::pipeline::final_lut(&r, 1.0, 0.8),
            &r.eval_cache,
            r.p3,
        );
        println!("  精度代价: ΔE00 {base:.3} → {sm:.3}  ({:+.3})", sm - base);
        // The subset is deliberately biased toward the hardest pairs, so its absolute ΔE
        // is NOT comparable to the dataset-wide figure — only the cost is used.
        assert!(
            base > r.agg_mean,
            "{dir_in} 子集应当比全体更难（{base} vs {}）",
            r.agg_mean
        );
        let cost = sm - base;
        assert!(cost.abs() < 0.25, "{dir_in} 平滑精度代价异常: {cost:+.3}");

        // The variance identity must hold: total == within + between.  If the residual
        // accumulation were wrong this would drift.
        let fd = &r.floor_decomp;
        let rel = (fd.total - fd.within - fd.between).abs() / fd.total.max(1e-30);
        println!(
            "  下界分解: 图内 {:.1}%  图间 {:.1}%  (恒等式偏差 {:.2e})",
            fd.within_share() * 100.0,
            fd.between_share() * 100.0,
            rel
        );
        assert!(rel < 1e-9, "{dir_in} 方差分解恒等式不成立，偏差 {rel}");
        assert!(
            fd.pixels > 0.0 && !fd.per_image.is_empty(),
            "{dir_in} 分解没有采到像素"
        );

        assert!(mean < 0.005, "{dir_in} 平均偏差过大 {mean}");
        assert!(p999 < 0.02, "{dir_in} 99.9 分位偏差过大 {p999}");
    }
}

/// The UI is entirely in Chinese, and egui bundles no CJK glyphs — without a system
/// font every character is a tofu box.  Make sure a loadable face actually contains
/// the glyphs, rather than discovering it visually.
#[test]
fn a_system_cjk_font_is_available() {
    use ab_glyph::{Font, FontRef};
    let mut found = Vec::new();
    for (path, idx) in lut3d::theme::cjk_candidates() {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let font = match FontRef::try_from_slice_and_index(&bytes, *idx) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut ok = 0;
        for ch in ['中', '文', '图', '像', '像', '强', '度', '预', '检'] {
            if font.glyph_id(ch).0 != 0 {
                ok += 1;
            }
        }
        found.push(format!("{path}#{idx} -> {ok}/9 CJK glyphs"));
        if ok >= 9 {
            println!("可用中文字体: {path}#{idx}（{ok}/9 字形齐全）");
            return;
        }
    }
    eprintln!(
        "跳过：本机没有含中文字形的系统字体。检查过：\n{}",
        found.join("\n")
    );
    eprintln!("（CI 上可安装 fonts-noto-cjk 来启用这项检查）");
}

/// Smoothing must actually reduce curvature, must not move the black/white end points
/// much (a LUT that no longer maps 0 to 0 is broken), and must be monotone in strength.
#[test]
fn smoothing_reduces_curvature() {
    if !have("tests/golden/lut_ds1.npy") {
        return;
    }
    use lut3d::core::project::{banding, max_second_difference, smooth_lut};
    let (_, lut) = read_npy("tests/golden/lut_ds1.npy");
    let n = 33;
    let base = max_second_difference(&lut, n);
    println!(
        "平滑前 色带风险(10bit) {:.2} 级",
        banding(&lut, n).levels_10bit()
    );

    let mut prev = f64::INFINITY;
    for s in [0.25f64, 0.5, 1.0] {
        let y = smooth_lut(&lut, n, s);
        let sec = max_second_difference(&y, n);
        let b = banding(&y, n);
        // 端点漂移：黑必须还是黑
        let mut end_drift = 0.0f64;
        for i in [0usize, n * n * n - 1] {
            for c in 0..3 {
                end_drift = end_drift.max((y[i * 3 + c] - lut[i * 3 + c]).abs());
            }
        }
        println!("  strength {s:.2}: 色带风险 {:>5.2} 级(10bit)  {:>5.2} 级(8bit)  端点漂移 {:.4}  折叠 {}",
                 b.levels_10bit(), b.levels_8bit(), end_drift, b.folds);
        assert!(sec < prev, "强度增加但曲率没有单调下降: {sec} !< {prev}");
        prev = sec;
        assert!(
            end_drift < 0.02,
            "端点漂移过大 {end_drift}——LUT 不再把黑映射到黑"
        );
    }
    assert!(prev < base, "最强平滑也没有降低曲率");
    println!("  平滑确实生效：{base:.5} -> {prev:.5}");
}

/// The banding-amplification view silently rendered a flat grey field because a fixed
/// 24x multiplier was two orders of magnitude too small for residuals of order 1e-3.
/// Guard the property that matters: the output must actually carry contrast.
#[test]
fn amplification_view_is_not_flat() {
    if !have("tests/golden/lut_ds1.npy") {
        return;
    }
    use lut3d::core::pipeline::apply_lut;
    use lut3d::core::project::{amplify_steps, smooth_lut};
    use lut3d::core::testchart::{sky, stripes, CH, CW, N_B, N_G};

    let (_, lut) = read_npy("tests/golden/lut_ds1.npy");
    let n = 33;
    for (tag, chart) in [("全色域", stripes(CW, CH, N_G, N_B)), ("天空", sky(CW, CH))] {
        for s in [0.0f64, 0.5] {
            let x = smooth_lut(&lut, n, s);
            let pred = apply_lut(&x, &chart, n);
            let (amp, gain, peak) = amplify_steps(&pred, CW, CH, 0.45);
            let g: Vec<f64> = amp.iter().step_by(3).map(|&v| v as f64).collect();
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            let sd = (g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / g.len() as f64).sqrt();
            // 用 1..254 之间的像素占比衡量"是否真的画出了东西"
            let usable =
                g.iter().filter(|&&v| v > 4.0 && v < 251.0).count() as f64 / g.len() as f64;
            println!("  {tag} 平滑{s:.1}: 增益×{gain:.0}  残差峰值 {peak:.2e}  灰度标准差 {sd:.1}  有效像素 {:.1}%",
                     usable * 100.0);
            assert!(
                sd > 3.0,
                "{tag} 平滑{s} 的放大图几乎是平的（标准差 {sd:.2}）"
            );
            assert!(
                usable > 0.02,
                "{tag} 平滑{s} 的放大图饱和了（有效像素仅 {:.1}%）",
                usable * 100.0
            );
            // A hard edge anywhere in the chart would make the residual peak orders of
            // magnitude larger than the LUT's curvature, and the auto-gain would then
            // scale the real banding down to a flat field.
            if tag == "全色域" {
                assert!(
                    peak < 0.05,
                    "{tag} 残差峰值 {peak:.2e} 过大——说明测试图里有硬边"
                );
            }
        }
    }
}

/// The sky ramp must actually span a usable range, otherwise the LUT barely acts on it
/// and smoothing shows no visible difference.
#[test]
fn sky_ramp_has_usable_range() {
    use lut3d::core::testchart::{sky, CH, CW};
    let s = sky(CW, CH);
    let lum = |i: usize| {
        0.2126 * s[i * 3] as f64 + 0.7152 * s[i * 3 + 1] as f64 + 0.0722 * s[i * 3 + 2] as f64
    };
    let top = lum(0);
    let bot = lum(CW * (CH - 1));
    println!(
        "  天空亮度范围 {top:.3} → {bot:.3}（跨度 {:.3}）",
        (bot - top).abs()
    );
    assert!((bot - top).abs() > 0.35, "天空渐变范围太窄，LUT 作用不起来");
}
