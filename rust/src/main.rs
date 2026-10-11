//! lutfit — 3D LUT fitting tool.

#![allow(clippy::field_reassign_with_default)]

use lutfit::app;
use lutfit::core::pipeline::{export, run_fit, PipelineOptions, Progress};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Scan for the flag rather than assuming a position: macOS prepends its own
    // `-psn_...` argument when the app is launched via `open`.
    // Fit the screen: a 940pt window does not fit a 982pt display once the menu bar
    // and title bar are accounted for, and the OS silently shrinks it -- which put the
    // bottom of the scroll area off-screen.
    let h = std::env::var("LUTFIT_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(860.0f32);
    if let Some(gi) = args.iter().position(|a| a == "--gallery") {
        let ind = args.get(gi + 1).cloned().unwrap_or_default();
        let tgt = args.get(gi + 2).cloned().unwrap_or_default();
        let lut: usize = args.get(gi + 3).and_then(|s| s.parse().ok()).unwrap_or(33);
        let opts = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(1320.0, h))
                .with_title("3D LUT 拟合工具"),
            ..Default::default()
        };
        let _ = eframe::run_native(
            "3D LUT 拟合工具",
            opts,
            Box::new(move |cc| Ok(Box::new(app::LutApp::with_paths(cc, &ind, &tgt, lut)))),
        );
        return;
    }
    if args.len() == 1 {
        let opts = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(1320.0, h))
                .with_min_inner_size([1000.0, 560.0])
                .with_title("3D LUT 拟合工具"),
            ..Default::default()
        };
        let _ = eframe::run_native(
            "3D LUT 拟合工具",
            opts,
            Box::new(|cc| Ok(Box::new(app::LutApp::new(cc)))),
        );
        return;
    }
    if args.len() < 2 || args[1] == "help" || args[1] == "--help" {
        eprintln!("用法:");
        eprintln!("  lutfit fit <输入文件夹> <目标文件夹> <输出.cube> [lut尺寸]");
        return;
    }
    match args[1].as_str() {
        "fit" => {
            if args.len() < 5 {
                eprintln!("需要：fit <输入> <目标> <输出.cube> [尺寸]");
                std::process::exit(1);
            }
            let (ind, tgt, out) = (&args[2], &args[3], &args[4]);
            let size: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(33);
            let mut opts = PipelineOptions::default();
            opts.core = lutfit::core::fit::FitOptions::new(size);
            let cancel = || false;
            let mut on_stage = |msg: &str, frac: f64| {
                print!("\r  [{:5.1}%] {:<56}", frac * 100.0, msg);
                let _ = std::io::stdout().flush();
            };
            let mut pr = Progress {
                on_stage: &mut on_stage,
                cancel: &cancel,
            };
            match run_fit(ind, tgt, &opts, &mut pr) {
                Ok(r) => {
                    println!("\n");
                    println!(
                        "图对 {}  像素 {:.1}M  耗时 {:.1}s",
                        r.n_pairs,
                        r.pixels / 1e6,
                        r.elapsed_s
                    );
                    println!(
                        "  阶段: 解码+统计 {:.1}s   拟合 {:.1}s   色差评估 {:.1}s",
                        r.t_stats, r.t_fit, r.t_eval
                    );
                    let mv =
                        lutfit::core::fit::MATVEC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
                    let per = r.t_fit / mv.max(1) as f64 * 1000.0;
                    println!(
                        "  求解器: matvec {mv} 次，平均 {:.3} ms/次，每次约 {:.0} 万次浮点运算",
                        per,
                        96624.0 * 4.0 * 3.0 / 1e4
                    );
                    for (nm, why) in &r.skipped {
                        println!("  跳过 {nm}: {why}");
                    }
                    println!("\n覆盖度");
                    println!(
                        "  节点 {:.1}%   像素质量 {:.3}%",
                        r.preflight.node_coverage * 100.0,
                        r.preflight.mass_coverage * 100.0
                    );
                    println!("  外推区改动 {:.5}（有数据区 {:.5}）", r.risk.0, r.risk.2);
                    println!("\n误差账本");
                    println!(
                        "  模型误差 {:.5}   不可约下界 {:.5}   总 {:.5}  (+{:.1}%)",
                        r.residual_model,
                        r.residual_floor,
                        (r.residual_model.powi(2) + r.residual_floor.powi(2)).sqrt(),
                        ((r.residual_model.powi(2) + r.residual_floor.powi(2)).sqrt()
                            / r.residual_floor
                            - 1.0)
                            * 100.0
                    );
                    println!("\n结构");
                    println!(
                        "  折叠 {} -> {}   亮度违反 {}   最大二阶差分 {:.5}",
                        r.folds_before, r.folds_after, r.luma_viol, r.max_second
                    );
                    println!("\n自适应尝试");
                    for t in &r.trials {
                        println!(
                            "  trial {}: lam_s={:.4} folds={} min_det={:+.4} model={:.5}",
                            t.trial, t.lam_s, t.folds, t.min_det, t.model
                        );
                    }
                    println!("  采用 lam_s = {}", r.chosen_lam_s);
                    println!("\n色差 ΔE00");
                    println!("  平均 {:.2}   95分位 {:.2}", r.agg_mean, r.agg_p95);
                    let smooth: f64 = std::env::var("LUTFIT_SMOOTH")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0);
                    if let Err(e) = export(&r, out, 1.0, smooth) {
                        eprintln!("导出失败: {e}");
                    } else {
                        println!("\n已导出 {out}");
                    }
                }
                Err(e) => eprintln!("\n失败: {e}"),
            }
        }
        _ => eprintln!("未知子命令 {}", args[1]),
    }
}
