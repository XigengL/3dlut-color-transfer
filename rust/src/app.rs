//! Application shell: background fitting, staged progress with an ETA, a plain-language
//! pre-flight card, a five-up preview contact sheet, and export.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use egui::{Color32, RichText};

use crate::core::fit::FitOptions;
use crate::core::pipeline::{
    apply_lut, export, final_lut, mean_de, run_fit, strength_preview, FitResult, PipelineOptions,
    Preview, Progress,
};
use crate::core::project::{amplify_steps, max_second_difference, smooth_lut};
use crate::core::testchart::{self, to_rgb8, CH, CW, N_B, N_G};
use crate::theme as th;

enum Msg {
    Progress(String, f64),
    Done(Box<FitResult>),
    Err(String),
}

enum State {
    Idle,
    Running(f64, String),
    Done,
    Error(String),
}

pub struct LutApp {
    in_dir: String,
    out_dir: String,
    lut_size: usize,
    preserve_neutral: bool,
    strength: f32,
    state: State,
    rx: Option<Receiver<Msg>>,
    cancel: Arc<AtomicBool>,
    started: Option<Instant>,
    result: Option<FitResult>,
    tex: Option<egui::TextureHandle>,
    tex_strength: f32,
    tex_key: (u32, u32),
    tex_dims: (usize, usize),
    sized: bool,
    scrolled: bool,
    smooth: f32,
    dither: bool,
    chart: Vec<f32>,
    chart_tex: Option<egui::TextureHandle>,
    chart_sm_tex: Option<egui::TextureHandle>,
    chart_key: (u32, bool, u32, u32),
    sky_mode: bool,
    amplify: bool,
    amp_tex: Option<egui::TextureHandle>,
    amp_gain: Option<(f64, f64)>,
    de_smooth: Option<f64>,
    de_baseline: Option<f64>,
    de_key: (u32, u32),
    fold_check: Option<(usize, usize, f64)>,
}

impl Default for LutApp {
    fn default() -> Self {
        Self {
            in_dir: String::new(),
            out_dir: String::new(),
            lut_size: 33,
            preserve_neutral: true,
            strength: 1.0,
            state: State::Idle,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            started: None,
            result: None,
            tex: None,
            tex_strength: -1.0,
            tex_key: (u32::MAX, u32::MAX),
            tex_dims: (0, 0),
            sized: false,
            scrolled: false,
            smooth: std::env::var("LUTFIT_SMOOTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            dither: std::env::var("LUTFIT_DITHER").is_ok(),
            chart: testchart::stripes(CW, CH, N_G, N_B),
            chart_tex: None,
            chart_sm_tex: None,
            chart_key: (u32::MAX, false, u32::MAX, u32::MAX),
            sky_mode: false,
            amplify: std::env::var("LUTFIT_AMP").is_ok(),
            amp_tex: None,
            amp_gain: None,
            de_smooth: None,
            de_baseline: None,
            de_key: (u32::MAX, u32::MAX),
            fold_check: None,
        }
    }
}

impl LutApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        th::apply(&cc.egui_ctx);
        Self::default()
    }

    /// Pre-fill the two folders and start immediately.  Used by the screenshot harness
    /// so the running and finished states can be inspected without input automation.
    pub fn with_paths(cc: &eframe::CreationContext<'_>, ind: &str, tgt: &str, lut: usize) -> Self {
        th::apply(&cc.egui_ctx);
        let mut a = Self::default();
        a.in_dir = ind.to_string();
        a.out_dir = tgt.to_string();
        a.lut_size = lut;
        a.start();
        a
    }

    fn start(&mut self) {
        let (ind, tgt, size, neutral) = (
            self.in_dir.clone(),
            self.out_dir.clone(),
            self.lut_size,
            self.preserve_neutral,
        );
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        self.state = State::Running(0.0, "准备中".into());
        self.started = Some(Instant::now());
        self.result = None;
        self.tex = None;
        self.tex_strength = -1.0;

        std::thread::spawn(move || {
            let mut opts = PipelineOptions::default();
            let mut core = FitOptions::new(size);
            core.preserve_neutral = neutral;
            opts.core = core;
            let mut on_stage = |msg: &str, f: f64| {
                let _ = tx.send(Msg::Progress(msg.to_string(), f));
            };
            let cancelf = move || cancel.load(Ordering::Relaxed);
            let mut pr = Progress {
                on_stage: &mut on_stage,
                cancel: &cancelf,
            };
            match run_fit(&ind, &tgt, &opts, &mut pr) {
                Ok(r) => {
                    let _ = tx.send(Msg::Done(Box::new(r)));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Err(e));
                }
            }
        });
    }

    fn poll(&mut self) {
        let msgs: Vec<Msg> = match &self.rx {
            Some(rx) => rx.try_iter().collect(),
            None => return,
        };
        for m in msgs {
            match m {
                Msg::Progress(msg, f) => self.state = State::Running(f, msg),
                Msg::Done(r) => {
                    self.state = State::Done;
                    let base = mean_de(
                        &final_lut(&r, self.strength as f64, 0.0),
                        &r.eval_cache,
                        r.p3,
                    );
                    self.de_smooth = Some(base);
                    self.de_baseline = Some(base);
                    self.de_key = (self.strength.to_bits(), 0);
                    self.result = Some(*r);
                    self.tex = None;
                    self.tex_strength = -1.0;
                    self.rx = None;
                }
                Msg::Err(e) => {
                    self.state = State::Error(e);
                    self.rx = None;
                }
            }
        }
    }

    /// Build the five-up contact sheet: one column per preview, three bands each
    /// (A on top, A+LUT in the middle, B at the bottom).
    fn refresh_texture(&mut self, ctx: &egui::Context) {
        let res = match &self.result {
            Some(r) => r,
            None => return,
        };
        if res.previews.is_empty() {
            return;
        }
        let alpha = self.strength;
        let sm = self.smooth;
        let key = (alpha.to_bits(), sm.to_bits());
        if self.tex_key == key && self.tex.is_some() {
            return;
        }
        let x = final_lut(res, alpha as f64, sm as f64);
        let n = res.n;

        let cols = res.previews.len();
        let (cw, ch) = (Preview::W, Preview::H);
        let gap = 5usize;
        let tw = cw * cols + gap * (cols - 1);
        let th_ = ch * 3 + gap * 2;
        let mut buf = vec![0xE8u8; tw * th_ * 3];

        for (ci, p) in res.previews.iter().enumerate() {
            let img: Vec<f32> = p.a.iter().map(|&v| v as f32 / 255.0).collect();
            let pred = apply_lut(&x, &img, n);
            let x0 = ci * (cw + gap);
            for row in 0..ch {
                for col in 0..cw {
                    let si = (row * cw + col) * 3;
                    for band in 0..3usize {
                        let y = row + band * (ch + gap);
                        let di = (y * tw + x0 + col) * 3;
                        for c in 0..3 {
                            buf[di + c] = match band {
                                0 => p.a[si + c],
                                2 => p.b[si + c],
                                _ => (pred[si + c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                            };
                        }
                    }
                }
            }
        }
        self.tex = Some(ctx.load_texture(
            "sheet",
            egui::ColorImage::from_rgb([tw, th_], &buf),
            egui::TextureOptions::LINEAR,
        ));
        self.tex_strength = alpha;
        self.tex_dims = (tw, th_);
    }
}

impl LutApp {
    /// Render the chart twice: through the raw LUT and through the smoothed one, so the
    /// slider's effect shows as a side-by-side instead of something to remember.
    fn refresh_chart(&mut self, ctx: &egui::Context) {
        let res = match &self.result {
            Some(r) => r,
            None => {
                self.chart_tex = None;
                self.chart_sm_tex = None;
                return;
            }
        };
        let key = (
            self.smooth.to_bits(),
            self.dither,
            self.strength.to_bits(),
            self.sky_mode as u32,
        );
        if key == self.chart_key && self.chart_tex.is_some() {
            return;
        }
        let n = res.n;
        let base = strength_preview(res, self.strength as f64);
        let mk = |ctx: &egui::Context, name: &str, x: &[f64], chart: &[f32], n: usize, d: bool| {
            let pred = apply_lut(x, chart, n);
            let rgb = to_rgb8(&pred, CW, CH, d);
            (
                ctx.load_texture(
                    name,
                    egui::ColorImage::from_rgb([CW, CH], &rgb),
                    egui::TextureOptions::NEAREST,
                ),
                pred,
            )
        };
        let (t0, _) = mk(ctx, "chart_raw", &base, &self.chart, n, self.dither);
        let sm = smooth_lut(&base, n, self.smooth as f64);
        let (t1, p1) = mk(ctx, "chart_sm", &sm, &self.chart, n, self.dither);
        self.chart_tex = Some(t0);
        self.chart_sm_tex = Some(t1);
        let (amp, gain, peak) = amplify_steps(&p1, CW, CH, 0.45);
        self.amp_gain = Some((gain, peak));
        self.amp_tex = Some(ctx.load_texture(
            "chart_amp",
            egui::ColorImage::from_rgb([CW, CH], &amp),
            egui::TextureOptions::NEAREST,
        ));
        self.chart_key = key;
    }
}

/// A labelled row of the pre-flight card: what it is, the number, and why it matters.
fn metric(ui: &mut egui::Ui, name: &str, value: &str, note: &str, color: Color32) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [150.0, 20.0],
            egui::Label::new(RichText::new(name).color(th::TEXT_DIM).size(13.0)),
        );
        ui.add_sized(
            [92.0, 20.0],
            egui::Label::new(RichText::new(value).size(15.0).strong().color(color)),
        );
        ui.label(RichText::new(note).size(12.0).color(th::TEXT_DIM));
    });
}

impl eframe::App for LutApp {
    /// eframe's default clear colour is transparent, so the few pixels between the two
    /// panel frames showed through as a black bar down the middle.  Paint the window
    /// background instead.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        th::BG.to_normalized_gamma_f32()
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Shrink to fit the monitor.  Asking for a window taller than the display makes
        // the OS silently clamp it, which used to push the bottom of the scroll area
        // off-screen.  Querying egui avoids needing any platform-specific code.
        if !self.sized {
            self.sized = true;
            if std::env::var("LUTFIT_H").is_err() {
                if let Some(mon) = ctx.input(|i| i.viewport().monitor_size) {
                    let want = egui::vec2(1320.0, (mon.y * 0.86).min(900.0));
                    if mon.y > 0.0 {
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(want));
                    }
                }
            }
        }
        self.poll();
        self.refresh_texture(ctx);
        self.refresh_chart(ctx);
        if matches!(self.state, State::Running(..)) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        // ---------------------------------------------------------------- sidebar
        egui::SidePanel::left("controls")
            .exact_width(352.0)
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::none()
                    .fill(th::BG)
                    .inner_margin(egui::Margin::symmetric(16.0, 16.0)),
            )
            .show(ctx, |ui| {
                // The side panel needs its own scroll area: in a short window the
                // 「开始拟合」button ended up below the fold with no way to reach it.
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .show(ui, |ui| {
                        ui.label(RichText::new("3D LUT 拟合").size(22.0).strong());
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(
                                "把相机的原图和修图后的成品喂进来，\n得到一个 .cube 色彩查找表。",
                            )
                            .size(12.5)
                            .color(th::TEXT_DIM),
                        );
                        ui.add_space(18.0);

                        th::section(ui, "① 选择两组图片");
                        ui.add_space(6.0);
                        for (label, hint, value) in [
                            ("原图", "相机直出 JPG", &mut self.in_dir),
                            ("修图后", "同一张图修好的版本", &mut self.out_dir),
                        ] {
                            ui.label(RichText::new(label).size(13.0).color(th::TEXT));
                            let path = if value.is_empty() {
                                "尚未选择文件夹".to_string()
                            } else {
                                value.clone()
                            };
                            let col = if value.is_empty() {
                                th::TEXT_DIM
                            } else {
                                th::TEXT
                            };
                            // One full-width button instead of a row with computed widths: any
                            // arithmetic here overflows the panel by the scroll bar's width and
                            // pushes the scroll bar off-screen.
                            let label = if value.is_empty() {
                                "选择文件夹…".to_string()
                            } else {
                                format!("{}   ⌄", th::ellipsize(&path, 26))
                            };
                            let w = (ui.available_width() - 18.0).max(80.0);
                            if ui
                                .add_sized(
                                    [w, 32.0],
                                    egui::Button::new(RichText::new(label).size(12.5).color(col))
                                        .fill(th::CARD),
                                )
                                .clicked()
                            {
                                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                    *value = p.display().to_string();
                                }
                            }
                            ui.label(RichText::new(hint).size(11.0).color(th::TEXT_DIM));
                            ui.add_space(10.0);
                        }

                        ui.add_space(8.0);
                        th::section(ui, "② 网格尺寸");
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            for s in [17usize, 33, 65] {
                                if ui
                                    .add_sized(
                                        [66.0, 30.0],
                                        egui::SelectableLabel::new(
                                            self.lut_size == s,
                                            format!("{s}³"),
                                        ),
                                    )
                                    .clicked()
                                {
                                    self.lut_size = s;
                                }
                            }
                        });
                        ui.label(
                            RichText::new("33³ 通用；画面色彩复杂用 65³，\n但需要更多图片支撑")
                                .size(11.0)
                                .color(th::TEXT_DIM),
                        );
                        ui.add_space(8.0);
                        ui.checkbox(
                            &mut self.preserve_neutral,
                            RichText::new("保持灰阶中性").size(13.0),
                        );

                        ui.add_space(18.0);
                        th::section(ui, "③ 开始");
                        ui.add_space(6.0);
                        let running = matches!(self.state, State::Running(..));
                        let can = !running && !self.in_dir.is_empty() && !self.out_dir.is_empty();
                        ui.horizontal(|ui| {
                            let btn = egui::Button::new(
                                RichText::new("开始拟合")
                                    .size(15.0)
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(if can {
                                th::ACCENT
                            } else {
                                th::SEP
                            });
                            if ui.add_sized([150.0, 42.0], btn).clicked() && can {
                                self.start();
                            }
                            if running
                                && ui
                                    .add_sized([74.0, 42.0], egui::Button::new("取消"))
                                    .clicked()
                            {
                                self.cancel.store(true, Ordering::Relaxed);
                            }
                        });

                        if let State::Running(f, msg) = &self.state {
                            ui.add_space(14.0);
                            let eta = self.started.map(|t0| {
                                let el = t0.elapsed().as_secs_f64();
                                if *f > 0.02 {
                                    (el, el * (1.0 - f) / f)
                                } else {
                                    (el, f64::NAN)
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!("{:>3.0}%", f * 100.0))
                                        .size(22.0)
                                        .strong()
                                        .color(th::ACCENT),
                                );
                                if let Some((_, e)) = eta {
                                    if e.is_finite() {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    RichText::new(format!("还剩 {}", th::mmss(e)))
                                                        .size(13.0)
                                                        .color(th::TEXT_DIM),
                                                );
                                            },
                                        );
                                    }
                                }
                            });
                            ui.add_space(6.0);
                            ui.add(
                                egui::ProgressBar::new(*f as f32)
                                    .desired_height(8.0)
                                    .fill(th::ACCENT)
                                    .rounding(4.0),
                            );
                            ui.add_space(6.0);
                            ui.label(RichText::new(msg).size(12.0).color(th::TEXT));
                            if let Some((el, _)) = eta {
                                ui.label(
                                    RichText::new(format!("已用 {}", th::mmss(el)))
                                        .size(11.5)
                                        .color(th::TEXT_DIM),
                                );
                            }
                        }
                        if let State::Error(e) = &self.state {
                            ui.add_space(10.0);
                            ui.colored_label(
                                th::BAD,
                                RichText::new(format!("出错：{e}")).size(12.5),
                            );
                        }
                        if std::env::var("LUTFIT_DEBUG_LAYOUT").is_ok() {
                            eprintln!(
                                "[sidebar] 内容高 {:.0}  视口高 {:.0}",
                                ui.min_rect().height(),
                                ui.clip_rect().height()
                            );
                        }
                    });
            });

        // ---------------------------------------------------------------- main
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(th::BG).inner_margin(egui::Margin::symmetric(18.0, 18.0)))
            .show(ctx, |ui| {
                let mut sa = egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    // A scroll bar that only appears when needed left no clue that
                    // there was more below the fold.
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible);
                // Screenshot harness: park the view at the bottom so the preview sheet
                // can be inspected without input automation.
                if !self.scrolled && std::env::var("LUTFIT_SCROLL").as_deref() == Ok("bottom")
                    && self.result.is_some() {
                    sa = sa.vertical_scroll_offset(100_000.0);
                    self.scrolled = true;
                }
                sa.show(ui, |ui| {
                    let Some(r) = &self.result else {
                        ui.add_space(120.0);
                        ui.vertical_centered(|ui| {
                            ui.label(RichText::new("左侧选好两个文件夹，然后点「开始拟合」")
                                .size(15.0).color(th::TEXT_DIM));
                            ui.add_space(6.0);
                            ui.label(RichText::new("整个过程约 1–3 分钟，取决于图片数量和大小")
                                .size(12.0).color(th::TEXT_DIM));
                        });
                        return;
                    };

                    // ---- pre-flight ----
                    let mc = r.preflight.mass_coverage;
                    let (mark, col) = if mc > 0.995 { ("✓", th::OK) }
                        else if mc > 0.98 { ("⚠", th::WARN) } else { ("✗", th::BAD) };
                    th::card(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(mark).size(22.0).color(col).strong());
                            ui.label(RichText::new("预检").size(16.0).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(format!("{} 组图 · {:.0}M 像素 · {:.0} 秒",
                                    r.n_pairs, r.pixels / 1e6, r.elapsed_s)).size(12.0).color(th::TEXT_DIM));
                            });
                        });
                        ui.add_space(10.0);
                        let ok_col = if mc > 0.995 { th::OK } else { th::WARN };
                        metric(ui, "像素质量覆盖", &format!("{:.2}%", mc * 100.0), "真实照片里落在有效数据上的像素比例", ok_col);
                        metric(ui, "色彩网格覆盖", &format!("{:.0}%", r.preflight.node_coverage * 100.0),
                               "色彩立方体的节点中有多少训练数据。其余是平滑推测", th::TEXT);
                        metric(ui, "外推区改动量", &format!("{:.3}", r.risk.0),
                               &format!("没数据的地方被改了多少。有数据处为 {:.3}", r.risk.2), th::TEXT);
                        if !r.skipped.is_empty() {
                            ui.add_space(4.0);
                            ui.label(RichText::new(format!("跳过 {} 个文件（{}）", r.skipped.len(), r.skipped[0].0))
                                .size(11.5).color(th::WARN));
                        }
                    });

                    // ---- accuracy ----
                    ui.add_space(12.0);
                    let total = (r.residual_model.powi(2) + r.residual_floor.powi(2)).sqrt();
                    th::card(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("拟合质量").size(16.0).strong());
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new(format!("{:.2}", r.agg_mean)).size(30.0).strong().color(th::ACCENT));
                                ui.label(RichText::new("ΔE00 平均色差").size(11.5).color(th::TEXT_DIM));
                            });
                            ui.add_space(30.0);
                            ui.vertical(|ui| {
                                ui.add_space(6.0);
                                metric(ui, "95 分位色差", &format!("{:.2}", r.agg_p95), "最差的 5% 像素", th::TEXT);
                                metric(ui, "模型误差", &format!("{:.5}", r.residual_model), "LUT 没能拟合的部分", th::TEXT);
                                metric(ui, "数据下界", &format!("{:.5}", r.residual_floor), "87 组图互相矛盾导致的、任何 LUT 都消不掉的部分", th::TEXT);
                                metric(ui, "总误差 / 下界", &format!("+{:.1}%", (total / r.residual_floor - 1.0) * 100.0),
                                       "算法离理论极限有多远，越接近 0 越好", if total / r.residual_floor < 1.2 { th::OK } else { th::TEXT });
                            });
                        });
                        ui.add_space(12.0);
                        // Where the error actually lives.  `within` is local adjustment inside
                        // a single photo, which no global LUT can remove; `between` is photos
                        // being edited differently from each other, which a set of LUTs could.
                        {
                            let fd = &r.floor_decomp;
                            ui.label(RichText::new("下界分解 · 误差花在哪").size(12.0).color(th::TEXT_DIM));
                            ui.add_space(4.0);
                            let bar = |ui: &mut egui::Ui, frac: f64, col: egui::Color32| {
                                let w = ui.available_width().min(420.0);
                                let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 10.0),
                                                                      egui::Sense::hover());
                                ui.painter().rect_filled(rect, 2.0, col.gamma_multiply(0.85));
                                let mut x = rect.left();
                                let seg = w * frac as f32;
                                let _ = seg;
                                x += 0.0;
                                let _ = x;
                            };
                            let _ = bar;
                            let betw = fd.between_share();
                            let with = fd.within_share();
                            ui.horizontal(|ui| {
                                ui.add_sized([168.0, 20.0], egui::Label::new(
                                    RichText::new(format!("图内差异 {:>4.1}%", with * 100.0)).size(13.0).color(th::TEXT)));
                                ui.label(RichText::new("同一张照片里的局部调整（渐变/画笔）——全局 LUT 原理上救不了").size(11.5).color(th::TEXT_DIM));
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized([168.0, 20.0], egui::Label::new(
                                    RichText::new(format!("图间差异 {:>4.1}%", betw * 100.0)).size(13.0)
                                        .color(if betw > 0.35 { th::WARN } else { th::TEXT })));
                                ui.label(RichText::new("不同照片修法不一致——这部分的多少，决定「多个 LUT」值不值得做").size(11.5).color(th::TEXT_DIM));
                            });
                        }
                        ui.add_space(8.0);
                        ui.label(RichText::new(if total / r.residual_floor < 1.2 {
                            "ΔE 参考：< 1 肉眼难辨，1–2 轻微，2–4 明显但可接受，> 4 偏色可见"
                        } else { "拟合效率偏低，考虑增加图片或提高网格尺寸" }).size(11.5).color(th::TEXT_DIM));
                    });

                    // ---- structural ----
                    ui.add_space(12.0);
                    th::card(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("结构检查").size(16.0).strong());
                        ui.add_space(8.0);
                        let fold_ok = r.folds_after == 0;
                        metric(ui, "色彩反转", &format!("{} 处", r.folds_after),
                               "LUT 里导致颜色翻转的区域，0 为最好", if fold_ok { th::OK } else { th::BAD });
                        metric(ui, "色带风险", &format!("{:.5}", r.max_second),
                               "亮度曲线的最大二阶差分，越小越不易出色带", if r.max_second < 0.004 { th::OK } else { th::WARN });
                        if !r.trials.is_empty() {
                            let t = &r.trials[r.trials.len() - 1];
                            metric(ui, "正则化强度", &format!("{:.0}", r.chosen_lam_s),
                                   &format!("自动搜索后采用；共尝试 {} 次", r.trials.len()), th::TEXT);
                            let _ = t;
                        }
                    });

                    // ---- preview sheet ----
                    if !r.previews.is_empty() {
                        ui.add_space(12.0);
                        th::card(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new("效果预览").size(16.0).strong());
                            ui.add_space(2.0);
                            ui.label(RichText::new("每列从上到下：原图 → 套用 LUT → 修图师成品。五张按明暗、饱和度和难度自动挑出，覆盖不同场景。")
                                .size(11.5).color(th::TEXT_DIM));
                            ui.add_space(10.0);
                            if let Some(t) = &self.tex {
                                let avail = ui.available_width();
                                let s = t.size_vec2();
                                let w = avail.min(1100.0);
                                ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(w, s.y * w / s.x)).rounding(6.0));
                            }
                            ui.add_space(6.0);
                            let cols = r.previews.len();
                            ui.columns(cols, |cs| {
                                for (i, c) in cs.iter_mut().enumerate() {
                                    let p = &r.previews[i];
                                    c.label(RichText::new(th::ellipsize(&p.name, 18)).size(11.0).color(th::TEXT));
                                    c.label(RichText::new(format!("ΔE {:.2} · {}", p.de, p.label)).size(11.0).color(th::TEXT_DIM));
                                }
                            });
                        });
                    }

                    // ---- smoothing ----
                    ui.add_space(12.0);
                    th::card(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("平滑").size(16.0).strong());
                            ui.add_space(8.0);
                            ui.label(RichText::new("抹平格点之间的折角，代价是损失一点精度")
                                .size(11.5).color(th::TEXT_DIM));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.checkbox(&mut self.sky_mode, RichText::new("天空渐变").size(12.0)).changed() {
                                    self.chart = if self.sky_mode {
                                        testchart::sky(CW, CH)
                                    } else {
                                        testchart::stripes(CW, CH, N_G, N_B)
                                    };
                                }
                                ui.checkbox(&mut self.amplify, RichText::new("色带放大").size(12.0));
                                ui.checkbox(&mut self.dither, RichText::new("抖动显示").size(12.0));
                            });
                        });
                        ui.add_space(8.0);
                        let sl = ui.horizontal(|ui| {
                            ui.add_sized([320.0, 22.0],
                                egui::Slider::new(&mut self.smooth, 0.0..=1.0).show_value(false));
                            ui.label(RichText::new(format!("{:.2}", self.smooth)).size(15.0).strong());
                            ui.add_space(14.0);
                            ui.label(RichText::new("预览是 8bit，10bit 交付请以右侧数字为准")
                                .size(11.0).color(th::TEXT_DIM));
                        }).response;
                        if sl.drag_stopped() {
                            let n = r.n;
                            let x = final_lut(r, self.strength as f64, self.smooth as f64);
                            let f = crate::core::project::fold_report(&x, n);
                            self.fold_check = Some((f.n_folded_cells, 0, f.min_det));
                            self.de_smooth = Some(mean_de(&x, &r.eval_cache, r.p3));
                            self.de_key = (self.strength.to_bits(), self.smooth.to_bits());
                        }

                        ui.add_space(10.0);
                        // Before / after, side by side.
                        let avail = ui.available_width();
                        let colw = ((avail - 190.0) / 2.0 - 8.0).max(160.0);
                        let imh = colw * CH as f32 / CW as f32;
                        ui.horizontal_top(|ui| {
                            let show_amp = self.amplify;
                            let tile = |ui: &mut egui::Ui, tex: Option<&egui::TextureHandle>, cap: &str| {
                                ui.vertical(|ui| {
                                    ui.label(RichText::new(cap).size(11.5).color(th::TEXT_DIM));
                                    ui.add_space(3.0);
                                    if let Some(t) = tex {
                                        ui.add(egui::Image::new(t)
                                            .fit_to_exact_size(egui::vec2(colw, imh)).rounding(5.0));
                                    } else {
                                        ui.allocate_space(egui::vec2(colw, imh));
                                    }
                                });
                            };
                            tile(ui, self.chart_tex.as_ref(), "未平滑");
                            ui.add_space(10.0);
                            if show_amp {
                                tile(ui, self.amp_tex.as_ref(), "色带放大 ×24");
                            } else {
                                tile(ui, self.chart_sm_tex.as_ref(), "已平滑");
                            }
                            ui.add_space(14.0);
                            ui.vertical(|ui| {
                                let x = smooth_lut(&strength_preview(r, self.strength as f64), r.n,
                                                   self.smooth as f64);
                                let sec0 = max_second_difference(&r.lut, r.n);
                                let sec1 = max_second_difference(&x, r.n);
                                let lv = |s: f64, bits: i32| s * (2f64.powi(bits) - 1.0);
                                let col = |v: f64| if v < 0.5 { th::OK } else if v < 1.5 { th::WARN } else { th::BAD };
                                ui.label(RichText::new("色带风险 · 10bit").size(12.0).color(th::TEXT_DIM));
                                ui.label(RichText::new(format!("{:.2}  →  {:.2} 级",
                                        lv(sec0, 10), lv(sec1, 10)))
                                    .size(19.0).strong().color(col(lv(sec1, 10))));
                                ui.add_space(8.0);
                                ui.label(RichText::new("色带风险 · 8bit").size(12.0).color(th::TEXT_DIM));
                                ui.label(RichText::new(format!("{:.2}  →  {:.2} 级",
                                        lv(sec0, 8), lv(sec1, 8)))
                                    .size(19.0).strong().color(col(lv(sec1, 8))));
                                ui.add_space(10.0);
                                {
                                    // Only the *cost* is trustworthy here.  The preview picker
                                    // deliberately chooses the hardest pairs, so the subset's
                                    // absolute ΔE (2.20) runs far above the dataset-wide 1.01.
                                    // A cost measured on hard pairs is a conservative upper
                                    // bound, which is the right side to err on.
                                    ui.label(RichText::new("ΔE00 精度代价").size(12.0).color(th::TEXT_DIM));
                                    match (self.de_smooth, self.de_baseline) {
                                        (Some(d), Some(b)) if self.smooth > 0.0 => {
                                            let delta = d - b;
                                            ui.label(RichText::new(format!("{delta:+.3}"))
                                                .size(19.0).strong()
                                                .color(if delta.abs() < 0.05 { th::OK } else { th::WARN }));
                                            ui.label(RichText::new(if delta.abs() < 0.05 {
                                                "可忽略（< 0.05 ΔE）"
                                            } else {
                                                "有代价，权衡后再拉"
                                            }).size(10.5).color(th::TEXT_DIM));
                                        }
                                        _ => {
                                            ui.label(RichText::new("0.000").size(19.0).strong().color(th::OK));
                                            ui.label(RichText::new("尚未平滑").size(10.5).color(th::TEXT_DIM));
                                        }
                                    }
                                    ui.label(RichText::new(format!("在 {} 张最难的图上实测，是上界",
                                            r.eval_cache.pairs.len()))
                                        .size(10.5).color(th::TEXT_DIM));
                                    ui.add_space(8.0);
                                }
                                match self.fold_check {
                                    Some((f, _, md)) => {
                                        ui.label(RichText::new(format!("色彩反转  {f} 处")).size(12.5)
                                            .color(if f == 0 { th::OK } else { th::BAD }));
                                        ui.label(RichText::new(format!("最小 det {md:+.4}")).size(11.0)
                                            .color(th::TEXT_DIM));
                                    }
                                    None => {
                                        ui.label(RichText::new("色彩反转").size(12.0).color(th::TEXT_DIM));
                                        ui.label(RichText::new("松手后校验").size(12.5).color(th::TEXT_DIM));
                                    }
                                }
                            });
                        });
                        ui.add_space(6.0);
                        let note = if self.amplify {
                            match self.amp_gain {
                                Some((gain, peak)) => format!(
                                    "「色带放大」= 亮度减去低频后按残差自动定标（×{gain:.0}，峰值 {peak:.2e}）。中灰=无台阶，越白/越黑=越陡。与显示位深无关。"),
                                None => "「色带放大」".to_string(),
                            }
                        } else if self.sky_mode {
                            "天空渐变：低彩度、窄范围，真实照片里最容易出色带的场景。".to_string()
                        } else {
                            format!("全色域：{N_G}×{N_B} = {} 条全宽 R 渐变，逐条换 G/B 取值，覆盖整个立方体。",
                                    N_G * N_B)
                        };
                        ui.label(RichText::new(note).size(11.0).color(th::TEXT_DIM));
                    });

                    // ---- strength + export ----
                    ui.add_space(12.0);
                    th::card(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("应用强度").size(16.0).strong());
                                ui.label(RichText::new("觉得效果太重就往左拉，右边预览实时变化")
                                    .size(11.5).color(th::TEXT_DIM));
                            });
                            ui.add_space(20.0);
                            ui.add_sized([260.0, 24.0],
                                egui::Slider::new(&mut self.strength, 0.0..=1.0).show_value(false));
                            ui.label(RichText::new(format!("{:>3.0}%", self.strength * 100.0)).size(17.0).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let btn = egui::Button::new(RichText::new("导出 .cube").size(14.0).strong().color(Color32::WHITE))
                                    .fill(th::ACCENT);
                                if ui.add_sized([140.0, 40.0], btn).clicked() {
                                    if let Some(path) = rfd::FileDialog::new()
                                        .set_file_name(format!("lut-{}.cube", (self.strength * 100.0) as i32))
                                        .add_filter("Cube LUT", &["cube"]).save_file() {
                                        let _ = export(r, &path.display().to_string(), self.strength as f64, self.smooth as f64);
                                    }
                                }
                            });
                        });
                    });
                    ui.add_space(20.0);
                    if std::env::var("LUTFIT_DEBUG_LAYOUT").is_ok() {
                        eprintln!("[layout] 内容高 {:.0}  视口高 {:.0}  滚动偏移 {:.0}  窗口 {:.0}x{:.0}",
                            ui.min_rect().height(),
                            ui.clip_rect().height(),
                            ui.min_rect().top() - ui.clip_rect().top(),
                            ui.ctx().screen_rect().width(), ui.ctx().screen_rect().height());
                    }
                });
            });
    }
}
