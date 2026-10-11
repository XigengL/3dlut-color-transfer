//! Apple-ish minimal visual design: a lot of whitespace, one accent colour, rounded
//! surfaces, and a clear type hierarchy.  Nothing decorative.

use egui::{Color32, FontFamily, FontId, Rounding, Stroke, TextStyle, Visuals};

pub const BG: Color32 = Color32::from_rgb(0xF5, 0xF5, 0xF7);
pub const CARD: Color32 = Color32::from_rgb(0xFF, 0xFF, 0xFF);
pub const TEXT: Color32 = Color32::from_rgb(0x1D, 0x1D, 0x1F);
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x6E, 0x6E, 0x73);
pub const SEP: Color32 = Color32::from_rgb(0xD2, 0xD2, 0xD7);
pub const ACCENT: Color32 = Color32::from_rgb(0x00, 0x71, 0xE3);
pub const OK: Color32 = Color32::from_rgb(0x1D, 0x8A, 0x3E);
pub const WARN: Color32 = Color32::from_rgb(0xB2, 0x5E, 0x00);
pub const BAD: Color32 = Color32::from_rgb(0xC7, 0x22, 0x1E);

/// egui ships no CJK glyphs, so every Chinese character renders as a tofu box.
/// Load a system font instead of bundling one: macOS, Windows and Linux all ship a
/// usable sans-serif CJK face, and it keeps the binary at 5.6 MB.
pub fn cjk_candidates() -> &'static [(&'static str, u32)] {
    &[
        // macOS
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", 0),
        // Windows
        ("C:/Windows/Fonts/msyh.ttc", 0),
        ("C:/Windows/Fonts/msyh.ttf", 0),
        ("C:/Windows/Fonts/simhei.ttf", 0),
        ("C:/Windows/Fonts/simsun.ttc", 0),
        // Linux
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
        ("/usr/share/fonts/truetype/arphic/uming.ttc", 0),
    ]
}

/// Install the first CJK font that exists.  Returns its path so the caller can log it.
pub fn install_cjk_font(ctx: &egui::Context) -> Option<String> {
    for (path, index) in cjk_candidates() {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut fd = egui::FontData::from_owned(bytes);
        fd.index = *index;
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert("cjk".to_owned(), fd);
        // Put it first so Latin is rendered by the same face and the two scripts
        // match; push it as a fallback for monospace.
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "cjk".to_owned());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .push("cjk".to_owned());
        ctx.set_fonts(fonts);
        return Some((*path).to_string());
    }
    None
}

pub fn apply(ctx: &egui::Context) {
    install_cjk_font(ctx);
    let mut style = (*ctx.style()).clone();

    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(26.0, FontFamily::Proportional),
        ),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (
            TextStyle::Button,
            FontId::new(14.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(12.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(13.0, FontFamily::Monospace),
        ),
    ]
    .into();

    let mut v = Visuals::light();
    v.panel_fill = BG;
    v.window_fill = CARD;
    // Track blends into the page, so only the handle reads as a control.
    v.extreme_bg_color = BG;
    v.override_text_color = Some(TEXT);
    v.window_rounding = Rounding::same(12.0);
    v.window_stroke = Stroke::new(1.0_f32, SEP);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.25);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);

    let r = Rounding::same(8.0);
    v.widgets.noninteractive.rounding = r;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, SEP);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_DIM);

    v.widgets.inactive.rounding = r;
    // The scroll-bar handle is drawn with `inactive.bg_fill` and its track with
    // `extreme_bg_color`.  Setting both to near-white made the scroll bar invisible
    // against the #F5F5F7 background -- which read as "the page will not scroll".
    v.widgets.inactive.bg_fill = Color32::from_rgb(0xC2, 0xC2, 0xC8);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(0xE8, 0xE8, 0xED);
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);

    v.widgets.hovered.rounding = r;
    v.widgets.hovered.bg_fill = Color32::from_rgb(0xA8, 0xA8, 0xAE);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0xDD, 0xDD, 0xE3);
    v.widgets.hovered.bg_stroke = Stroke::NONE;

    v.widgets.active.rounding = r;
    v.widgets.active.bg_fill = ACCENT.gamma_multiply(0.18);
    v.widgets.active.weak_bg_fill = ACCENT.gamma_multiply(0.18);
    v.widgets.active.bg_stroke = Stroke::NONE;

    v.widgets.open.rounding = r;

    style.visuals = v;
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 9.0);
    style.spacing.interact_size.y = 30.0;
    ctx.set_style(style);
}

/// A white rounded card with a generous inner margin.
pub fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let pad = 18.0;
    egui::Frame::none()
        .fill(CARD)
        .rounding(Rounding::same(12.0))
        .stroke(Stroke::new(1.0_f32, SEP))
        .inner_margin(egui::Margin::same(pad))
        .show(ui, |ui| add(ui))
        .inner
}

pub fn heading(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(26.0).strong().color(TEXT));
}

pub fn sub(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(13.0).color(TEXT_DIM));
}

pub fn section(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(12.0)
            .strong()
            .color(TEXT_DIM),
    );
    ui.add_space(2.0);
}

/// Shorten a path for display without hiding which file it is: keep the leading
/// directory component and the tail.
pub fn ellipsize(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let head: String = chars[..max / 3].iter().collect();
    let tail: String = chars[chars.len() - (max - max / 3 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

pub fn mmss(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        format!("{s} 秒")
    } else {
        format!("{} 分 {:02} 秒", s / 60, s % 60)
    }
}
