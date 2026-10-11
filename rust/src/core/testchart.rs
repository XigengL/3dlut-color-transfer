//! Synthetic test pattern for judging LUT smoothness.
//!
//! The pattern supplies *code values*; it is colour-space agnostic.  Reading those
//! values as Display P3 covers the P3 gamut, reading them as sRGB covers the sRGB
//! gamut — the colour space only enters when we *display* the result.
//!
//! Layout is full-width horizontal ramps, not a tile grid.  That choice is forced by
//! the arithmetic: LUT contouring appears at the scale of one grid interval, so a
//! 33-node LUT needs each ramp to span a large fraction of the image.  An earlier
//! 6x6 tile version gave only 85px per ramp, i.e. 2.7px per node interval, and 1.6px
//! once scaled into the panel — far below what the eye can resolve, which made the
//! chart useless for its one job.  Here each ramp spans the full width.

pub const CW: usize = 768;
pub const W: usize = CW;
pub const CH: usize = 512;
pub const H: usize = CH;
/// (G, B) combinations stacked as stripes; R sweeps across the whole width.
pub const N_G: usize = 8;
pub const N_B: usize = 4;

/// `(G, B)` for stripe `i`, in boustrophedon order.
///
/// Plain raster order made G run 0..1 then jump straight back to 0 between consecutive
/// stripes — a full-range discontinuity, whose blur residual swamped the LUT's own
/// curvature and flattened the amplification view.  Reversing every other row removes
/// that jump, and only B steps between rows.
fn gb(i: usize, ng: usize, nb: usize) -> (f32, f32) {
    let i = i.min(ng * nb - 1);
    let (row, col) = (i / ng, i % ng);
    let c = if row % 2 == 0 { col } else { ng - 1 - col };
    (c as f32 / (ng - 1) as f32, row as f32 / (nb - 1).max(1) as f32)
}

/// Stripes with a smooth cross-fade instead of a hard edge.
///
/// Hard edges produce residuals two orders of magnitude larger than the LUT's own
/// curvature, so the auto-gain in `amplify_steps` was set by the stripe boundaries and
/// the actual banding got scaled down to a flat grey.  Blending removes the
/// discontinuity, leaving the curvature as the dominant residual.  Each stripe still
/// holds its pure (G, B) through its middle half, so coverage is preserved.
pub fn stripes(w: usize, h: usize, ng: usize, nb: usize) -> Vec<f32> {
    let rows = ng * nb;
    // Cross-fade across the whole stripe: the (G, B) trajectory is then a continuous
    // path that still passes through every grid point, with no fast transitions.
    let tw = 1.0f64;
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let fy = (y as f64 + 0.5) / h as f64 * rows as f64;
        let i = fy.floor() as usize;
        let t = fy - i as f64;
        let (g0, b0) = gb(i, ng, nb);
        let (g1, b1) = gb(i + 1, ng, nb);
        let k = if t > 1.0 - tw {
            let u = (t - (1.0 - tw)) / tw;
            (u * u * (3.0 - 2.0 * u)) as f32 // smoothstep
        } else {
            0.0
        };
        let g = g0 + (g1 - g0) * k;
        let b = b0 + (b1 - b0) * k;
        for x in 0..w {
            let o = (y * w + x) * 3;
            out[o] = x as f32 / (w - 1) as f32;
            out[o + 1] = g;
            out[o + 2] = b;
        }
    }
    out
}

/// A realistic sky: deep saturated blue at the top grading to a pale, slightly warm
/// horizon.  The first version used a narrow, uniformly light band, which the LUT barely
/// touched, so smoothing produced no visible difference — the opposite of useful.
pub fn sky(w: usize, h: usize) -> Vec<f32> {
    // top = deep blue, bottom = pale horizon
    const TOP: [f32; 3] = [0.10, 0.26, 0.52];
    const BOT: [f32; 3] = [0.84, 0.87, 0.92];
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let t = y as f32 / (h - 1).max(1) as f32;
        // slight ease so the upper third changes fastest, as real skies do
        let s = t * t * (3.0 - 2.0 * t);
        for x in 0..w {
            let u = x as f32 / (w - 1).max(1) as f32;
            let o = (y * w + x) * 3;
            // a touch of horizontal drift so it is not a purely 1-D ramp
            let d = (u - 0.5) * 0.03;
            out[o] = (TOP[0] + (BOT[0] - TOP[0]) * s + d).clamp(0.0, 1.0);
            out[o + 1] = (TOP[1] + (BOT[1] - TOP[1]) * s + d * 0.5).clamp(0.0, 1.0);
            out[o + 2] = (TOP[2] + (BOT[2] - TOP[2]) * s - d * 0.5).clamp(0.0, 1.0);
        }
    }
    out
}

/// 8x8 ordered Bayer dither at 1/255, so the 8-bit preview stops stacking its own
/// contouring on top of the LUT's.  Off by default.
const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

pub fn to_rgb8(v: &[f64], w: usize, h: usize, dither: bool) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let t = if dither { (BAYER8[y & 7][x & 7] as f64 / 64.0 - 0.5) / 255.0 } else { 0.0 };
            for c in 0..3 {
                out[i + c] = ((v[i + c] + t).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    out
}
