//! Colour science: sRGB / Display P3, transfer functions, Oklab, CIELAB, CIEDE2000.
//!
//! Ported 1:1 from `lutfit/color.py`; the golden tests compare against the Python
//! output bit-for-bit (see `tests/golden.rs`).

pub type Px = [f64; 3];

pub const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4123908, 0.3575843, 0.1804808],
    [0.2126390, 0.7151687, 0.0721923],
    [0.0193308, 0.1191948, 0.9505322],
];
pub const P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];
/// Display P3 and sRGB share the sRGB transfer curve and the D65 white point, so
/// converting between them is a single 3x3 matrix — no chromatic adaptation.
pub const P3_TO_SRGB: [[f64; 3]; 3] = [
    [1.2249401, -0.2249402, 0.0000000],
    [-0.0420569, 1.0420571, 0.0000000],
    [-0.0196376, -0.0786361, 1.0982737],
];

const D65: Px = [0.95047, 1.0, 1.08883];
pub const LUMA: Px = [0.2126, 0.7152, 0.0722];

#[inline]
pub fn mat3(m: &[[f64; 3]; 3], v: Px) -> Px {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let id = 1.0 / det;
    [
        [
            (e * i - f * h) * id,
            (c * h - b * i) * id,
            (b * f - c * e) * id,
        ],
        [
            (f * g - d * i) * id,
            (a * i - c * g) * id,
            (c * d - a * f) * id,
        ],
        [
            (d * h - e * g) * id,
            (b * g - a * h) * id,
            (a * e - b * d) * id,
        ],
    ]
}

#[inline]
pub fn decode_srgb(c: f64) -> f64 {
    let c = if c < 0.0 { 0.0 } else { c };
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn encode_srgb(l: f64) -> f64 {
    let l = if l < 0.0 { 0.0 } else { l };
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

pub fn decode_srgb_px(p: Px) -> Px {
    [decode_srgb(p[0]), decode_srgb(p[1]), decode_srgb(p[2])]
}
pub fn encode_srgb_px(p: Px) -> Px {
    [encode_srgb(p[0]), encode_srgb(p[1]), encode_srgb(p[2])]
}

const M1: [[f64; 3]; 3] = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
];
const M2: [[f64; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

pub fn code_to_oklab(p: Px, space_is_p3: bool) -> Px {
    let rgb_to_xyz = if space_is_p3 {
        &P3_TO_XYZ
    } else {
        &SRGB_TO_XYZ
    };
    let xyz = mat3(rgb_to_xyz, decode_srgb_px(p));
    let lms = mat3(&M1, xyz);
    let lms = [
        lms[0].max(0.0).cbrt(),
        lms[1].max(0.0).cbrt(),
        lms[2].max(0.0).cbrt(),
    ];
    mat3(&M2, lms)
}

pub fn code_to_lab(p: Px, space_is_p3: bool) -> Px {
    let rgb_to_xyz = if space_is_p3 {
        &P3_TO_XYZ
    } else {
        &SRGB_TO_XYZ
    };
    let xyz = mat3(rgb_to_xyz, decode_srgb_px(p));
    let eps = (6.0f64 / 29.0).powi(3);
    let kap = 1.0 / (3.0 * (6.0f64 / 29.0).powi(2));
    let mut f = [0.0f64; 3];
    for i in 0..3 {
        let t = xyz[i] / D65[i];
        f[i] = if t > eps {
            t.max(0.0).cbrt()
        } else {
            t * kap + 4.0 / 29.0
        };
    }
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        200.0 * (f[1] - f[2]),
    ]
}

pub fn oklab_to_code(lab: Px, space_is_p3: bool) -> Px {
    let m2i = mat3_inv(&M2);
    let m1i = mat3_inv(&M1);
    let xyz_to_rgb = mat3_inv(if space_is_p3 {
        &P3_TO_XYZ
    } else {
        &SRGB_TO_XYZ
    });
    let lms = mat3(&m2i, lab);
    let lms3 = [
        lms[0].max(0.0).powi(3),
        lms[1].max(0.0).powi(3),
        lms[2].max(0.0).powi(3),
    ];
    let xyz = mat3(&m1i, lms3);
    encode_srgb_px(mat3(&xyz_to_rgb, xyz))
}

pub fn in_gamut(p: Px, tol: f64) -> bool {
    p.iter().all(|&c| c >= -tol && c <= 1.0 + tol)
}

/// CIEDE2000 (Sharma et al.).  Verified against the published 34-pair reference set.
pub fn de00(l1: Px, l2: Px) -> f64 {
    let (l1v, a1, b1) = (l1[0], l1[1], l1[2]);
    let (l2v, a2, b2) = (l2[0], l2[1], l2[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let cb = 0.5 * (c1 + c2);
    let cb7 = cb.powi(7);
    let g = 0.5 * (1.0 - (cb7 / (cb7 + 25.0f64.powi(7) + 1e-30)).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let mut h1p = b1.atan2(a1p).to_degrees();
    if h1p < 0.0 {
        h1p += 360.0;
    }
    let mut h2p = b2.atan2(a2p).to_degrees();
    if h2p < 0.0 {
        h2p += 360.0;
    }
    if a1p.abs() + b1.abs() == 0.0 {
        h1p = 0.0;
    }
    if a2p.abs() + b2.abs() == 0.0 {
        h2p = 0.0;
    }
    let dlp = l2v - l1v;
    let dcp = c2p - c1p;
    let mut dhp = h2p - h1p;
    if dhp > 180.0 {
        dhp -= 360.0;
    }
    if dhp < -180.0 {
        dhp += 360.0;
    }
    if c1p * c2p == 0.0 {
        dhp = 0.0;
    }
    let dhp_term = 2.0 * (c1p * c2p).sqrt() * (dhp / 2.0).to_radians().sin();
    let lbp = 0.5 * (l1v + l2v);
    let cbp = 0.5 * (c1p + c2p);
    let hsum = h1p + h2p;
    let habs = (h1p - h2p).abs();
    let hbp = if c1p * c2p == 0.0 {
        hsum
    } else if habs <= 180.0 {
        0.5 * hsum
    } else if hsum < 360.0 {
        0.5 * (hsum + 360.0)
    } else {
        0.5 * (hsum - 360.0)
    };
    let t = 1.0 - 0.17 * (hbp - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hbp).to_radians().cos()
        + 0.32 * (3.0 * hbp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hbp - 63.0).to_radians().cos();
    let dtheta = 30.0 * (-(((hbp - 275.0) / 25.0).powi(2))).exp();
    let cbp7 = cbp.powi(7);
    let rc = 2.0 * (cbp7 / (cbp7 + 25.0f64.powi(7) + 1e-30)).sqrt();
    let sl = 1.0 + (0.015 * (lbp - 50.0).powi(2)) / (20.0 + (lbp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * dtheta).to_radians().sin() * rc;
    let l_t = dlp / sl;
    let c_t = dcp / sc;
    let h_t = dhp_term / sh;
    (l_t * l_t + c_t * c_t + h_t * h_t + rt * c_t * h_t).sqrt()
}

#[inline]
pub fn luma(p: Px) -> f64 {
    p[0] * LUMA[0] + p[1] * LUMA[1] + p[2] * LUMA[2]
}
