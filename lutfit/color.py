"""Colour science: transfer functions, RGB<->XYZ for sRGB and Display P3, CIELAB, Oklab, CIEDE2000.

No external dependencies beyond numpy.  No colour management is performed on load:
the pixel code values are used verbatim, as specified in DESIGN.md.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- primaries ---
# Both spaces are D65, so no chromatic adaptation is ever required.

RGB_TO_XYZ = {
    "srgb": np.array([
        [0.4123908, 0.3575843, 0.1804808],
        [0.2126390, 0.7151687, 0.0721923],
        [0.0193308, 0.1191948, 0.9505322],
    ], dtype=np.float64),
    "p3": np.array([
        [0.4865709, 0.2656677, 0.1982173],
        [0.2289746, 0.6917385, 0.0792869],
        [0.0000000, 0.0451134, 1.0439444],
    ], dtype=np.float64),
}
XYZ_TO_RGB = {k: np.linalg.inv(v) for k, v in RGB_TO_XYZ.items()}

# Display P3 and sRGB share the sRGB transfer curve, so only the matrix differs.
P3_TO_SRGB = XYZ_TO_RGB["srgb"] @ RGB_TO_XYZ["p3"]

D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


# ------------------------------------------------------------ transfer fn ----
def encode_srgb(lin: np.ndarray) -> np.ndarray:
    """Linear -> sRGB code value (the Display P3 curve is identical)."""
    lin = np.clip(lin, 0.0, None)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def decode_srgb(code: np.ndarray) -> np.ndarray:
    """sRGB code value -> linear."""
    code = np.clip(code, 0.0, None)
    return np.where(code <= 0.04045, code / 12.92,
                    np.power((code + 0.055) / 1.055, 2.4))


# ------------------------------------------------------------------ oklab ----
_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
], dtype=np.float64)
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float64)


def code_to_linear(code: np.ndarray, space: str = "srgb") -> np.ndarray:
    return decode_srgb(code)


def linear_to_xyz(lin: np.ndarray, space: str = "srgb") -> np.ndarray:
    return lin @ RGB_TO_XYZ[space].T


def code_to_oklab(code: np.ndarray, space: str = "srgb") -> np.ndarray:
    xyz = linear_to_xyz(decode_srgb(code), space)
    lms = np.cbrt(np.maximum(xyz @ _M1.T, 0.0))
    return lms @ _M2.T


def code_to_lab(code: np.ndarray, space: str = "srgb") -> np.ndarray:
    xyz = linear_to_xyz(decode_srgb(code), space) / D65
    eps = (6.0 / 29.0) ** 3
    kap = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), xyz * kap + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


# ---------------------------------------------------------------- ciede2000 --
def de00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. implementation)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7 + 1e-30)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where((np.abs(a1p) + np.abs(b1)) == 0.0, 0.0, h1p)
    h2p = np.where((np.abs(a2p) + np.abs(b2)) == 0.0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    hbp = np.where(C1p * C2p == 0.0, hsum,
                   np.where(habs <= 180.0, 0.5 * hsum,
                            np.where(hsum < 360.0, 0.5 * (hsum + 360.0),
                                     0.5 * (hsum - 360.0))))

    T = (1.0
         - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# ------------------------------------------------------------------ gamut ----
def oklab_to_code(lab: np.ndarray, space: str = "srgb") -> np.ndarray:
    """Oklab -> code value (may fall outside [0,1]; caller decides what to do)."""
    lms = lab @ np.linalg.inv(_M2).T
    xyz = np.maximum(lms, 0.0) ** 3 @ np.linalg.inv(_M1).T
    lin = xyz @ XYZ_TO_RGB[space].T
    return encode_srgb(lin)


def in_gamut(code: np.ndarray, tol: float = 0.0) -> np.ndarray:
    return np.all((code >= -tol) & (code <= 1.0 + tol), axis=-1)
