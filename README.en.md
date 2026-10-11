# lut3d

Fit a **single 3D LUT** from paired before/after photographs — camera JPEG in,
retouched JPEG out. Rust, cross-platform, offline.

The point of this project is not the idea (fitting a LUT from image pairs is not new)
but the **error accounting**: it tells you how much of the remaining error the fitting
algorithm could still recover, and how much no global colour transform can ever touch.

- Rust port is verified **numerically equivalent** to an independent Python reference
  (model error matching to 5 decimal places).
- ~**9× faster** than the reference implementation (142 s vs 22 min for 87 pairs).
- Ships as a **5.9 MB** self-contained binary — no Python, no runtime, no bundled fonts.

---

## What it does

Give it two folders of the same scenes — one straight out of the camera, one retouched —
and it produces a `.cube` LUT plus a report on whether that LUT can be trusted.

```text
┌─────────────────────────┬──────────────────────────────────────────────┐
│  3D LUT 拟合             │  ✓ 预检                    46 组图 · 155M 像素 │
│                         │                                              │
│  ① 选择两组图片          │    像素质量覆盖   99.99%   真实照片落在有效数据 │
│  原图     [ 选择文件夹… ] │    色彩网格覆盖   31%      其余是平滑推测      │
│  修图后   [ 选择文件夹… ] │    外推区改动量   0.074    有数据处为 0.111    │
│                         │                                              │
│  ② 网格尺寸              │  拟合质量                                     │
│  [17³] [33³] [65³]      │  1.01        95 分位色差  2.11                │
│    33³ 通用              │  ΔE00        模型误差     0.01301            │
│  ☑ 保持灰阶中性          │              数据下界     0.02692            │
│                         │              总误差/下界  +11.1%              │
│  ③ 开始                  │                                              │
│  [   开始拟合   ]        │  结构检查                                     │
│                         │  色彩反转 0 处                             │
│                         │  色带风险 0.00168                          │
│                         │                                              │
│                         │  效果预览   （5 张，每列 原图→LUT→成品）      │
│                         │  平滑      [========|----]  0.60             │
│                         │  应用强度  [======|------] 100%  [ 导出 .cube ]│
└─────────────────────────┴──────────────────────────────────────────────┘
```

The interface is in Chinese; the code, comments and docs are in English.

---

## What it does *not* do

Being explicit about this matters more than the feature list:

- **Ignores ICC profiles.** JPEG code values are treated as the working space verbatim.
  This is a deliberate choice for data with no trustworthy embedded profile, but it means
  a Display P3 working space only enters at the display step, not the fit.
- **JPEG only.** HEIC is unsupported and reported as skipped.
- **No spatial awareness.** The output is one global colour transform. Local adjustments
  (gradient masks, brushes, dodge & burn) are unrepresentable *by construction* — this is
  measured, not assumed: see the error budget below.
- **No ICC / no colour management on export.** A `.cube` has no colour space tag.
- Needs **≥ 10 well-aligned pairs** with matching file stems and identical dimensions.
- The GUI preview is **8-bit** regardless of your display. For 10-bit delivery the numeric
  readout is authoritative, not the preview.

---

## Build

Requires a recent stable Rust (1.76+).

```bash
cd rust
cargo build --release
```

**macOS** — produces a self-contained `.app`:

```bash
./build_mac.sh          # → dist/lut3d.app  (5.9 MB)
```

**Windows**:

```bat
build_win.bat           :: → dist\lut3d.exe
```

No bundled assets. The CJK font is loaded from the system at startup (PingFang /
Hiragino / Microsoft YaHei / Noto CJK, in that order), which is why the binary stays
under 6 MB and sidesteps font redistribution licensing entirely.

---

## Use

```bash
./target/release/lut3d                                  # GUI
./target/release/lut3d fit <原图目录> <修图目录> out.cube 33   # CLI
```

Put the same scene's two files under the **same stem** (`A.jpg` / `A.png` both work);
the extensions may differ.

Then import the `.cube` into Resolve, Premiere, Photoshop, or anything else that reads
the format.

---

## How it works

The whole method rests on one observation:

> **Tetrahedral interpolation is linear in the grid values.**

Which cell a query falls into depends only on the input coordinate, never on the values
stored in the grid. So a colour transform becomes `output = T(u)·x` — a sparse linear
least-squares problem in ~108k unknowns for a 33³ LUT.

Everything else — multiscale coarse-to-fine, Laplacian regularisation, box constraints,
fold repair, isotonic projection, adaptive regularisation — is there to keep that solution
usable where there is no data, where it is ill-posed, and where it would otherwise produce
colour inversions or banding.

**Full write-up: [`rust/DESIGN.md`](rust/DESIGN.md)** — including the error-budget algebra
and an explicit list of directions that have been *ruled out* with measurements.

---

## Verification

Two independent layers, both runnable in CI:

**Numerical core** (`cargo test --release`, seconds) — checked against a separate Python
implementation and against published references:

| Test | Bound |
|---|---|
| CIEDE2000 vs Sharma et al. 34-pair reference | max error < 1e-4 |
| sRGB transfer functions vs Python | bit-identical (< 1e-15) |
| Oklab / CIELAB vs Python | < 1e-12 |
| Tetrahedral indices and weights (3000 points) | zero mismatches, < 1e-15 |
| Identity LUT reconstruction, 2³…33³ | < 1e-11 |
| Laplacian on constant / linear fields | exactly zero |

**Synthetic recovery** (`tests/synthetic.rs`) — asks the stronger question: given data
generated by a *known* smooth transform, does the fit recover it? No private data needed.

```
17³ grid: max error 0.0198   mean 0.003913   folds 0
33³ grid: max error 0.0098   mean 0.001636   folds 0
```

The error halves as the grid doubles — the expected behaviour, and a check that the
multiscale hand-off is not losing information.

**End-to-end vs the reference** (`cargo test --release -- --ignored end_to_end`) is
`#[ignore]`d because it needs paired photographs, which are not in this repository.

---

## Accuracy: what is actually achievable

Error decomposes exactly:

```
total² = model² + irreducible²
```

|  | model error | irreducible floor | total / floor |
|---|---|---|---|
| Dataset 1 — 46 pairs, gentle grade | 0.01299 | 0.02692 | +11.0% |
| Dataset 2 — 87 pairs, strong grade | 0.03893 | 0.07357 | +13.1% |

The model error is only **18.5% of the squared error**. So a *perfect* fitting algorithm
would improve RMS by at most **~10%**. Improving the solver further is close to pointless.

The floor itself splits into the part a *set* of LUTs could remove (photos edited
differently from one another) and the part nothing can (local adjustments inside a single
photo):

|  | within-image | between-image |
|---|---|---|
| Dataset 1 | **68.3%** | 31.7% |
| Dataset 2 | 39.3% | **60.7%** |

Which is why the honest answer to "is there room to optimise?" is **dataset-dependent** —
and why the tool reports this split on every run.

---

## Repository layout

```
rust/               the implementation (primary)
  src/core/         colour science, tetrahedral operator, solver, constraints
  src/app.rs        GUI
  tests/            golden + synthetic tests
  DESIGN.md         method and error analysis
lut3d/             Python reference implementation
tools/              dataset analysis and comparison scripts
```

The Python tree is kept as a **reference implementation** for cross-checking, not as a
supported path.

---

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
