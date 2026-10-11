"""Fit -> project -> export -> measure ΔE on real pairs -> render comparison images."""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from lutfit.accumulate import summarize
from lutfit.color import code_to_lab, de00
from lutfit.cube import write_cube
from lutfit.fit import fit_multiscale
from lutfit.loader import REC709, build_mask, find_pairs, load_code
from lutfit.project import (enforce_monotone, fold_report, gamut_compress,
                            monotonicity_report, repair_folds)
from lutfit.tetra import evaluate, nodes_flat
from lutfit.verify import axis_sweeps, jacobian_check, neutral_scan

from tools.fit_lut import gray_curve

OUT = "output"
os.makedirs(OUT, exist_ok=True)


def apply_lut(x, img, n=33, chunk=400_000):
    h, w, _ = img.shape
    flat = np.ascontiguousarray(img.reshape(-1, 3), dtype=np.float64)
    out = np.empty_like(flat)
    for i in range(0, len(flat), chunk):
        out[i:i + chunk] = evaluate(x, flat[i:i + chunk], n)
    return np.clip(out, 0.0, 1.0).reshape(h, w, 3)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def label_bar(w, text, height=22):
    bar = Image.new("RGB", (w, height), (24, 24, 24))
    ImageDraw.Draw(bar).text((8, 5), text, fill=(235, 235, 235))
    return bar


def stack(images, texts, width):
    rows = []
    for im, tx in zip(images, texts):
        scale = width / im.width
        im2 = im.resize((width, max(1, int(im.height * scale))), Image.LANCZOS)
        rows.append(label_bar(width, tx))
        rows.append(im2)
    W = width
    H = sum(r.height for r in rows)
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    return canvas


def main():
    with open("work/stats.pkl", "rb") as f:
        blob = pickle.load(f)
    stats = blob["stats"]
    s = summarize(stats)
    from lutfit.coverage import preflight, scaled_min_count, extrapolation_risk, verdict
    total_px = float(s["w"].sum())
    mc = scaled_min_count(total_px)
    print(f"min_count = {mc} (scaled to {total_px/1e6:.1f}M pixels)")
    keep = s["w"] >= mc
    a_bar, b_bar, w = s["a_bar"][keep], s["b_bar"][keep], s["w"][keep]
    core = w >= 200
    t, v, ginfo = gray_curve(stats)

    print("=== fitting ===", flush=True)
    t0 = time.time()
    x, hist = fit_multiscale(a_bar, b_bar, w, levels=(5, 9, 17, 33),
                             lam_s=6.0, beta=2.0, gamma=20.0,
                             mono_mu=1000.0, mono_outer=8,
                             gray_curve_t=t, gray_curve_v=v, chroma_max=0.10,
                             maxiter=400, verbose=True)
    print(f"fit {time.time()-t0:.0f}s", flush=True)

    x, gp = gamut_compress(x, "srgb")
    x = enforce_monotone(x, 33)
    f_before = fold_report(x, 33)
    x = repair_folds(x, 33, iters=25, target=0.02)
    x = enforce_monotone(x, 33)
    f_after = fold_report(x, 33)

    pf = preflight(a_bar, w, n=33, min_support=4)
    risk = extrapolation_risk(x, pf, 33)
    print("\n=== coverage ===")
    print(f"  node coverage {pf['node_coverage']*100:.1f}%   "
          f"pixel-mass coverage {pf['mass_coverage']*100:.3f}%")
    print(f"  bare mean shift {risk['bare_mean_abs_shift']:.5f} vs "
          f"supported {risk['supported_mean_abs_shift']:.5f}")
    print("  " + verdict(pf, risk).replace("\n", "\n  "))

    mo = monotonicity_report(x, 33)
    jc = jacobian_check(x, 33)
    ns = neutral_scan(x, 33, curve_t=t, curve_v=v)
    sw = axis_sweeps(x, 33, n_lines=5, n_steps=512)
    r = evaluate(x, a_bar, 33) - b_bar
    core_err = np.sqrt((w[core, None] * r[core] ** 2).sum() / w[core].sum())
    floor = np.sqrt((s["b_var"][keep] * w[:, None]).sum() / w.sum())

    print("\n=== structural ===")
    print(f"  gamut compress      : {gp['n_bad']} nodes, max excursion {gp['max_excursion']:.4f}")
    print(f"  folds before repair : {f_before['n_folded_cells']} cells / "
          f"{f_before['n_folded_tetra']} tetra, min det {f_before['min_det']:+.4f}")
    print(f"  folds after repair  : {f_after['n_folded_cells']} cells / "
          f"{f_after['n_folded_tetra']} tetra, min det {f_after['min_det']:+.4f}")
    print(f"  jacobian            : {jc['n_nonpositive']}/{jc['cells']} non-positive, "
          f"min det {jc['min_det']:+.4f}")
    print(f"  mono diag negatives : {mo['n_negative']}  min slope {mo['min_slope']:+.6f}")
    print(f"  luma negatives      : {mo['luma_n_negative']}  min {mo['luma_min_slope']:+.6f}")
    print(f"  neutral: dev from measured curve max {ns.get('dev_from_curve', 0):.4f}  "
          f"black {ns['black_point']:.4f}  white {ns['white_point']:.4f}")
    print(f"  max 2nd difference  : {max(q['max_second_diff'] for q in sw.values()):.6f}")
    print(f"  model err {core_err:.5f}  floor {floor:.5f}  total {np.hypot(core_err, floor):.5f}")

    write_cube(f"{OUT}/lut33.cube", x, 33, title="3dlut A->B grade")

    # ---------------------------------------------------------------- dE ----
    print("\n=== ΔE00 on real pairs (LUT applied to A, vs B) ===", flush=True)
    pairs = find_pairs("input", "target")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    demo_idx = [int(i) for i in order[:6]]

    print(f"{'name':<32}{'mean':>7}{'p95':>7}{'max':>7} | {'neutral':>8}{'shadow':>8}"
          f"{'mid':>7}{'high':>7}{'sat':>7}")
    allrows = []
    demo_imgs = {}
    for i, (name, pa, pb) in enumerate(pairs):
        a = load_code(pa, 1200)
        b = load_code(pb, 1200)
        if a.shape != b.shape:
            continue
        m = build_mask(a, b)
        pred = apply_lut(x, a)
        la = code_to_lab(pred, "srgb")
        lb = code_to_lab(b, "srgb")
        de = de00(la, lb)
        dm = de[m]
        chroma = b[m].max(1) - b[m].min(1)
        luma = b[m] @ REC709
        reg = {
            "neutral": dm[chroma < 0.05],
            "shadow": dm[luma < 0.25],
            "mid": dm[(luma >= 0.25) & (luma < 0.7)],
            "high": dm[luma >= 0.7],
            "sat": dm[chroma > 0.3],
        }
        row = dict(name=name, mean=float(dm.mean()), p95=float(np.percentile(dm, 95)),
                   max=float(dm.max()),
                   **{k: (float(vv.mean()) if len(vv) else float("nan")) for k, vv in reg.items()})
        allrows.append(row)
        print(f"{name:<32}{row['mean']:>7.2f}{row['p95']:>7.2f}{row['max']:>7.2f} | "
              f"{row['neutral']:>8.2f}{row['shadow']:>8.2f}{row['mid']:>7.2f}"
              f"{row['high']:>7.2f}{row['sat']:>7.2f}", flush=True)

        if i in demo_idx:
            demo_imgs[name] = (a, b, pred, de)

    key = ["mean", "p95", "max", "neutral", "shadow", "mid", "high", "sat"]
    agg = {k: float(np.mean([r[k] for r in allrows])) for k in key}
    print(f"{'OVERALL MEAN':<32}{agg['mean']:>7.2f}{agg['p95']:>7.2f}{agg['max']:>7.2f} | "
          f"{agg['neutral']:>8.2f}{agg['shadow']:>8.2f}{agg['mid']:>7.2f}"
          f"{agg['high']:>7.2f}{agg['sat']:>7.2f}")

    # -------------------------------------------------------------- demo ----
    print("\n=== rendering demo images ===", flush=True)
    tiles = []
    for name, (a, b, pred, de) in demo_imgs.items():
        for arr, tx in ((a, f"A (input)  {name}"),
                        (pred, "A + LUT33"),
                        (b, "B (target)")):
            tiles.append((to_u8(arr), tx))
    W = 900
    rows = []
    for arr, tx in tiles:
        im = Image.fromarray(arr)
        sc = W / im.width
        rows.append(label_bar(W, tx))
        rows.append(im.resize((W, max(1, int(im.height * sc))), Image.LANCZOS))
    H = sum(r.height for r in rows)
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(f"{OUT}/demo_pairs.png")
    print(f"  wrote {OUT}/demo_pairs.png  ({sheet.width}x{sheet.height})")

    # ----------------------------------------------------- banding test ----
    H2, W2 = 700, 1400
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float64)
    sky = np.zeros((H2, W2, 3))
    sky[..., 0] = 0.20 + 0.45 * (1 - yy / H2) + 0.05 * (xx / W2)
    sky[..., 1] = 0.32 + 0.45 * (1 - yy / H2) + 0.03 * (xx / W2)
    sky[..., 2] = 0.55 + 0.40 * (1 - yy / H2)
    test = np.clip(sky, 0, 1)
    out = apply_lut(x, test)

    blur = np.asarray(Image.fromarray(to_u8(out)).filter(ImageFilter_Gaussian(6)),
                      dtype=np.float64) / 255.0
    resid = (out - blur) * 30.0 + 0.5

    strip = np.zeros((H2, W2, 3))
    strip[..., 0] = xx / W2
    strip[..., 1] = xx / W2
    strip[..., 2] = xx / W2
    strip_out = apply_lut(x, strip)

    imgs = [
        (to_u8(test), "synthetic smooth gradient (input)"),
        (to_u8(out), "after LUT33  — look for contouring / banding"),
        (to_u8(resid), "high-frequency residual x30  — bands would show as contours"),
        (to_u8(strip), "neutral ramp (input)"),
        (to_u8(strip_out), "neutral ramp after LUT33"),
    ]
    rows = []
    for arr, tx in imgs:
        im = Image.fromarray(arr)
        sc = W / im.width
        rows.append(label_bar(W, tx))
        rows.append(im.resize((W, max(1, int(im.height * sc))), Image.LANCZOS))
    H = sum(r.height for r in rows)
    sheet2 = Image.new("RGB", (W, H), (20, 20, 20))
    y = 0
    for r in rows:
        sheet2.paste(r, (0, y))
        y += r.height
    sheet2.save(f"{OUT}/demo_banding.png")
    print(f"  wrote {OUT}/demo_banding.png  ({sheet2.width}x{sheet2.height})")

    np.save(f"{OUT}/lut33.npy", x)
    print("\ndone.")


from PIL import ImageFilter


def ImageFilter_Gaussian(r):
    return ImageFilter.GaussianBlur(r)


if __name__ == "__main__":
    main()
