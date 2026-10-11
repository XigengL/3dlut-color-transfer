"""按图对做 K 折交叉验证 —— 回答「样本内的 ΔE 能不能代表新照片」。

每折：只用训练组拟合 → 在训练组和留出组上分别测 ΔE00。
如果留出 ≈ 训练，说明没有过拟合。
如果留出明显更差，说明 33^3 / 10.8 万自由参数相对 6.6 万样本过拟合了，
需要在重写之前先加正则化。

用法：
    python3 tools/cv_holdout.py            # 默认 5 折
    python3 tools/cv_holdout.py 3          # 3 折（更快）
"""
import os
import sys
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lutfit.accumulate import build_statistics, summarize
from lutfit.color import code_to_lab, de00
from lutfit.coverage import preflight, scaled_min_count
from lutfit.fit import fit_multiscale
from lutfit.grayaxis import gray_curve
from lutfit.loader import build_mask, find_pairs, load_code
from lutfit.pipeline import apply_lut
from lutfit.project import (enforce_monotone, fold_report, gamut_compress,
                            repair_folds)
from lutfit.tetra import evaluate

GRID = 65
TARGET_MAX = 1800
EVAL_MAX = 1200
LUT_N = 33
SEED = 20250930


def fit_on(train_pairs):
    stats = build_statistics(train_pairs, grid=GRID, target_max=TARGET_MAX, verbose=False)
    s = summarize(stats)
    mc = scaled_min_count(float(s["w"].sum()))
    keep = s["w"] >= mc
    a_bar, b_bar, w = s["a_bar"][keep], s["b_bar"][keep], s["w"][keep]
    t, v, _ = gray_curve(stats)

    x, _ = fit_multiscale(a_bar, b_bar, w, levels=(5, 9, 17, 25, 33),
                          lam_s=6.0, beta=2.0, gamma=20.0,
                          mono_mu=1000.0, mono_outer=8,
                          gray_curve_t=t, gray_curve_v=v, chroma_max=0.10,
                          maxiter=400, verbose=False)
    x, _ = gamut_compress(x, "srgb")
    x = enforce_monotone(x, LUT_N)
    if fold_report(x, LUT_N)["n_folded_cells"] > 0:
        x = repair_folds(x, LUT_N, iters=40, target=0.02)
        x = enforce_monotone(x, LUT_N)

    r = evaluate(x, a_bar, LUT_N) - b_bar
    model = float(np.sqrt((w[:, None] * r ** 2).sum() / w.sum()))
    floor = float(np.sqrt((s["b_var"][keep] * w[:, None]).sum() / w.sum()))
    pf = preflight(a_bar, w, n=LUT_N, min_support=4)
    return x, model, floor, pf, len(w)


def de_of(x, pairs):
    """ΔE00 of LUT(A) vs B for each pair."""
    rows = []
    for name, pa, pb in pairs:
        a = load_code(pa, EVAL_MAX)
        b = load_code(pb, EVAL_MAX)
        if a.shape != b.shape:
            continue
        m = build_mask(a, b)
        pred = apply_lut(x, a, LUT_N)
        de = de00(code_to_lab(pred, "srgb"), code_to_lab(b, "srgb"))[m]
        if de.size:
            rows.append(dict(name=name, mean=float(de.mean()),
                             p95=float(np.percentile(de, 95))))
    return rows


def agg(rows):
    if not rows:
        return float("nan")
    return float(np.mean([r["mean"] for r in rows]))


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pairs = find_pairs("input", "target")
    print(f"{len(pairs)} 组图对，{k} 折交叉验证\n")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    folds = np.array_split(order, k)

    print(f"\n{'fold':>4} {'trainN':>7} {'holdN':>6} {'model':>8} {'floor':>8} "
          f"{'in-sample ΔE':>13} {'holdout ΔE':>11} {'gap':>7} {'folds':>6} {'sec':>5}")
    t_start = time.time()
    gaps, tr_all, ho_all = [], [], []
    for fi, hold_idx in enumerate(folds):
        t0 = time.time()
        hold_names = {pairs[i][0] for i in hold_idx}
        train = [p for p in pairs if p[0] not in hold_names]
        hold = [pairs[i] for i in hold_idx]

        x, model, floor, pf, nsamp = fit_on(train)
        tr = de_of(x, train)
        ho = de_of(x, hold)
        tr_m, ho_m = agg(tr), agg(ho)
        if np.isfinite(tr_m) and np.isfinite(ho_m):
            gaps.append(ho_m - tr_m)
            tr_all.append(tr_m)
            ho_all.append(ho_m)
        ff = fold_report(x, LUT_N)
        print(f"{fi:>4} {len(train):>7} {len(hold):>6} {model:>8.5f} {floor:>8.5f} "
              f"{tr_m:>13.3f} {ho_m:>11.3f} {ho_m-tr_m:>+7.3f} "
              f"{ff['n_folded_cells']:>6} {time.time()-t0:>5.0f}", flush=True)

    print("\n" + "=" * 74)
    print(f"训练集 ΔE00 平均 : {np.mean(tr_all):.3f}")
    print(f"留出集 ΔE00 平均 : {np.mean(ho_all):.3f}")
    print(f"过拟合间隙       : {np.mean(gaps):+.3f}  "
          f"（{(np.mean(gaps)/np.mean(tr_all))*100:+.1f}%）")
    print(f"总耗时           : {(time.time()-t_start)/60:.1f} 分钟")
    if np.mean(gaps) < 0.15:
        print("\n结论：留出与训练基本一致 -> 没有过拟合，样本内数字可信。")
    elif np.mean(gaps) < 0.5:
        print("\n结论：留出略差 -> 有轻微过拟合，正则化可以再紧一点。")
    else:
        print("\n结论：留出明显更差 -> 过拟合，需要先加正则化再考虑重写。")


if __name__ == "__main__":
    main()
