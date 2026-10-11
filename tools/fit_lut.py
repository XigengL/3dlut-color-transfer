"""Fit the 33^3 LUT from the accumulated statistics, project it, verify it, export it."""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lutfit.accumulate import summarize
from lutfit.grayaxis import gray_curve
from lutfit.coverage import preflight, scaled_min_count, verdict
from lutfit.cube import read_cube, write_cube
from lutfit.fit import fit_multiscale, neutral_targets
from lutfit.project import (enforce_monotone, fold_report, gamut_compress,
                            monotonicity_report, repair_folds)
from lutfit.tetra import nodes_flat
from lutfit.verify import (axis_sweeps, jacobian_check, neutral_scan, range_check)

OUT = "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="work/stats.pkl")
    ap.add_argument("--min-count", type=int, default=None,
                    help="bin sample threshold; default scales with total pixels")
    ap.add_argument("--lam-s", type=float, default=6.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=20.0)
    ap.add_argument("--mono-mu", type=float, default=1000.0)
    ap.add_argument("--mono-outer", type=int, default=8)
    ap.add_argument("--repair-folds", type=int, default=25)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--tag", default="lut33")
    ap.add_argument("--space", default="srgb")
    ap.add_argument("--no-project", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    with open(args.stats, "rb") as f:
        blob = pickle.load(f)
    stats = blob["stats"]
    s = summarize(stats)

    total_px = float(s["w"].sum())
    if args.min_count is None:
        args.min_count = scaled_min_count(total_px)
    keep = s["w"] >= args.min_count
    a_bar, b_bar, w = s["a_bar"][keep], s["b_bar"][keep], s["w"][keep]
    print(f"samples: {keep.sum():,} bins (>= {args.min_count} px), "
          f"{w.sum()/1e6:.1f}M pixels, {100*w.sum()/stats['cnt'].sum():.2f}% of mass")

    pf = preflight(a_bar, w, n=33, min_support=4)
    print(f"\n--- pre-flight coverage (33^3) ---")
    print(f"  node coverage : {pf['node_coverage']*100:.1f}%  "
          f"({pf['n_bare']} / {pf['n_nodes']} nodes have no support)")
    print(f"  pixel mass in supported nodes : {pf['mass_coverage']*100:.3f}%")
    print(f"  median bin support : {pf['median_bin_count']:.0f}")
    print("  " + verdict(pf).replace("\n", "\n  "))

    t, v, ginfo = gray_curve(stats)
    if t is not None:
        print(f"neutral curve: black_lift={ginfo['black_lift']:.4f}  "
              f"white_end={ginfo['white_end']:.4f}  "
              f"tint R-G={ginfo['tint_rg']:+.4f} G-B={ginfo['tint_gb']:+.4f}  "
              f"(valid luma {ginfo['t_valid_lo']:.3f}..{ginfo['t_valid_hi']:.3f})")
    np.savez(os.path.join(OUT, "gray_curve.npz"), t=t, v=v)

    t0 = time.time()
    x, hist = fit_multiscale(a_bar, b_bar, w, levels=(5, 9, 17, 33),
                             lam_s=args.lam_s, beta=args.beta, gamma=args.gamma,
                             mono_mu=args.mono_mu, mono_outer=args.mono_outer,
                             gray_curve_t=t, gray_curve_v=v,
                             maxiter=args.maxiter, verbose=True)
    print(f"fit done in {time.time()-t0:.1f}s")

    raw = x.copy()
    report = {"config": vars(args), "neutral_curve": ginfo}

    x, gp = gamut_compress(x, args.space)
    x = enforce_monotone(x, 33)
    folds_before = fold_report(x, 33)
    if args.repair_folds > 0:
        x = repair_folds(x, 33, iters=args.repair_folds, target=0.02)
        x = enforce_monotone(x, 33)
    folds_after = fold_report(x, 33)
    report["projection"] = {"gamut": gp, "folds_before": folds_before, "folds_after": folds_after}
    print(f"gamut compress : {gp['n_bad']} nodes, max excursion {gp['max_excursion']:.4f}")
    print(f"folds before   : {folds_before['n_folded_cells']} cells / "
          f"{folds_before['n_folded_tetra']} tetra, min det {folds_before['min_det']:+.4f}")
    print(f"folds after    : {folds_after['n_folded_cells']} cells / "
          f"{folds_after['n_folded_tetra']} tetra, min det {folds_after['min_det']:+.4f}")

    np.save(os.path.join(OUT, f"{args.tag}.npy"), x)
    write_cube(os.path.join(OUT, f"{args.tag}.cube"), x, 33,
               title=f"3dlut {args.tag}")

    # ---- structural verification -------------------------------------------
    print("\n--- structural verification ---")
    for name, ax in (("R", 0), ("G", 1), ("B", 2)):
        pass
    sw = axis_sweeps(x, 33)
    for k, val in sw.items():
        print(f"  axis {k}: min first diff {val['min_first_diff']:+.6f}   "
              f"max 2nd diff {val['max_second_diff']:.6f}")
    ns = neutral_scan(x, 33, curve_t=t, curve_v=v)
    print(f"  neutral: min first diff {ns['min_first_diff']:+.6f}  "
          f"max 2nd diff {ns['max_second_diff']:.6f}  "
          f"dev from measured curve {ns.get('dev_from_curve', float('nan')):.5f}  "
          f"chroma gain near neutral {ns['near_neutral_chroma_gain']:.2f}")
    jc = jacobian_check(x, 33)
    print(f"  jacobian: min det {jc['min_det']:+.4f}  non-positive {jc['n_nonpositive']}/{jc['cells']}  "
          f"singular range {jc['min_singular']:.5f}..{jc['max_singular']:.4f}")
    rc = range_check(x, 33)
    print(f"  range: [{rc['min']:.5f}, {rc['max']:.5f}]  below0={rc['n_below']} above1={rc['n_above']}")
    report.update(structural=dict(axis_sweeps=sw, neutral=ns, jacobian=jc, range=rc))

    # ---- residual decomposition --------------------------------------------
    from lutfit.tetra import evaluate
    pred = evaluate(x, a_bar, 33)
    resid = pred - b_bar
    model = float(np.sqrt((w[:, None] * resid ** 2).sum() / w.sum()))
    floor = float(np.sqrt((s["b_var"][keep] * w[:, None]).sum() / w.sum()))
    print(f"\n--- residual decomposition (code units, RMS) ---")
    print(f"  model error   (33^3 LUT cannot represent) : {model:.5f}")
    print(f"  irreducible   (local edits + JPEG noise)  : {floor:.5f}")
    print(f"  expected total                            : {np.hypot(model, floor):.5f}")
    report["residual"] = dict(model=model, irreducible=floor, total=float(np.hypot(model, floor)))

    with open(os.path.join(OUT, f"{args.tag}_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {OUT}/{args.tag}.cube  and  {OUT}/{args.tag}_report.json")


if __name__ == "__main__":
    main()
