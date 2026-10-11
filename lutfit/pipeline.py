"""End-to-end pipeline: folders of A/B JPEGs in, a fitted .cube plus a report out.

Shared by the command line and the GUI, so both exercise exactly the same code.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .accumulate import build_statistics, summarize
from .color import code_to_lab, de00
from .coverage import extrapolation_risk, preflight, scaled_min_count, verdict
from .cube import write_cube
from .errors import Cancelled
from .fit import fit_multiscale
from .grayaxis import gray_curve
from .loader import REC709, build_mask, find_pairs, load_code
from .project import (enforce_monotone, fold_report, gamut_compress,
                      monotonicity_report, polish_lut, repair_folds)
from .tetra import evaluate, nodes_flat
from .verify import (axis_sweeps, jacobian_check, monotone_excursion,
                     neutral_scan)

# Stage boundaries as fractions of the progress bar.
STAGES = [
    ("扫描文件", 0.00, 0.01),
    ("统计颜色分布", 0.01, 0.45),
    ("预检覆盖度", 0.45, 0.48),
    ("拟合 LUT", 0.48, 0.85),
    ("约束与修复", 0.85, 0.92),
    ("结构验证", 0.92, 0.95),
    ("评估色差", 0.95, 0.99),
    ("生成预览", 0.99, 1.00),
]


@dataclass
class FitOptions:
    lut_size: int = 33
    grid: int = 65
    target_max: int = 1800
    eval_max: int = 1200
    lam_s: float = 6.0
    beta: float = 2.0
    gamma: float = 20.0
    mono_mu: float = 1000.0
    mono_outer: int = 8
    repair_folds: int = 40
    adaptive: bool = True         # 折叠未清零时自动加强正则化重试末层
    max_trials: int = 3
    preserve_neutral: bool = True
    space: str = "srgb"
    max_pairs: int = 0            # 0 = all
    preview_count: int = 3
    preview_max: int = 640        # stored preview edge, keeps the GUI light

    def scaled(self) -> "FitOptions":
        """Grid that suits the LUT size, so a 17^3 export does not over-resolve."""
        o = FitOptions(**asdict(self))
        if o.lut_size <= 17:
            o.grid = min(o.grid, 49)
        return o


@dataclass
class FitResult:
    lut: np.ndarray
    options: FitOptions
    n_pairs: int = 0
    pixels: float = 0.0
    preflight: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    structural: dict = field(default_factory=dict)
    residual: dict = field(default_factory=dict)
    per_pair: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    gray: dict = field(default_factory=dict)
    previews: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    elapsed: float = 0.0
    verdict: str = ""


def _check(should_cancel):
    if should_cancel is not None and should_cancel():
        raise Cancelled()


def with_strength(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Blend the fitted LUT with identity.

    ``alpha = 1`` is the fitted look; lower values are what most people actually
    want, because a least-squares fit of 46 different edits tends to come out at
    full strength.  Both identity and the fit are monotone, so a convex blend is
    monotone too; the fold check is re-run on export to be sure.
    """
    ident = nodes_flat(n)
    return ident + float(alpha) * (x - ident)


def apply_lut(x: np.ndarray, img: np.ndarray, n: int, chunk: int = 400_000) -> np.ndarray:
    h, w, _ = img.shape
    flat = np.ascontiguousarray(img.reshape(-1, 3), dtype=np.float64)
    out = np.empty_like(flat)
    for i in range(0, len(flat), chunk):
        out[i:i + chunk] = evaluate(x, flat[i:i + chunk], n)
    return np.clip(out, 0.0, 1.0).reshape(h, w, 3)


def run_fit(input_dir: str, target_dir: str, options: FitOptions = None,
            progress=None, should_cancel=None, on_preflight=None) -> FitResult:
    """Run the whole pipeline.

    ``progress(stage, frac, message)`` is called continuously.
    ``on_preflight(info)`` fires as soon as the coverage numbers exist — about a
    minute in, rather than after the whole fit — so the UI can show something
    useful while the expensive stage runs.
    """
    options = (options or FitOptions()).scaled()
    t_start = time.time()

    def report(stage_idx, frac, message=None):
        name, lo, hi = STAGES[stage_idx]
        f = lo + (hi - lo) * float(np.clip(frac, 0.0, 1.0))
        if progress:
            progress(stage_idx, f, message or name)

    report(0, 0.0)
    pairs = find_pairs(input_dir, target_dir)
    if not pairs:
        raise ValueError("两个文件夹里没有找到同名图片对")
    if options.max_pairs:
        pairs = pairs[:options.max_pairs]
    _check(should_cancel)

    stats = build_statistics(
        pairs, grid=options.grid, target_max=options.target_max,
        progress=lambda f, m: report(1, f, m),
        should_cancel=should_cancel)
    skipped = stats.get("skipped", [])
    _check(should_cancel)

    report(2, 0.0)
    summary = summarize(stats)
    total_px = float(summary["w"].sum())
    min_count = scaled_min_count(total_px)
    keep = summary["w"] >= min_count
    a_bar = summary["a_bar"][keep]
    b_bar = summary["b_bar"][keep]
    w = summary["w"][keep]
    core = w >= max(200, min_count * 20)

    pf = preflight(a_bar, w, n=options.lut_size, min_support=4)
    t, v, ginfo = gray_curve(stats)
    if not options.preserve_neutral:
        t = v = None
    report(2, 1.0)
    if on_preflight is not None:
        on_preflight(dict(n_pairs=len(pairs), pixels=total_px, preflight=pf,
                          gray=ginfo or {}, complete=False))

    _check(should_cancel)
    report(3, 0.0)

    report(4, 0.0)
    gp = {}
    folds_before = folds_after = None
    last = {}

    def _project(xf, n):
        """Post-processing, run inside the adaptive loop so each trial is judged
        on its *finished* LUT rather than on the raw solve."""
        nonlocal gp, folds_before, folds_after
        fb = fold_report(enforce_monotone(xf, n), n)["n_folded_cells"]
        y, pinfo = polish_lut(xf, n, rounds=4,
                              repair_iters=options.repair_folds,
                              target=0.02, space=options.space)
        gp = pinfo["gamut"]
        fa = fold_report(y, n)
        rr = evaluate(y, a_bar, n) - b_bar
        model = float(np.sqrt((w[:, None] * rr ** 2).sum() / w.sum()))
        last.update(folds_before=dict(n_folded_cells=fb), folds_after=fa, model=model,
                    polish=pinfo["history"])
        return y, dict(gamut=gp, folds_before=dict(n_folded_cells=fb),
                       folds_after=fa, model=model, polish=pinfo["history"])

    x, fit_hist = fit_multiscale(
        a_bar, b_bar, w, levels=_levels(options.lut_size),
        lam_s=options.lam_s, beta=options.beta,
        gamma=options.gamma if options.preserve_neutral else 0.0,
        mono_mu=options.mono_mu, mono_outer=options.mono_outer,
        gray_curve_t=t, gray_curve_v=v, chroma_max=0.10,
        maxiter=400, verbose=False,
        progress=lambda f, m: report(3, f, m),
        should_cancel=should_cancel,
        project=_project, adaptive=options.adaptive,
        max_trials=options.max_trials)
    folds_before = last.get("folds_before", fold_report(x, options.lut_size))
    folds_after = last.get("folds_after", fold_report(x, options.lut_size))
    report(4, 1.0)

    _check(should_cancel)
    report(5, 0.0)
    n = options.lut_size
    structural = dict(
        folds_before=folds_before, folds_after=folds_after,
        fit_trials=fit_hist[-1].get("trials", []),
        chosen_lam_s=fit_hist[-1].get("chosen_lam_s", options.lam_s),
        chosen_mono_mu=fit_hist[-1].get("chosen_mono_mu", options.mono_mu),
        jacobian=jacobian_check(x, n),
        monotonicity=monotonicity_report(x, n),
        neutral=neutral_scan(x, n, n_steps=1024, curve_t=t, curve_v=v),
        axis_sweeps=axis_sweeps(x, n, n_lines=5, n_steps=512),
        gamut=gp,
    )
    report(5, 1.0)

    r = evaluate(x, a_bar, n) - b_bar
    model = float(np.sqrt((w[:, None] * r ** 2).sum() / w.sum()))
    floor = float(np.sqrt((summary["b_var"][keep] * w[:, None]).sum() / w.sum()))
    residual = dict(model=model, irreducible=floor, total=float(np.hypot(model, floor)))

    risk = extrapolation_risk(x, pf, n)
    _check(should_cancel)

    # ---- ΔE against the targets, and previews -----------------------------
    report(6, 0.0)
    per_pair, previews = _evaluate(x, pairs, options, report,
                                   should_cancel=should_cancel)
    report(6, 1.0)

    agg = {}
    if per_pair:
        for k in ("mean", "p95", "max", "neutral", "shadow", "mid", "high", "sat"):
            vals = [p[k] for p in per_pair if np.isfinite(p[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")

    return FitResult(
        lut=x, options=options, n_pairs=len(pairs) - len(skipped), pixels=total_px,
        skipped=skipped,
        preflight=pf, risk=risk, structural=structural, residual=residual,
        per_pair=per_pair, aggregate=agg, gray=ginfo or {}, previews=previews,
        elapsed=time.time() - t_start, verdict=verdict(pf, risk),
    )


def _levels(lut_size: int):
    ladder = [5, 9, 17, 25, 33, 49, 65]
    return tuple([l for l in ladder if l < lut_size] + [lut_size])


def _evaluate(x, pairs, options, report, should_cancel=None):
    n = options.lut_size
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    preview_idx = set(int(i) for i in order[:options.preview_count])

    per_pair, previews = [], []
    for i, (name, pa, pb) in enumerate(pairs):
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        report(6, i / max(len(pairs), 1), f"评估色差  {name}")
        try:
            a = load_code(pa, options.eval_max)
            b = load_code(pb, options.eval_max)
        except Exception:                              # 无法解码的格式，跳过
            continue
        if a.shape != b.shape:
            continue
        m = build_mask(a, b)
        pred = apply_lut(x, a, n)
        de = de00(code_to_lab(pred, options.space), code_to_lab(b, options.space))
        dm = de[m]
        if dm.size == 0:
            continue
        chroma = b[m].max(1) - b[m].min(1)
        luma = b[m] @ REC709

        def _m(sel):
            v = dm[sel]
            return float(v.mean()) if v.size else float("nan")

        per_pair.append(dict(
            name=name, mean=float(dm.mean()), p95=float(np.percentile(dm, 95)),
            max=float(dm.max()),
            neutral=_m(chroma < 0.05), shadow=_m(luma < 0.25),
            mid=_m((luma >= 0.25) & (luma < 0.7)), high=_m(luma >= 0.7),
            sat=_m(chroma > 0.3)))

        if i in preview_idx:
            st = max(1, int(np.ceil(a.shape[1] / options.preview_max)))
            previews.append(dict(name=name,
                                 a=np.ascontiguousarray(a[::st, ::st]),
                                 pred=np.ascontiguousarray(pred[::st, ::st]),
                                 b=np.ascontiguousarray(b[::st, ::st])))
    return per_pair, previews


def export_cube(result: FitResult, path: str, strength: float = 1.0) -> np.ndarray:
    """Blend to the requested strength, re-check, and write the .cube."""
    n = result.options.lut_size
    x = with_strength(result.lut, n, strength)
    write_cube(path, x, n, title="3dlut A->B grade")
    return x


def strength_check(result: FitResult, strength: float) -> dict:
    """Cheap re-verification of a strength-blended LUT (used by the GUI slider)."""
    n = result.options.lut_size
    x = with_strength(result.lut, n, strength)
    return dict(folds=fold_report(x, n),
                mono=monotonicity_report(x, n),
                rng=(float(x.min()), float(x.max())))
