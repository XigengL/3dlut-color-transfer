"""在两个数据集上验证自适应正则化：数据集1 应零额外成本，数据集2 应触发重试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lutfit.pipeline import FitOptions, run_fit, export_cube

IND, TGD, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
opts = FitOptions()
res = run_fit(IND, TGD, opts, progress=lambda s,f,m: None)
st, rs, agg = res.structural, res.residual, res.aggregate
print(f"\n===== {TAG} =====")
print(f"图对 {res.n_pairs}  像素 {res.pixels/1e6:.1f}M  耗时 {res.elapsed/60:.1f} 分钟")
print("自适应尝试历史:")
for tr in st.get('fit_trials', []):
    print(f"   trial {tr['trial']}: lam_s={tr['lam_s']:<6g} folds={tr['folds']:<5} "
          f"min_det={tr['min_det']:+.4f} model={tr['model']:.5f} max2nd={tr['max_second']:.5f}")
print(f"采用 lam_s = {st.get('chosen_lam_s')}   尝试次数 = {len(st.get('fit_trials', []))}")
print(f"模型误差 {rs['model']:.5f}  下界 {rs['irreducible']:.5f}  总 {rs['total']:.5f} "
      f"(+{(rs['total']/rs['irreducible']-1)*100:.1f}%)")
print(f"折叠 {st['folds_before']['n_folded_cells']} -> {st['folds_after']['n_folded_cells']}")
if agg: print(f"ΔE00 平均 {agg['mean']:.2f}  95分位 {agg['p95']:.2f}")
export_cube(res, f'output/lut33_{TAG}.cube', 1.0)
np.save(f'output/lut33_{TAG}.npy', res.lut)
