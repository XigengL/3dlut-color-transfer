import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lutfit.loader import find_pairs
from lutfit.accumulate import build_statistics, summarize

pairs = find_pairs('input', 'target')
print(f"{len(pairs)} pairs")
t0 = time.time()
stats = build_statistics(pairs, grid=65, target_max=1800)
print(f"accumulation done in {time.time()-t0:.1f}s")

s = summarize(stats)
print(f"\noccupied bins      : {s['n_occupied']:,} / {s['total_nodes']:,}  ({s['coverage']*100:.1f}%)")
print(f"bins with >=50 px  : {s['coverage_strong']*100:.1f}% of grid, "
      f"{s['mass_in_strong']*100:.2f}% of pixel mass")
print(f"median bin count   : {s['median_count']:.0f}")
print(f"irreducible floor  : R={s['irreducible_rms'][0]:.5f} "
      f"G={s['irreducible_rms'][1]:.5f} B={s['irreducible_rms'][2]:.5f} (code units)")

with open('work/stats.pkl','wb') as f:
    pickle.dump({'stats': stats, 'summary': s}, f)
print("saved work/stats.pkl")
