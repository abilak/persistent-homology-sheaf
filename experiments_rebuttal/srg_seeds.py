"""Robustness of the SRG separation to random initialization (addresses the
reviewer's 'SRG uses random init' concern): report min/median/max separation
over many seeds for baseline vs topo. Separation must stay far above noise for
EVERY seed."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from srg_graphs import all_pairs
from srg_analysis import model_separation

N = 30
THR = 1e-6  # 1000x the ~1e-9 baseline noise floor
rows = []
print(f"SRG separation robustness over {N} random seeds (threshold {THR:.0e})")
print(f"{'pair':<26}{'base max':<11}{'topo med':<11}{'topo min':<11}"
      f"{'topo max':<11}{'frac>thr':<9}")
for name, (G1, G2, params) in all_pairs().items():
    _, b_max, b_all = model_separation(G1, G2, use_topology=False, n_seeds=N)
    t_med, t_max, t_all = model_separation(G1, G2, use_topology=True, n_seeds=N)
    t_all = np.array(t_all)
    frac = float((t_all > THR).mean())
    rows.append(dict(pair=name, baseline_max=b_max, topo_median=t_med,
                     topo_min=float(t_all.min()), topo_max=t_max,
                     frac_above_thr=frac, topo_all=t_all.tolist()))
    print(f"{name:<26}{b_max:<11.1e}{t_med:<11.1e}{t_all.min():<11.1e}"
          f"{t_max:<11.1e}{frac*100:>5.0f}%")
json.dump(rows, open('experiments_rebuttal/srg_seeds_results.json', 'w'),
          indent=2, default=str)
print(f"\nmean frac of seeds separating (>{THR:.0e}): "
      f"{np.mean([r['frac_above_thr'] for r in rows])*100:.0f}%")
