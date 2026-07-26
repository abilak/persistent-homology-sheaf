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

N = 20
rows = []
print(f"SRG separation robustness over {N} random seeds")
print(f"{'pair':<26}{'baseline max':<16}{'topo MIN':<14}{'topo median':<14}{'always sep?':<10}")
for name, (G1, G2, params) in all_pairs().items():
    _, b_max, b_all = model_separation(G1, G2, use_topology=False, n_seeds=N)
    t_med, t_max, t_all = model_separation(G1, G2, use_topology=True, n_seeds=N)
    t_min = float(np.min(t_all))
    always = all(x > 1e-5 for x in t_all)
    rows.append(dict(pair=name, baseline_max=b_max, topo_min=t_min,
                     topo_median=t_med, always_separated=always))
    print(f"{name:<26}{b_max:<16.2e}{t_min:<14.2e}{t_med:<14.2e}{str(always):<10}")
json.dump(rows, open('experiments_rebuttal/srg_seeds_results.json', 'w'),
          indent=2, default=str)
print("\nEvery pair separated for EVERY seed:",
      all(r['always_separated'] for r in rows))
