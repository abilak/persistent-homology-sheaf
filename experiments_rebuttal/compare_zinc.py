"""Score ZINC runs: test MAE at the best-validation epoch, per model variant,
mean +/- std over seeds, and the seed-paired difference to the baseline.

usage: python experiments_rebuttal/compare_zinc.py [baseline_variant]   (default: fast)"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_variant = sys.argv[1] if len(sys.argv) > 1 else "fast"
pools = defaultdict(dict)
for fp in glob.glob(os.path.join(ROOT, "rebuttal_results", "ZINC", "*.json")):
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    if r.get("task") != "regression":
        continue
    parts = os.path.basename(fp)[:-5].split("_")
    variant = "_".join(parts[3:])
    pools[(r["model"], variant)][r["seed"]] = r["test_mae_at_best_val"]

if ("baseline", base_variant) not in pools:
    sys.exit(f"no ZINC baseline runs with variant '{base_variant}' yet")
b = pools[("baseline", base_variant)]
print(f"ZINC test MAE at best-val epoch (lower is better); baseline/{base_variant}: "
      f"{np.mean(list(b.values())):.4f} +/- {np.std(list(b.values()), ddof=1) if len(b) > 1 else 0:.4f} "
      f"(n={len(b)} seeds)\n")
print(f"{'variant':<18}{'test MAE':<20}{'paired vs baseline (seeds)':<30}")
print("-" * 68)
for (m, v), runs in sorted(pools.items()):
    if m != "topo":
        continue
    seeds = sorted(set(runs) & set(b))
    vals = np.array([runs[s] for s in runs.keys()])
    d = np.array([runs[s] - b[s] for s in seeds])
    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
    paired = (f"{d.mean():+.4f} (n={len(d)}, {int((d < 0).sum())}/{len(d)} seeds better)"
              if len(d) else "no matching seeds yet")
    print(f"{v:<18}{vals.mean():.4f} +/- {sd:<10.4f}{paired}")
