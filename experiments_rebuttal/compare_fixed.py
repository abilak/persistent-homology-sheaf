"""Score fixed-split runs (ZINC: test MAE, lower is better; MOLHIV: test ROC-AUC,
higher is better) at the best-validation epoch, per model variant, mean +/- std
over seeds, and the seed-paired difference to the baseline.

usage: python experiments_rebuttal/compare_fixed.py ZINC|MOLHIV [baseline_variant]   (default: fast)"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import boot_ci, perm_p, stable_seed
from paper_tables import exact_wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = sys.argv[1] if len(sys.argv) > 1 else "ZINC"
base_variant = sys.argv[2] if len(sys.argv) > 2 else "fast"
task, key, name, lower = {"ZINC": ("regression", "test_mae_at_best_val", "test MAE", True),
                          "MOLHIV": ("auc", "test_auc_at_best_val", "test ROC-AUC", False)}[dataset]
pools = defaultdict(dict)
for fp in glob.glob(os.path.join(ROOT, "rebuttal_results", dataset, "*.json")):
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    if r.get("task") != task:
        continue
    parts = os.path.basename(fp)[:-5].split("_")
    variant = "_".join(parts[3:])
    pools[(r["model"], variant)][r["seed"]] = r[key]

if ("baseline", base_variant) not in pools:
    sys.exit(f"no {dataset} baseline runs with variant '{base_variant}' yet")
b = pools[("baseline", base_variant)]
bv = list(b.values())
print(f"{dataset} {name} at best-val epoch ({'lower' if lower else 'higher'} is better); "
      f"baseline/{base_variant}: {np.mean(bv):.4f} +/- {np.std(bv, ddof=1) if len(bv) > 1 else 0:.4f} "
      f"(n={len(b)} seeds)\n")
print(f"{'variant':<18}{name:<20}{'paired vs baseline (seeds)':<34}{'95% CI':<20}{'perm p':<9}{'Wilcoxon p'}")
print("-" * 112)
for (m, v), runs in sorted(pools.items()):
    if m == "baseline":
        continue
    if m != "topo":
        v = f"{m}/{v}"
    seeds = sorted(set(runs) & set(b))
    vals = np.array(list(runs.values()))
    d = np.array([runs[s] - b[s] for s in seeds])
    better = int((d < 0).sum() if lower else (d > 0).sum())
    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
    if len(d):
        paired = f"{d.mean():+.4f} (n={len(d)}, {better}/{len(d)} better)"
        lo, hi = boot_ci(d, seed=stable_seed(dataset, v))
        stats = f"[{lo:+.4f},{hi:+.4f}]   {perm_p(d):<9.4f}{exact_wilcoxon(d):.4f}"
    else:
        paired, stats = "no matching seeds yet", ""
    print(f"{v:<18}{vals.mean():.4f} +/- {sd:<10.4f}{paired:<34}{stats}")
