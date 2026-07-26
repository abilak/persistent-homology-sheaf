"""Summarize hp_check results: mean val accuracy per variant over its folds, so
you can pick the best topo config (and the fair baseline) before the big runs."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results", "NCI1")

by = defaultdict(list)
for fp in glob.glob(os.path.join(RESULTS, "*.json")):
    if fp.endswith(".log"):
        continue
    name = os.path.basename(fp)[:-5]
    # only hp_check files carry a trailing variant tag after the fold
    parts = name.split("_")
    if len(parts) < 4:
        continue
    model, variant = parts[0], "_".join(parts[3:])
    if not variant:
        continue
    try:
        r = json.load(open(fp))
        if "best_val_acc" in r:
            by[(model, variant)].append((r["best_val_acc"], r.get("sec_per_train_epoch")))
    except Exception:
        pass

if not by:
    print("No hp_check results yet in", RESULTS)
    sys.exit()

print(f"{'model':<9}{'variant':<10}{'mean val%':<12}{'per-fold':<26}{'s/ep':<7}{'n'}")
print("-" * 72)
rows = []
for (model, variant), vals in sorted(by.items()):
    accs = np.array([v[0] for v in vals]) * 100
    spe = np.median([v[1] for v in vals if v[1] is not None]) if vals else 0
    rows.append((model, variant, accs.mean(), accs, spe, len(accs)))
    pf = ",".join(f"{a:.1f}" for a in accs)
    print(f"{model:<9}{variant:<10}{accs.mean():<12.2f}{pf:<26}{spe:<7.1f}{len(accs)}")

topo = [r for r in rows if r[0] == "topo"]
base = [r for r in rows if r[0] == "baseline"]
if topo:
    best = max(topo, key=lambda r: r[2])
    print(f"\nBEST topo config: {best[1]}  (mean val {best[2]:.2f}%, {best[4]:.1f}s/ep)")
if base:
    bb = max(base, key=lambda r: r[2])
    print(f"BEST baseline:    {bb[1]}  (mean val {bb[2]:.2f}%)")
    if topo:
        print(f"topo - baseline (best each): {best[2]-bb[2]:+.2f}%")
print("\nTo run the big datasets with the winning topo config, edit the "
      "`big_heavy` plan's topo jobs to add its overrides, e.g. "
      "overrides={'topo_gate_bias':-2.0,'topo_node_level':False}.")
