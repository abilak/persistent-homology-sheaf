"""Same-code training-time comparison across models (Q3).
Reads `sec_per_train_epoch` from the main run JSONs and reports median +/- IQR
per (dataset, model).
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")

by = defaultdict(list)
for fp in glob.glob(os.path.join(RESULTS, "*", "*.json")):
    base = os.path.basename(fp)[:-5]
    parts = base.split("_")
    if len(parts) != 3 or not parts[1].startswith("s") or not parts[2].startswith("f"):
        continue
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    if r.get("overrides") or "sec_per_train_epoch" not in r:
        continue
    by[(r["dataset"], r["model"])].append(r["sec_per_train_epoch"])

MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']
datasets = sorted({d for d, _ in by})

print(f"{'':<10}" + "".join(f"{m:<12}" for m in MODELS))
print("-" * 80)
for ds in datasets:
    row = f"{ds:<10}"
    for m in MODELS:
        vals = by.get((ds, m))
        if vals:
            v = np.median(vals)
            row += f"{v:<12.3f}"
        else:
            row += f"{'-':<12}"
    print(row)

# ratio table vs the lightest MP model (GCN)
print("\nratio vs GCN (per-graph training cost multiplier):")
print(f"{'':<10}" + "".join(f"{m:<12}" for m in MODELS))
print("-" * 80)
for ds in datasets:
    gcn = by.get((ds, 'gcn'))
    if not gcn:
        continue
    gcn_v = np.median(gcn)
    row = f"{ds:<10}"
    for m in MODELS:
        vals = by.get((ds, m))
        if vals:
            v = np.median(vals) / gcn_v
            row += f"{v:<12.1f}"
        else:
            row += f"{'-':<12}"
    print(row)
