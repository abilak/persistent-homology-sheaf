"""
Paired statistics under the paper's own metric (max val acc per fold) —
the protocol PPGN uses, matching the paper's Table 1. Reports:
  * per-model mean +/- std across folds (averaged over seeds)
  * paired topo - baseline on matched (seed, fold), fold-blocked
    (n = #folds), with 95% bootstrap CI + sign-flip permutation p-value
    + midrank+tie-corrected Wilcoxon signed-rank p-value
  * same for every same-code baseline vs the PPGN baseline (so the
    reviewer sees GCN/GIN/MLP/GSN paired against baseline too)
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import boot_ci, perm_p, wilcoxon_p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")


def load_main(dataset):
    """Return {model: {(seed,fold): best_val_acc}} for main (non-swept) runs."""
    out = defaultdict(dict)
    for fp in glob.glob(os.path.join(RESULTS, dataset, "*.json")):
        base = os.path.basename(fp)[:-5]
        parts = base.split("_")
        if len(parts) != 3 or not parts[1].startswith("s") or not parts[2].startswith("f"):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        if r.get("overrides") or "best_val_acc" not in r:
            continue
        out[r["model"]][(r["seed"], r["fold"])] = r["best_val_acc"]
    return out


def per_fold(runs):
    """seed-averaged per-fold accuracies (array of length = #folds)."""
    seeds = sorted({s for s, _ in runs})
    folds = sorted({f for _, f in runs})
    return np.array([np.mean([runs[(s, f)] for s in seeds if (s, f) in runs])
                     for f in folds]), seeds, folds


for ds in ("MUTAG", "PTC"):
    data = load_main(ds)
    if not data:
        continue
    print("=" * 78)
    print(f"{ds}  (paper metric = max val acc per fold)")
    print("=" * 78)
    print(f"{'model':<12}{'mean +/- std':<18}{'n':<6}{'seeds':<12}{'folds'}")
    means = {}
    for m in ('mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo'):
        if m not in data:
            continue
        pf, seeds, folds = per_fold(data[m])
        means[m] = pf
        acc = pf * 100
        print(f"{m:<12}{acc.mean():.2f} +/- {acc.std(ddof=1):.2f}   "
              f"{len(data[m]):<6}{len(seeds):<12}{len(folds)}")

    if 'baseline' not in means:
        continue
    base = means['baseline']

    print(f"\nPAIRED vs baseline  (fold-blocked; n = {len(base)} folds)")
    print("-" * 78)
    print(f"{'contrast':<20}{'mean diff':<14}{'95% CI':<24}{'perm p':<10}{'Wilcoxon p'}")
    for m in ('topo', 'gcn', 'gin', 'mlp', 'gsn'):
        if m not in means:
            continue
        d = (means[m] - base) * 100
        lo, hi = boot_ci(d, seed=abs(hash((ds, m))) % 2**31)
        print(f"{m + ' - baseline':<20}{d.mean():+7.2f}       "
              f"[{lo:+.2f}, {hi:+.2f}]     "
              f"{perm_p(d):<10.4f}{wilcoxon_p(d):.4f}")
    print()
