"""
Independently verify two claims before acting on them:
 (1) our headline metric (per-fold max over epochs) differs materially from the
     standard Xu/GIN protocol (average the fold curves, pick ONE epoch by the
     mean curve, report the fold accuracies at that epoch);
 (2) paired significance changes when we respect the fold blocking (10 folds
     re-used across 5 seeds -> effective n is 10, not 50).
Prints both metrics side by side so we can see whether conclusions flip.
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")


def load(dataset):
    """Return {model: {(seed,fold): curve}} for MAIN runs only (no variants)."""
    out = defaultdict(dict)
    for fp in glob.glob(os.path.join(RESULTS, dataset, "*.json")):
        name = os.path.basename(fp)[:-5]
        parts = name.split("_")
        # main runs are exactly model_s{seed}_f{fold}
        if len(parts) != 3 or not parts[1].startswith('s') or not parts[2].startswith('f'):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        if 'val_curve' not in r or 'dataset' not in r:
            continue
        out[r['model']][(r['seed'], r['fold'])] = np.asarray(r['val_curve'])
    return out


def xu_protocol(curves_by_sf, seeds, folds):
    """Per seed: pick epoch maximizing the mean curve over folds; return the
    per-fold accuracies at that epoch, averaged over seeds -> array over folds."""
    per_fold = defaultdict(list)
    for s in seeds:
        mats = [curves_by_sf[(s, f)] for f in folds if (s, f) in curves_by_sf]
        if len(mats) != len(folds):
            return None
        L = min(len(m) for m in mats)
        M = np.stack([m[:L] for m in mats])          # (folds, epochs)
        ep = int(np.argmax(M.mean(axis=0)))
        for i, f in enumerate(folds):
            per_fold[f].append(M[i, ep])
    return np.array([np.mean(per_fold[f]) for f in folds])


def maxep_protocol(curves_by_sf, seeds, folds):
    per_fold = defaultdict(list)
    for s in seeds:
        for f in folds:
            if (s, f) in curves_by_sf:
                per_fold[f].append(curves_by_sf[(s, f)].max())
    return np.array([np.mean(per_fold[f]) for f in folds])


def perm_p(d, nperm=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    obs = abs(d.mean())
    sg = rng.choice([-1.0, 1.0], size=(nperm, len(d)))
    return float((np.sum(np.abs((sg * d).mean(1)) >= obs - 1e-12) + 1) / (nperm + 1))


def boot_ci(x, nb=20000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (nb, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


for ds in ("MUTAG", "PTC"):
    data = load(ds)
    if not data:
        continue
    seeds = sorted({s for m in data.values() for (s, f) in m})
    folds = sorted({f for m in data.values() for (s, f) in m})
    print("=" * 78)
    print(f"{ds}   seeds={seeds} folds={folds}")
    print("=" * 78)
    print(f"{'model':<10}{'max-over-epoch':<18}{'Xu protocol':<18}{'inflation'}")
    xu, mx = {}, {}
    for model in ('mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo'):
        if model not in data:
            continue
        a = maxep_protocol(data[model], seeds, folds)
        b = xu_protocol(data[model], seeds, folds)
        if b is None:
            continue
        mx[model], xu[model] = a, b
        print(f"{model:<10}{a.mean()*100:6.2f} +/- {a.std()*100:<8.2f}"
              f"{b.mean()*100:6.2f} +/- {b.std()*100:<8.2f}"
              f"{(a.mean()-b.mean())*100:+.2f}")

    # paired topo - baseline, fold-blocked (n = #folds), under BOTH metrics
    print(f"\n{'paired topo-baseline':<24}{'mean':<10}{'95% CI':<22}{'perm p'}")
    for tag, dd in (("max-over-epoch", mx), ("Xu protocol", xu)):
        if 'topo' in dd and 'baseline' in dd:
            d = (dd['topo'] - dd['baseline']) * 100
            lo, hi = boot_ci(d)
            print(f"  {tag:<22}{d.mean():+6.2f}    "
                  f"[{lo:+6.2f},{hi:+6.2f}]      {perm_p(d):.4f}   (n={len(d)} folds)")
    print()
