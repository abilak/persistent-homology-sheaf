"""
Aggregate rebuttal_results/*/*.json into:
  * per-(dataset,model) mean best-val-acc, std, 95% bootstrap CI over all
    (seed,fold) runs, and mean-of-per-seed-fold-means.
  * PAIRED comparison topo vs baseline on matched (seed,fold): mean paired
    difference, 95% bootstrap CI, sign-flip permutation p-value, Wilcoxon
    signed-rank p (normal approx). This directly addresses "gains comparable
    to fold std": the paired test removes fold-to-fold variance.
All statistics implemented in numpy (no scipy available).
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")
RNG = np.random.default_rng(12345)


def load():
    rows = []
    for fp in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        if fp.endswith(".log"):
            continue
        try:
            with open(fp) as f:
                rows.append(json.load(f))
        except Exception:
            pass
    return rows


def bootstrap_ci(x, nb=10000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float('nan'),) * 2
    idx = RNG.integers(0, len(x), size=(nb, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def perm_pvalue(d, nperm=20000):
    """Two-sided sign-flip permutation test on paired diffs d (H0: mean 0)."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return float('nan')
    obs = abs(d.mean())
    signs = RNG.choice([-1, 1], size=(nperm, len(d)))
    stats = np.abs((signs * d).mean(1))
    return float((np.sum(stats >= obs - 1e-12) + 1) / (nperm + 1))


def wilcoxon_p(d):
    """Wilcoxon signed-rank test, normal approximation (two-sided)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return float('nan')
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    W = np.sum(r[d > 0])
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sigma
    # two-sided normal tail
    return float(2 * 0.5 * (1 - _erf(abs(z) / np.sqrt(2))))


def _erf(x):
    # Abramowitz-Stegun 7.1.26
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return y


def main():
    rows = load()
    by = {}
    for r in rows:
        by.setdefault((r['dataset'], r['model']), []).append(r)

    datasets = sorted({d for d, _ in by})
    models = ['mlp', 'gcn', 'gin', 'baseline', 'topo']

    print("=" * 100)
    print("PER-MODEL VALIDATION ACCURACY (mean +/- std over all seed x fold runs)"
          "  [n runs]")
    print("=" * 100)
    summary = {}
    for ds in datasets:
        print(f"\n{ds}:")
        for m in models:
            rs = by.get((ds, m))
            if not rs:
                continue
            accs = np.array([x['best_val_acc'] for x in rs]) * 100
            lo, hi = bootstrap_ci(accs)
            summary[(ds, m)] = dict(mean=accs.mean(), std=accs.std(), n=len(accs),
                                    ci=(lo * 1, hi * 1))
            print(f"  {m:<9} {accs.mean():6.2f} +/- {accs.std():4.2f}   "
                  f"95%CI[{lo:.2f},{hi:.2f}]   n={len(accs)}")

    print("\n" + "=" * 100)
    print("PAIRED topo - baseline  (matched seed x fold; removes fold variance)")
    print("=" * 100)
    for ds in datasets:
        base = {(x['seed'], x['fold']): x['best_val_acc']
                for x in by.get((ds, 'baseline'), [])}
        topo = {(x['seed'], x['fold']): x['best_val_acc']
                for x in by.get((ds, 'topo'), [])}
        keys = sorted(set(base) & set(topo))
        if len(keys) < 5:
            print(f"\n{ds}: insufficient paired runs ({len(keys)})")
            continue
        d = np.array([topo[k] - base[k] for k in keys]) * 100
        lo, hi = bootstrap_ci(d)
        p_perm = perm_pvalue(d)
        p_wil = wilcoxon_p(d)
        wins = int(np.sum(d > 0))
        print(f"\n{ds}: n_pairs={len(keys)}  mean diff={d.mean():+.2f}%  "
              f"95%CI[{lo:+.2f},{hi:+.2f}]  topo wins {wins}/{len(keys)}")
        print(f"   permutation p={p_perm:.4g}   Wilcoxon p={p_wil:.4g}   "
              f"{'SIGNIFICANT' if p_perm < 0.05 else 'n.s.'} at 0.05")

    print("\n" + "=" * 100)
    print("TRAINING COST: mean seconds / train epoch (NOTE: taken during parallel"
          " runs -> use runtime_bench.py for clean single-process numbers)")
    print("=" * 100)
    for ds in datasets:
        line = f"{ds:<10}"
        for m in models:
            rs = by.get((ds, m))
            if rs:
                spe = np.median([x['sec_per_train_epoch'] for x in rs])
                line += f" {m}={spe:.2f}s"
        print(line)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump({f"{k[0]}/{k[1]}": v for k, v in summary.items()}, f,
                  indent=2, default=str)


if __name__ == "__main__":
    main()
