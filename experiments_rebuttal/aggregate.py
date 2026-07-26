"""
Aggregate rebuttal_results/*/*.json under the STANDARD Xu et al. / GIN TU
protocol, which is what the published baselines (GIN, PPGN, GSN, CIN, SIN) use:

    for each seed:
        average the per-fold accuracy curves over the 10 folds,
        pick the SINGLE epoch e* maximizing that averaged curve,
        take each fold's accuracy at e*.
    report mean +/- std across folds (averaged over seeds).

This differs from "best epoch per fold" (max over the epoch axis independently
in each fold), which is upward-biased by an amount that grows with curve noise
and therefore favours noisier models. We report that quantity too, clearly
labelled, so the difference is visible rather than hidden.

Significance: the same 10 fold splits are reused for every seed, so the 50
seed x fold runs are NOT independent. All tests are therefore FOLD-BLOCKED --
paired differences are averaged over seeds within a fold, giving n = #folds
independent units -- via a sign-flip permutation test and a percentile
bootstrap. Wilcoxon uses midranks with a tie correction (accuracies on a
~19-graph fold are multiples of 1/19, so ties are pervasive).
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")
MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']


# ------------------------------- loading ---------------------------------- #
def load(dataset=None):
    """Load MAIN runs only. Hyperparameter-sweep variants (extra filename
    suffix) and short-epoch smoke runs are excluded so they cannot pollute the
    headline numbers. Returns {(dataset, model): {(seed,fold): record}}."""
    out = defaultdict(dict)
    bad = 0
    pat = os.path.join(RESULTS, dataset or "*", "*.json")
    for fp in glob.glob(pat):
        base = os.path.basename(fp)[:-5]
        parts = base.split("_")
        # main run filenames are exactly  <model>_s<seed>_f<fold>
        if (len(parts) != 3 or not parts[1].startswith("s")
                or not parts[2].startswith("f")):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            bad += 1
            print(f"  [warn] unreadable/truncated result: {fp}")
            continue
        if not all(k in r for k in ("dataset", "model", "val_curve")):
            bad += 1
            print(f"  [warn] incomplete result: {fp}")
            continue
        if r.get("overrides"):
            continue                                    # swept config
        key = (r["dataset"], r["model"])
        sf = (r["seed"], r["fold"])
        if sf in out[key]:
            print(f"  [warn] duplicate run {key} {sf}: {fp}")
        out[key][sf] = r
    if bad:
        print(f"  [warn] skipped {bad} unusable result file(s)\n")
    return out


# ------------------------------- protocols -------------------------------- #
def per_fold_scores(runs, xu=True):
    """runs: {(seed,fold): record} -> array of per-fold accuracy (mean over
    seeds). xu=True uses the standard shared-epoch protocol."""
    seeds = sorted({s for s, _ in runs})
    folds = sorted({f for _, f in runs})
    per_fold = defaultdict(list)
    for s in seeds:
        curves = {f: np.asarray(runs[(s, f)]["val_curve"])
                  for f in folds if (s, f) in runs}
        if not curves:
            continue
        if xu:
            if len(curves) != len(folds):
                continue                    # need all folds to pick a shared epoch
            L = min(len(c) for c in curves.values())
            M = np.stack([curves[f][:L] for f in folds])
            e = int(np.argmax(M.mean(axis=0)))
            for i, f in enumerate(folds):
                per_fold[f].append(M[i, e])
        else:
            for f, c in curves.items():
                per_fold[f].append(c.max())
    if not per_fold:
        return None, 0
    n_runs = sum(len(v) for v in per_fold.values())
    return np.array([np.mean(per_fold[f]) for f in sorted(per_fold)]), n_runs


# ------------------------------ statistics -------------------------------- #
def boot_ci(x, nb=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)          # per-call RNG => reproducible
    x = np.asarray(x, float)
    if len(x) < 2:
        return float('nan'), float('nan')
    idx = rng.integers(0, len(x), (nb, len(x)))
    m = x[idx].mean(1)
    return (float(np.percentile(m, 100 * alpha / 2)),
            float(np.percentile(m, 100 * (1 - alpha / 2))))


def perm_p(d, nperm=20000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) == 0:
        return float('nan')
    obs = abs(d.mean())
    sg = rng.choice([-1.0, 1.0], size=(nperm, len(d)))
    return float((np.sum(np.abs((sg * d).mean(1)) >= obs - 1e-12) + 1) / (nperm + 1))


def _midranks(a):
    """Average ranks for ties (scipy.stats.rankdata equivalent)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank, normal approx with midranks, tie
    correction and continuity correction."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return float('nan')
    r = _midranks(np.abs(d))
    W = float(np.sum(r[d > 0]))
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = np.sum(counts ** 3 - counts) / 48.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term)
    if sigma == 0:
        return float('nan')
    z = (abs(W - mu) - 0.5) / sigma
    return float(2 * 0.5 * (1 - _erf(abs(z) / np.sqrt(2))))


def _erf(x):
    t = 1 / (1 + 0.3275911 * x)
    return 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)


# --------------------------------- main ----------------------------------- #
def main():
    by = load()
    datasets = sorted({d for d, _ in by})
    summary = {}

    print("=" * 96)
    print("ACCURACY under the STANDARD Xu/GIN protocol (shared best epoch per "
          "seed)\nmean +/- std ACROSS FOLDS.  [brackets] = the upward-biased "
          "'best epoch per fold' number.")
    print("=" * 96)
    for ds in datasets:
        print(f"\n{ds}:")
        for m in MODELS:
            runs = by.get((ds, m))
            if not runs:
                continue
            xu, n = per_fold_scores(runs, xu=True)
            mx, _ = per_fold_scores(runs, xu=False)
            if xu is None:
                continue
            lo, hi = boot_ci(xu * 100, seed=abs(hash((ds, m))) % 2**31)
            summary[f"{ds}/{m}"] = dict(xu_mean=float(xu.mean() * 100),
                                        xu_std=float(xu.std(ddof=1) * 100),
                                        maxep_mean=float(mx.mean() * 100),
                                        n_runs=n, n_folds=len(xu))
            print(f"  {m:<9} {xu.mean()*100:6.2f} +/- {xu.std(ddof=1)*100:4.2f}"
                  f"   95%CI[{lo:.2f},{hi:.2f}]"
                  f"   [biased: {mx.mean()*100:.2f}]   runs={n}")

    print("\n" + "=" * 96)
    print("PAIRED topo - baseline, FOLD-BLOCKED (seeds averaged within a fold; "
          "n = #folds)")
    print("=" * 96)
    for ds in datasets:
        b, t = by.get((ds, 'baseline')), by.get((ds, 'topo'))
        if not b or not t:
            continue
        for tag, use_xu in (("Xu protocol (headline)", True),
                            ("best-epoch-per-fold (biased)", False)):
            sb, _ = per_fold_scores(b, xu=use_xu)
            st, _ = per_fold_scores(t, xu=use_xu)
            if sb is None or st is None or len(sb) != len(st):
                continue
            d = (st - sb) * 100
            lo, hi = boot_ci(d, seed=abs(hash((ds, tag))) % 2**31)
            pp, pw = perm_p(d), wilcoxon_p(d)
            w = int(np.sum(d > 0)); l = int(np.sum(d < 0)); tie = int(np.sum(d == 0))
            print(f"\n{ds}  [{tag}]  n={len(d)} folds")
            print(f"   mean diff {d.mean():+.2f}%   95%CI[{lo:+.2f},{hi:+.2f}]"
                  f"   W/L/T {w}/{l}/{tie}")
            print(f"   permutation p={pp:.4f}   Wilcoxon p={pw:.4f}"
                  f"   -> {'SIGNIFICANT' if pp < 0.05 else 'n.s.'} at 0.05")

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'summary.json')}")


if __name__ == "__main__":
    main()
