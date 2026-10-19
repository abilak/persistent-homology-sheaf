import os, sys, json, glob, itertools, zlib
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')

FIXED = {'ZINC': ('regression', 'test_mae_at_best_val', 'test MAE', True), 'MOLHIV': ('auc', 'test_auc_at_best_val', 'test ROC-AUC', False)}

TU = ["NCI1", "NCI109", "MUTAG", "PTC", "PROTEINS", "ENZYMES", "IMDBBINARY", "IMDBMULTI"]
NAMES = {"IMDBBINARY": "IMDB-B", "IMDBMULTI": "IMDB-M"}
PH = r"\placeholder{--}"

ROWS = [("MLP (no message passing)", "mlp"), ("GCN", "gcn"), ("GIN", "gin"), ("GSN (cycles 3--6)", "gsn"), ("PPGN backbone", "base"), ("PPGN backbone, widened", "wide"), ("PPGN+PH, static control", "static"), ("PPGN + Betti numbers", "betti"), ("PPGN+PH (ours)", "full"), ("Paired gain", "gain"), ("Paired 95\\% CI", "ci"), ("Permutation $p$", "perm"), ("Wilcoxon $p$ (exact)", "wil"), ("Folds better", "pos")]

MOL_ROWS = [("MLP (no message passing)", "mlp", "fast"), ("GCN", "gcn", "fast"), ("GIN", "gin", "fast"), ("GSN (cycles 3--6)", "gsn", "fast"), ("PPGN backbone", "baseline", "fast"), ("PPGN backbone, widened", "baseline", "wide_fast"), ("PPGN+PH, static control", "topo", "full_static_fast"), ("PPGN + Betti numbers", "baseline", "betti_fast"), ("PPGN+PH (ours)", "topo", "full_fast")]

def load_runs(ds, need=None):
    P = defaultdict(dict)
    for fp in glob.glob(os.path.join(RESULTS, ds, '*.json')):
        name = os.path.basename(fp)[:-5]
        parts = name.split('_')
        if(len(parts) < 3 or not parts[1].startswith('s') or not parts[2].startswith('f')):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        if(need and need not in r):
            continue
        variant = '_'.join(parts[3:])
        P[(r['model'], variant)][(r['seed'], r['fold'])] = r
    return P

def fold_scores(runs):
    seeds = sorted({s for s, _ in runs})
    folds = sorted({f for _, f in runs})
    pf = defaultdict(list)
    for s in seeds:
        curves = {}
        for f in folds:
            if((s, f) in runs):
                curves[f] = np.asarray(runs[(s, f)]['val_curve'])
        if(not curves):
            continue
        # xu et al: one epoch per seed, picked on the fold averaged curve
        if(len(curves) != len(folds)):
            continue
        L = min(len(c) for c in curves.values())
        M = np.stack([curves[f][:L] for f in folds])
        best_epoch = int(np.argmax(M.mean(axis=0)))
        for(i, f) in enumerate(folds):
            pf[f].append(M[i, best_epoch])
    if(not pf):
        return None, 0
    n = sum(len(v) for v in pf.values())
    scores = np.array([np.mean(pf[f]) for f in sorted(pf)])
    return scores, n

def fmt(m, s):
    return "%.2f{\\scriptsize$\\pm$%.2f}" % (m, s)

def tu_scores(P, key):
    if(key not in P or len(P[key]) < 10):
        return None
    s, _ = fold_scores(P[key])
    if(s is None or len(s) < 10):
        return None
    return s * 100

def seed_of(*key):
    # not hash(), it's salted per process
    return zlib.crc32(repr(key).encode())

def boot_ci(x, nb=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if(len(x) < 2):
        return float('nan'), float('nan')
    idx = rng.integers(0, len(x), (nb, len(x)))
    means = x[idx].mean(1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def perm_p(d, nperm=20000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if(len(d) == 0):
        return float('nan')
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(nperm, len(d)))
    flipped = np.abs((signs * d).mean(1))
    count = np.sum(flipped >= obs - 1e-12)
    return float((count + 1) / (nperm + 1))

def exact_wil(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if(n == 0):
        return float('nan')
    r = rankdata(np.abs(d))
    half = r.sum() / 2
    obs = abs(np.sum(r[d > 0]) - half)
    hits = 0
    for s in itertools.product((0, 1), repeat=n):
        if(abs(float(np.dot(r, s)) - half) >= obs - 1e-12):
            hits += 1
    return hits / 2 ** n

def tu_table():
    rows = defaultdict(list)
    for ds in TU:
        P = load_runs(ds)
        cells = {"base": ("baseline", "fast"), "wide": ("baseline", "wide_fast"), "full": ("topo", "full_fast"), "static": ("topo", "full_static_fast"), "betti": ("baseline", "betti_fast")}
        for mp in ("mlp", "gcn", "gin", "gsn"):
            cells[mp] = (mp, "fast")
        sc = {}
        for(k, key) in cells.items():
            sc[k] = tu_scores(P, key)
        for(k, x) in sc.items():
            if(x is not None):
                rows[k].append(fmt(x.mean(), x.std(ddof=1)))
            else:
                rows[k].append(PH)

        b, t = sc["base"], sc["full"]
        if(b is None or t is None):
            for k in ("gain", "ci", "perm", "wil", "pos"):
                rows[k].append(PH)
            continue
        d = t - b
        lo, hi = boot_ci(d, seed=seed_of("full_fast"))
        rows["gain"].append("$%+.2f$" % d.mean())
        rows["ci"].append("$[%+.2f, %+.2f]$" % (lo, hi))
        rows["perm"].append("$%.3f$" % perm_p(d))
        rows["wil"].append("$%.3f$" % exact_wil(d))
        rows["pos"].append("%d/%d" % (int((d > 0).sum()), len(d)))

    print("% ---- TU table (paste into tab:tu) ----")
    print("% columns: " + " & ".join(NAMES.get(d, d) for d in TU))
    for(label, k) in ROWS:
        print(label + " & " + " & ".join(rows[k]) + r" \\")

def compare_tu(ds, bvar):
    pools = load_runs(ds, need='val_curve')
    found = []
    for((m, v), d) in sorted(pools.items()):
        if(v):
            found.append('{}/{}(n={})'.format(m, v, len(d)))
        else:
            found.append('{}(n={})'.format(m, len(d)))
    print(f"{ds}: pools found -> " + ", ".join(found))

    bkey = ('baseline', bvar or '')
    if(bkey not in pools):
        print(f"no baseline pool {bkey}")
        sys.exit(1)
    sb, _ = fold_scores(pools[bkey])
    bname = 'baseline/' + bkey[1] if bkey[1] else 'baseline'
    print(f"\n{bname}: {sb.mean()*100:.2f} +/- {sb.std(ddof=1)*100:.2f}  (n={len(sb)} folds)\n")

    print(f"{'topo variant':<18}{'Xu acc':<18}{'paired vs baseline':<24}{'perm p'}")
    print("-" * 76)
    for((m, v), runs) in sorted(pools.items()):
        if(m == 'baseline'):
            continue
        if(m != 'topo'):
            v = f"{m}/{v}"
        st, _ = fold_scores(runs)
        label = v or '(main)'
        if(st is None or len(st) != len(sb)):
            nf = 0 if st is None else len(st)
            print(f"{label:<18}incomplete ({nf} folds)")
            continue
        d = (st - sb) * 100
        lo, hi = boot_ci(d, seed=seed_of(v))
        print(f"{label:<18}{st.mean()*100:6.2f} +/- {st.std(ddof=1)*100:<8.2f}{d.mean():+6.2f}  [{lo:+.2f},{hi:+.2f}]   {perm_p(d):.4f}")

def seed_vals(P, key, stat):
    out = {}
    for r in P.get(key, {}).values():
        if(stat in r):
            out[r["seed"]] = r[stat]
    return out

def compare_fixed(ds, bvar):
    task, key, name, lower = FIXED[ds]
    pools = defaultdict(dict)
    for(k, runs) in load_runs(ds).items():
        for((s, f), r) in runs.items():
            if(r.get('task') == task):
                pools[k][s] = r[key]

    if(('baseline', bvar) not in pools):
        sys.exit(f"no {ds} baseline runs with variant '{bvar}' yet")
    b = pools[('baseline', bvar)]
    bv = list(b.values())
    bsd = np.std(bv, ddof=1) if len(bv) > 1 else 0
    direction = 'lower' if lower else 'higher'
    print(f"{ds} {name} at best-val epoch ({direction} is better); baseline/{bvar}: {np.mean(bv):.4f} +/- {bsd:.4f} (n={len(b)} seeds)\n")
    print(f"{'variant':<18}{name:<20}{'paired vs baseline (seeds)':<34}{'95% CI':<20}{'perm p':<9}{'Wilcoxon p'}")
    print("-" * 112)
    for((m, v), runs) in sorted(pools.items()):
        if(m == 'baseline'):
            continue
        if(m != 'topo'):
            v = f"{m}/{v}"
        seeds = sorted(set(runs) & set(b))
        vals = np.array(list(runs.values()))
        d = np.array([runs[s] - b[s] for s in seeds])
        if(lower):
            better = int((d < 0).sum())
        else:
            better = int((d > 0).sum())
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        if(len(d)):
            paired = "%+.4f (n=%d, %d/%d better)" % (d.mean(), len(d), better, len(d))
            lo, hi = boot_ci(d, seed=seed_of(ds, v))
            stats = f"[{lo:+.4f},{hi:+.4f}]   {perm_p(d):<9.4f}{exact_wil(d):.4f}"
        else:
            paired, stats = "no matching seeds yet", ""
        print(f"{v:<18}{vals.mean():.4f} +/- {sd:<10.4f}{paired:<34}{stats}")

def fixed_table():
    print("\n% ---- fixed-split table (paste into tab:mol): rows = models, columns = ZINC MAE, molhiv ROC-AUC ----")
    specs = [("ZINC", "test_mae_at_best_val", 1.0), ("MOLHIV", "test_auc_at_best_val", 100.0)]
    pools = {}
    for(ds, _, _) in specs:
        pools[ds] = load_runs(ds)
    for(label, m, v) in MOL_ROWS:
        cells = []
        for(ds, stat, scale) in specs:
            x = np.array(list(seed_vals(pools[ds], (m, v), stat).values())) * scale
            if(not len(x)):
                cells.append(PH)
                continue
            sd = x.std(ddof=1) if len(x) > 1 else 0
            cells.append("%.3f{\\scriptsize$\\pm$%.3f}" % (x.mean(), sd))
        print(label + " & " + " & ".join(cells) + r" \\")

    for(ds, stat, scale) in specs:
        bb = seed_vals(pools[ds], ("baseline", "fast"), stat)
        tt = seed_vals(pools[ds], ("topo", "full_fast"), stat)
        common = sorted(set(bb) & set(tt))
        if(len(common) < 2):
            continue
        d = np.array([tt[k] - bb[k] for k in common]) * scale
        lo, hi = boot_ci(d, seed=seed_of(ds, "full_fast"))
        print("%%   %s paired (full - baseline): %+.4f  95%% CI [%+.4f, %+.4f]  perm p %.4f  Wilcoxon p %.4f  n=%d" % (ds, d.mean(), lo, hi, perm_p(d), exact_wil(d), len(d)))

if(__name__ == '__main__'):
    if(len(sys.argv) < 2):
        tu_table()
        fixed_table()
    else:
        ds = sys.argv[1]
        base = sys.argv[2] if len(sys.argv) > 2 else None
        if(ds.upper() in ('ZINC', 'MOLHIV')):
            compare_fixed(ds.upper(), base or 'fast')
        else:
            compare_tu(ds, base)
