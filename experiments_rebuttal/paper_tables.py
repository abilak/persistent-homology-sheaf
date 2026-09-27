"""
Print the paper's result tables as LaTeX rows, straight from rebuttal_results/.

  python experiments_rebuttal/paper_tables.py            # TU table + fixed-split table
  python experiments_rebuttal/paper_tables.py --ema      # the *_ema runs instead

TU datasets: Xu et al. 10-fold protocol with the shared best epoch; one seed.
Per dataset: baseline, parameter-matched (widened) baseline, full model
(mean +/- std over folds), and the fold-blocked paired comparison of the full
model against the baseline: mean difference, 95% bootstrap CI, two-sided
sign-flip permutation p, EXACT two-sided Wilcoxon signed-rank p (midranks,
all 2^n sign patterns), and the number of folds on which the full model is
better. Missing runs print as \\placeholder{--}.

ZINC (test MAE, lower is better) and MOLHIV (test ROC-AUC): mean +/- std over
seeds at the best-validation epoch, plus the seed-paired comparison.
"""
import os, sys, json, glob, itertools
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import per_fold_scores, boot_ci, perm_p, stable_seed, _midranks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "rebuttal_results")
EMA = "--ema" in sys.argv
SUF = "_ema" if EMA else ""
TU = ["NCI1", "NCI109", "MUTAG", "PTC", "PROTEINS", "ENZYMES", "IMDBBINARY", "IMDBMULTI"]
NAMES = {"IMDBBINARY": "IMDB-B", "IMDBMULTI": "IMDB-M"}
PH = r"\placeholder{--}"


def pools(ds):
    P = defaultdict(dict)
    for fp in glob.glob(os.path.join(RES, ds, "*.json")):
        parts = os.path.basename(fp)[:-5].split("_")
        if len(parts) < 3 or not parts[1].startswith("s") or not parts[2].startswith("f"):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        P[(r["model"], "_".join(parts[3:]))][(r["seed"], r["fold"])] = r
    return P


def exact_wilcoxon(d):
    """exact two-sided Wilcoxon signed-rank p over all 2^n sign patterns (midranks)"""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    r = _midranks(np.abs(d))
    half = r.sum() / 2
    obs = abs(np.sum(r[d > 0]) - half)
    hits = sum(abs(float(np.dot(r, s)) - half) >= obs - 1e-12
               for s in itertools.product((0, 1), repeat=n))
    return hits / 2 ** n


def fmt(m, s):
    return f"{m:.2f}{{\\scriptsize$\\pm${s:.2f}}}"


def tu_scores(P, key):
    if key not in P or len(P[key]) < 10:
        return None
    s, _ = per_fold_scores(P[key], xu=True)
    return None if s is None or len(s) < 10 else s * 100


def tu_table():
    rows = defaultdict(list)
    for ds in TU:
        P = pools(ds)
        b = tu_scores(P, ("baseline", "fast" + SUF))
        w = tu_scores(P, ("baseline", "wide_fast" + SUF))
        t = tu_scores(P, ("topo", "full_fast" + SUF))
        rows["base"].append(fmt(b.mean(), b.std(ddof=1)) if b is not None else PH)
        rows["wide"].append(fmt(w.mean(), w.std(ddof=1)) if w is not None else PH)
        rows["full"].append(fmt(t.mean(), t.std(ddof=1)) if t is not None else PH)
        if b is not None and t is not None:
            d = t - b
            lo, hi = boot_ci(d, seed=stable_seed("full_fast" + SUF))
            rows["gain"].append(f"${d.mean():+.2f}$")
            rows["ci"].append(f"$[{lo:+.2f}, {hi:+.2f}]$")
            rows["perm"].append(f"${perm_p(d):.3f}$")
            rows["wil"].append(f"${exact_wilcoxon(d):.3f}$")
            rows["pos"].append(f"{int((d > 0).sum())}/{len(d)}")
        else:
            for k in ("gain", "ci", "perm", "wil", "pos"):
                rows[k].append(PH)
    print("% ---- TU table (paste into tab:tu) ----")
    print("% columns: " + " & ".join(NAMES.get(d, d) for d in TU))
    for label, key in [("PPGN backbone", "base"), ("PPGN backbone, widened", "wide"),
                       ("PPGN+PH (ours)", "full"), ("Paired gain", "gain"),
                       ("Paired 95\\% CI", "ci"), ("Permutation $p$", "perm"),
                       ("Wilcoxon $p$ (exact)", "wil"), ("Folds better", "pos")]:
        print(f"{label} & " + " & ".join(rows[key]) + r" \\")


def seed_values(P, key, stat):
    return {r["seed"]: r[stat] for r in P.get(key, {}).values() if stat in r}


def fixed_table():
    print("\n% ---- fixed-split table (paste into tab:mol) ----")
    for ds, stat, scale in [("ZINC", "test_mae_at_best_val", 1.0), ("MOLHIV", "test_auc_at_best_val", 100.0)]:
        P = pools(ds)
        cells = []
        for key in [("baseline", "fast" + SUF), ("baseline", "wide_fast" + SUF), ("topo", "full_fast" + SUF)]:
            v = np.array(list(seed_values(P, key, stat).values())) * scale
            cells.append(f"{v.mean():.3f}{{\\scriptsize$\\pm${v.std(ddof=1) if len(v) > 1 else 0:.3f}}} ($n={len(v)}$)"
                         if len(v) else PH)
        print(f"{ds} & " + " & ".join(cells) + r" \\")
        bb = seed_values(P, ("baseline", "fast" + SUF), stat)
        tt = seed_values(P, ("topo", "full_fast" + SUF), stat)
        common = sorted(set(bb) & set(tt))
        if len(common) >= 2:
            d = np.array([tt[k] - bb[k] for k in common]) * scale
            lo, hi = boot_ci(d, seed=stable_seed(ds, "full_fast" + SUF))
            print(f"%   {ds} paired (full - baseline): {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"perm p {perm_p(d):.4f}  Wilcoxon p {exact_wilcoxon(d):.4f}  n={len(d)}")


if __name__ == "__main__":
    tu_table()
    fixed_table()
