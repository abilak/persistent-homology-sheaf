"""
Paired significance test from per-fold accuracies you already have.

Use this for NCI1 / NCI109 (and anything else) where you have the per-fold
numbers in your own spreadsheet rather than as run JSONs in this repo. These
are the datasets where the gain is large (+2.67, +4.12) and the fold std is
small (~1.1), so the paired test should be strongly significant -- which
directly answers the "gains are comparable to fold std" critique.

Input: two space/comma-separated lists of 10 per-fold accuracies (baseline
then topo), either on the command line or edited into DATA below.

  python paired_from_folds.py NCI1 "77.1 78.2 ..." "80.0 80.4 ..."

Reports mean difference, 95% bootstrap CI, two-sided sign-flip permutation
p-value, midrank+tie-corrected Wilcoxon p-value, Cohen's d_z, and W/L/T --
fold-blocked (n = number of folds), which is the correct unit of analysis
because the same folds are reused across seeds.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import boot_ci, perm_p, wilcoxon_p

# Optionally hard-code your numbers here instead of passing them on the CLI.
# Values are PER-FOLD accuracies (already averaged over seeds), in percent.
DATA = {
    # "NCI1":   dict(baseline=[...10 numbers...], topo=[...10 numbers...]),
    # "NCI109": dict(baseline=[...],              topo=[...]),
}


def parse(s):
    return [float(x) for x in s.replace(",", " ").split()]


def report(name, base, topo):
    base, topo = np.asarray(base, float), np.asarray(topo, float)
    if len(base) != len(topo):
        print(f"{name}: length mismatch ({len(base)} vs {len(topo)})")
        return
    d = topo - base
    n = len(d)
    lo, hi = boot_ci(d, seed=abs(hash(name)) % 2**31)
    pp, pw = perm_p(d), wilcoxon_p(d)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float('inf')
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    print(f"\n=== {name}  (n = {n} folds, fold-blocked) ===")
    print(f"  baseline      {base.mean():6.2f} +/- {base.std(ddof=1):.2f}")
    print(f"  topo          {topo.mean():6.2f} +/- {topo.std(ddof=1):.2f}")
    print(f"  paired diff   {d.mean():+6.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  permutation p {pp:.4g}      Wilcoxon p {pw:.4g}")
    print(f"  Cohen's d_z   {dz:.2f}        W/L/T {w}/{l}/{t}")
    verdict = "SIGNIFICANT at 0.05" if pp < 0.05 else "n.s. at 0.05"
    print(f"  -> {verdict}")
    print(f"  per-fold diffs: {np.round(d, 2).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        report(sys.argv[1], parse(sys.argv[2]), parse(sys.argv[3]))
    elif DATA:
        for name, v in DATA.items():
            report(name, v["baseline"], v["topo"])
    else:
        print(__doc__)
        print("No data supplied. Either pass them:\n"
              '  python paired_from_folds.py NCI1 "77.1 78.2 ..." "80.0 80.4 ..."\n'
              "or fill in the DATA dict at the top of this file.")
