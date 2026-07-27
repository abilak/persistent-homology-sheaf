"""
Score any variant-tagged topo runs against the matched baseline under the
standard Xu protocol with fold-blocked paired statistics.

usage: python compare_variant.py DATASET [baseline_variant]
       (baseline_variant defaults to the untagged main baseline)
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import per_fold_scores, boot_ci, perm_p, wilcoxon_p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")

ds = sys.argv[1]
base_variant = sys.argv[2] if len(sys.argv) > 2 else None

pools = defaultdict(dict)
for fp in glob.glob(os.path.join(RESULTS, ds, "*.json")):
    base = os.path.basename(fp)[:-5]
    parts = base.split("_")
    if len(parts) < 3 or not parts[1].startswith("s") or not parts[2].startswith("f"):
        continue
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    if "val_curve" not in r:
        continue
    variant = "_".join(parts[3:]) if len(parts) > 3 else ""
    pools[(r["model"], variant)][(r["seed"], r["fold"])] = r

print(f"{ds}: pools found -> " + ", ".join(
    f"{m}{'/'+v if v else ''}(n={len(d)})" for (m, v), d in sorted(pools.items())))

bkey = ("baseline", base_variant or "")
if bkey not in pools:
    print(f"no baseline pool {bkey}"); sys.exit(1)
sb, _ = per_fold_scores(pools[bkey], xu=True)
print(f"\nbaseline{'/'+bkey[1] if bkey[1] else ''}: "
      f"{sb.mean()*100:.2f} +/- {sb.std(ddof=1)*100:.2f}  (n={len(sb)} folds)\n")

print(f"{'topo variant':<18}{'Xu acc':<18}{'paired vs baseline':<24}{'perm p'}")
print("-" * 76)
for (m, v), runs in sorted(pools.items()):
    if m != "topo":
        continue
    st, _ = per_fold_scores(runs, xu=True)
    if st is None or len(st) != len(sb):
        print(f"{v or '(main)':<18}incomplete ({0 if st is None else len(st)} folds)")
        continue
    d = (st - sb) * 100
    lo, hi = boot_ci(d, seed=abs(hash(v)) % 2**31)
    print(f"{v or '(main)':<18}{st.mean()*100:6.2f} +/- {st.std(ddof=1)*100:<8.2f}"
          f"{d.mean():+6.2f}  [{lo:+.2f},{hi:+.2f}]   {perm_p(d):.4f}")
