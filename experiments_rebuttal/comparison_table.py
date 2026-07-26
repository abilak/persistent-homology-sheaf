"""
Build the Q2/Q3 comparison table: our same-code, same-protocol baselines
(MLP, GCN, GIN, GSN, PPGN, PPGN+PH) merged with PUBLISHED numbers for recent
topological / subgraph-counting GNNs under the identical standard Xu et al.
10-fold TU protocol (so the rows are directly comparable). Published numbers
are cited, not re-run.
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")

# Published, same-protocol (standard Xu/GIN 10-fold) accuracies.
# Sources: GIN (Xu 2019); PPGN (Maron 2019); GSN (Bouritsas, arXiv:2006.09252,
# subgraph counting); CIN & SIN (Bodnar, arXiv:2106.12575, topological
# cellular/simplicial nets); WL kernel; DGCNN. "-" = not reported.
PUBLISHED = {
    #                MUTAG        PTC          PROTEINS     NCI1         NCI109       IMDB-B
    "WL kernel  [pub]":  ["90.4±5.7","59.9±4.3","75.0±3.1","86.0±1.8","-",       "73.8±3.9"],
    "DGCNN     [pub]":   ["85.8±1.8","58.6±2.5","75.5±0.9","74.4±0.5","-",       "70.0±0.9"],
    "GIN       [pub]":   ["89.4±5.6","64.6±7.0","76.2±2.8","82.7±1.7","-",       "75.1±5.1"],
    "PPGN      [pub]":   ["90.6±8.7","66.2±6.6","77.2±4.7","83.2±1.1","82.2±1.4","73.0±5.8"],
    "GSN (subgraph) [pub]":["92.2±7.5","68.2±7.2","76.6±5.0","83.5±2.0","-",     "77.8±3.3"],
    "SIN (topolog.) [pub]":["-",      "-",      "76.4±3.3","82.7±2.1","-",       "75.6±3.2"],
    "CIN (topolog.) [pub]":["92.7±6.1","68.2±5.6","77.0±4.3","83.6±1.4","84.0±1.6","75.6±3.7"],
}
COLS = ["MUTAG", "PTC", "PROTEINS", "NCI1", "NCI109", "IMDBBINARY"]
OURS = [("MLP (no MP)", "mlp"), ("GCN", "gcn"), ("GIN", "gin"),
        ("GSN (subgraph)", "gsn"), ("PPGN (baseline)", "baseline"),
        ("PPGN+PH (ours)", "topo")]


def load():
    by = {}
    for fp in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        if fp.endswith(".log"):
            continue
        try:
            r = json.load(open(fp))
            by.setdefault((r['dataset'], r['model']), []).append(r['best_val_acc'])
        except Exception:
            pass
    return by


def cell(accs):
    if not accs:
        return "-"
    a = np.array(accs) * 100
    return f"{a.mean():.1f}±{a.std():.1f} (n={len(a)})"


def main():
    by = load()
    header = "Model".ljust(22) + "".join(c[:9].ljust(12) for c in
                                          ["MUTAG", "PTC", "PROTEINS", "NCI1", "NCI109", "IMDB-B"])
    print("=" * len(header))
    print("OUR SAME-CODE, SAME-PROTOCOL RESULTS (mean±std over seeds×folds)")
    print("=" * len(header)); print(header); print("-" * len(header))
    for name, key in OURS:
        row = name.ljust(22)
        for ds in COLS:
            row += cell(by.get((ds, key))).ljust(12)
        print(row)
    print("\n" + "=" * len(header))
    print("PUBLISHED SAME-PROTOCOL BASELINES (standard Xu 10-fold; cited)")
    print("=" * len(header)); print(header); print("-" * len(header))
    for name, vals in PUBLISHED.items():
        print(name.ljust(22) + "".join(v.ljust(12) for v in vals))


if __name__ == "__main__":
    main()
