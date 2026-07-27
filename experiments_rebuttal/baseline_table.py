"""
The rebuttal's baseline comparison table. Under the standard Xu et al. TU
10-fold protocol, on the identical splits (`data/benchmark_graphs/*/10fold_idx/`).
Combines:
  * OUR SAME-CODE numbers (paper's max-per-fold metric, our 10-fold x 5 seeds)
  * PUBLISHED numbers for baselines that use the same protocol
    (Xu 2019, Maron 2019, Bouritsas 2020 [GSN], Bodnar 2021 [SIN, CIN],
     Horn 2022 [TOGL]). NOT reruns -- cited directly from their papers.
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "rebuttal_results")


def load_main(ds):
    out = defaultdict(list)
    for fp in glob.glob(os.path.join(RESULTS, ds, "*.json")):
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
        out[r["model"]].append(r["best_val_acc"])
    return out


# Same-protocol PUBLISHED numbers.
# Sources:
#   WL kernel: Shervashidze 2011 (Xu 2019 Table 1)
#   DGCNN: Zhang 2018 (Xu 2019)
#   GIN: Xu et al. 2019 (arXiv:1810.00826), Table 1
#   PPGN: Maron et al. 2019 (arXiv:1905.11136), Table 2
#   GSN: Bouritsas et al. 2020 (arXiv:2006.09252), Table 1
#   SIN: Bodnar et al. 2021 (arXiv:2103.03212), Table 1
#   CIN: Bodnar et al. 2021 (arXiv:2106.12575), Table 1
# All: mean +/- std across 10 folds, same Xu splits.
PUBLISHED = {
    "WL kernel":            {"MUTAG": "90.4±5.7", "PTC": "59.9±4.3", "PROTEINS": "75.0±3.1", "NCI1": "86.0±1.8", "IMDB-B": "73.8±3.9"},
    "DGCNN":                {"MUTAG": "85.8±1.8", "PTC": "58.6±2.5", "PROTEINS": "75.5±0.9", "NCI1": "74.4±0.5", "IMDB-B": "70.0±0.9"},
    "GIN [Xu 2019]":         {"MUTAG": "89.4±5.6", "PTC": "64.6±7.0", "PROTEINS": "76.2±2.8", "NCI1": "82.7±1.7", "IMDB-B": "75.1±5.1"},
    "PPGN [Maron 2019]":     {"MUTAG": "90.6±8.7", "PTC": "66.2±6.6", "PROTEINS": "77.2±4.7", "NCI1": "83.2±1.1", "IMDB-B": "73.0±5.8"},
    "GSN [Bouritsas 2020]":  {"MUTAG": "92.2±7.5", "PTC": "68.2±7.2", "PROTEINS": "76.6±5.0", "NCI1": "83.5±2.0", "IMDB-B": "77.8±3.3"},
    "SIN [Bodnar 2021]":     {"MUTAG": "-",         "PTC": "-",         "PROTEINS": "76.4±3.3", "NCI1": "82.7±2.1", "IMDB-B": "75.6±3.2"},
    "CIN [Bodnar 2021]":     {"MUTAG": "92.7±6.1", "PTC": "68.2±5.6", "PROTEINS": "77.0±4.3", "NCI1": "83.6±1.4", "IMDB-B": "75.6±3.7"},
}

COLS = ["MUTAG", "PTC", "PROTEINS", "NCI1", "IMDB-B"]

# ---- our same-code numbers ---- #
print("=" * 90)
print("OUR SAME-CODE RESULTS  (paper's metric = best epoch per fold, mean +/- std over 10 folds x 5 seeds)")
print("=" * 90)
print(f"{'model':<30}" + "".join(f"{c:<12}" for c in COLS))
print("-" * 90)
ours = {}
for m, label in [
    ("mlp", "MLP (no MP)"),
    ("gcn", "GCN"),
    ("gin", "GIN"),
    ("gsn", "GSN (ours)"),
    ("baseline", "PPGN (baseline)"),
    ("topo", "PPGN + PH (ours)"),
]:
    row = f"{label:<30}"
    for ds in COLS:
        vals = load_main(ds).get(m, [])
        if vals:
            a = np.array(vals) * 100
            row += f"{a.mean():.1f}±{a.std(ddof=1):.1f}    "[:12]
        else:
            row += f"{'-':<12}"
    print(row)

print("\n" + "=" * 90)
print("PUBLISHED BASELINES  (same Xu protocol, cited)")
print("=" * 90)
print(f"{'model':<30}" + "".join(f"{c:<12}" for c in COLS))
print("-" * 90)
for name, row in PUBLISHED.items():
    print(f"{name:<30}" + "".join(f"{row[c]:<12}" for c in COLS))
