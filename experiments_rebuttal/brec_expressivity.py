"""
BREC expressivity benchmark -- the gold-standard, third-party stress test.

BREC (Wang, Yao, Wang, Zhang & Zhang, "Towards Better Evaluation of GNN
Expressiveness with the BREC Dataset", arXiv:2304.07702) contains 400 pairs of
non-isomorphic graphs in four difficulty classes -- Basic (60), Regular (100,
incl. strongly-regular, 4-vertex-condition, distance-regular), Extension (100),
and CFI (100) -- almost all of which are 1-WL AND 3-WL indistinguishable by
construction. A model "distinguishes" a pair only if it maps the two graphs to
reliably different embeddings while staying invariant across isomorphic
(permuted) copies. The strongly-regular block of *Regular* is exactly the
3-WL-hard regime our persistent-homology branch targets (same phenomenon as our
Rook/Shrikhande and SR-classification results), so BREC is the standardized,
un-cherry-picked confirmation of Corollary 6.

This script reuses the EXACT model zoo of the rebuttal (srg_classification.cfg
+ norm_adj_input), so every model here -- MLP, GCN, GIN, GSN, PPGN, PPGN+PH --
is bit-for-bit the same architecture, input pipeline and config as the SR / CSL
experiments. No model is re-implemented for BREC.

Evaluation: official Reliable Paired Comparison (RPC).
  For each pair (G1, G2):
    1. Draw SAMPLE_NUM random node-permutations of each graph.
    2. Fit a fresh per-pair model contrastively (push the two graphs' embedding
       means apart, keep permutations of one graph tight).
    3. MAJOR test  : two-sample Hotelling T^2 between {emb(perm G1)} and
                     {emb(perm G2)}.  Large  => the two graphs are separated.
    4. RELIABILITY : Hotelling T^2 between two independent permutation sets of
                     the SAME graph.  Small => separation is not permutation /
                     numerical noise (the model is permutation-invariant).
    Distinguished iff MAJOR > THRESHOLD and RELIABILITY < THRESHOLD.

Usage
-----
  # smoke test (few pairs, tiny sample):
  python experiments_rebuttal/brec_expressivity.py --model topo --pairs 0-6 \
         --sample-num 16 --epochs 8

  # full 400-pair sweep, one model per GPU:
  CCD_DEVICE=cuda python experiments_rebuttal/brec_expressivity.py --model gin  \
         --sample-num 400 --epochs 20 --out rebuttal_results/brec/gin.json
  CCD_DEVICE=cuda python experiments_rebuttal/brec_expressivity.py --model ppgn \
         --sample-num 400 --epochs 20 --out rebuttal_results/brec/ppgn.json
  CCD_DEVICE=cuda python experiments_rebuttal/brec_expressivity.py --model topo \
         --safe --sample-num 400 --epochs 20 --out rebuttal_results/brec/topo.json

  # just the (headline) Regular category:
  CCD_DEVICE=cuda python experiments_rebuttal/brec_expressivity.py --model topo \
         --pairs 60-160

Data
----
Point --data at the official BREC graph collection (800 graphs = 400 consecutive
pairs) as a .npy of graph6 byte-strings or a .g6 text file, or drop it at
experiments_rebuttal/data/brec_v3.npy. Obtain it from the official release:
https://github.com/GraphPKU/BREC  (BREC_data_all.zip -> the .npy of pairs).
"""
import os
import sys
import json
import time
import argparse

# repo-root importable + experiments_rebuttal on path (mirror srg_classification)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

import srg_classification as srg           # reuse cfg / norm_adj_input / SAFE_GATE
from srg_classification import cfg, norm_adj_input
from models.base_model import BaseModel
from models.baseline_models import BaselineModel
from utils.device import get_device

DEV = get_device()


# ----------------------------------------------------------------------------
# Official BREC category ranges (in PAIR index; 400 pairs total)
# ----------------------------------------------------------------------------
PART_DICT = {
    "Basic":               (0, 60),
    "Regular":             (60, 160),
    "Extension":           (160, 260),
    "CFI":                 (260, 360),
    "4-Vertex_Condition":  (360, 380),
    "Distance_Regular":    (380, 400),
}

# RPC constants (Wang et al. 2023). OUTPUT_DIM is the embedding width fed to the
# T^2 test; THRESHOLD is the Hotelling critical value at the paper's
# significance level (Bonferroni-conservative over the 400 pairs).
OUTPUT_DIM = 16
THRESHOLD  = 72.34
DEFAULT_LR = 1e-3
COV_EPS    = 1e-3   # ridge AFTER per-dimension standardization (unit variance),
                    # so it regularizes without swamping genuine signal

# map CLI name -> srg_classification model_type
MODEL_ALIASES = {"ppgn": "baseline", "topo": "topo",
                 "mlp": "mlp", "gcn": "gcn", "gin": "gin", "gsn": "gsn"}
PPGN_LIKE = {"baseline", "topo"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _default_data_path():
    for cand in ["experiments_rebuttal/data/brec_v3.npy",
                 "experiments_rebuttal/data/brec_v3.g6",
                 "data/brec_v3.npy"]:
        if os.path.exists(cand):
            return cand
    return None


def load_brec_graphs(path):
    """Return a list of networkx graphs (length should be 800 = 400 pairs)."""
    if path is None:
        raise FileNotFoundError(
            "BREC data not found. Download the 400-pair collection from "
            "https://github.com/GraphPKU/BREC (BREC_data_all.zip) and pass "
            "--data <brec_v3.npy> (or place it at "
            "experiments_rebuttal/data/brec_v3.npy)."
        )
    if path.endswith(".npy"):
        items = list(np.load(path, allow_pickle=True))
    else:  # .g6 text, one graph6 string per line
        with open(path, "rb") as f:
            items = [ln.strip() for ln in f if ln.strip()]

    graphs = []
    for it in items:
        if isinstance(it, nx.Graph):
            g = it
        else:
            if isinstance(it, str):
                it = it.encode()
            g = nx.from_graph6_bytes(bytes(it))
        graphs.append(nx.convert_node_labels_to_integers(g))
    return graphs


def _same_unordered_pair(a1, a2, b1, b2):
    """True if {a1,a2} and {b1,b2} are the same graph pair up to isomorphism,
    IGNORING order (BREC dumps store some instances with the two graphs
    swapped). Node-count guards short-circuit the expensive iso checks."""
    n = lambda g: g.number_of_nodes()
    straight = (n(a1) == n(b1) and n(a2) == n(b2)
                and nx.is_isomorphic(a1, b1) and nx.is_isomorphic(a2, b2))
    if straight:
        return True
    return (n(a1) == n(b2) and n(a2) == n(b1)
            and nx.is_isomorphic(a1, b2) and nx.is_isomorphic(a2, b1))


def detect_pair_stride(graphs, max_scan=2048):
    """Some BREC dumps are 'pair-major': each real pair is stored as R
    consecutive permuted instances (e.g. 51200 = 400 pairs x 64 instances x 2
    graphs), and instances may swap the order of the two graphs. Return R =
    number of consecutive storage-pairs that are the SAME unordered pair as
    storage-pair 0 (R = 1 if pairs are already distinct)."""
    def pair(p):
        return graphs[2 * p], graphs[2 * p + 1]
    g0a, g0b = pair(0)
    total = len(graphs) // 2
    R = 1
    while R < min(max_scan, total):
        ga, gb = pair(R)
        if not _same_unordered_pair(ga, gb, g0a, g0b):
            break
        R += 1
    return R


def make_perm_batch(G, num, rng):
    """Stack `num` random node-permutations -> (num, 1, n, n) via the SAME
    normalized-adjacency encoding the SR/CSL experiments use."""
    n = G.number_of_nodes()
    tensors = [norm_adj_input(G, rng.permutation(n)) for _ in range(num)]
    return torch.stack(tensors, 0).to(DEV) if tensors[0].dim() == 3 \
        else torch.cat(tensors, 0).to(DEV)


# ----------------------------------------------------------------------------
# Models (reuse the rebuttal zoo; embedding = the model's num_classes logits)
# ----------------------------------------------------------------------------
def build_model(cli_name):
    model_type = MODEL_ALIASES[cli_name]
    conf = cfg(OUTPUT_DIM, model_type)               # num_classes := OUTPUT_DIM
    model = BaseModel(conf) if model_type in PPGN_LIKE else BaselineModel(conf)
    return model.to(DEV)


# ----------------------------------------------------------------------------
# Hotelling two-sample T^2
# ----------------------------------------------------------------------------
def _cov(X):
    Xc = X - X.mean(0, keepdim=True)
    return (Xc.t() @ Xc) / max(1, X.shape[0] - 1)


def hotelling_t2(X, Y, eps=COV_EPS):
    nx_, ny = X.shape[0], Y.shape[0]
    d = X.shape[1]
    diff = X.mean(0) - Y.mean(0)
    Sp = ((nx_ - 1) * _cov(X) + (ny - 1) * _cov(Y)) / max(1, nx_ + ny - 2)
    Sp = Sp + eps * torch.eye(d, device=Sp.device, dtype=Sp.dtype)
    inv = torch.linalg.pinv(Sp)
    return float((nx_ * ny) / (nx_ + ny) * (diff @ inv @ diff))


# ----------------------------------------------------------------------------
# Per-pair Reliable Paired Comparison
# ----------------------------------------------------------------------------
def evaluate_pair(cli_name, g1, g2, args, rng):
    model = build_model(cli_name)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    S, bs = args.sample_num, args.batch

    def batched_embed(g, num, grad):
        outs = []
        for start in range(0, num, bs):
            k = min(bs, num - start)
            batch = make_perm_batch(g, k, rng)
            if grad:
                outs.append(model(batch))
            else:
                with torch.no_grad():
                    outs.append(model(batch))
        return torch.cat(outs, 0)

    # ---- fit: push the two graphs apart on the unit sphere, tighten each ----
    # Unit-normalising the embeddings keeps the objective scale-free (the model
    # can't cheat by inflating logits), so `between` in [0,4] and `within`
    # measures genuine permutation scatter. Maximise separation, minimise
    # within-graph scatter.
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        e1 = F.normalize(batched_embed(g1, bs, grad=True), dim=1)
        e2 = F.normalize(batched_embed(g2, bs, grad=True), dim=1)
        between = (e1.mean(0) - e2.mean(0)).pow(2).sum()
        within = e1.var(0).sum() + e2.var(0).sum()
        loss = -between + 0.5 * within
        loss.backward()
        opt.step()

    # ---- test: Hotelling T^2 with COMMON standardization ----
    # The fix for spurious "reliability": measure the within-graph drift of a
    # (near-)permutation-invariant model against the SAME per-dimension scale
    # used for the between-graph test. We standardize E1, E2 AND the reliability
    # replica E1b by the mean/std of the pooled major sample [E1; E2]. If the
    # separation is real, that scale is O(1) and the ~1e-3 floating-point
    # non-invariance across permutations lands near zero -> reliability ~ 0. If
    # there is no separation, the scale is float-noise and BOTH stats sit at the
    # null F-level (well below THRESHOLD) -> correctly not distinguished.
    model.eval()
    E1  = batched_embed(g1, S, grad=False)
    E2  = batched_embed(g2, S, grad=False)
    E1b = batched_embed(g1, S, grad=False)   # independent perms of the SAME graph

    pooled = torch.cat([E1, E2], 0)
    mu = pooled.mean(0, keepdim=True)
    sd = pooled.std(0, keepdim=True).clamp_min(1e-8)
    z = lambda E: (E - mu) / sd

    major = hotelling_t2(z(E1), z(E2))
    reliability = hotelling_t2(z(E1), z(E1b))
    distinguished = (major > THRESHOLD) and (reliability < THRESHOLD)
    return {"major": major, "reliability": reliability,
            "distinguished": bool(distinguished)}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def parse_pairs_arg(s, total):
    if s is None:
        return list(range(total))
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), min(int(b), total)))
    return [int(x) for x in s.split(",")]


def category_of(pair_idx):
    for name, (lo, hi) in PART_DICT.items():
        if lo <= pair_idx < hi:
            return name
    return "Unknown"


def main():
    ap = argparse.ArgumentParser(description="BREC expressivity benchmark")
    ap.add_argument("--model", choices=list(MODEL_ALIASES), default="topo",
                    help="mlp/gcn/gin (1-WL), gsn (subgraph), ppgn (3-WL), "
                         "topo (PPGN+PH, ours)")
    ap.add_argument("--data", default=None, help="path to brec_v3.npy / .g6")
    ap.add_argument("--pairs", default=None,
                    help="'lo-hi' or comma list of UNIQUE pair indices (default: all)")
    ap.add_argument("--pair-stride", type=int, default=0,
                    help="instances per real pair for pair-major dumps "
                         "(0 = auto-detect; 1 = pairs already distinct)")
    ap.add_argument("--sample-num", type=int, default=32,
                    help="permutations per graph for the T^2 test (official 400)")
    ap.add_argument("--batch", type=int, default=16,
                    help="perm minibatch size (also #perms per fit step)")
    ap.add_argument("--epochs", type=int, default=30, help="per-pair fit epochs")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--safe", action="store_true",
                    help="topo: scalar zero-init gate (starts at PPGN baseline)")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="skip pairs with a graph larger than this (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-pair results to JSON")
    args = ap.parse_args()

    if args.safe:
        srg.SAFE_GATE = True   # cfg() reads this global when building topo
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_path = args.data or _default_data_path()
    graphs = load_brec_graphs(data_path)
    total_pairs = len(graphs) // 2
    print(f"Loaded {len(graphs)} graphs = {total_pairs} storage-pairs "
          f"from {data_path}")

    # collapse pair-major permutation blocks to the unique pairs
    stride = args.pair_stride or detect_pair_stride(graphs)
    num_pairs = total_pairs // stride
    if stride > 1:
        note = "" if total_pairs % stride == 0 else \
            f"  [WARN: {total_pairs} not divisible by {stride}]"
        print(f"Pair-major layout: {stride} permuted instances/pair "
              f"-> {num_pairs} UNIQUE pairs (using first instance of each).{note}")
    else:
        print(f"Distinct pairs: {num_pairs}")
    print(f"Device: {DEV} | model: {args.model} | safe={args.safe} | "
          f"sample_num={args.sample_num} epochs={args.epochs}")

    pair_ids = parse_pairs_arg(args.pairs, num_pairs)
    per_cat, records, skipped = {}, [], 0
    t_start = time.time()
    for pid in pair_ids:                        # pid = UNIQUE pair index
        sp = pid * stride                        # -> storage-pair index
        g1, g2 = graphs[2 * sp], graphs[2 * sp + 1]
        n = max(g1.number_of_nodes(), g2.number_of_nodes())
        if args.max_nodes and n > args.max_nodes:
            skipped += 1
            continue
        res = evaluate_pair(args.model, g1, g2, args, rng)
        cat = category_of(pid)
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += int(res["distinguished"])
        per_cat[cat][1] += 1
        records.append({"pair": pid, "category": cat, "n": n, **res})
        flag = "OK " if res["distinguished"] else "   "
        print(f"[{flag}] pair {pid:>3} ({cat:<18}) n={n:<3} "
              f"major={res['major']:.2f} reliab={res['reliability']:.2f}")

    # ---- summary ----
    print("\n" + "=" * 66)
    print(f"BREC results  --  model = {args.model}"
          f"  ({time.time() - t_start:.0f}s)")
    print("=" * 66)
    print(f"{'category':<20}{'distinguished / seen':>24}{'rate':>12}")
    tot_d = tot_s = 0
    for name in PART_DICT:
        if name in per_cat:
            d, s = per_cat[name]
            tot_d += d; tot_s += s
            print(f"{name:<20}{f'{d} / {s}':>24}{f'{100*d/max(1,s):.1f}%':>12}")
    print("-" * 56)
    print(f"{'TOTAL':<20}{f'{tot_d} / {tot_s}':>24}"
          f"{f'{100*tot_d/max(1,tot_s):.1f}%':>12}")
    if skipped:
        print(f"(skipped {skipped} pairs exceeding --max-nodes={args.max_nodes})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "config": vars(args),
                       "per_category": per_cat, "total": [tot_d, tot_s],
                       "skipped": skipped, "records": records}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
