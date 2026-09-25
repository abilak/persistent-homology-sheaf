"""Sanity check for the random-initialization SR separation experiment.

A graph and a relabeled copy of itself are isomorphic, so any
permutation-invariant model must give them identical outputs (up to float
rounding, ~1e-8 in float32). If a model separates G from relabeled-G about as
strongly as it separates G1 from G2, its G1-vs-G2 "separation" is not evidence
of distinguishing non-isomorphic graphs.

Uses only srg_analysis.py's configuration, seeds, and the BaseModel API, so it
runs unchanged on both the `rebuttal` and `impl-containment` branches:
    python experiments_rebuttal/test_isomorphic_copies.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import torch, numpy as np, networkx as nx
import srg_analysis as SA
from srg_graphs import all_pairs
from models.base_model import BaseModel


def relabel(G, seed):
    p = np.random.default_rng(seed).permutation(G.number_of_nodes())
    return nx.relabel_nodes(G, {i: int(p[i]) for i in range(G.number_of_nodes())})


def diff(Ga, Gb, seeds=5, float64=False):
    """median over srg_analysis seeds of max |f(Ga) - f(Gb)|. float64=True
    evaluates the SAME weights (initialized in float32, then cast) in double
    precision: a genuine topological difference survives the cast, float
    rounding amplified at ties does not."""
    xa, xb = SA.graph_to_input(Ga), SA.graph_to_input(Gb)
    d = []
    for s in range(seeds):
        torch.manual_seed(1000 + s); np.random.seed(1000 + s)   # srg_analysis seeds
        m = BaseModel(SA.make_config(True)).eval()
        if float64:
            torch.set_default_dtype(torch.float64)
            m = m.double(); a, b = xa.double(), xb.double()
        else:
            a, b = xa, xb
        with torch.no_grad():
            d.append((m(a) - m(b)).abs().max().item())
        torch.set_default_dtype(torch.float32)
    return float(np.median(d))


if __name__ == '__main__':
    print("float32 (as in the paper) and the SAME weights in float64:")
    print(f"{'pair':26s} |   G1 vs G2  f32 / f64   | G vs relabeled G, f32 (G1; G2) | f64 (G1; G2)")
    for name, (G1, G2, _) in all_pairs().items():
        r1, r2 = relabel(G1, 7), relabel(G2, 7)
        a32, a64 = diff(G1, G2), diff(G1, G2, float64=True)
        s32 = (diff(G1, r1), diff(G2, r2)); s64 = (diff(G1, r1, float64=True), diff(G2, r2, float64=True))
        print(f"{name:26s} | {a32:9.1e} / {a64:9.1e}   | {s32[0]:9.1e}; {s32[1]:9.1e}           | "
              f"{s64[0]:8.1e}; {s64[1]:8.1e}", flush=True)
    print("A genuine separation is large in BOTH precisions and far above the")
    print("G-vs-relabeled-G values (which must be ~0 for an invariant model).")
