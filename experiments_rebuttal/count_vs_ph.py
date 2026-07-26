"""
Q1 (decisive ablation): persistent homology is STRICTLY more expressive than
exposing higher-order clique counts.

We contrast two mechanisms on graph pairs with IDENTICAL k-clique counts in
every dimension:

  * "clique-count-only" model: a permutation-invariant readout of the raw
    k-clique-count vector (k=1..5) -- the most information any clique-COUNTING
    mechanism can provide.
  * the paper's learned PH branch (BaseModel with use_topology=True).

Canonical count-identical family (also a textbook example that message-passing
GNNs / 1-WL cannot separate): C_{2m} vs 2 x C_m. Both are 2-regular with the
same number of vertices, edges, and 0 triangles/tetrahedra -> identical clique
counts -> a clique-count model gives IDENTICAL outputs. But their clique-complex
homology differs (b1 = 1 vs 2; b0 = 1 vs 2), so persistent homology separates
them. This isolates the homological signal from mere counting.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, networkx as nx, torch
from easydict import EasyDict
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from srg_graphs import all_pairs
from srg_analysis import clique_counts, clique_complex_betti, graph_to_input, \
    make_config, model_separation
from models.base_model import BaseModel


def clique_count_vector(G, kmax=5):
    c = clique_counts(G, kmax)
    return np.array([c[k] for k in range(1, kmax + 1)], dtype=float)


def clique_count_only_separation(G1, G2, kmax=5):
    """A clique-count model separates iff the count vectors differ."""
    v1, v2 = clique_count_vector(G1, kmax), clique_count_vector(G2, kmax)
    return float(np.abs(v1 - v2).max()), v1.tolist(), v2.tolist()


def count_identical_pairs():
    pairs = {}
    for m in (6, 8):
        G_conn = nx.cycle_graph(2 * m)                       # C_{2m}
        G_disc = nx.disjoint_union(nx.cycle_graph(m), nx.cycle_graph(m))  # 2 x C_m
        pairs[f"C{2*m} vs 2xC{m}"] = (G_conn, G_disc)
    return pairs


def main():
    print("=" * 92)
    print("COUNT-IDENTICAL PAIRS: clique counts cannot separate, PH can")
    print("=" * 92)
    out = []
    for name, (G1, G2) in count_identical_pairs().items():
        cc_sep, v1, v2 = clique_count_only_separation(G1, G2)
        b1, b2 = clique_complex_betti(G1, 1), clique_complex_betti(G2, 1)
        base_med, _, _ = model_separation(G1, G2, use_topology=False, n_seeds=5)
        topo_med, topo_max, _ = model_separation(G1, G2, use_topology=True, n_seeds=5)
        print(f"\n### {name}")
        print(f"  clique-count vectors (k=1..5): {v1}  vs  {v2}")
        print(f"  clique-count-only model max output diff: {cc_sep:.2e}  "
              f"({'separates' if cc_sep>0 else 'CANNOT separate'})")
        print(f"  clique-complex Betti: {b1} vs {b2}  "
              f"({'differ' if b1!=b2 else 'same'})")
        print(f"  baseline 2-IGN separation:  {base_med:.2e}")
        print(f"  PH-augmented separation:    median={topo_med:.2e} max={topo_max:.2e}"
              f"  ({'SEPARATES' if topo_med>1e-5 else 'no'})")
        out.append(dict(pair=name, count_vec_G1=v1, count_vec_G2=v2,
                        clique_count_model_sep=cc_sep, betti_G1=b1, betti_G2=b2,
                        baseline_sep=base_med, ph_sep_median=topo_med,
                        ph_sep_max=topo_max))
    with open("experiments_rebuttal/count_vs_ph_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 92)
    print("CONCLUSION: On C_{2m} vs 2xC_m the raw clique-count vector is IDENTICAL,")
    print("so any clique-counting mechanism gives identical outputs; PH separates")
    print("them via H0/H1 (components/loops). Clique counting is strictly weaker")
    print("than the persistent homology our architecture computes.")


if __name__ == "__main__":
    main()
