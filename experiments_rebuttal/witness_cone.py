"""
Witness that the LEARNED filtration does work (not only filtration-independent
homology).

For a strongly regular graph, 2-FWL has three pair classes (diagonal, edge,
non-edge), so the PPGN features and hence the learned filtration are constant
within each simplex dimension, and the persistence diagrams are determined by
simplex counts and boundary ranks alone. The coned graph cone(G) = G + an apex
joined to every vertex breaks this: apex edges get their own 2-FWL colour, so
the filtration can order the triangles of G before the apex triangles.

For R = Rook(4,4), S = Shrikhande at D = 2: cone(R), cone(S) are 2-FWL- (3-WL-)
equivalent, have identical simplex counts (17, 64, 80) and identical Betti
numbers (1, 0, 32) of K_2, so every filtration-independent invariant of K_2
agrees -- yet under a filtration in which the triangles of G enter first, they
kill rank d_2(K_2(G)) = 24 (R) versus 31 (S) cycles before the apex triangles
kill the rest. Only a learned, non-constant filtration can expose this.

For every pair (the strongly regular pairs of the paper and their cones) this
script reports:
  - 2-FWL equivalence (direct refinement on the disjoint union),
  - clique counts by size and Betti numbers of K_D(G) for every D,
  - whether ALL filtration-independent statistics of K_D agree for D <= 2,
  - the number of cycles killed by the triangles of G (cones: the mechanism),
  - output differences at random initialization (float64) for PPGN and for
    PPGN+PH at D=2,P=1 (training configuration) and D=3,P=2, over --seeds seeds.

  python experiments_rebuttal/witness_cone.py --seeds=30
Outputs rebuttal_results/witness_cone.json and a LaTeX table on stdout.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
import numpy as np, networkx as nx, torch, gudhi
import srg_analysis as S
from srg_graphs import all_pairs

OUT = os.path.join("rebuttal_results", "witness_cone.json")
THRESH = 1e-5          # separation threshold (as in the paper)


def cone(G):
    H = nx.convert_node_labels_to_integers(G)
    n = H.number_of_nodes()
    H.add_node(n)
    H.add_edges_from((n, v) for v in range(n))
    return H


def two_fwl_equivalent(G1, G2, max_rounds=50):
    """2-FWL (folklore, = 3-WL) on the disjoint union; True iff the stable
    colour histograms of the two graphs coincide."""
    G = nx.disjoint_union(G1, G2)
    n1, N = G1.number_of_nodes(), G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(N)).astype(np.int8)
    comp = np.array([0] * n1 + [1] * (N - n1))
    # initial colour: isomorphism type of the ordered pair (equal, adjacent) + same component
    col = (np.eye(N, dtype=np.int64) * 2 + A).astype(np.int64)
    col = col * 2 + (comp[:, None] == comp[None, :])
    def relabel(keys):
        uniq = {}
        return np.array([uniq.setdefault(k, len(uniq)) for k in keys], dtype=np.int64)
    col = relabel(col.ravel().tolist()).reshape(N, N)
    for _ in range(max_rounds):
        keys = []
        for u in range(N):
            for v in range(N):
                ms = tuple(sorted(zip(col[u, :].tolist(), col[:, v].tolist())))
                keys.append((col[u, v], hash(ms)))
        new = relabel(keys).reshape(N, N)
        if len(np.unique(new)) == len(np.unique(col)):
            col = new
            break
        col = new
    h1 = sorted(col[:n1, :n1].ravel().tolist())
    h2 = sorted(col[n1:, n1:].ravel().tolist())
    return h1 == h2


def clique_counts(G):
    c = {}
    for q in nx.enumerate_all_cliques(G):
        c[len(q)] = c.get(len(q), 0) + 1
    return [c[k] for k in sorted(c)]


def betti_KD(G, D):
    st = gudhi.SimplexTree()
    for q in nx.enumerate_all_cliques(G):
        if len(q) - 1 > D:
            break
        st.insert(q, filtration=0.0)
    st.compute_persistence(persistence_dim_max=True)
    b = list(st.betti_numbers())[:D + 1]
    return b + [0] * (D + 1 - len(b))


def cycles_killed_by_own_triangles(G):
    """rank of the boundary map d_2 of K_2(G) = #edges - #vertices + #components - beta_1(K_2(G))"""
    b = betti_KD(G, 2)
    return G.number_of_edges() - G.number_of_nodes() + b[0] - b[1]


def separation(G1, G2, use_topology, D, P, seeds):
    torch.set_default_dtype(torch.float64)
    x1, x2 = S.graph_to_input(G1), S.graph_to_input(G2)
    out = []
    for s in range(seeds):
        torch.manual_seed(1000 + s); np.random.seed(1000 + s)
        cfg = S.make_config(use_topology)
        cfg.architecture.topo_max_simplex_dim = D
        cfg.architecture.topo_max_ph_dim = P
        m = S.BaseModel(cfg).eval()
        with torch.no_grad():
            out.append(float((m(x1) - m(x2)).abs().max()))
    out = np.array(out)
    return dict(median=float(np.median(out)), max=float(out.max()),
                frac_separated=float((out > THRESH).mean()))


def analyze(name, G1, G2, seeds):
    w = max(max(len(c) for c in nx.find_cliques(G1)), max(len(c) for c in nx.find_cliques(G2)))
    cc1, cc2 = clique_counts(G1), clique_counts(G2)
    betti = {D: (betti_KD(G1, D), betti_KD(G2, D)) for D in range(1, w)}
    def agree_upto(D):
        return (cc1[:D + 1] == cc2[:D + 1]) and all(betti[d][0] == betti[d][1] for d in range(1, D + 1))
    r = dict(
        pair=name, n=[G1.number_of_nodes(), G2.number_of_nodes()],
        two_fwl_equivalent=two_fwl_equivalent(G1, G2),
        clique_counts=[cc1, cc2],
        betti_by_D={str(D): v for D, v in betti.items()},
        betti_equal_every_D=all(b1 == b2 for b1, b2 in betti.values()),
        filtration_independent_agree_D1=agree_upto(1),
        filtration_independent_agree_D2=agree_upto(2),
        ppgn=separation(G1, G2, False, 2, 1, seeds),
        ph_D2P1=separation(G1, G2, True, 2, 1, seeds),
        ph_D3P2=separation(G1, G2, True, 3, 2, seeds),
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    results = []
    for name, v in all_pairs().items():
        G1, G2 = v[0], v[1]
        for label, a, b in [(name, G1, G2), (f"cone({name})", cone(G1), cone(G2))]:
            r = analyze(label, a, b, args.seeds)
            if label.startswith("cone("):
                r["cycles_killed_by_own_triangles"] = [cycles_killed_by_own_triangles(G1),
                                                       cycles_killed_by_own_triangles(G2)]
            results.append(r)
            print(f"{label:38s} 2-FWL-eq {r['two_fwl_equivalent']!s:5s} "
                  f"K_2 stats agree {r['filtration_independent_agree_D2']!s:5s} "
                  f"Betti equal every D {r['betti_equal_every_D']!s:5s} | "
                  f"PPGN {r['ppgn']['median']:.1e}  PH(2,1) {r['ph_D2P1']['median']:.1e} "
                  f"[{r['ph_D2P1']['frac_separated']:.0%}]  PH(3,2) {r['ph_D3P2']['median']:.1e} "
                  f"[{r['ph_D3P2']['frac_separated']:.0%}]", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(seeds=args.seeds, threshold=THRESH, results=results), open(OUT, "w"), indent=1)
    print("\n% LaTeX rows: pair & K_2 statistics agree & PPGN & PPGN+PH (D=2,P=1) & PPGN+PH (D=3,P=2)")
    for r in results:
        fmt = lambda d: f"${d['median']:.1e}$ ({d['frac_separated']:.0%})".replace("%", "\\%")
        print(f"{r['pair']} & {'yes' if r['filtration_independent_agree_D2'] else 'no'} & "
              f"{fmt(r['ppgn'])} & {fmt(r['ph_D2P1'])} & {fmt(r['ph_D3P2'])} \\\\")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
