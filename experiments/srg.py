import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import networkx as nx
import torch, gudhi
from itertools import combinations, product
from easydict import EasyDict

from models.base_model import BaseModel
from runner import FULL

OUT_DIR = os.path.join(ROOT, 'results', 'expr')
THRESH = 1e-5

def gf_mul(x, y, p, n):
    # GF(p^2) as pairs (a, b) = a + b t with t^2 = n
    a, b = x
    c, d = y
    return ((a * c + b * d * n) % p, (a * d + b * c) % p)

def cayley(p, conn):
    els = [(a, b) for a in range(p) for b in range(p)]
    g = nx.Graph()
    g.add_nodes_from(range(len(els)))
    for(i, j) in combinations(range(len(els)), 2):
        x, y = els[i], els[j]
        diff = ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
        if(diff in conn):
            g.add_edge(i, j)
    return g

def rook_graph(n):
    cells = list(product(range(n), range(n)))
    idx = {c: i for i, c in enumerate(cells)}
    g = nx.Graph()
    g.add_nodes_from(range(n * n))
    for((r1, c1), (r2, c2)) in combinations(cells, 2):
        if(r1 == r2 or c1 == c2):
            g.add_edge(idx[(r1, c1)], idx[(r2, c2)])
    return g

def shrikhande_graph():
    verts = list(product(range(4), range(4)))
    idx = {v: i for i, v in enumerate(verts)}
    shifts = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    g = nx.Graph()
    g.add_nodes_from(range(16))
    for v in verts:
        for c in shifts:
            w = ((v[0] + c[0]) % 4, (v[1] + c[1]) % 4)
            g.add_edge(idx[v], idx[w])
    return g

def triangular_graph(n):
    # line graph of K_n
    pairs = list(combinations(range(n), 2))
    idx = {p: i for i, p in enumerate(pairs)}
    g = nx.Graph()
    g.add_nodes_from(range(len(pairs)))
    for(a, b) in combinations(pairs, 2):
        if(len(set(a) & set(b)) == 1):
            g.add_edge(idx[a], idx[b])
    return g

def seidel_switch(G, S):
    S = set(S)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for(u, v) in combinations(G.nodes(), 2):
        e = G.has_edge(u, v)
        if((u in S) != (v in S)):
            e = not e
        if(e):
            H.add_edge(u, v)
    return H

def chang_graphs():
    # Seidel switch T(8) on a perfect matching, C8, and K3+C5 inside K8
    idx = {p: i for i, p in enumerate(combinations(range(8), 2))}
    T = triangular_graph(8)
    switch_sets = [[(0, 1), (2, 3), (4, 5), (6, 7)], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)], [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 6), (6, 7), (3, 7)]]
    chang = []
    for es in switch_sets:
        verts = [idx[tuple(sorted(e))] for e in es]
        chang.append(seidel_switch(T, verts))
    return chang

def paley_graph(q):
    p, n = {25: (5, 2), 49: (7, 3)}[q]
    els = [(a, b) for a in range(p) for b in range(p)]
    squares = set()
    for e in els:
        if(e != (0, 0)):
            squares.add(gf_mul(e, e, p, n))
    return cayley(p, squares)

def latin_square_L3_5():
    n = 5
    cells = list(product(range(n), range(n)))
    idx = {c: i for i, c in enumerate(cells)}
    L = [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2], [2, 3, 4, 0, 1], [3, 4, 1, 2, 0], [4, 2, 0, 1, 3]]
    g = nx.Graph()
    g.add_nodes_from(range(n * n))
    for((r1, c1), (r2, c2)) in combinations(cells, 2):
        if(r1 == r2 or c1 == c2 or L[r1][c1] == L[r2][c2]):
            g.add_edge(idx[(r1, c1)], idx[(r2, c2)])
    return g

def peisert_graph(q=49):
    # connection set is g^j for j = 0,1 mod 4, g primitive in GF(49)
    p, n = 7, 3
    cands = [(a, b) for a in range(p) for b in range(p)][1:]
    for gen in cands:
        x = (1, 0)
        seen = set()
        for _ in range(q - 1):
            x = gf_mul(x, gen, p, n)
            seen.add(x)
        if(len(seen) == q - 1):
            break
    conn = set()
    x = (1, 0)
    for j in range(q - 1):
        if(j % 4 in (0, 1)):
            conn.add(x)
        x = gf_mul(x, gen, p, n)
    return cayley(p, conn)

def cone(G):
    H = nx.convert_node_labels_to_integers(G)
    n = H.number_of_nodes()
    H.add_node(n)
    H.add_edges_from((n, v) for v in range(n))
    return H

def all_pairs():
    T8 = triangular_graph(8)
    ch = chang_graphs()
    return {"Rook(4,4) vs Shrikhande": (rook_graph(4), shrikhande_graph(), (16, 6, 2, 2)), "T(8) vs Chang_1": (T8, ch[0], (28, 12, 6, 4)), "T(8) vs Chang_2": (T8, ch[1], (28, 12, 6, 4)), "T(8) vs Chang_3": (T8, ch[2], (28, 12, 6, 4)), "Paley(25) vs L3(5)": (paley_graph(25), latin_square_L3_5(), (25, 12, 5, 6)), "Paley(49) vs Peisert(49)": (paley_graph(49), peisert_graph(49), (49, 24, 11, 12))}

def srg_params(G):
    n = G.number_of_nodes()
    degs = set(d for _, d in G.degree())
    if(len(degs) != 1):
        return None
    A = nx.to_numpy_array(G, nodelist=range(n))
    A2 = A @ A
    lam = set()
    mu = set()
    for(u, v) in combinations(range(n), 2):
        common = int(A2[u, v])
        if(A[u, v] > 0):
            lam.add(common)
        else:
            mu.add(common)
    if(len(lam) == 1 and len(mu) == 1):
        return (n, degs.pop(), lam.pop(), mu.pop())
    return None

def spectrum(G):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    return np.round(np.sort(np.linalg.eigvalsh(A)), 4)

def check_pairs():
    bad = 0
    for(name, (G1, G2, exp)) in all_pairs().items():
        p1 = srg_params(G1)
        p2 = srg_params(G2)
        cosp = np.allclose(spectrum(G1), spectrum(G2))
        iso = nx.is_isomorphic(G1, G2)
        ok = p1 == exp and p2 == exp and cosp and not iso
        if(not ok):
            bad += 1
        status = 'ok' if ok else 'FAIL'
        print("%s: %s %s (want %s) cospectral=%s iso=%s  %s" % (name, p1, p2, exp, cosp, iso, status))
    if(not bad):
        print("all good")
    else:
        print("%d pairs failed" % bad)

def clique_counts(G, kmax=5):
    counts = {k: 0 for k in range(1, kmax + 1)}
    for c in nx.enumerate_all_cliques(G):
        if(len(c) <= kmax):
            counts[len(c)] += 1
    return counts

def cc_betti(G, max_dim=3):
    st = gudhi.SimplexTree()
    for c in nx.enumerate_all_cliques(G):
        if(len(c) <= max_dim + 1):
            st.insert(list(c), filtration=0.0)
    st.compute_persistence(persistence_dim_max=True)
    return st.betti_numbers()

def to_input(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)).astype(np.float32)
    deg = np.sqrt(A.sum(0))
    dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    norm = np.diag(dinv) @ A @ np.diag(dinv)
    x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    return x.to(torch.get_default_dtype())

def make_config(use_topo):
    cfg = EasyDict({"dataset_name": "SRG", "node_labels": 0, "num_classes": 2, "architecture": {"block_features": [64, 64], "depth_of_mlp": 2, "use_topology": use_topo, "topo_hidden_dim": 32, "topo_max_ph_dim": 2, "topo_num_stats": 16, "topo_max_simplex_dim": 3}})
    if(use_topo):
        cfg.architecture.update(FULL)
    return cfg

def model_sep(G1, G2, use_topo, n_seeds=5):
    x1 = to_input(G1)
    x2 = to_input(G2)
    diffs = []
    for s in range(n_seeds):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        m = BaseModel(make_config(use_topo))
        m.eval()
        with torch.no_grad():
            o1 = m(x1)
            o2 = m(x2)
        diffs.append((o1 - o2).abs().max().item())
    return float(np.median(diffs)), float(np.max(diffs)), diffs

def analysis():
    rows = []
    for(name, (G1, G2, params)) in all_pairs().items():
        c1 = clique_counts(G1)
        c2 = clique_counts(G2)
        b1 = cc_betti(G1)
        b2 = cc_betti(G2)
        base_med, base_max, _ = model_sep(G1, G2, False)
        topo_med, topo_max, _ = model_sep(G1, G2, True)
        row = dict(pair=name, params=params, cliques_G1=c1, cliques_G2=c2, counts_equal=(c1 == c2), betti_G1=b1, betti_G2=b2, betti_equal=(b1 == b2), baseline_sep_median=base_med, baseline_sep_max=base_max, topo_sep_median=topo_med, topo_sep_max=topo_max)
        rows.append(row)

        print("\n%s  srg%s" % (name, params))
        print("  cliques k=1..5: %s | %s  same=%s" % (c1, c2, c1 == c2))
        print("  betti: {} | {}  same={}".format(b1, b2, b1 == b2))
        print("  ppgn sep  med={:.2e} max={:.2e}".format(base_med, base_max))
        if(topo_med > THRESH):
            verdict = 'separated'
        else:
            verdict = 'not separated'
        print(f"  ph sep    med={topo_med:.2e} max={topo_max:.2e}  {verdict}")
    with open(os.path.join(OUT_DIR, 'srg_results.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n%-28s%-16s%-16s" % ('pair', 'counts differ', 'ph separates'))
    for r in rows:
        cdiff = 'no' if r['counts_equal'] else 'yes'
        ph_sep = 'yes' if r['topo_sep_median'] > THRESH else 'no'
        print("%-28s%-16s%-16s" % (r['pair'], cdiff, ph_sep))

def seed_sweep(N, thr=1e-6):
    rows = []
    print("separation over {} seeds, thr {:.0e}".format(N, thr))
    for(name, (G1, G2, _)) in all_pairs().items():
        _, b_max, _ = model_sep(G1, G2, False, n_seeds=N)
        t_med, t_max, t_all = model_sep(G1, G2, True, n_seeds=N)
        t_all = np.array(t_all)
        frac = float((t_all > thr).mean())
        rows.append(dict(pair=name, baseline_max=b_max, topo_median=t_med, topo_min=float(t_all.min()), topo_max=t_max, frac_above_thr=frac, topo_all=t_all.tolist()))
        print(f"{name:<26} base max {b_max:.1e}  topo med {t_med:.1e} min {t_all.min():.1e} max {t_max:.1e}  {frac * 100:.0f}% above")
    json.dump(rows, open(os.path.join(OUT_DIR, 'srg_seeds_results.json'), 'w'), indent=2, default=str)

def count_vec(G, kmax=5):
    c = clique_counts(G, kmax)
    vec = []
    for k in range(1, kmax + 1):
        vec.append(c[k])
    return np.array(vec, dtype=float)

def count_vs_ph():
    # C_2m vs 2 x C_m: same clique counts in every dim but different b0, b1
    out = []
    for m in (6, 8):
        name = f"C{2*m} vs 2xC{m}"
        G1 = nx.cycle_graph(2 * m)
        G2 = nx.disjoint_union(nx.cycle_graph(m), nx.cycle_graph(m))

        v1 = count_vec(G1)
        v2 = count_vec(G2)
        cc_sep = float(np.abs(v1 - v2).max())
        b1 = cc_betti(G1, 1)
        b2 = cc_betti(G2, 1)

        base_med, _, _ = model_sep(G1, G2, use_topo=False, n_seeds=5)
        topo_med, topo_max, _ = model_sep(G1, G2, use_topo=True, n_seeds=5)
        print(name)
        print("  counts", v1.tolist(), v2.tolist(), " count model diff %.2e" % cc_sep)
        print("  betti", b1, b2)
        print(f"  ppgn {base_med:.2e}   ph med {topo_med:.2e} max {topo_max:.2e}")
        out.append(dict(pair=name, count_vec_G1=v1.tolist(), count_vec_G2=v2.tolist(), clique_count_model_sep=cc_sep, betti_G1=b1, betti_G2=b2, baseline_sep=base_med, ph_sep_median=topo_med, ph_sep_max=topo_max))
    with open(os.path.join(OUT_DIR, "count_vs_ph_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

def relabel(keys):
    seen = {}
    labels = []
    for k in keys:
        if(k not in seen):
            seen[k] = len(seen)
        labels.append(seen[k])
    return np.array(labels, dtype=np.int64)

def fwl2_equiv(G1, G2, max_rounds=50):
    G = nx.disjoint_union(G1, G2)
    n1 = G1.number_of_nodes()
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(N)).astype(np.int8)
    comp = np.array([0] * n1 + [1] * (N - n1))

    col = (np.eye(N, dtype=np.int64) * 2 + A).astype(np.int64)
    col = col * 2 + (comp[:, None] == comp[None, :])
    col = relabel(col.ravel().tolist()).reshape(N, N)

    for _ in range(max_rounds):
        keys = []
        for u in range(N):
            for v in range(N):
                ms = tuple(sorted(zip(col[u, :].tolist(), col[:, v].tolist())))
                keys.append((col[u, v], hash(ms)))
        new = relabel(keys).reshape(N, N)
        done = len(np.unique(new)) == len(np.unique(col))
        col = new
        if(done):
            break
    hist1 = sorted(col[:n1, :n1].ravel().tolist())
    hist2 = sorted(col[n1:, n1:].ravel().tolist())
    return hist1 == hist2

def cc_list(G):
    c = {}
    for q in nx.enumerate_all_cliques(G):
        c[len(q)] = c.get(len(q), 0) + 1
    return [c[k] for k in sorted(c)]

def betti_KD(G, D):
    st = gudhi.SimplexTree()
    for q in nx.enumerate_all_cliques(G):
        if(len(q) - 1 > D):
            break
        st.insert(q, filtration=0.0)
    st.compute_persistence(persistence_dim_max=True)
    b = list(st.betti_numbers())[:D + 1]
    return b + [0] * (D + 1 - len(b))

def own_tri_kills(G):
    # rank d_2 of K_2(G) = E - V + b0 - b1
    b = betti_KD(G, 2)
    return G.number_of_edges() - G.number_of_nodes() + b[0] - b[1]

def sep(G1, G2, use_topo, D, P, seeds):
    torch.set_default_dtype(torch.float64)
    x1 = to_input(G1)
    x2 = to_input(G2)
    out = []
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        cfg = make_config(use_topo)
        cfg.architecture.topo_max_simplex_dim = D
        cfg.architecture.topo_max_ph_dim = P
        m = BaseModel(cfg).eval()
        with torch.no_grad():
            diff = (m(x1) - m(x2)).abs().max()
        out.append(float(diff))
    out = np.array(out)
    return dict(median=float(np.median(out)), max=float(out.max()), frac_separated=float((out > THRESH).mean()))

def analyze_pair(name, G1, G2, seeds):
    w = 0
    for G in (G1, G2):
        w = max(w, max(len(c) for c in nx.find_cliques(G)))
    cc1 = cc_list(G1)
    cc2 = cc_list(G2)
    betti = {}
    for D in range(1, w):
        betti[D] = (betti_KD(G1, D), betti_KD(G2, D))

    def agree(D):
        if(cc1[:D + 1] != cc2[:D + 1]):
            return False
        return all(betti[d][0] == betti[d][1] for d in range(1, D + 1))

    return dict(pair=name, n=[G1.number_of_nodes(), G2.number_of_nodes()], two_fwl_equivalent=fwl2_equiv(G1, G2), clique_counts=[cc1, cc2], betti_by_D={str(D): v for D, v in betti.items()}, betti_equal_every_D=all(b1 == b2 for b1, b2 in betti.values()), filtration_independent_agree_D1=agree(1), filtration_independent_agree_D2=agree(2), ppgn=sep(G1, G2, False, 2, 1, seeds), ph_D2P1=sep(G1, G2, True, 2, 1, seeds), ph_D3P2=sep(G1, G2, True, 3, 2, seeds))

def tex_cell(d):
    s = "${:.1e}$ ({:.0%})".format(d['median'], d['frac_separated'])
    return s.replace("%", "\\%")

def witness_cone(n_seeds):
    res = []
    for(name, (G1, G2, _)) in all_pairs().items():
        jobs = [(name, G1, G2), ("cone(%s)" % name, cone(G1), cone(G2))]
        for(label, a, b) in jobs:
            r = analyze_pair(label, a, b, n_seeds)
            if(label.startswith("cone(")):
                r["cycles_killed_by_own_triangles"] = [own_tri_kills(G1), own_tri_kills(G2)]
            res.append(r)
            ph21, ph32 = r['ph_D2P1'], r['ph_D3P2']
            print(f"{label:38s} 2fwl-eq {r['two_fwl_equivalent']!s:5s} K2 agree {r['filtration_independent_agree_D2']!s:5s} betti eq {r['betti_equal_every_D']!s:5s} | ppgn {r['ppgn']['median']:.1e}  ph(2,1) {ph21['median']:.1e} [{ph21['frac_separated']:.0%}]  ph(3,2) {ph32['median']:.1e} [{ph32['frac_separated']:.0%}]", flush=True)
    out = os.path.join(OUT_DIR, "witness_cone.json")
    json.dump(dict(seeds=n_seeds, threshold=THRESH, results=res), open(out, "w"), indent=1)

    for r in res:
        k2 = 'yes' if r['filtration_independent_agree_D2'] else 'no'
        print("%s & %s & %s & %s & %s \\\\" % (r['pair'], k2, tex_cell(r['ppgn']), tex_cell(r['ph_D2P1']), tex_cell(r['ph_D3P2'])))
    print("wrote", out)

if(__name__ == "__main__"):
    argv = sys.argv[1:]
    mode = 'srg'
    if(len(argv) > 0 and not argv[0].startswith('-')):
        mode = argv.pop(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None)
    args = ap.parse_args(argv)

    if(mode not in ('srg', 'count', 'cone', 'check')):
        print("unknown mode %s, use one of: count, cone, check (or nothing)" % mode)
        sys.exit(1)
    if(mode == 'check'):
        check_pairs()
        sys.exit(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    if(mode == 'srg'):
        if(args.seeds):
            seed_sweep(args.seeds)
        else:
            analysis()
    elif(mode == 'count'):
        count_vs_ph()
    elif(mode == 'cone'):
        if(args.seeds is None):
            args.seeds = 30
        witness_cone(args.seeds)
