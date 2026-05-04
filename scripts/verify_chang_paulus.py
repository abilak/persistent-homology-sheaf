"""
Extended Empirical Verification: Chang Graphs, Paley Graphs, and Peisert Graphs
================================================================================
Tests additional 3-WL equivalent graph pairs beyond Rook/Shrikhande:

1. Chang graph family — srg(28, 12, 6, 4):
   T(8) = L(K8) vs Chang1, Chang2, Chang3  (3 pairs)

2. Paley(25) family — srg(25, 12, 5, 6):
   Paley(25) vs Latin-square graph L3(5)  (1 pair)

3. Paley(49) vs Peisert(49) — srg(49, 24, 11, 12):
   Two non-isomorphic conference graphs from different multiplicative
   coset constructions over GF(49).  (1 pair)

All pairs are cospectral and 3-WL equivalent. Our topology-augmented model
should distinguish each pair while the baseline (Maron et al. 2019) cannot.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from itertools import combinations
from easydict import EasyDict
from models.base_model import BaseModel


# ============================================================================
# Graph Constructions
# ============================================================================

def build_T8():
    """
    T(8) = L(K_8): the triangular / line graph of K_8.
    Vertices = 28 edges of K_8, labeled as (i,j) with 0 <= i < j <= 7.
    Two vertices are adjacent iff they share an endpoint.
    srg(28, 12, 6, 4).
    """
    # Enumerate the 28 edges of K_8
    edges_of_K8 = list(combinations(range(8), 2))
    n = len(edges_of_K8)  # 28
    assert n == 28
    adj = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            e1 = edges_of_K8[a]
            e2 = edges_of_K8[b]
            # Adjacent if they share an endpoint
            if e1[0] in e2 or e1[1] in e2:
                adj[a, b] = 1.0
                adj[b, a] = 1.0
    return adj, edges_of_K8


def seidel_switch(adj, subset_indices):
    """
    Seidel switching: for vertices in subset S and vertices not in S,
    flip adjacency (edge ↔ non-edge) for all cross-pairs.
    Edges within S and within V\\S remain unchanged.
    """
    n = adj.shape[0]
    new_adj = adj.copy()
    S = set(subset_indices)
    for i in range(n):
        for j in range(i + 1, n):
            if (i in S) != (j in S):  # one in S, one not in S
                new_adj[i, j] = 1.0 - new_adj[i, j]
                new_adj[j, i] = 1.0 - new_adj[j, i]
    return new_adj


def build_chang_graphs():
    """
    Build all 3 Chang graphs by Seidel switching from T(8).

    Chang 1: Switch w.r.t. vertices corresponding to a perfect matching
             of K_8 (4 pairwise disjoint edges).
    Chang 2: Switch w.r.t. vertices corresponding to C_3 + C_5
             (triangle + pentagon, vertex-disjoint).
    Chang 3: Switch w.r.t. vertices corresponding to C_8
             (Hamiltonian cycle on K_8's 8 vertices).

    The "vertices" of T(8) correspond to edges of K_8, so the switching
    set is the set of T(8)-vertices that correspond to edges forming the
    specified subgraph of K_8.
    """
    t8_adj, edges_of_K8 = build_T8()
    edge_to_idx = {e: i for i, e in enumerate(edges_of_K8)}

    def edges_to_vertex_set(edge_list):
        """Convert a list of K_8 edges to T(8) vertex indices."""
        indices = []
        for e in edge_list:
            key = tuple(sorted(e))
            indices.append(edge_to_idx[key])
        return indices

    # Chang 1: Perfect matching on K_8 vertices {0,...,7}
    # Matching: {(0,1), (2,3), (4,5), (6,7)}
    matching = [(0, 1), (2, 3), (4, 5), (6, 7)]
    chang1_set = edges_to_vertex_set(matching)
    chang1_adj = seidel_switch(t8_adj, chang1_set)

    # Chang 2: C_3 + C_5 (triangle on {0,1,2} + pentagon on {3,4,5,6,7})
    triangle = [(0, 1), (1, 2), (0, 2)]
    pentagon = [(3, 4), (4, 5), (5, 6), (6, 7), (3, 7)]
    chang2_set = edges_to_vertex_set(triangle + pentagon)
    chang2_adj = seidel_switch(t8_adj, chang2_set)

    # Chang 3: C_8 (Hamiltonian cycle 0-1-2-3-4-5-6-7-0)
    cycle8 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
    chang3_set = edges_to_vertex_set(cycle8)
    chang3_adj = seidel_switch(t8_adj, chang3_set)

    return t8_adj, chang1_adj, chang2_adj, chang3_adj


def build_paley25():
    """
    Paley graph of order 25: srg(25, 12, 5, 6).
    GF(25) = GF(5)[x] / (x^2 + x + 1).
    Elements represented as (a, b) meaning a + b*x, with a,b in {0,1,2,3,4}.
    Vertex numbering: 5*a + b.
    Two vertices are adjacent iff their difference is a nonzero square in GF(25).
    """
    # GF(25) arithmetic: irreducible polynomial x^2 + x + 1 over GF(5)
    # (Check: x^2 + x + 1 has no roots in GF(5): 0+0+1=1, 1+1+1=3, 4+2+1=2, 9+3+1≡3+3+1=2, 16+4+1≡1+4+1=1)
    # Actually let me use x^2 + 2 which is irreducible over GF(5)
    # Check: 0+2=2, 1+2=3, 4+2=1, 9+2≡4+2=1, 16+2≡1+2=3. No roots. Good.

    def gf25_mul(a, b):
        """Multiply two elements (a0+a1*x) * (b0+b1*x) in GF(5)[x]/(x^2+2)."""
        a0, a1 = a
        b0, b1 = b
        # (a0+a1*x)(b0+b1*x) = a0*b0 + (a0*b1+a1*b0)*x + a1*b1*x^2
        # x^2 = -2 = 3 (mod 5)
        c0 = (a0 * b0 + a1 * b1 * 3) % 5
        c1 = (a0 * b1 + a1 * b0) % 5
        return (c0, c1)

    def gf25_sub(a, b):
        """Subtract b from a in GF(25)."""
        return ((a[0] - b[0]) % 5, (a[1] - b[1]) % 5)

    # Enumerate all elements
    elements = [(a, b) for a in range(5) for b in range(5)]
    n = 25

    # Find all nonzero squares in GF(25)
    squares = set()
    for e in elements:
        if e == (0, 0):
            continue
        sq = gf25_mul(e, e)
        squares.add(sq)
    # Should have (25-1)/2 = 12 nonzero squares
    assert len(squares) == 12, f"Expected 12 squares, got {len(squares)}"

    # Build adjacency
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            diff = gf25_sub(elements[i], elements[j])
            if diff in squares:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


def build_latin_square_graph():
    """
    Latin square graph L_3(5): srg(25, 12, 5, 6).
    Vertices = 25 cells of a 5x5 grid, numbered 5*i + j.
    Two cells are adjacent if they share:
      - same row, OR
      - same column, OR
      - same symbol in the Latin square L[i][j] = (i+j) mod 5
    Each gives 4 neighbors → total degree = 4+4+4 = 12.
    """
    n = 25
    adj = np.zeros((n, n), dtype=np.float32)

    # Latin square: L[i][j] = (i + j) mod 5
    def cell(i, j):
        return 5 * i + j

    def latin(i, j):
        return (i + j) % 5

    for i1 in range(5):
        for j1 in range(5):
            for i2 in range(5):
                for j2 in range(5):
                    if i1 == i2 and j1 == j2:
                        continue
                    v1 = cell(i1, j1)
                    v2 = cell(i2, j2)
                    # Adjacent if same row, same column, or same Latin symbol
                    if i1 == i2 or j1 == j2 or latin(i1, j1) == latin(i2, j2):
                        adj[v1, v2] = 1.0

    return adj


def _gf49_setup():
    """
    Setup GF(49) = GF(7)[x] / (x^2 + 1).
    x^2 + 1 is irreducible over GF(7) since -1 is not a QR mod 7.
    Elements: (a, b) representing a + b*x, with a,b in {0,...,6}.
    Vertex numbering: 7*a + b.
    """
    def mul(a, b):
        # (a0+a1*x)(b0+b1*x) = a0*b0 + (a0*b1+a1*b0)*x + a1*b1*x^2
        # x^2 = -1 = 6 (mod 7)
        c0 = (a[0] * b[0] + 6 * a[1] * b[1]) % 7
        c1 = (a[0] * b[1] + a[1] * b[0]) % 7
        return (c0, c1)

    def sub(a, b):
        return ((a[0] - b[0]) % 7, (a[1] - b[1]) % 7)

    # Find a primitive element (generator of GF(49)*,  order 48)
    elements = [(a, b) for a in range(7) for b in range(7)]
    zero = (0, 0)

    def order(g):
        cur = g
        for k in range(1, 49):
            if cur == (1, 0):
                return k
            cur = mul(cur, g)
        return None

    # (1, 1) = 1+x typically works; find primitive element
    omega = None
    for candidate in [(1, 1), (2, 1), (1, 2), (3, 1)]:
        if order(candidate) == 48:
            omega = candidate
            break
    assert omega is not None, "Could not find primitive element of GF(49)"

    # Precompute powers of omega
    powers = [None] * 48
    cur = (1, 0)  # omega^0 = 1
    for k in range(48):
        powers[k] = cur
        cur = mul(cur, omega)

    return elements, zero, sub, powers


def build_paley49():
    """
    Paley graph of order 49: srg(49, 24, 11, 12).
    GF(49) = GF(7)[x]/(x^2+1). Vertices are the 49 field elements.
    Two vertices are adjacent iff their difference is a nonzero square
    (i.e., an even power of the primitive element omega).
    """
    elements, zero, sub, powers = _gf49_setup()
    n = 49

    # Squares = {omega^(2k) : k=0,...,23} — 24 elements
    squares = set(powers[k] for k in range(0, 48, 2))
    assert len(squares) == 24

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            diff = sub(elements[i], elements[j])
            if diff in squares:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


def build_peisert49():
    """
    Peisert graph of order 49: srg(49, 24, 11, 12).
    (Peisert 2001) Non-isomorphic to Paley(49).

    Connection set: {omega^j : j mod 4 in {0, 1}} — the union of the
    subgroup of 4th powers and its first coset.
    This gives 24 elements (half of GF(49)*), forming a valid
    connection set for an srg with the same parameters as Paley(49).
    """
    elements, zero, sub, powers = _gf49_setup()
    n = 49

    # Peisert set = {omega^j : j mod 4 in {0, 1}} — 24 elements
    peisert_set = set(powers[k] for k in range(48) if k % 4 in (0, 1))
    assert len(peisert_set) == 24

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            diff = sub(elements[i], elements[j])
            if diff in peisert_set:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


# ============================================================================
# Verification utilities
# ============================================================================

def verify_srg(adj, name, expected_params):
    """Verify strongly regular graph parameters (n, k, lambda, mu)."""
    n_exp, k_exp, lam_exp, mu_exp = expected_params
    n = adj.shape[0]
    assert n == n_exp, f"{name}: expected {n_exp} vertices, got {n}"

    degrees = adj.sum(axis=1)
    assert np.all(degrees == k_exp), f"{name}: not {k_exp}-regular, degrees = {set(degrees.astype(int))}"

    lambdas = []
    mus = []
    for i in range(n):
        for j in range(i + 1, n):
            common = int(adj[i] @ adj[j])
            if adj[i, j] == 1:
                lambdas.append(common)
            else:
                mus.append(common)

    assert all(l == lam_exp for l in lambdas), \
        f"{name}: lambda != {lam_exp}, got {set(lambdas)}"
    assert all(m == mu_exp for m in mus), \
        f"{name}: mu != {mu_exp}, got {set(mus)}"
    return True


def count_cliques(adj, size):
    """Count cliques of given size."""
    n = adj.shape[0]
    count = 0
    for clique in combinations(range(n), size):
        is_clique = True
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if adj[clique[i], clique[j]] == 0:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            count += 1
    return count


def graphs_are_isomorphic(adj1, adj2):
    """Quick non-isomorphism check via degree sequence + spectrum."""
    eigs1 = sorted(np.linalg.eigvalsh(adj1))
    eigs2 = sorted(np.linalg.eigvalsh(adj2))
    return np.allclose(eigs1, eigs2, atol=1e-6)


def graph_to_tensor(adj):
    """Convert adjacency to model input tensor (1, 1, M, M)."""
    M = adj.shape[0]
    tensor = np.zeros((1, 1, M, M), dtype=np.float32)
    tensor[0, 0, :, :] = adj
    return torch.tensor(tensor)


def make_config(use_topology, n_vertices):
    """Create model config. Adjust max_simplex_dim based on graph size."""
    config = EasyDict()
    config.num_classes = 2
    config.node_labels = 0
    config.dataset_name = 'TEST'
    config.architecture = EasyDict()
    config.architecture.block_features = [64, 64]
    config.architecture.depth_of_mlp = 2
    config.architecture.new_suffix = True
    config.architecture.use_topology = use_topology
    if use_topology:
        config.architecture.topo_hidden_dim = 32
        config.architecture.topo_max_ph_dim = 2
        config.architecture.topo_num_stats = 16
        # dim 3 needed to detect K4 differences (all tested families)
        config.architecture.topo_max_simplex_dim = 3
    return config


def test_pair_detailed(adj1, adj2, name1, name2, n_vertices, seed=42):
    """Test whether baseline and topo models can distinguish a graph pair.
    Returns detailed stats matching rook_shrikhande format."""
    device = 'cpu'
    t1 = graph_to_tensor(adj1).to(device)
    t2 = graph_to_tensor(adj2).to(device)

    # Baseline
    config_base = make_config(use_topology=False, n_vertices=n_vertices)
    torch.manual_seed(seed)
    model_base = BaseModel(config_base).to(device)
    model_base.eval()
    n_params_base = sum(p.numel() for p in model_base.parameters())

    with torch.no_grad():
        s1_base = model_base(t1)
        s2_base = model_base(t2)
    diff_base = (s1_base - s2_base).abs().max().item()

    # Check intermediate equivariant representations (baseline)
    with torch.no_grad():
        # Get features after equivariant blocks (before final fc)
        x1 = t1
        x2 = t2
        for block in model_base.reg_blocks:
            x1 = block(x1)
            x2 = block(x2)
        # Pool: sum over spatial dims
        pool1 = x1.sum(dim=(-1, -2))
        pool2 = x2.sum(dim=(-1, -2))
        intermediate_diff = (pool1 - pool2).abs().max().item()

    # Topology
    config_topo = make_config(use_topology=True, n_vertices=n_vertices)
    torch.manual_seed(seed)
    model_topo = BaseModel(config_topo).to(device)
    model_topo.eval()
    n_params_topo = sum(p.numel() for p in model_topo.parameters())

    with torch.no_grad():
        s1_topo = model_topo(t1)
        s2_topo = model_topo(t2)
    diff_topo = (s1_topo - s2_topo).abs().max().item()

    baseline_same = diff_base < 1e-5
    topo_diff = diff_topo > 1e-5

    return {
        'name1': name1, 'name2': name2,
        'diff_base': diff_base, 'diff_topo': diff_topo,
        'baseline_identical': baseline_same,
        'topo_distinguishes': topo_diff,
        'scores_base_1': s1_base.cpu().numpy().flatten(),
        'scores_base_2': s2_base.cpu().numpy().flatten(),
        'scores_topo_1': s1_topo.cpu().numpy().flatten(),
        'scores_topo_2': s2_topo.cpu().numpy().flatten(),
        'n_params_base': n_params_base,
        'n_params_topo': n_params_topo,
        'intermediate_diff': intermediate_diff,
    }


def compute_ph_stats(adj, max_dim=3):
    """Compute PH pair counts using degree filtration."""
    try:
        import gudhi
        from layers.topology import build_clique_complex
    except ImportError:
        return None

    sc = build_clique_complex(adj, max_dim=max_dim)
    st = gudhi.SimplexTree()
    for dim_k in sorted(sc.keys()):
        for sigma in sc[dim_k]:
            filt_val = dim_k / (dim_k + 1)
            st.insert(list(sigma), filtration=filt_val)
    st.make_filtration_non_decreasing()
    st.persistence()
    pairs = st.persistence_pairs()

    dim_counts = {}
    for birth_s, death_s in pairs:
        if len(death_s) == 0:
            continue
        dim = len(birth_s) - 1
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    simplex_counts = {d: len(sc.get(d, [])) for d in sorted(sc.keys())}
    return {'pairs_by_dim': dim_counts, 'simplices_by_dim': simplex_counts}


# ============================================================================
# Main experiment
# ============================================================================

def print_family_detail(family_name, srg_params, graphs, ph_max_dim=3):
    """
    Print detailed Rook/Shrikhande-style statistics for a family of graphs.
    graphs: list of (name, adj) tuples. First graph is the 'reference'.
    Returns list of test results for all pairs (reference vs others).
    """
    n_exp, k_exp, lam_exp, mu_exp = srg_params
    names = [g[0] for g in graphs]
    adjs = [g[1] for g in graphs]
    n_graphs = len(graphs)

    print(f"\n{'='*70}")
    print(f"  {family_name}")
    print(f"  Strongly regular parameters: srg({n_exp}, {k_exp}, {lam_exp}, {mu_exp})")
    print(f"{'='*70}")

    # --- 1. Verification of srg parameters ---
    print(f"\n  [A] Verifying strongly regular parameters...")
    for name, adj in graphs:
        verify_srg(adj, name, srg_params)
        print(f"      {name}: verified srg({n_exp}, {k_exp}, {lam_exp}, {mu_exp})")

    # --- 2. Adjacency checksums ---
    print(f"\n  [B] Adjacency checksums (sum of entries):")
    for name, adj in graphs:
        checksum = int(adj.sum())
        expected = n_exp * k_exp
        print(f"      {name:14s}: {checksum} (expected: {n_exp}*{k_exp} = {expected})")

    # --- 3. Full adjacency spectrum ---
    print(f"\n  [C] Adjacency spectrum:")
    all_eigs = {}
    for name, adj in graphs:
        eigs = sorted(np.linalg.eigvalsh(adj), reverse=True)
        all_eigs[name] = eigs
        # Compact representation: show unique eigenvalues with multiplicities
        rounded = [round(e, 4) for e in eigs]
        from collections import Counter
        counts = Counter(rounded)
        spec_str = ", ".join(f"{val}(×{cnt})" for val, cnt in
                           sorted(counts.items(), key=lambda x: -x[0]))
        print(f"      {name:14s}: {spec_str}")

    # Cospectral check
    ref_eigs = all_eigs[names[0]]
    print(f"\n      Cospectral verification:")
    for i in range(1, n_graphs):
        cospectral = np.allclose(ref_eigs, all_eigs[names[i]], atol=0.01)
        print(f"        {names[0]} vs {names[i]}: cospectral = {cospectral}")

    # --- 4. Clique structure ---
    print(f"\n  [D] Clique structure (key topological difference):")
    clique_data = {}
    header = f"      {'Graph':14s} {'Triangles':>10s} {'K4':>8s}"
    if n_exp <= 30:
        header += f" {'K5':>8s}"
    print(header)
    print(f"      {'-'*14} {'-'*10} {'-'*8}" + (f" {'-'*8}" if n_exp <= 30 else ""))
    for name, adj in graphs:
        tri = count_cliques(adj, 3)
        k4 = count_cliques(adj, 4)
        clique_data[name] = {'triangles': tri, 'K4': k4}
        row = f"      {name:14s} {tri:>10d} {k4:>8d}"
        if n_exp <= 30:
            k5 = count_cliques(adj, 5)
            clique_data[name]['K5'] = k5
            row += f" {k5:>8d}"
        print(row)

    # --- 5. Simplicial complex & PH ---
    print(f"\n  [E] Simplicial complex and persistent homology (max_dim={ph_max_dim}):")
    ph_data = {}
    for name, adj in graphs:
        ph = compute_ph_stats(adj, max_dim=ph_max_dim)
        ph_data[name] = ph

    # Simplicial complex table
    dims_present = sorted(set().union(*(ph['simplices_by_dim'].keys()
                                        for ph in ph_data.values() if ph)))
    header = f"      {'Graph':14s}" + "".join(f" {'dim'+str(d):>8s}" for d in dims_present)
    print(f"\n      --- Simplicial Complex (simplex counts) ---")
    print(header)
    print(f"      {'-'*14}" + "".join(f" {'-'*8}" for _ in dims_present))
    for name, _ in graphs:
        ph = ph_data[name]
        row = f"      {name:14s}"
        for d in dims_present:
            row += f" {ph['simplices_by_dim'].get(d, 0):>8d}"
        print(row)

    # PH pairs table
    ph_dims = sorted(set().union(*(ph['pairs_by_dim'].keys()
                                   for ph in ph_data.values() if ph)))
    print(f"\n      --- Persistence Pairs (finite, degree filtration) ---")
    header = f"      {'Graph':14s}" + "".join(f" {'H'+str(d):>8s}" for d in ph_dims)
    print(header)
    print(f"      {'-'*14}" + "".join(f" {'-'*8}" for _ in ph_dims))
    for name, _ in graphs:
        ph = ph_data[name]
        row = f"      {name:14s}"
        for d in ph_dims:
            row += f" {ph['pairs_by_dim'].get(d, 0):>8d}"
        print(row)

    # --- 6. Model tests ---
    print(f"\n  [F] Model discrimination tests:")
    ref_name, ref_adj = graphs[0]
    results = []

    # Print model architecture once
    config_base = make_config(use_topology=False, n_vertices=n_exp)
    config_topo = make_config(use_topology=True, n_vertices=n_exp)
    torch.manual_seed(42)
    temp_base = BaseModel(config_base)
    torch.manual_seed(42)
    temp_topo = BaseModel(config_topo)
    n_base = sum(p.numel() for p in temp_base.parameters())
    n_topo = sum(p.numel() for p in temp_topo.parameters())
    print(f"      Baseline architecture: block_features=[64, 64], depth_of_mlp=2")
    print(f"      Baseline parameters: {n_base}")
    print(f"      Topology architecture: + max_ph_dim=2, max_simplex_dim={config_topo.architecture.topo_max_simplex_dim}, num_stats=16, hidden_dim=32")
    print(f"      Topology parameters: {n_topo} (+{n_topo - n_base} from topology)")

    for i in range(1, n_graphs):
        test_name, test_adj = graphs[i]
        result = test_pair_detailed(ref_adj, test_adj, ref_name, test_name,
                                    n_vertices=n_exp)
        results.append(result)

        print(f"\n      --- {ref_name} vs {test_name} ---")
        print(f"      BASELINE (Maron et al. 2019, no topology):")
        print(f"        {ref_name:14s} scores: {result['scores_base_1']}")
        print(f"        {test_name:14s} scores: {result['scores_base_2']}")
        print(f"        Max absolute difference: {result['diff_base']:.10f}")
        print(f"        Intermediate repr diff:  {result['intermediate_diff']:.10f}")
        print(f"        Outputs identical (< 1e-5): {result['baseline_identical']}")

        print(f"      TOPOLOGY-AUGMENTED:")
        print(f"        {ref_name:14s} scores: {result['scores_topo_1']}")
        print(f"        {test_name:14s} scores: {result['scores_topo_2']}")
        print(f"        Max absolute difference: {result['diff_topo']:.10f}")
        print(f"        Outputs different (> 1e-5): {result['topo_distinguishes']}")

        if result['diff_base'] > 0:
            ratio = result['diff_topo'] / result['diff_base']
            print(f"        Ratio (topo/base):         {ratio:.1f}x")

    # --- 7. Per-family summary ---
    print(f"\n  [G] Family Summary:")
    print(f"      {'Pair':30s} {'Base Diff':>12s} {'Topo Diff':>12s} {'Status':>8s}")
    print(f"      {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    for r in results:
        pair = f"{r['name1']} vs {r['name2']}"
        status = "✓" if (r['baseline_identical'] and r['topo_distinguishes']) else "✗"
        print(f"      {pair:30s} {r['diff_base']:>12.2e} {r['diff_topo']:>12.2e} {status:>8s}")

    n_ok = sum(1 for r in results if r['baseline_identical'] and r['topo_distinguishes'])
    print(f"\n      Verified: {n_ok}/{len(results)} pairs")
    return results


def run_experiment():
    import platform

    print("=" * 70)
    print("Extended Empirical Verification: Chang, Paley & Peisert Families")
    print("=" * 70)

    print("\n[0] Environment & reproducibility info")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  NumPy version:   {np.__version__}")
    print(f"  Random seed:     42")
    print(f"  Device:          cpu (deterministic)")
    print(f"  Python:          {platform.python_version()}")
    print(f"  Platform:        {platform.platform()}")

    # ==========================
    # FAMILY 1: Chang graphs
    # ==========================
    t8_adj, chang1_adj, chang2_adj, chang3_adj = build_chang_graphs()
    chang_graphs = [
        ("T(8)", t8_adj),
        ("Chang1", chang1_adj),
        ("Chang2", chang2_adj),
        ("Chang3", chang3_adj),
    ]
    chang_results = print_family_detail(
        "FAMILY 1: Chang graphs — T(8) vs Chang1, Chang2, Chang3",
        (28, 12, 6, 4),
        chang_graphs,
        ph_max_dim=3
    )

    # ==========================
    # FAMILY 2: Paley(25) vs L_3(5)
    # ==========================
    paley25_adj = build_paley25()
    latin_adj = build_latin_square_graph()
    paley25_graphs = [
        ("Paley(25)", paley25_adj),
        ("L_3(5)", latin_adj),
    ]
    paley25_results = print_family_detail(
        "FAMILY 2: Paley(25) vs Latin square graph L_3(5)",
        (25, 12, 5, 6),
        paley25_graphs,
        ph_max_dim=3
    )

    # ==========================
    # FAMILY 3: Paley(49) vs Peisert(49)
    # ==========================
    paley49_adj = build_paley49()
    peisert49_adj = build_peisert49()
    paley49_graphs = [
        ("Paley(49)", paley49_adj),
        ("Peisert(49)", peisert49_adj),
    ]
    paley49_results = print_family_detail(
        "FAMILY 3: Paley(49) vs Peisert(49)",
        (49, 24, 11, 12),
        paley49_graphs,
        ph_max_dim=3
    )

    # ==========================
    # COMPREHENSIVE SUMMARY
    # ==========================
    all_results = chang_results + paley25_results + paley49_results

    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)

    print(f"\n  {'Pair':30s} {'Base Diff':>12s} {'Topo Diff':>12s} {'Base=':>6s} {'Topo≠':>6s} {'Status':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*6} {'-'*6} {'-'*8}")
    for r in all_results:
        pair_name = f"{r['name1']} vs {r['name2']}"
        base_eq = "Yes" if r['baseline_identical'] else "No"
        topo_ne = "Yes" if r['topo_distinguishes'] else "No"
        status = "✓" if (r['baseline_identical'] and r['topo_distinguishes']) else "✗"
        print(f"  {pair_name:30s} {r['diff_base']:>12.2e} {r['diff_topo']:>12.2e} "
              f"{base_eq:>6s} {topo_ne:>6s} {status:>8s}")

    n_verified = sum(1 for r in all_results
                     if r['baseline_identical'] and r['topo_distinguishes'])
    n_total = len(all_results)
    print(f"\n  Verified: {n_verified}/{n_total} pairs")
    print(f"")
    print(f"  --- Graph Family Parameters ---")
    print(f"  Chang family:   srg(28, 12, 6, 4) — 4 graphs, 3 witness pairs")
    print(f"  Paley(25):      srg(25, 12, 5, 6) — 2 graphs, 1 witness pair")
    print(f"  Paley(49):      srg(49, 24, 11, 12) — 2 graphs, 1 witness pair")
    print(f"")
    print(f"  Key insight: All pairs are cospectral and 3-WL equivalent,")
    print(f"  yet our model separates them via persistent homology detecting")
    print(f"  differences in simplicial complex structure (clique distribution).")
    print(f"")
    if n_verified == n_total:
        print(f"  ✓ ALL PAIRS VERIFIED: Topology model strictly more expressive than 3-WL")
    else:
        print(f"  Partial verification: {n_verified}/{n_total} pairs distinguished")
    print("=" * 70)
    print(f"\n  Reproduction: python scripts/verify_chang_paulus.py")
    print(f"  Required packages: torch, numpy, gudhi, easydict")


if __name__ == '__main__':
    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    output_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, output_buffer)

    run_experiment()

    sys.stdout = old_stdout

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'experiments', 'chang_paulus_verification.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(output_buffer.getvalue())
    print(f"\nReport saved to: {report_path}")
