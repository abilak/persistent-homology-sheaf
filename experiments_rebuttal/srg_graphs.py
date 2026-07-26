"""
Constructions of the strongly regular graph (SRG) pairs used in Table 2 /
Corollary 6, with programmatic verification of SRG parameters, cospectrality
and non-isomorphism. All graphs returned as networkx.Graph on integer nodes.

Pairs (all 3-WL / 2-FWL equivalent, cospectral, non-isomorphic):
  Rook(4,4)  vs Shrikhande         srg(16, 6, 2, 2)
  T(8)=J(8,2) vs Chang_1,2,3       srg(28,12, 6, 4)
  Paley(25)  vs L3(5) Latin-sq     srg(25,12, 5, 6)
  Paley(49)  vs Peisert(49)        srg(49,24,11,12)
"""
import numpy as np
import networkx as nx
from itertools import combinations, product


# ----------------------------- finite fields ------------------------------ #
class GF:
    """GF(p^2) = GF(p)[t]/(t^2 - n), elements as (a,b) meaning a + b t."""
    def __init__(self, p, nonresidue):
        self.p = p
        self.n = nonresidue  # t^2 = nonresidue
        self.elements = [(a, b) for a in range(p) for b in range(p)]

    def add(self, x, y):
        return ((x[0] + y[0]) % self.p, (x[1] + y[1]) % self.p)

    def sub(self, x, y):
        return ((x[0] - y[0]) % self.p, (x[1] - y[1]) % self.p)

    def mul(self, x, y):
        a, b = x
        c, d = y
        # (a+bt)(c+dt) = ac + bd t^2 + (ad+bc) t = (ac + bd n) + (ad+bc) t
        return ((a * c + b * d * self.n) % self.p, (a * d + b * c) % self.p)

    def pow(self, x, k):
        r = (1, 0)
        for _ in range(k):
            r = self.mul(r, x)
        return r

    def squares(self):
        """Set of nonzero squares (quadratic residues) in GF(p^2)."""
        s = set()
        for e in self.elements:
            if e == (0, 0):
                continue
            s.add(self.mul(e, e))
        return s

    def primitive_element(self):
        """Find a generator of the multiplicative group (order p^2-1)."""
        order = self.p * self.p - 1
        for e in self.elements:
            if e == (0, 0):
                continue
            seen = set()
            x = (1, 0)
            for _ in range(order):
                x = self.mul(x, e)
                seen.add(x)
            if len(seen) == order:
                return e
        raise RuntimeError("no primitive element found")


# ------------------------------ constructions ------------------------------ #
def rook_graph(n):
    """Rook's graph R(n,n) = L2(n): n^2 cells, adjacent iff same row or col."""
    G = nx.Graph()
    cells = list(product(range(n), range(n)))
    G.add_nodes_from(range(n * n))
    idx = {c: i for i, c in enumerate(cells)}
    for (r1, c1), (r2, c2) in combinations(cells, 2):
        if r1 == r2 or c1 == c2:
            G.add_edge(idx[(r1, c1)], idx[(r2, c2)])
    return G


def shrikhande_graph():
    """Cayley graph on Z4 x Z4 with connection set +-{(1,0),(0,1),(1,1)}."""
    G = nx.Graph()
    verts = list(product(range(4), range(4)))
    idx = {v: i for i, v in enumerate(verts)}
    G.add_nodes_from(range(16))
    conn = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for v in verts:
        for c in conn:
            w = ((v[0] + c[0]) % 4, (v[1] + c[1]) % 4)
            G.add_edge(idx[v], idx[w])
    return G


def triangular_graph(n):
    """T(n) = Johnson J(n,2) = line graph of K_n. Vertices = 2-subsets."""
    G = nx.Graph()
    pairs = list(combinations(range(n), 2))
    idx = {p: i for i, p in enumerate(pairs)}
    G.add_nodes_from(range(len(pairs)))
    for a, b in combinations(pairs, 2):
        if len(set(a) & set(b)) == 1:
            G.add_edge(idx[a], idx[b])
    return G


def _seidel_switch(G, S):
    """Seidel switching w.r.t. vertex subset S (set)."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    Sset = set(S)
    for u, v in combinations(G.nodes(), 2):
        adj = G.has_edge(u, v)
        if (u in Sset) ^ (v in Sset):
            adj = not adj  # switch edges between S and complement
        if adj:
            H.add_edge(u, v)
    return H


def chang_graphs():
    """
    The three Chang graphs: Seidel-switch T(8) w.r.t. the vertex set of
    (a) 4 disjoint edges (perfect matching on 8 pts),
    (b) an 8-cycle,
    (c) a K_3 + C_5 (triangle plus 5-cycle).
    Here "vertex set" means the T(8)-vertices (=2-subsets of [8]) corresponding
    to the edges of the chosen subgraph of K_8. Switching class is what matters;
    these give the three graphs cospectral with, non-isomorphic to, T(8).
    """
    pairs = list(combinations(range(8), 2))
    idx = {p: i for i, p in enumerate(pairs)}
    T = triangular_graph(8)

    def edgeset_to_S(edges):
        return [idx[tuple(sorted(e))] for e in edges]

    # (a) perfect matching: {01,23,45,67}
    S1 = edgeset_to_S([(0, 1), (2, 3), (4, 5), (6, 7)])
    # (b) 8-cycle 0-1-2-3-4-5-6-7-0
    S2 = edgeset_to_S([(0, 1), (1, 2), (2, 3), (3, 4),
                       (4, 5), (5, 6), (6, 7), (0, 7)])
    # (c) triangle {0,1,2} + 5-cycle 3-4-5-6-7-3
    S3 = edgeset_to_S([(0, 1), (1, 2), (0, 2),
                       (3, 4), (4, 5), (5, 6), (6, 7), (3, 7)])
    return [_seidel_switch(T, S1), _seidel_switch(T, S2), _seidel_switch(T, S3)]


def paley_graph(q):
    """Paley graph on GF(q). q=25 or 49 (prime power p^2)."""
    if q == 25:
        F = GF(5, 2)   # t^2 = 2 (2 is a non-residue mod 5)
    elif q == 49:
        F = GF(7, 3)   # t^2 = 3 (3 is a non-residue mod 7)
    else:
        raise ValueError("only q in {25,49} supported")
    sq = F.squares()
    idx = {e: i for i, e in enumerate(F.elements)}
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for x, y in combinations(F.elements, 2):
        if F.sub(x, y) in sq:
            G.add_edge(idx[x], idx[y])
    return G


def latin_square_L3_5():
    """
    L3(5): rook's graph R(5,5) augmented with "same symbol" adjacency from one
    Latin square L (cyclic: L[r][c] = (r+c) mod 5). srg(25,12,5,6), a Latin
    square graph on 25 cells that is cospectral with but not isomorphic to
    Paley(25).
    """
    n = 5
    cells = list(product(range(n), range(n)))
    idx = {c: i for i, c in enumerate(cells)}
    # A Latin square NOT isotopic to the cyclic group Z5. The cyclic table
    # yields a graph isomorphic to Paley(25); this non-cyclic square yields
    # the genuinely distinct srg(25,12,5,6) that is cospectral with but
    # non-isomorphic to Paley(25). (Found by search over order-5 Latin squares.)
    L = [[0, 1, 2, 3, 4],
         [1, 0, 3, 4, 2],
         [2, 3, 4, 0, 1],
         [3, 4, 1, 2, 0],
         [4, 2, 0, 1, 3]]
    G = nx.Graph()
    G.add_nodes_from(range(n * n))
    for (r1, c1), (r2, c2) in combinations(cells, 2):
        same_row = (r1 == r2)
        same_col = (c1 == c2)
        same_sym = (L[r1][c1] == L[r2][c2])
        if same_row or same_col or same_sym:
            G.add_edge(idx[(r1, c1)], idx[(r2, c2)])
    return G


def peisert_graph(q=49):
    """
    Peisert graph P*(q), q=p^2 with p = 3 mod 4. Cayley graph on GF(q) with
    connection set = {g^j : j = 0 or 1 (mod 4)} where g is a primitive root
    (union of the 4th-power cosets of index 0 and 1). srg(49,24,11,12),
    cospectral with, non-isomorphic to Paley(49).
    """
    assert q == 49
    F = GF(7, 3)
    g = F.primitive_element()
    order = q - 1
    # powers of g
    powers = []
    x = (1, 0)
    for j in range(order):
        powers.append((j, x))
        x = F.mul(x, g)
    conn = set(val for (j, val) in powers if j % 4 in (0, 1))
    idx = {e: i for i, e in enumerate(F.elements)}
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for xg, yg in combinations(F.elements, 2):
        if F.sub(xg, yg) in conn:
            G.add_edge(idx[xg], idx[yg])
    return G


# ------------------------------ verification ------------------------------- #
def srg_parameters(G):
    """Return (n,k,lambda,mu) if G is strongly regular, else None."""
    n = G.number_of_nodes()
    degs = [d for _, d in G.degree()]
    if len(set(degs)) != 1:
        return None
    k = degs[0]
    lam = set()
    mu = set()
    A = nx.to_numpy_array(G, nodelist=range(n))
    A2 = A @ A
    for u, v in combinations(range(n), 2):
        common = int(A2[u, v])
        if A[u, v] > 0:
            lam.add(common)
        else:
            mu.add(common)
    if len(lam) == 1 and len(mu) == 1:
        return (n, k, lam.pop(), mu.pop())
    return None


def spectrum(G):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    ev = np.linalg.eigvalsh(A)
    return np.round(np.sort(ev), 4)


def all_pairs():
    """Return dict name->(G1,G2, expected_params)."""
    return {
        "Rook(4,4) vs Shrikhande": (rook_graph(4), shrikhande_graph(), (16, 6, 2, 2)),
        "T(8) vs Chang_1": (triangular_graph(8), chang_graphs()[0], (28, 12, 6, 4)),
        "T(8) vs Chang_2": (triangular_graph(8), chang_graphs()[1], (28, 12, 6, 4)),
        "T(8) vs Chang_3": (triangular_graph(8), chang_graphs()[2], (28, 12, 6, 4)),
        "Paley(25) vs L3(5)": (paley_graph(25), latin_square_L3_5(), (25, 12, 5, 6)),
        "Paley(49) vs Peisert(49)": (paley_graph(49), peisert_graph(49), (49, 24, 11, 12)),
    }


if __name__ == "__main__":
    print("Verifying SRG constructions...\n")
    ok_all = True
    for name, (G1, G2, exp) in all_pairs().items():
        p1 = srg_parameters(G1)
        p2 = srg_parameters(G2)
        cosp = np.allclose(spectrum(G1), spectrum(G2))
        iso = nx.is_isomorphic(G1, G2)
        ok = (p1 == exp) and (p2 == exp) and cosp and (not iso)
        ok_all &= ok
        print(f"{name}")
        print(f"   params G1={p1} G2={p2} expected={exp}")
        print(f"   cospectral={cosp}  isomorphic={iso}  "
              f"-> {'OK' if ok else '**FAIL**'}")
    print("\nALL PAIRS VALID" if ok_all else "\nSOME PAIRS INVALID")
