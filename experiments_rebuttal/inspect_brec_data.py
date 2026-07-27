"""
Diagnose the layout of a BREC data file and recommend the correct pairing.

`brec_expressivity.py` assumes 800 graph6 entries = 400 consecutive pairs
(graph 2i, 2i+1). Some BREC distributions instead store many permutation
copies per graph (e.g. 51200 = 800 x 64), which breaks that assumption -- you
end up "pairing" isomorphic copies, and every pair reads at the Hotelling null
(major ~= reliab ~= p). This script figures out the real structure.

  python experiments_rebuttal/inspect_brec_data.py [path]
"""
import sys
import numpy as np
import networkx as nx


def to_graph(x):
    if isinstance(x, nx.Graph):
        return nx.convert_node_labels_to_integers(x)
    if isinstance(x, str):
        x = x.encode()
    return nx.convert_node_labels_to_integers(nx.from_graph6_bytes(bytes(x)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 \
        else "experiments_rebuttal/data/brec_v3.npy"
    raw = np.load(path, allow_pickle=True)
    print(f"file:  {path}")
    print(f"numpy: dtype={raw.dtype} shape={raw.shape} len={len(raw)}")
    flat = list(raw.reshape(-1)) if raw.ndim > 1 else list(raw)
    L = len(flat)
    print(f"flattened entries: {L}")
    print(f"first entry: type={type(flat[0]).__name__} {repr(flat[0])[:70]}")

    # parse a prefix
    n_parse = min(300, L)
    G = [to_graph(flat[i]) for i in range(n_parse)]
    sizes = [g.number_of_nodes() for g in G]
    print(f"\nfirst {n_parse} graphs: node counts min={min(sizes)} "
          f"max={max(sizes)}; first 12 = {sizes[:12]}")

    # (1) run-length of consecutive isomorphic entries starting at 0
    B = 1
    while B < n_parse and G[0].number_of_nodes() == G[B].number_of_nodes() \
            and nx.is_isomorphic(G[0], G[B]):
        B += 1
    print(f"\nconsecutive isomorphic run from entry 0: {B}")
    if B > 1:
        print(f"  -> looks like {B} permutation-copies stored per graph.")
        if L % B == 0:
            n_graphs = L // B
            print(f"  -> {n_graphs} distinct graphs = {n_graphs // 2} pairs; "
                  f"pair k = entries ({B}*2k, {B}*(2k+1)).")

    # (2) test candidate pairings: count non-isomorphic (good) among first 40
    def frac_noniso(pairs):
        good = tot = 0
        for a, b in pairs:
            if a >= n_parse or b >= n_parse:
                break
            tot += 1
            if G[a].number_of_nodes() == G[b].number_of_nodes():
                good += 0 if nx.is_isomorphic(G[a], G[b]) else 1
            else:
                good += 1
        return good, tot

    print("\ncandidate pairings (want: most pairs NON-isomorphic):")
    cons = [(2 * k, 2 * k + 1) for k in range(20)]
    g, t = frac_noniso(cons)
    print(f"  consecutive (2k,2k+1):           {g}/{t} non-isomorphic")
    if B > 1:
        blk = [(B * 2 * k, B * (2 * k + 1)) for k in range(20)]
        g, t = frac_noniso(blk)
        print(f"  block-of-{B} ({B}*2k, {B}*(2k+1)):   {g}/{t} non-isomorphic")
    half = [(k, L // 2 + k) for k in range(20)]
    g, t = frac_noniso(half)
    print(f"  halves (k, L/2+k):               {g}/{t} non-isomorphic")

    # (3) repeat structure: does graph 0 (and pair 0) reappear later? A BREC
    # file of 51200 = 800 x 64 is very likely the 400 canonical pairs repeated
    # under 64 GLOBAL permutations (perm-major: period = #graphs-per-block), or
    # each pair stored with N perm-instances back-to-back (pair-major).
    print("\nrepeat structure:")
    print(f"  G[0] isomorphic to G[2]?  {nx.is_isomorphic(G[0], G[2])}"
          f"   (True => pair-major: pair 0 stored as consecutive perm-copies)")
    # smallest stride P>0 where graph 0 reappears (bounded, node-count guarded)
    n0 = G[0].number_of_nodes()
    stride = None
    limit = min(L, 4000)
    for P in range(1, limit):
        gp = flat[P]
        gP = to_graph(gp)
        if gP.number_of_nodes() == n0 and nx.is_isomorphic(G[0], gP):
            stride = P
            break
    print(f"  smallest stride with G[0] reappearing: {stride}"
          + (f"  (scanned first {limit})" if stride is None else ""))
    # UNORDERED run length: how many consecutive storage-pairs are the same
    # unordered pair {G1,G2} as storage-pair 0 (instances may swap the two
    # graphs' order). This is the true instances-per-pair R the loader uses.
    def same_unordered(a1, a2, b1, b2):
        nn = lambda g: g.number_of_nodes()
        if nn(a1) == nn(b1) and nn(a2) == nn(b2) \
                and nx.is_isomorphic(a1, b1) and nx.is_isomorphic(a2, b2):
            return True
        return (nn(a1) == nn(b2) and nn(a2) == nn(b1)
                and nx.is_isomorphic(a1, b2) and nx.is_isomorphic(a2, b1))

    p0a = to_graph(flat[0]); p0b = to_graph(flat[1])
    R = 1
    while 2 * R + 1 < L and R < 4000:
        qa = to_graph(flat[2 * R]); qb = to_graph(flat[2 * R + 1])
        if not same_unordered(qa, qb, p0a, p0b):
            break
        R += 1
    n_unique = (L // 2) // R if R else L // 2
    print(f"\n  UNORDERED instances-per-pair R = {R}  ->  {n_unique} unique pairs")
    if L % (2 * R) == 0:
        print(f"=> LAYOUT: {n_unique} unique pairs, each stored as {R} permuted "
              f"instances (some order-swapped).")
        print(f"=> The loader auto-detects this (unordered stride) and "
              f"collapses to {n_unique} pairs. Expect ~400 for standard BREC.")
        if n_unique == 400:
            print("=> 400 pairs — matches canonical BREC. Category ranges align.")
        else:
            print(f"=> NOTE: {n_unique} != 400. If not canonical BREC, paste "
                  f"this so we can confirm the category map.")
    else:
        print(f"  (L not divisible by 2R={2*R}; paste this and I'll adjust.)")


if __name__ == "__main__":
    main()
