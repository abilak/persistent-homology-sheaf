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
    if stride:
        # verify the whole pair repeats at this stride
        gpair = (to_graph(flat[stride]), to_graph(flat[stride + 1]))
        pair_repeats = nx.is_isomorphic(G[0], gpair[0]) and \
            nx.is_isomorphic(G[1], gpair[1])
        print(f"  pair 0 repeats at stride {stride}? {pair_repeats}")
        if pair_repeats and L % stride == 0:
            uniq_graphs = stride
            print(f"\n=> LAYOUT: {L // stride} permutation-blocks of "
                  f"{stride} graphs. {uniq_graphs // 2} UNIQUE pairs.")
            print(f"=> FIX: evaluate only the first block -> "
                  f"`--pairs 0-{uniq_graphs // 2}` "
                  f"(categories map correctly to those {uniq_graphs // 2} "
                  f"pairs).")
        elif not pair_repeats:
            print("  (graph 0 recurs but the PAIR doesn't repeat at that "
                  "stride -- paste this and I'll match the loader.)")
    else:
        print("  => no repeat found: looks like genuinely distinct pairs; "
              "consecutive pairing is correct as-is.")


if __name__ == "__main__":
    main()
