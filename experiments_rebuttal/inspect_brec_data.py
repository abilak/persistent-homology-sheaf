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

    print("\nInterpretation: the pairing whose pairs are (almost) all "
          "NON-isomorphic is the correct one. If 'block-of-B' wins, the file "
          "stores B perms/graph -- reshape to one representative per graph "
          "before pairing (I can patch the loader once you paste this output).")


if __name__ == "__main__":
    main()
