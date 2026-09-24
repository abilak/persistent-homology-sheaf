"""Optimized topology layer must match the frozen reference implementation
(_topology_ref.py = the pre-optimization code) in forward values AND gradients,
for every option combination, when both hold identical parameters."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
import numpy as np, torch
# float64: the optimized path reorders sums, so float32 agreement is only
# ~1e-4 relative on large parameter grads; in float64 it must be exact-ish.
torch.set_default_dtype(torch.float64)
import layers.topology as F
import _topology_ref as R
from test_batching import make_batch

OPTS = ['node_level', 'multiplicity', 'scale_stats', 'norm_stats', 'essential', 'filt_squash']


def tied_batch(B, M, d):
    """Features depend only on adjacency (no noise): symmetric graphs give
    many exactly tied filtration values, exercising tie-merging."""
    import networkx as nx
    gs = [nx.cycle_graph(M), nx.circulant_graph(M, [1, 2]),
          nx.disjoint_union(nx.cycle_graph(M // 2), nx.cycle_graph(M - M // 2)),
          nx.circulant_graph(M, [1, 3]), nx.cycle_graph(M)][:B]
    adjs = [nx.to_numpy_array(g).astype(np.float64) for g in gs]
    x = torch.zeros(B, d, M, M)
    for b, A in enumerate(adjs):
        At = torch.from_numpy(A).to(x.dtype)
        for c in range(d):
            x[b, c] = At * (c + 1) / d + torch.eye(M) * 0.5
    structs = [F.build_graph_structs([A], 2)[0] for A in adjs]
    return x, structs, adjs


def compare(opts, gate_mode='conv', max_ph_dim=1, B=5, M=12, d=8, tied=False):
    x, structs_f, adjs = (tied_batch(B, M, d) if tied else make_batch(B, M, d, seed=3))
    structs_r = [R.build_graph_structs([A], 2)[0] for A in adjs]
    torch.manual_seed(5)
    Lf = F.TopologyLayer(d, 16, max_ph_dim, 8, gate_bias=1.0, gate_mode=gate_mode,
                         gate_init=0.3, **opts)
    Lr = R.TopologyLayer(d, 16, max_ph_dim, 8, gate_bias=1.0, gate_mode=gate_mode,
                         gate_init=0.3, **opts)
    Lr.load_state_dict(Lf.state_dict())
    for m in (Lf, Lr):   # make last fusion layer non-trivial so grads flow everywhere
        torch.manual_seed(9); torch.nn.init.normal_(m.fusion[2].weight, std=0.3)
    xf = x.clone().requires_grad_(True); xr = x.clone().requires_grad_(True)
    of = Lf(xf, structs_f); orr = Lr(xr, structs_r)
    (of ** 2).sum().backward(); (orr ** 2).sum().backward()
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1.0)).item()
    dx = rel(of, orr)
    dg = rel(xf.grad, xr.grad)
    dp = max(rel(pf.grad, pr.grad)
             for (nf, pf), (nr, pr) in zip(Lf.named_parameters(), Lr.named_parameters())
             if pf.grad is not None and pr.grad is not None)
    gv = (Lf.last_graph_vec - Lr.last_graph_vec).abs().max().item()
    return dx, dg, dp, gv


if __name__ == '__main__':
    ok = True; worst = 0.0; n = 0
    for bits in itertools.product([False, True], repeat=len(OPTS)):
        opts = dict(zip(OPTS, bits))
        for gm in ('conv', 'scalar'):
            for P, tied in ((1, False), (2, False), (1, True)):
                r = compare(opts, gm, P, tied=tied)
                m = max(r); worst = max(worst, m); n += 1
                if not (m < 1e-6):
                    ok = False; print("MISMATCH", opts, gm, P, r)
    print(f"{n} configurations, worst relative |fast - ref| over outputs/grads = {worst:.2e}")
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
