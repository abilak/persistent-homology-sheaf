"""GPU-resident persistence (layers/torch_ph.py) vs the gudhi path: same
features and gradients (float64) for every option combination, on random
graphs with triangles, disconnected graphs, tied filtrations, triangle-free
cycles, and real TU batches."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import numpy as np, torch, networkx as nx
torch.set_default_dtype(torch.float64)
import layers.topology as T
T.BatchPlan.TORCH_PH_MAX_TRIANGLES = 64   # exercise large reductions too


def graphs_random(B, M, seed, p=0.35):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        A = np.triu((rng.random((M, M)) < p).astype(float), 1); out.append(A + A.T)
    return out


def graphs_special(M):
    gs = [nx.cycle_graph(M), nx.disjoint_union(nx.cycle_graph(M // 2), nx.cycle_graph(M - M // 2)),
          nx.circulant_graph(M, [1, 2]), nx.empty_graph(M), nx.disjoint_union(nx.complete_graph(5), nx.empty_graph(M - 5)),
          nx.disjoint_union(nx.complete_graph(4), nx.path_graph(M - 4))]
    return [nx.to_numpy_array(g) for g in gs]


def run(adjs, backend, opts, P, filt, tied=False):
    structs = T.build_graph_structs(adjs, 2)
    torch.manual_seed(0)
    ph = T.DifferentiablePH(P, 6, **opts)
    fs = [f.clone().requires_grad_(True) for f in filt]
    plan = T.get_plan(structs, torch.device('cpu'), 1, P, backend=backend)
    assert plan.use_torch_ph == (backend == 'torch'), (backend, plan.use_torch_ph)
    M = structs[0].M
    g, n = ph([(st, f) for st, f in zip(structs, fs)], device='cpu',
              num_nodes=M if opts.get('node', True) else None, plan=plan)
    loss = (g ** 2).sum() + ((n ** 2).sum() if n is not None else 0)
    loss.backward()
    return g.detach(), (n.detach() if n is not None else None), [f.grad.clone() for f in fs]


def compare(adjs, opts, P, tied):
    structs = T.build_graph_structs(adjs, 2)
    rng = np.random.default_rng(1)
    if tied:     # values depend on dimension only -> massive ties
        filt = [torch.tensor([0.1 * len(s) + (0.0 if i % 3 else 0.05) for i, s in
                              enumerate(st.simplices_list)]) for st in structs]
    else:
        filt = [torch.tensor(rng.standard_normal(st.S)) for st in structs]
    node = opts.pop('node')
    o = dict(opts)
    res = {}
    for be in ('gudhi', 'torch'):
        T._PLAN_CACHE.clear()
        res[be] = run(adjs, be, dict(o), P, filt) if not node else None
    return res


if __name__ == '__main__':
    ok = True; worst = 0.0; n_cfg = 0
    OPTS = ['multiplicity', 'essential', 'scale_stats']
    sets = [('random', graphs_random(6, 11, 0)), ('random_dense', graphs_random(4, 9, 3, 0.6)),
            ('special', graphs_special(10))]
    for (name, adjs), bits, node, P, tied in itertools.product(
            sets, itertools.product([False, True], repeat=3), [True, False], [0, 1], [False, True]):
        opts = dict(zip(OPTS, bits))
        structs = T.build_graph_structs(adjs, 2)
        rng = np.random.default_rng(7)
        if tied:
            filt = [torch.tensor([0.25 * (len(s) - 1) + (0.1 if (i % 4 == 0) else 0.0)
                                  for i, s in enumerate(st.simplices_list)]) for st in structs]
        else:
            filt = [torch.tensor(rng.standard_normal(st.S)) for st in structs]
        # the max-correction runs inside; raw values may violate monotonicity
        outs = {}
        for be in ('gudhi', 'torch'):
            T._PLAN_CACHE.clear()
            torch.manual_seed(0)
            ph = T.DifferentiablePH(P, 6, **opts)
            fs = [f.clone().requires_grad_(True) for f in filt]
            plan = T.get_plan(structs, torch.device('cpu'), 1, P, backend=be)
            if be == 'torch' and not plan.use_torch_ph:
                print("torch path not taken for", name); ok = False
            M = structs[0].M
            g, nv = ph([(st, f) for st, f in zip(structs, fs)], device='cpu',
                       num_nodes=M if node else None, plan=plan)
            ((g ** 2).sum() + ((nv ** 2).sum() if nv is not None else 0)).backward()
            outs[be] = (g.detach(), None if nv is None else nv.detach(),
                        torch.cat([f.grad for f in fs]))
        a, b = outs['gudhi'], outs['torch']
        rel = lambda x, y: ((x - y).abs().max() / y.abs().max().clamp_min(1)).item()
        dg = rel(b[0], a[0]); dn = rel(b[1], a[1]) if node else 0.0
        dgr = rel(b[2], a[2])
        # under ties, WHICH tied simplex gets the gradient is implementation-
        # defined (diagrams and features are not); compare grads without ties
        m = max(dg, dn) if tied else max(dg, dn, dgr)
        worst = max(worst, m); n_cfg += 1
        if not (m < 1e-9):
            ok = False
            print(f"MISMATCH {name} {opts} node={node} P={P} tied={tied}: feat={dg:.1e} node={dn:.1e} grad={dgr:.1e}")
    print(f"{n_cfg} configurations, worst relative |torch - gudhi| = {worst:.2e}")
    print("PASS" if ok else "FAIL")
    if not ok: sys.exit(1)


def real_batches(ds, n_batches=6, heads=3):
    """Full TopologyLayer on real TU batches: torch backend vs gudhi."""
    from run_job import build_config
    from data_loader.data_generator import DataGenerator
    cfg = build_config(ds, 'topo'); cfg.num_fold = 1
    data = DataGenerator(cfg); data.initialize('train')
    worst = 0.0; used = 0
    for _ in range(n_batches):
        g = torch.as_tensor(data.next_batch()[0]).double()
        structs = T.build_graph_structs(g[:, 0].numpy(), 2)
        torch.manual_seed(3)
        proj = torch.nn.Conv2d(g.shape[1], 8, 1)
        # molecules have massive EXACT ties (same atom types), where which tied
        # simplex receives the gradient is tie-break dependent (both backends
        # valid). Forward is compared as is; gradients on tie-broken inputs.
        jitter = 1e-4 * torch.randn(g.shape[0], 8, g.shape[2], g.shape[3])
        outs = {}
        for be in ('gudhi', 'torch'):
            T._PLAN_CACHE.clear()
            torch.manual_seed(5)
            L = T.TopologyLayer(8, 16, 1, 8, multiplicity=True, essential=True, norm_stats=True,
                                filt_squash=True, num_filtrations=heads, ph_backend=be)
            torch.nn.init.normal_(L.fusion[2].weight, std=0.3)
            with torch.no_grad():
                o_tied = L(torch.relu(proj(g)), structs)
            gg = g.clone().requires_grad_(True)
            o = L(torch.relu(proj(gg)) + jitter, structs)
            (o ** 2).mean().backward()
            outs[be] = (o_tied, gg.grad.clone(), o.detach())
            if be == 'torch':
                used += int(get_plan_used(structs, heads))
        rel = lambda x, y: ((x - y).abs().max() / y.abs().max().clamp_min(1)).item()
        worst = max(worst, rel(outs['torch'][0], outs['gudhi'][0]),
                    rel(outs['torch'][1], outs['gudhi'][1]),
                    rel(outs['torch'][2], outs['gudhi'][2]))
    return worst, used


def get_plan_used(structs, heads):
    return T.get_plan(structs, torch.device('cpu'), heads, 1, backend='torch').use_torch_ph


if __name__ == '__main__':
    ok2 = True
    for ds in ('MUTAG', 'NCI1', 'PTC'):
        w, used = real_batches(ds)
        good = w < 1e-9; ok2 &= good
        print(f"{ds}: full layer (3 heads, node-level, essential) torch vs gudhi (fwd tied+untied, grad untied): "
              f"{w:.1e} on {used}/6 batches via torch [{'OK' if good else 'FAIL'}]")
    print("PASS" if ok2 else "FAIL"); sys.exit(0 if ok2 else 1)
