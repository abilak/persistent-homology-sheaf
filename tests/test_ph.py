import numpy as np, torch, networkx as nx, gudhi, pytest
from test_model import T, CPU, float64, rand_adj, rand_graphs, jiggle, tu_batches
import layers.torch_ph as TP

pytestmark = pytest.mark.usefixtures('float64')

def bstructs(x, D=2):
    adjs = [x._host_adj[b, :k, :k] for b, k in enumerate(x._n_real)]
    return T.build_structs(adjs, D)

def filt_vals(rng, structs, tied):
    vals = []
    for st in structs:
        if(tied):
            vals.append(torch.tensor(rng.integers(0, 3, st['S']) * 0.5))
        else:
            vals.append(torch.tensor(rng.standard_normal(st['S'])))
    return vals

def corrected(st, f):
    idx = {s: i for i, s in enumerate(st['simplices'])}
    v = np.array(f, dtype=float)
    for(i, s) in enumerate(st['simplices']):
        if(len(s) > 1):
            faces = [s[:k] + s[k + 1:] for k in range(len(s))]
            v[i] = max(v[i], max(v[idx[fc]] for fc in faces))
    return v

def gudhi_dgms(st, vals, P):
    t = gudhi.SimplexTree()
    for(i, s) in enumerate(st['simplices']):
        t.insert(list(s), filtration=float(vals[i]))
    t.persistence(persistence_dim_max=True, min_persistence=TP.PERS_EPS)
    fin, ess = {}, {}
    for d in range(P + 1):
        iv = t.persistence_intervals_in_dimension(d)
        fin[d] = [(b, e) for b, e in iv if np.isfinite(e) and e - b > TP.PERS_EPS]
        ess[d] = [b for b, e in iv if np.isinf(e)]
    return fin, ess

def torch_dgms(structs, vals, P, n):
    plan = T.get_plan(structs, CPU, pdim=P, backend='torch', M_pad=n)
    assert plan['use_torch']
    adj = torch.cat([torch.from_numpy(v) for v in vals])
    blocks = TP.torch_pers(adj, plan, P + 1, True, False)
    fin = [{d: [] for d in range(P + 1)} for _ in structs]
    ess = [{d: [] for d in range(P + 1)} for _ in structs]
    for((kind, d), blk) in blocks.items():
        ok = blk['valid'].numpy()
        b = blk['b'].detach().numpy()
        e = blk['d'].detach().numpy() if kind == 'fin' else None
        for i in range(len(structs)):
            for r in np.nonzero(ok[i])[0]:
                if(kind == 'fin'):
                    fin[i][d].append((b[i, r], e[i, r]))
                else:
                    ess[i][d].append(b[i, r])
    return fin, ess

def same(a, b):
    a, b = sorted(a), sorted(b)
    return len(a) == len(b) and all(np.allclose(x, y, atol=1e-12) for x, y in zip(a, b))

def fuzz(seed, trials, nmax=37):
    rng = np.random.default_rng(seed)
    bad = []
    n_pairs = 0
    for t in range(trials):
        D = int(rng.integers(1, 4))
        P = int(rng.integers(0, min(D, 2) + 1))
        n = int(rng.integers(4, nmax if D < 3 else 15))
        p = float(rng.uniform(0.1, 0.9))
        tied = bool(rng.integers(2))
        structs = T.build_structs(rand_graphs(rng, 3, n, p), D)
        fs = filt_vals(rng, structs, tied)
        vals = [corrected(st, f.numpy()) for st, f in zip(structs, fs)]
        tf, te = torch_dgms(structs, vals, P, n)
        for(i, st) in enumerate(structs):
            gf, ge = gudhi_dgms(st, vals[i], P)
            for d in range(P + 1):
                n_pairs += len(gf[d]) + len(ge[d])
                if(not (same(gf[d], tf[i][d]) and same(ge[d], te[i][d]))):
                    bad.append((t, n, D, P, tied, d))
    return bad, n_pairs

def test_vs_gudhi():
    bad, n_pairs = fuzz(0, 150)
    assert n_pairs > 1000
    assert not bad, bad[:5]

def test_relax(monkeypatch):
    # Forces the edge list H0 path
    monkeypatch.setattr(TP, 'SQUARING_MAX_N', 6)
    bad, _ = fuzz(1, 60)
    assert not bad, bad[:5]

def test_big_sparse():
    rng = np.random.default_rng(9)
    adjs = [rand_adj(n, 3.0 / n, rng) for n in (200, 170)]
    structs = T.build_structs(adjs, 2)
    vals = [corrected(st, rng.standard_normal(st['S'])) for st in structs]
    tf, te = torch_dgms(structs, vals, 1, 200)
    for(i, st) in enumerate(structs):
        gf, ge = gudhi_dgms(st, vals[i], 1)
        for d in range(2):
            assert same(gf[d], tf[i][d]) and same(ge[d], te[i][d])

def test_schedules(monkeypatch):
    rng = np.random.default_rng(2)
    structs = T.build_structs(rand_graphs(rng, 4, 11, 0.5), 3)
    vals = [corrected(st, rng.standard_normal(st['S'])) for st in structs]
    monkeypatch.setattr(TP, 'FIXED_STEPS_MAX', 10 ** 9)
    a = torch_dgms(structs, vals, 2, 11)
    monkeypatch.setattr(TP, 'FIXED_STEPS_MAX', 0)
    b = torch_dgms(structs, vals, 2, 11)
    for k in range(2):
        for i in range(4):
            for d in range(3):
                assert same(a[k][i][d], b[k][i][d])

def ph_feats(structs, fs, backend, P, node, use_ess):
    torch.manual_seed(0)
    ph = T.DifferentiablePH(P, 6, use_ess=use_ess)
    xs = [f.clone().requires_grad_(True) for f in fs]
    M = structs[0]['M']
    plan = T.get_plan(structs, CPU, pdim=P, backend=backend, M_pad=M)
    g, nv = ph(list(zip(structs, xs)), device=CPU, nnodes=M if node else None, plan=plan)
    loss = (g ** 2).sum()
    if(node):
        loss = loss + (nv ** 2).sum()
    loss.backward()
    nv_out = None if nv is None else nv.detach()
    return g.detach(), nv_out, torch.cat([x.grad for x in xs]), plan['use_torch']

def rel(x, y):
    return ((x - y).abs().max() / y.abs().max().clamp_min(1)).item()

def test_grads_vs_gudhi():
    rng = np.random.default_rng(1)
    worst = 0.0
    for t in range(120):
        D = int(rng.integers(1, 4))
        P = int(rng.integers(0, min(D, 2) + 1))
        n = int(rng.integers(4, 30 if D < 3 else 15))
        p = float(rng.uniform(0.1, 0.9))
        tied = bool(rng.integers(2))
        node = bool(rng.integers(2))
        ess = bool(rng.integers(2))
        structs = T.build_structs(rand_graphs(rng, 4, n, p), D)
        fs = filt_vals(rng, structs, tied)
        a = ph_feats(structs, fs, 'gudhi', P, node, ess)
        b = ph_feats(structs, fs, 'torch', P, node, ess)
        assert b[3]
        err = max(rel(b[0], a[0]), rel(b[1], a[1]) if node else 0.0)
        if(not tied):
            err = max(err, rel(b[2], a[2]))
        worst = max(worst, err)
    assert worst < 1e-9

@pytest.mark.parametrize('ds', ['MUTAG', 'PTC'])
def test_multiplicities_exact_on_tu(ds):
    for x in tu_batches(ds):
        structs = bstructs(x)
        M = x.shape[-1]
        torch.manual_seed(3)
        h = torch.relu(torch.nn.Conv2d(x.shape[1], 8, 1)(x))
        res = {}
        for be in ('gudhi', 'torch'):
            torch.manual_seed(5)
            filt = T.LearnedFiltration(8, 16)
            ph = T.DifferentiablePH(1, 8, use_ess=True)
            plan = T.get_plan(structs, CPU, pdim=1, backend=be, M_pad=M)
            assert be == 'gudhi' or plan['use_torch']
            g, nv = ph(filt(h, structs, plan), device=CPU, nnodes=M, plan=plan)
            # pooled finite (8), log1p #finite, pooled essential (8), log1p #essential
            g = g.detach().reshape(len(structs), 2, -1)
            nv = nv.detach().reshape(len(structs), 2, -1, M)
            res[be] = torch.cat([g[:, :, [8, -1]].reshape(-1), nv[:, :, [8, -1]].reshape(-1)])
        assert torch.equal(res['torch'], res['gudhi'])

@pytest.mark.parametrize('ds', ['MUTAG', 'PTC'])
def test_layer_tu(ds):
    for x in tu_batches(ds, n_batches=2):
        structs = bstructs(x)
        torch.manual_seed(3)
        proj = torch.nn.Conv2d(x.shape[1], 8, 1)
        jit = 1e-4 * torch.randn(x.shape[0], 8, x.shape[2], x.shape[3])
        outs = {}
        for be in ('gudhi', 'torch'):
            torch.manual_seed(5)
            L = jiggle(T.TopologyLayer(8, hid=16, pdim=1, nstat=8, norm_st=True, use_ess=True, squash=True, ph_backend=be))
            with torch.no_grad():
                o_tied = L(torch.relu(proj(x)), structs)
            xx = x.clone().requires_grad_(True)
            o = L(torch.relu(proj(xx)) + jit, structs)
            (o ** 2).mean().backward()
            outs[be] = (o_tied, xx.grad, o.detach())
        for k in range(3):
            assert rel(outs['torch'][k], outs['gudhi'][k]) < 1e-7

@pytest.mark.parametrize('backend', ['gudhi', 'torch'])
def test_betti(backend):
    ph = T.DifferentiablePH(1, 4, use_ess=True)
    for(G, betti) in [(nx.cycle_graph(6), (1, 1)), (nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4)), (2, 2)), (nx.complete_graph(5), (1, 0))]:
        st = T.build_structs([nx.to_numpy_array(G)], 2)[0]
        plan = T.get_plan([st], CPU, pdim=1, backend=backend, M_pad=st['M'])
        g, _ = ph([(st, torch.rand(st['S']))], device=CPU, nnodes=None, plan=plan)
        counts = torch.expm1(g[0].reshape(2, -1)[:, -1]).tolist()
        got = tuple(round(v) for v in counts)
        assert got == betti

def test_static_counts():
    rng = np.random.default_rng(4)
    structs = T.build_structs([rand_adj(12, 0.4, rng) for _ in range(3)], 2)
    ph = T.DifferentiablePH(2, 4, static=True)
    plan = T.get_plan(structs, CPU, pdim=2, backend='torch', M_pad=12)
    g, nv = ph([(st, torch.randn(st['S'])) for st in structs], device=CPU, nnodes=12, plan=plan)
    assert torch.isfinite(g).all() and torch.isfinite(nv).all()
    cnt = torch.expm1(g.detach().reshape(3, 3, -1)[:, :, 4])
    want = torch.tensor([[sum(len(s) == d + 1 for s in st['simplices']) for d in range(3)] for st in structs])
    assert torch.allclose(cnt, want.to(cnt.dtype))
