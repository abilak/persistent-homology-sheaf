import os, sys
from types import SimpleNamespace
import numpy as np, torch, networkx as nx, pytest
from easydict import EasyDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import layers.topology as T

CPU = torch.device('cpu')
DEV = torch.device(os.environ.get('DEVICE', 'cuda'))
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')

FINAL = dict(use_topology=True, topo_multiplicity=True, topo_norm_stats=True, topo_essential=True, topo_filt_squash=True, topo_readout=True, topo_hidden_dim=32, topo_num_stats=32, topo_max_ph_dim=1, topo_max_simplex_dim=2)

@pytest.fixture
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)

pytestmark = pytest.mark.usefixtures('float64')

def rand_adj(n, p, rng):
    a = np.triu(rng.random((n, n)) < p, 1).astype(float)
    return a + a.T

def rand_graphs(rng, B, n, p):
    out = []
    for _ in range(B):
        kind = rng.integers(6)
        if(kind == 0):
            a = np.zeros((n, n))
        elif(kind == 1):
            a = np.zeros((n, n))
            k = max(2, n // 3)
            a[:k, :k] = 1
            a[k:2 * k, k:2 * k] = rng.random((k, k)) < 0.7
        else:
            a = rng.random((n, n)) < p
        a = np.triu(a, 1).astype(float)
        out.append(((a + a.T) > 0).astype(float))
    return out

def permute(x, perm):
    return x[:, :, perm][:, :, :, perm]

def onehot(adj, labels, n_labels):
    n = adj.shape[0]
    x = torch.zeros(1, n_labels + 1, n, n)
    x[0, 0] = torch.from_numpy(adj)
    for v in range(n):
        x[0, 1 + int(labels[v]), v, v] = 1.0
    return x

def make_model(n_labels=3, n_classes=2, block=(16, 16), seed=0, **arch):
    from models.base_model import BaseModel
    a = dict(block_features=list(block), depth_of_mlp=2)
    a.update(arch)
    cfg = EasyDict(dict(architecture=a, node_labels=n_labels, num_classes=n_classes))
    torch.manual_seed(seed)
    return BaseModel(cfg)

def jiggle(m, s=0.1, seed=1):
    # zero init heads hide bugs
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            noise = torch.randn(p.shape, generator=g, dtype=p.dtype)
            p.add_(s * noise.to(p.device))
    return m

def pad_input(graphs, pad_to=None):
    n = pad_to or max(g.shape[1] for g in graphs)
    out = np.zeros((len(graphs), graphs[0].shape[0], n, n))
    sizes = np.asarray([g.shape[1] for g in graphs])
    for(k, g) in enumerate(graphs):
        out[k, :, :g.shape[1], :g.shape[1]] = g
    x = torch.from_numpy(out).to(torch.get_default_dtype())
    x._host_adj = x[:, 0].numpy()
    x._n_real = sizes
    x._node_mask = torch.from_numpy(np.arange(n)[None, :] < sizes[:, None])
    return x

def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1)).item()

def sym_input(adj, d, seed=None):
    if(seed is not None):
        torch.manual_seed(seed)
    n = adj.shape[0]
    x = torch.randn(1, d, n, n)
    x[0, 0] = torch.from_numpy(adj)
    return (x + x.transpose(-1, -2)) / 2

@pytest.mark.parametrize('backend', ['gudhi', 'torch'])
def test_ph_perm(backend):
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    filt = T.LearnedFiltration(3, 16).eval()
    ph = T.DifferentiablePH(pdim=1, vec_dim=32, use_ess=True).eval()
    M = 9
    for t in range(5):
        A = rand_adj(M, 0.4, rng)
        x = sym_input(A, 3)
        perm = torch.randperm(M)
        Ap = A[np.ix_(perm.numpy(), perm.numpy())]
        res = []
        for(adj, xx) in ((A, x), (Ap, permute(x, perm))):
            st = T.build_structs([adj], 2)
            plan = T.get_plan(st, CPU, pdim=1, backend=backend, M_pad=M)
            with torch.no_grad():
                res.append(ph(filt(xx, st, plan), device=CPU, nnodes=M, plan=plan))
        g, nv = res[0]
        gp, nvp = res[1]
        assert (g - gp).abs().max() < 1e-9
        assert (nv[:, :, perm] - nvp).abs().max() < 1e-9

LAYER_OPTS = [dict(), dict(use_ess=True), dict(norm_st=True, use_ess=True, squash=True)]

@pytest.mark.parametrize('backend', ['gudhi', 'torch'])
@pytest.mark.parametrize('opts', LAYER_OPTS)
def test_layer_perm(opts, backend):
    M, d = 11, 6
    torch.manual_seed(11)
    L = jiggle(T.TopologyLayer(d, hid=16, pdim=1, nstat=8, ph_backend=backend, **opts)).eval()
    worst = 0.0
    for t in range(3):
        A = rand_adj(M, 0.3, np.random.default_rng(500 + t))
        x = sym_input(A, d)
        perm = torch.randperm(M)
        Ap = A[np.ix_(perm.numpy(), perm.numpy())]
        with torch.no_grad():
            o = L(x, T.build_structs([A], 2))
            op = L(permute(x, perm), T.build_structs([Ap], 2))
        worst = max(worst, (permute(o, perm) - op).abs().max().item())
    assert worst < 1e-9

MODELS = {'baseline': dict(), 'gudhi': dict(FINAL, topo_ph_backend='gudhi'), 'torch': dict(FINAL, topo_ph_backend='torch'), 'd3': dict(FINAL, topo_max_ph_dim=2, topo_max_simplex_dim=3, topo_ph_backend='gudhi'), 'd3_torch': dict(FINAL, topo_max_ph_dim=2, topo_max_simplex_dim=3, topo_ph_backend='torch'), 'betti': dict(betti_control=True, topo_max_simplex_dim=2), 'static': dict(FINAL, topo_static=True)}

@pytest.mark.parametrize('name', list(MODELS))
def test_invariance(name):
    arch = MODELS[name]
    m = jiggle(make_model(**arch)).eval()
    D = arch.get('topo_max_simplex_dim', 2)
    rng = np.random.default_rng(4)
    torch.manual_seed(1)
    worst = 0.0
    for t in range(5):
        if(D == 2):
            n = int(rng.integers(10, 20))
        else:
            n = int(rng.integers(8, 14))
        A = rand_graphs(rng, 1, n, float(rng.uniform(0.2, 0.5)))[0]
        x = onehot(A, rng.integers(0, 3, n), 3)
        with torch.no_grad():
            o = m(x)
            assert torch.isfinite(o).all()
            for _ in range(2):
                perm = torch.randperm(n)
                worst = max(worst, (o - m(permute(x, perm))).abs().max().item())
    assert worst < 1e-9

def test_topo_not_dead():
    # topo branch shouldn't be dead
    rng = np.random.default_rng(0)
    A = rand_adj(12, 0.4, rng)
    x = onehot(A, rng.integers(0, 3, 12), 3)
    a = jiggle(make_model(**FINAL))
    b = jiggle(make_model(**dict(FINAL, topo_ph_backend='torch')))
    with torch.no_grad():
        oa = a(x)
        ob = b(x)
        for L in a.topo_layers:
            for p in L.filtration.parameters():
                p.mul_(1.5)
        oa2 = a(x)
    assert torch.allclose(oa, ob, atol=1e-9)
    assert (oa - oa2).abs().max() > 1e-6

def test_inv_ln():
    n = T.InvertibleLayerNorm(9)
    x = torch.randn(5, 9) * 3 + 1
    y = n(x)
    assert y.shape == (5, 11)
    xr = y[:, 9:10] + torch.exp(y[:, 10:11]) * y[:, :9]
    assert (xr - x).abs().max() < 1e-4

def test_betti_vec():
    cases = [(nx.cycle_graph(6), 2, (1, 1, 0)), (nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4)), 2, (2, 2, 0)), (nx.complete_graph(4), 2, (1, 0, 1)), (nx.complete_graph(4), 3, (1, 0, 0, 0)), (nx.empty_graph(3), 2, (3, 0, 0))]
    for(G, D, want) in cases:
        st = T.get_struct(nx.to_numpy_array(G), D)
        assert tuple(T.betti_vec(st, D)) == want

def test_struct_is_clique_complex():
    rng = np.random.default_rng(0)
    for t in range(40):
        n = int(rng.integers(1, 25))
        D = int(rng.integers(1, 4))
        A = rand_adj(n, float(rng.uniform(0, 1)), rng)
        st = T.graph_struct(A, D)
        cl = set()
        for c in nx.enumerate_all_cliques(nx.from_numpy_array(A)):
            if(len(c) <= D + 1):
                cl.add(tuple(sorted(c)))
        assert st['S'] == len(st['simplices']) == len(cl)
        assert set(map(tuple, st['simplices'])) == cl
        assert st['M'] == n
    K5 = nx.to_numpy_array(nx.complete_graph(5))
    s2, s3 = T.get_struct(K5, 2), T.get_struct(K5, 3)
    assert max(map(len, s2['simplices'])) == 3 and max(map(len, s3['simplices'])) == 4

def same_size(B, M, d, seed):
    torch.manual_seed(seed)
    adjs = [rand_adj(M, 0.3, np.random.default_rng(seed * 100 + b)) for b in range(B)]
    x = torch.randn(B, d, M, M)
    for(b, A) in enumerate(adjs):
        x[b, 0] = torch.from_numpy(A)
    x = (x + x.transpose(-1, -2)) / 2
    return x, T.build_structs(adjs, 2)

@pytest.mark.parametrize('backend', ['gudhi', 'torch'])
@pytest.mark.parametrize('opts', [dict(), dict(use_ess=True), dict(norm_st=True), dict(norm_st=True, use_ess=True, squash=True)])
def test_batch_vs_single(opts, backend):
    x, structs = same_size(6, 12, 8, seed=1)
    torch.manual_seed(7)
    L = jiggle(T.TopologyLayer(8, hid=16, pdim=1, nstat=8, ph_backend=backend, **opts)).eval()
    xb = x.clone().requires_grad_(True)
    ob = L(xb, structs)
    ob.sum().backward()
    outs = []
    one_grads = []
    for b in range(6):
        xs = x[b:b + 1].clone().requires_grad_(True)
        o = L(xs, structs[b:b + 1])
        o.sum().backward()
        outs.append(o)
        one_grads.append(xs.grad)
    assert rel(ob, torch.cat(outs)) < 1e-9
    assert rel(xb.grad, torch.cat(one_grads)) < 1e-9

def test_batch_order():
    x, structs = same_size(6, 12, 8, seed=2)
    torch.manual_seed(3)
    L = T.TopologyLayer(8, hid=16, pdim=1, nstat=8, use_ess=True).eval()
    perm = list(reversed(range(6)))
    with torch.no_grad():
        o1 = L(x, structs)
        o2 = L(x[perm], [structs[i] for i in perm])
    assert (o1[perm] - o2).abs().max() < 1e-9

def mixed(n_graphs=8, C=4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in rng.integers(6, 18, n_graphs):
        g = np.zeros((C, n, n))
        g[0] = rand_adj(n, 0.3, rng)
        for c in range(1, C):
            g[c][np.diag_indices(n)] = rng.standard_normal(n)
        out.append(g)
    return out

PAD_MODELS = {'baseline': dict(), 'gudhi': dict(FINAL, topo_ph_backend='gudhi'), 'torch': dict(FINAL, topo_ph_backend='torch'), 'D3': dict(FINAL, topo_max_ph_dim=2, topo_max_simplex_dim=3, topo_ph_backend='torch'), 'betti': dict(betti_control=True)}

def grads(m):
    return [None if p.grad is None else p.grad.clone() for p in m.parameters()]

@pytest.mark.parametrize('name', list(PAD_MODELS))
def test_padding(name):
    gs = mixed()
    m = jiggle(make_model(**PAD_MODELS[name]))
    solo = []
    solo_g = []
    for g in gs:
        m.zero_grad()
        out = m(pad_input([g]))
        out.square().sum().backward()
        solo.append(out.detach()[0])
        solo_g.append(grads(m))
        with torch.no_grad():
            unpadded = m(torch.from_numpy(g)[None])[0]
            assert (unpadded - solo[-1]).abs().max() < 1e-10
    m.zero_grad()
    out = m(pad_input(gs, pad_to=20))
    out.square().sum().backward()
    worst = 0.0
    for k in range(len(gs)):
        worst = max(worst, (out[k].detach() - solo[k]).abs().max().item())
    assert worst < 1e-10
    for(i, bg) in enumerate(grads(m)):
        if(bg is None):
            continue
        sg = sum(s[i] for s in solo_g if s[i] is not None)
        assert rel(bg, sg) < 1e-10

@pytest.mark.parametrize('arch', [dict(FINAL, topo_ph_backend='gudhi'), dict(FINAL, topo_ph_backend='torch'), dict(baseline_type='gin')])
def test_microbatch(arch):
    from models.base_model import build_model
    from trainers.trainer import Trainer
    x = pad_input(mixed(9, seed=3))
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1])
    data = SimpleNamespace(next_batch=lambda: (x, y))
    res = []
    for mb in (0, 4):
        a = dict(block_features=[16, 16], depth_of_mlp=2, **arch)
        cfg = EasyDict(dict(dataset_name='MUTAG', node_labels=3, num_classes=2, architecture=a, hyperparams=dict(learning_rate=1e-3, decay_rate=0.5, micro_batch=mb)))
        torch.manual_seed(0)
        net = jiggle(build_model(cfg))
        if('baseline_type' in arch):
            net.eval()
        loss, correct = Trainer(net, data, cfg).train_step()
        res.append((float(loss), int(correct), grads(net)))
    (l0, c0, g0), (l1, c1, g1) = res
    assert abs(l0 - l1) < 1e-9 * max(1, abs(l0)) and c0 == c1
    for(a, b) in zip(g0, g1):
        if(a is None):
            continue
        assert rel(b, a) < 1e-9

@pytest.mark.parametrize('kind', ['mlp', 'gcn', 'gin', 'gsn'])
def test_baseline_pad(kind):
    from models.baseline_models import BaselineModel
    gs = mixed(6, seed=5)
    cfg = EasyDict(dict(node_labels=3, num_classes=2, architecture=dict(block_features=[16, 16], baseline_type=kind)))
    torch.manual_seed(0)
    m = jiggle(BaselineModel(cfg))
    m.eval()
    with torch.no_grad():
        out = m(pad_input(gs))
        for(k, g) in enumerate(gs):
            assert (out[k] - m(torch.from_numpy(g)[None])[0]).abs().max() < 1e-10
    m.train()
    res = []
    for pad in (None, 25):
        torch.manual_seed(1)
        res.append(m(pad_input(gs, pad_to=pad)))
    assert (res[0] - res[1]).abs().max() < 1e-10

def tu_batches(name, n_batches=3, bs=16):
    if(not os.path.exists(os.path.join(ROOT, 'data', 'benchmark_graphs', name, name + '.txt'))):
        pytest.skip('%s not in data/benchmark_graphs' % name)
    import data_loader.data_helper as helper
    graphs, _ = helper.load_dataset(name)
    order = np.argsort([g.shape[1] for g in graphs], kind='stable')
    starts = np.linspace(0, len(order) - bs, n_batches).astype(int)
    return [pad_input([graphs[i].astype(np.float64) for i in order[s:s + bs]]) for s in starts]

def to_dev(x):
    # .to() drops the extra attributes
    y = x.to(DEV)
    for k in ('_host_adj', '_n_real'):
        if(hasattr(x, k)):
            setattr(y, k, getattr(x, k))
    if(hasattr(x, '_node_mask')):
        y._node_mask = x._node_mask.to(DEV)
    return y

@needs_cuda
@pytest.mark.parametrize('ds', ['MUTAG', 'PTC', 'NCI1', 'PROTEINS', 'IMDBBINARY'])
def test_cuda_vs_gudhi(ds):
    for x in tu_batches(ds, n_batches=4):
        adjs = [x._host_adj[b, :k, :k] for b, k in enumerate(x._n_real)]
        structs = T.build_structs(adjs, 2)
        M = x.shape[-1]
        torch.manual_seed(3)
        h = torch.relu(torch.nn.Conv2d(x.shape[1], 8, 1)(x)).detach()
        res = {}
        for(be, d) in (('gudhi', CPU), ('torch', DEV)):
            torch.manual_seed(5)
            filt = T.LearnedFiltration(8, 16).to(d)
            ph = T.DifferentiablePH(1, 8, use_ess=True).to(d)
            plan = T.get_plan(structs, d, pdim=1, backend=be, M_pad=M)
            assert be == 'gudhi' or plan['use_torch']
            g, nv = ph(filt(h.to(d), structs, plan), device=d, nnodes=M, plan=plan)
            g = g.detach().cpu().reshape(len(structs), 2, -1)
            nv = nv.detach().cpu().reshape(len(structs), 2, -1, M)
            res[be] = (g, nv)
        ga, na = res['gudhi']
        gb, nb = res['torch']
        assert max(rel(gb, ga), rel(nb, na)) < 1e-7
        assert torch.equal(gb[:, :, [8, -1]], ga[:, :, [8, -1]]) and torch.equal(nb[:, :, [8, -1]], na[:, :, [8, -1]])

@needs_cuda
def test_cuda_invariance():
    m = jiggle(make_model(**dict(FINAL, topo_ph_backend='torch'))).to(DEV).eval()
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(6):
        n = 18
        x = onehot(rand_adj(n, 0.35, rng), rng.integers(0, 3, n), 3)
        perm = torch.randperm(n)
        xp = permute(x, perm)
        ha, hb = x[:, 0].numpy().copy(), xp[:, 0].numpy().copy()
        x, xp = x.to(DEV), xp.to(DEV)
        x._host_adj, xp._host_adj = ha, hb
        with torch.no_grad():
            worst = max(worst, (m(x) - m(xp)).abs().max().item())
    assert worst < 1e-9

@needs_cuda
def test_no_sync():
    torch.set_default_dtype(torch.float32)
    batches = [to_dev(x) for x in tu_batches('MUTAG', n_batches=6, bs=20)]
    m = make_model(n_labels=7, block=(64, 64), **dict(FINAL, topo_ph_backend='torch')).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, fused=True)
    ys = [torch.randint(0, 2, (x.shape[0],), device=DEV) for x in batches]
    for(x, y) in zip(batches[:2], ys[:2]):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y, reduction='sum').backward()
        opt.step()
    torch.cuda.synchronize()
    for(x, y) in zip(batches[2:], ys[2:]):
        torch.cuda.set_sync_debug_mode('error')
        try:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(x), y, reduction='sum').backward()
            opt.step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
