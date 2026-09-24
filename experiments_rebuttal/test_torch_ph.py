"""Verification of the GPU-resident persistent homology (layers/torch_ph.py).

 1. DIAGRAM FUZZ against gudhi's own API (independent oracle): random graphs
    (sizes 4-36, densities 0.1-0.9, disconnected, empty, complete pieces),
    clique complexes of dimension 1-3, homology up to 2, random and heavily
    TIED filtrations. For every graph and dimension, the multiset of finite
    (birth, death) pairs and of essential births must equal gudhi's.
 2. FEATURE + GRADIENT agreement of the full vectorization (graph- and
    node-level, multiplicities, essential blocks) against the gudhi path.
 3. Both reduction schedules (sync-free fixed steps / convergence checks)
    agree.
 4. Real TU batches (all six datasets, torch path forced, 3 filtration heads):
    forward on tied inputs, forward + gradients on tie-broken inputs.
 5. Permutation invariance of the full model on the torch backend.
 6. Pairing-determined coordinates (multiplicities) EXACTLY equal on real
    batches of all six datasets, on tied (unjittered) inputs.
All in float64."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import numpy as np, torch, networkx as nx, gudhi, functools
print = functools.partial(print, flush=True)
torch.set_default_dtype(torch.float64)
import layers.topology as T
import layers.torch_ph as TP
T.BatchPlan.TORCH_PH_MAX_ELEMS = 1 << 24   # keep CPU test runs quick
TP.MAX_BLOCK_ELEMS = 1 << 22             # force chunked reductions in the tests
CPU = torch.device('cpu')


def rand_graphs(rng, B, n, p):
    out = []
    for _ in range(B):
        kind = rng.integers(6)
        if kind == 0:
            A = np.zeros((n, n))
        elif kind == 1:        # disjoint dense blocks + isolated vertices
            A = np.zeros((n, n)); k = max(2, n // 3)
            A[:k, :k] = 1; A[k:2 * k, k:2 * k] = (rng.random((k, k)) < 0.7)
        else:
            A = (rng.random((n, n)) < p).astype(float)
        A = np.triu(A, 1); A = A + A.T; A = (A > 0).astype(float)
        out.append(A)
    return out


def filtration(rng, structs, tied):
    fs = []
    for st in structs:
        if tied:     # few distinct values -> massive ties (after max-correction too)
            fs.append(torch.tensor(rng.integers(0, 3, st.S) * 0.5))
        else:
            fs.append(torch.tensor(rng.standard_normal(st.S)))
    return fs


def corrected(structs, fs):
    ph = T.DifferentiablePH(0, 2)
    out = []
    for st, f in zip(structs, fs):
        tabs = [st.on(CPU)[3][d] for d in sorted(st.on(CPU)[3])]
        out.append(ph._make_non_decreasing(f, tabs).detach().numpy())
    return out


def gudhi_diagrams(st, vals, P):
    t = gudhi.SimplexTree()
    for i, sg in enumerate(st.simplices_list):
        t.insert(list(sg), filtration=float(vals[i]))
    t.make_filtration_non_decreasing()
    t.persistence(persistence_dim_max=True)
    fin = {d: [] for d in range(P + 1)}; ess = {d: [] for d in range(P + 1)}
    for d in range(P + 1):
        for b, dd in t.persistence_intervals_in_dimension(d):
            if np.isinf(dd):
                ess[d].append(b)
            elif dd > b:
                fin[d].append((b, dd))
    return fin, ess


def torch_diagrams(structs, fs, P):
    plan = T.BatchPlan(structs, CPU, 1, P, None, 'torch', structs[0].M)
    assert plan.use_torch_ph
    ph = T.DifferentiablePH(P, 2)
    tabs = [(plan.t[f'sim{d}'], plan.t[f'face{d}']) for d in plan.face_dims]
    adj = ph._make_non_decreasing(torch.cat(fs), tabs)
    blocks = TP.torch_persistence(adj, plan, P + 1, True, False, None)
    B = len(structs)
    fin = [{d: [] for d in range(P + 1)} for _ in range(B)]
    ess = [{d: [] for d in range(P + 1)} for _ in range(B)]
    for (kind, d), blk in blocks.items():
        v = blk['valid'].numpy(); b = blk['b'].detach().numpy()
        dd = blk['d'].detach().numpy() if kind == 'fin' else None
        for i in range(B):
            for r in np.nonzero(v[i])[0]:
                if kind == 'fin':
                    fin[i][d].append((b[i, r], dd[i, r]))
                else:
                    ess[i][d].append(b[i, r])
    return fin, ess


def same(a, b):
    a, b = sorted(a), sorted(b)
    return len(a) == len(b) and all(np.allclose(x, y, atol=1e-12) for x, y in zip(a, b))


def part1(n_trials=150):
    rng = np.random.default_rng(0)
    bad = 0; cnt = 0; stats = dict(fin=[0, 0, 0], ess=[0, 0, 0])
    for trial in range(n_trials):
        D = int(rng.integers(1, 4)); P = int(rng.integers(0, min(D, 2) + 1))
        n = int(rng.integers(4, 37 if D < 3 else 17)); p = float(rng.uniform(0.1, 0.9))
        tied = bool(rng.integers(2))
        adjs = rand_graphs(rng, 3, n, p)
        structs = T.build_graph_structs(adjs, D)
        plan = T.BatchPlan(structs, CPU, 1, P, None, 'torch', n)
        if not plan.use_torch_ph:
            continue
        fs = filtration(rng, structs, tied)
        tf, te = torch_diagrams(structs, fs, P)
        cv = corrected(structs, fs)
        for i, st in enumerate(structs):
            gf, ge = gudhi_diagrams(st, cv[i], P)
            for d in range(P + 1):
                cnt += 1
                stats['fin'][d] += len(gf[d]); stats['ess'][d] += len(ge[d])
                if not (same(gf[d], tf[i][d]) and same(ge[d], te[i][d])):
                    bad += 1
                    if bad <= 5:
                        print(f"  DIAGRAM MISMATCH trial {trial} n={n} p={p:.2f} D={D} P={P} tied={tied} dim {d}: "
                              f"gudhi fin {len(gf[d])} ess {len(ge[d])} | torch fin {len(tf[i][d])} ess {len(te[i][d])}")
    print(f"[1] diagrams vs gudhi API: {cnt - bad}/{cnt} (graph, dim) diagrams identical; "
          f"pairs checked: finite {stats['fin']}, essential {stats['ess']} (dims 0,1,2)")
    return bad == 0


def features(structs, fs, be, opts, P, node):
    T._PLAN_CACHE.clear()
    torch.manual_seed(0)
    ph = T.DifferentiablePH(P, 6, **opts)
    xs = [f.clone().requires_grad_(True) for f in fs]
    plan = T.get_plan(structs, CPU, 1, P, backend=be, M_pad=structs[0].M)
    M = structs[0].M
    g, nv = ph([(st, f) for st, f in zip(structs, xs)], device='cpu',
               num_nodes=M if node else None, plan=plan)
    ((g ** 2).sum() + ((nv ** 2).sum() if nv is not None else 0)).backward()
    return g.detach(), None if nv is None else nv.detach(), torch.cat([x.grad for x in xs]), plan.use_torch_ph


def part2(n_trials=120):
    rng = np.random.default_rng(1); bad = 0; cnt = 0; worst = 0.0
    for trial in range(n_trials):
        D = int(rng.integers(1, 4)); P = int(rng.integers(0, min(D, 2) + 1))
        n = int(rng.integers(4, 30 if D < 3 else 15)); p = float(rng.uniform(0.1, 0.9))
        tied = bool(rng.integers(2)); node = bool(rng.integers(2))
        opts = dict(multiplicity=bool(rng.integers(2)), essential=bool(rng.integers(2)),
                    scale_stats=bool(rng.integers(2)))
        structs = T.build_graph_structs(rand_graphs(rng, 4, n, p), D)
        fs = filtration(rng, structs, tied)
        a = features(structs, fs, 'gudhi', opts, P, node)
        b = features(structs, fs, 'torch', opts, P, node)
        if not b[3]:
            continue
        rel = lambda x, y: ((x - y).abs().max() / y.abs().max().clamp_min(1)).item()
        m = max(rel(b[0], a[0]), rel(b[1], a[1]) if node else 0.0)
        if not tied:     # gradient routing among tied simplices is tie-break defined
            m = max(m, rel(b[2], a[2]))
        cnt += 1; worst = max(worst, m)
        if not (m < 1e-9):
            bad += 1
            if bad <= 5:
                print(f"  FEATURE MISMATCH trial {trial}: n={n} D={D} P={P} tied={tied} node={node} {opts}: {m:.1e}")
    print(f"[2] features/grads vs gudhi path: {cnt - bad}/{cnt} configurations agree (worst {worst:.1e})")
    return bad == 0


def part3():
    rng = np.random.default_rng(2)
    structs = T.build_graph_structs(rand_graphs(rng, 4, 11, 0.5), 3)
    fs = filtration(rng, structs, False)
    old = TP.FIXED_STEPS_MAX
    TP.FIXED_STEPS_MAX = 10 ** 9; a = torch_diagrams(structs, fs, 2)
    TP.FIXED_STEPS_MAX = 0;       b = torch_diagrams(structs, fs, 2)
    TP.FIXED_STEPS_MAX = old
    ok = all(same(a[k][i][d], b[k][i][d]) for k in range(2) for i in range(4) for d in range(3))
    print(f"[3] fixed-step and convergence-checked reductions agree: {ok}")
    return ok


def part4(heads=3, n_batches=3):
    from run_job import build_config
    from data_loader.data_generator import DataGenerator
    ok = True
    for ds in ('MUTAG', 'PTC', 'NCI1', 'NCI109', 'PROTEINS', 'IMDBBINARY'):
        cfg = build_config(ds, 'topo', {'padded_batching': True}); cfg.num_fold = 1
        data = DataGenerator(cfg); data.initialize('train')
        worst = 0.0; used = 0
        for _ in range(n_batches):
            g0 = data.next_batch()[0]
            g = g0.double(); nr = g0._n_real
            structs = T.build_graph_structs([g0._host_adj[b, :k, :k] for b, k in enumerate(nr)], 2)
            torch.manual_seed(3)
            proj = torch.nn.Conv2d(g.shape[1], 8, 1)
            jitter = 1e-4 * torch.randn(g.shape[0], 8, g.shape[2], g.shape[3])
            outs = {}
            for be in ('gudhi', 'torch'):
                T._PLAN_CACHE.clear(); torch.manual_seed(5)
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
                    used += T.get_plan(structs, CPU, heads, 1, backend='torch', M_pad=g.shape[-1]).use_torch_ph
            rel = lambda x, y: ((x - y).abs().max() / y.abs().max().clamp_min(1)).item()
            worst = max(worst, *(rel(outs['torch'][k], outs['gudhi'][k]) for k in range(3)))
        # 1e-7: the per-graph standardization (x - mu) / (std + 1e-8) of the
        # pair inputs amplifies float64 summation-order noise when a segment's
        # pairs are nearly equal (std ~ 1e-10). Pairing-determined quantities
        # (multiplicities, essential counts) are checked EXACTLY in part 6.
        good = worst < 1e-7; ok &= good
        print(f"[4] {ds:10s}: full layer torch vs gudhi {worst:.1e}, torch path on {used}/{n_batches} batches "
              f"[{'OK' if good else 'FAIL'}]")
    return ok


def part6(n_batches=3):
    """Pairing-determined coordinates (finite and essential multiplicities)
    must agree EXACTLY on real batches of every dataset."""
    from run_job import build_config
    from data_loader.data_generator import DataGenerator
    ok = True
    for ds in ('MUTAG', 'PTC', 'NCI1', 'NCI109', 'PROTEINS', 'IMDBBINARY'):
        cfg = build_config(ds, 'topo', {'padded_batching': True}); cfg.num_fold = 1
        data = DataGenerator(cfg); data.initialize('train')
        worst = 0.0; used = 0
        for _ in range(n_batches):
            g0 = data.next_batch()[0]; g = g0.double(); nr = g0._n_real
            structs = T.build_graph_structs([g0._host_adj[b, :k, :k] for b, k in enumerate(nr)], 2)
            torch.manual_seed(3); proj = torch.nn.Conv2d(g.shape[1], 8, 1)
            x = torch.relu(proj(g))
            res = {}
            for be in ('gudhi', 'torch'):
                T._PLAN_CACHE.clear(); torch.manual_seed(5)
                filt = T.LearnedFiltration(8, 16)
                ph = T.DifferentiablePH(1, 8, multiplicity=True, essential=True)
                plan = T.get_plan(structs, CPU, 1, 1, backend=be, M_pad=g.shape[-1])
                used += int(be == 'torch' and plan.use_torch_ph)
                gv, nv = ph(filt(x, structs, plan), device='cpu', num_nodes=g.shape[-1], plan=plan)
                w = ph.per_dim_graph; wn = ph.per_dim_node
                gb = gv.detach().reshape(gv.shape[0], 2, w)
                nb = nv.detach().reshape(nv.shape[0], 2, wn, -1)
                res[be] = torch.cat([gb[:, :, 8].reshape(-1), gb[:, :, -1].reshape(-1),
                                     nb[:, :, 8].reshape(-1), nb[:, :, -1].reshape(-1)])
            worst = max(worst, (res['torch'] - res['gudhi']).abs().max().item())
        good = worst == 0.0; ok &= good
        print(f"[6] {ds:10s}: multiplicities (graph + node, finite + essential) torch vs gudhi: "
              f"max diff {worst:.1e} on {used}/{n_batches} torch batches [{'OK' if good else 'FAIL'}]")
    return ok


def part5():
    from easydict import EasyDict
    from models.base_model import BaseModel
    arch = dict(block_features=[16, 16], depth_of_mlp=2, new_suffix=True, use_topology=True,
                topo_hidden_dim=16, topo_max_ph_dim=2, topo_num_stats=8, topo_max_simplex_dim=3,
                topo_multiplicity=True, topo_norm_stats=True, topo_essential=True,
                topo_filt_squash=True, topo_readout=True, topo_num_filtrations=2, topo_ph_backend='torch')
    torch.manual_seed(0); m = BaseModel(EasyDict(dict(architecture=arch, node_labels=3, num_classes=2))).eval()
    for fc in m.topo_fc:
        if isinstance(fc, torch.nn.Linear):
            torch.nn.init.normal_(fc.weight)
    rng = np.random.default_rng(4); worst = 0.0
    for t in range(4):
        A = rand_graphs(rng, 1, 14, 0.45)[0]
        x = torch.zeros(1, 4, 14, 14); x[0, 0] = torch.from_numpy(A)
        lab = rng.integers(0, 3, 14)
        for v in range(14): x[0, 1 + lab[v], v, v] = 1.0
        perm = torch.randperm(14)
        xp = x[:, :, perm][:, :, :, perm]
        x._host_adj = x[:, 0].numpy(); xp._host_adj = xp[:, 0].numpy()
        with torch.no_grad():
            T._PLAN_CACHE.clear(); o1 = m(x)
            T._PLAN_CACHE.clear(); o2 = m(xp)
        worst = max(worst, (o1 - o2).abs().max().item())
    print(f"[5] full model (D=3, H0-H2, 2 heads, torch backend) permutation invariance: {worst:.1e}")
    return worst < 1e-9


if __name__ == '__main__':
    parts = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    fns = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6}
    results = [fns[k]() for k in parts]
    print("PASS" if all(results) else "FAIL"); sys.exit(0 if all(results) else 1)
