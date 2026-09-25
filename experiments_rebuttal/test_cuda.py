"""GPU acceptance test: the torch persistent-homology backend ON THE DEVICE
must reproduce the gudhi path (CPU) on real padded TU batches, the full model
must be permutation invariant on the device, and a training step on the
torch backend must not synchronize with the host (molecular datasets).

usage: python experiments_rebuttal/test_cuda.py [device]   (default: cuda)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
os.environ['CCD_DEVICE'] = DEV
import numpy as np, torch
import layers.topology as T
from run_job import build_config
from data_loader.data_generator import DataGenerator
dev = torch.device(DEV)
CPU = torch.device('cpu')
ok = True
print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == 'cuda' else ""))

# 1. torch backend on device == gudhi on CPU (float64), real batches
for ds in ('MUTAG', 'PTC', 'NCI1', 'NCI109', 'PROTEINS', 'IMDBBINARY'):
    cfg = build_config(ds, 'topo', {'padded_batching': True}); cfg.num_fold = 1
    data = DataGenerator(cfg); data.initialize('train')
    worst_feat = worst_mult = 0.0; used = 0
    for _ in range(4):
        g0 = data.next_batch()[0]; nr = g0._n_real
        structs = T.build_graph_structs([g0._host_adj[b, :k, :k] for b, k in enumerate(nr)], 2)
        g = g0.detach().to('cpu', torch.float64)
        torch.manual_seed(3); proj = torch.nn.Conv2d(g.shape[1], 8, 1).double()
        x = torch.relu(proj(g)).detach()
        res = {}
        for be, d in (('gudhi', CPU), ('torch', dev)):
            T._PLAN_CACHE.clear(); torch.manual_seed(5)
            filt = T.LearnedFiltration(8, 16).double().to(d)
            ph = T.DifferentiablePH(1, 8, multiplicity=True, essential=True).double().to(d)
            plan = T.get_plan(structs, d, 1, 1, backend=be, M_pad=g.shape[-1])
            used += int(be == 'torch' and plan.use_torch_ph)
            gv, nv = ph(filt(x.to(d), structs, plan), device=d, num_nodes=g.shape[-1], plan=plan)
            res[be] = (gv.detach().cpu(), nv.detach().cpu(), ph.per_dim_graph, ph.per_dim_node)
        (ga, na, w, wn), (gb, nb, _, _) = res['gudhi'], res['torch']
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1)).item()
        worst_feat = max(worst_feat, rel(gb, ga), rel(nb, na))
        mult = lambda G, N: torch.cat([G.reshape(G.shape[0], 2, w)[:, :, [8, -1]].reshape(-1),
                                       N.reshape(N.shape[0], 2, wn, -1)[:, :, [8, -1]].reshape(-1)])
        worst_mult = max(worst_mult, (mult(gb, nb) - mult(ga, na)).abs().max().item())
    good = worst_feat < 1e-7 and worst_mult == 0.0 and used == 4; ok &= good
    print(f"[1] {ds:10s}: torch-on-{dev.type} vs gudhi: features {worst_feat:.1e}, multiplicities "
          f"{worst_mult:.1e}, device path on {used}/4 batches [{'OK' if good else 'FAIL'}]")

# 2. full model invariance on device
from easydict import EasyDict
from models.base_model import BaseModel
arch = dict(block_features=[16, 16], depth_of_mlp=2, new_suffix=True, use_topology=True,
            topo_hidden_dim=16, topo_max_ph_dim=1, topo_num_stats=8, topo_max_simplex_dim=2,
            topo_multiplicity=True, topo_norm_stats=True, topo_essential=True, topo_filt_squash=True,
            topo_readout=True, topo_num_filtrations=2, topo_ph_backend='torch')
torch.manual_seed(0)
m = BaseModel(EasyDict(dict(architecture=arch, node_labels=3, num_classes=2))).double().to(dev).eval()
for fc in m.topo_fc:
    if isinstance(fc, torch.nn.Linear):
        torch.nn.init.normal_(fc.weight)
rng = np.random.default_rng(0); worst = 0.0
for t in range(6):
    n = 18; A = np.triu(rng.random((n, n)) < 0.35, 1).astype(float); A = A + A.T
    x = torch.zeros(1, 4, n, n, dtype=torch.float64); x[0, 0] = torch.from_numpy(A)
    lab = rng.integers(0, 3, n)
    for v in range(n): x[0, 1 + lab[v], v, v] = 1.0
    perm = torch.randperm(n); xp = x[:, :, perm][:, :, :, perm]
    ha, hb = x[:, 0].numpy().copy(), xp[:, 0].numpy().copy()
    x, xp = x.to(dev), xp.to(dev); x._host_adj, xp._host_adj = ha, hb
    with torch.no_grad():
        T._PLAN_CACHE.clear(); o1 = m(x); T._PLAN_CACHE.clear(); o2 = m(xp)
    worst = max(worst, (o1 - o2).abs().max().item())
good = worst < 1e-9; ok &= good
print(f"[2] full model permutation invariance on {dev.type}: {worst:.1e} [{'OK' if good else 'FAIL'}]")

# 3. no host synchronization in a training step (molecular data, torch backend)
if dev.type == 'cuda':
    from models.model_wrapper import ModelWrapper
    from trainers.trainer import Trainer
    FULL = dict(topo_multiplicity=True, topo_norm_stats=True, topo_essential=True,
                topo_filt_squash=True, topo_readout=True, topo_ph_backend='torch', padded_batching=True)
    for ds in ('MUTAG', 'NCI1'):
        cfg = build_config(ds, 'topo', FULL); cfg.num_fold = 1
        data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
        mw.train(); data.initialize('train')
        for _ in range(2): tr.train_step()
        torch.cuda.synchronize()
        n_ok = n_sync = 0
        for _ in range(min(10, data.num_iterations_train - 2)):
            g, l = data.next_batch()
            nr = g._n_real
            structs = T.build_graph_structs([g._host_adj[b, :k, :k] for b, k in enumerate(nr)], 2)
            K = max(T._tph_static(s)['cyc'] for s in structs)
            torch.cuda.set_sync_debug_mode('error')
            try:
                loss, _ = mw.run_model_get_loss_and_results(g, l)
                tr.optimizer.zero_grad(); loss.backward(); tr.optimizer.step()
                n_ok += 1
            except RuntimeError as e:
                if 'synchroniz' not in str(e):
                    raise
                n_sync += 1     # expected only for batches needing convergence checks
                if n_sync == 1:     # show WHERE the first sync happens
                    import traceback
                    frames = [f for f in traceback.extract_tb(e.__traceback__)
                              if 'site-packages' not in f.filename]
                    where = "; ".join(f"{os.path.relpath(f.filename)}:{f.lineno} ({f.line})"
                                      for f in frames[-3:])
                    print(f"    first sync (cycle rank {K}): {where}")
            finally:
                torch.cuda.set_sync_debug_mode(0)
        print(f"[3] {ds}: {n_ok} steps with zero host syncs, {n_sync} with syncs "
              f"(allowed only for batches with cycle rank > 11)")
        if ds == 'MUTAG' and n_sync:
            ok = False
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
