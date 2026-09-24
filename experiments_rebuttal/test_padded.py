"""Padded batching must be EXACT: each graph's output (and the parameter
gradients of a per-graph loss) in a padded mixed-size batch equal those of the
graph processed alone. Baseline PPGN and PPGN+PH (both PH backends), float64."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from easydict import EasyDict
import data_loader.data_helper as helper
from data_loader.data_generator import DataGenerator
import layers.topology as T
from models.base_model import BaseModel

graphs, labels = helper.load_dataset('MUTAG')
rng = np.random.default_rng(0)
idx = rng.choice(len(graphs), 10, replace=False)
sel = [graphs[i].astype(np.float64).copy() for i in idx]
# continuous node features on the label channels: removes the EXACT ties that
# one-hot labels create (under exact ties, 1e-16 summation-order differences
# between batch layouts legitimately flip argmax / PH tie-breaks and hence
# gradient routing; forward values are unaffected either way)
for g in sel:
    for c in range(1, g.shape[0]):
        g[c][np.diag_indices(g.shape[1])] = rng.standard_normal(g.shape[1])
print("sizes:", sorted(g.shape[1] for g in sel))
C = sel[0].shape[0]


def as_input(arr, n_real=None):
    x = torch.from_numpy(np.asarray(arr, dtype=np.float64))
    x._host_adj = x[:, 0].numpy()
    if n_real is not None:
        x._n_real = np.asarray(n_real)
        x._node_mask = torch.from_numpy(np.arange(x.shape[-1])[None, :] < x._n_real[:, None])
    return x


def build(model_kind, backend):
    arch = dict(block_features=[16, 16], depth_of_mlp=2, new_suffix=True,
                use_topology=model_kind == 'topo', topo_hidden_dim=16, topo_max_ph_dim=1,
                topo_num_stats=8, topo_max_simplex_dim=2)
    if model_kind == 'topo':
        arch.update(topo_multiplicity=True, topo_norm_stats=True, topo_essential=True,
                    topo_filt_squash=True, topo_readout=True, topo_num_filtrations=2,
                    topo_ph_backend=backend)
    cfg = EasyDict(dict(architecture=arch, node_labels=C - 1, num_classes=2))
    torch.manual_seed(0); m = BaseModel(cfg)
    for fc in getattr(m, 'topo_fc', []):
        if isinstance(fc, torch.nn.Linear):
            torch.nn.init.normal_(fc.weight, std=0.3)
    return m


ok = True
for kind, be in [('baseline', None), ('topo', 'gudhi'), ('topo', 'torch')]:
    m = build(kind, be)
    # alone
    solo, solo_grads = [], []
    for g in sel:
        T._PLAN_CACHE.clear()
        m.zero_grad()
        # alone, through the same (masked) readout: the original readout's
        # off-diagonal shift is batch-dependent, so its gradient at degenerate
        # diagonal/off-diagonal ties depends on batch composition even without
        # padding; the masked readout is batch-independent (same forward).
        out = m(as_input(g[None], [g.shape[1]]))
        out.square().sum().backward()
        solo.append(out.detach()[0]); solo_grads.append([p.grad.clone() if p.grad is not None else None for p in m.parameters()])
    # padded batch
    T._PLAN_CACHE.clear()
    padded, n_real = DataGenerator._pad_batch(sel)
    m.zero_grad()
    out = m(as_input(padded, n_real))
    out.square().sum().backward()
    batch_grads = [p.grad for p in m.parameters()]
    d_out = max((out[k].detach() - solo[k]).abs().max().item() for k in range(len(sel)))
    gsum = [sum(sg[i] for sg in solo_grads) if solo_grads[0][i] is not None else None
            for i in range(len(batch_grads))]
    d_grad = max(((bg - gs).abs().max() / gs.abs().max().clamp_min(1)).item()
                 for bg, gs in zip(batch_grads, gsum) if bg is not None and gs is not None)
    good = d_out < 1e-10 and d_grad < 1e-10; ok &= good
    print(f"{kind:8s} {be or '':6s}: max |padded - alone| output {d_out:.1e}, param-grad (rel) {d_grad:.1e} "
          f"[{'OK' if good else 'FAIL'}]")
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
