"""
Tests for the post-rebuttal architecture options:
  topo_norm_stats  (invertible LayerNorm), topo_essential (essential classes),
  topo_filt_squash (tanh-bounded filtration), topo_readout (topological head).
Checks: batched == per-graph (values + grads), permutation equivariance of the
layer, invariance of the full model, invertibility of the normalization,
essential-class counts on known complexes, and that tanh squashing leaves the
persistence pairing unchanged.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
# float64: persistence pairing (and tie-merging, which rounds to 1e-6) is
# discontinuous at ties. With several filtration heads, randomly initialized
# ReLU MLPs produce exact / near ties, and float32 summation-order noise
# (~1e-7) between batched / permuted computations can flip a pairing. That is
# the measure-zero discontinuity of PH, not an implementation difference; in
# float64 both paths agree to ~1e-15.
torch.set_default_dtype(torch.float64)
from easydict import EasyDict
import layers.topology as T
from models.base_model import BaseModel
from test_batching import rand_adj, make_batch

ALL = dict(multiplicity=True, norm_stats=True, essential=True, filt_squash=True)


def layer(d, **kw):
    torch.manual_seed(7)
    L = T.TopologyLayer(d, 16, 1, 8, gate_bias=1.0, node_level=kw.pop('node_level', True), **kw)
    L.eval(); return L


def batch_vs_single(**kw):
    x, structs, _ = make_batch(6, 12, 8, seed=1)
    L = layer(8, **kw)
    xb = x.clone().requires_grad_(True); ob = L(xb, structs); ob.sum().backward()
    outs, grads = [], []
    for b in range(6):
        xs = x[b:b+1].clone().requires_grad_(True); o = L(xs, [structs[b]])
        o.sum().backward(); outs.append(o); grads.append(xs.grad)
    os_, gs = torch.cat(outs), torch.cat(grads)
    return (((ob - os_).abs().max() / os_.abs().max().clamp_min(1)).item(),
            ((xb.grad - gs).abs().max() / gs.abs().max().clamp_min(1)).item())


def equivariance(**kw):
    worst = 0.0
    for t in range(3):
        M, d = 11, 6
        A = rand_adj(M, 500 + t)
        x = torch.randn(1, d, M, M); x[0, 0] = torch.from_numpy(A)
        x = (x + x.transpose(-1, -2)) / 2
        L = layer(d, **dict(kw))
        perm = torch.randperm(M)
        Ap = A[np.ix_(perm.numpy(), perm.numpy())]
        xp = x[:, :, perm, :][:, :, :, perm]
        with torch.no_grad():
            o = L(x, T.build_graph_structs([A], 2)); op = L(xp, T.build_graph_structs([Ap], 2))
        worst = max(worst, (o[:, :, perm, :][:, :, :, perm] - op).abs().max().item())
    return worst


def model_invariance():
    arch = dict(block_features=[16, 16], depth_of_mlp=2, new_suffix=True, use_topology=True,
                topo_hidden_dim=16, topo_max_ph_dim=1, topo_num_stats=8, topo_max_simplex_dim=2,
                topo_multiplicity=True, topo_norm_stats=True, topo_essential=True,
                topo_filt_squash=True, topo_readout=True, topo_num_filtrations=2)
    cfg = EasyDict(dict(architecture=arch, node_labels=3, num_classes=2))
    torch.manual_seed(0); m = BaseModel(cfg).eval()
    for fc in m.topo_fc:                      # make the zero-init head nonzero
        if isinstance(fc, torch.nn.Linear):
            torch.nn.init.normal_(fc.weight); torch.nn.init.normal_(fc.bias)
    worst = 0.0
    for t in range(3):
        M = 10; A = rand_adj(M, 900 + t)
        x = torch.zeros(1, 4, M, M); x[0, 0] = torch.from_numpy(A)
        lab = torch.randint(0, 3, (M,))
        for v in range(M): x[0, 1 + lab[v], v, v] = 1.0
        perm = torch.randperm(M)
        with torch.no_grad():
            worst = max(worst, (m(x) - m(x[:, :, perm][:, :, :, perm])).abs().max().item())
    return worst


def invertible_ln():
    n = T.InvertibleLayerNorm(9); x = torch.randn(5, 9) * 3 + 1
    y = n(x); ln, mu, logs = y[:, :9], y[:, 9:10], y[:, 10:11]
    xr = mu + torch.exp(logs) * (ln - n.ln.bias) / n.ln.weight
    return (xr - x).abs().max().item()


def essential_counts():
    """C_6 (b0=1,b1=1) and 2xC_4 (b0=2,b1=2): essential multiplicities must
    equal the Betti numbers of the 2-skeleton."""
    ph = T.DifferentiablePH(1, 4, multiplicity=True, essential=True)
    out = {}
    for name, G in [("C6", nx.cycle_graph(6)),
                    ("2xC4", nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4)))]:
        A = nx.to_numpy_array(G).astype(np.float32); st = T.build_graph_structs([A], 2)[0]
        f = torch.rand(st.S)
        g, _ = ph([(st, f)], device='cpu', num_nodes=None)
        w = ph.per_dim_graph
        # per-dim layout: [finite pooled(4), log(1+N), ess pooled(4), log(1+E)]
        e0 = torch.expm1(g[0, w - 1 - 0 * w + 0]).item() if False else None
        blk = g[0].reshape(2, w)
        out[name] = tuple(round(torch.expm1(blk[p, -1]).item()) for p in range(2))
    return out


def squash_pairing():
    """tanh is strictly increasing: gudhi pairing must be identical."""
    A = rand_adj(12, 3); st = T.build_graph_structs([A], 2)[0]
    f = torch.randn(st.S)
    ph = T.DifferentiablePH(1, 4)
    def pairs(vals):
        v = ph._make_non_decreasing(vals, [st.on('cpu')[3][d] for d in sorted(st.on('cpu')[3])])
        import gudhi
        t = gudhi.SimplexTree()
        for i, s in enumerate(st.simplices_list): t.insert(list(s), filtration=float(v[i]))
        t.make_filtration_non_decreasing(); t.persistence()
        return sorted((tuple(sorted(b)), tuple(sorted(d))) for b, d in t.persistence_pairs())
    return pairs(f) == pairs(torch.tanh(f))


def heads_contain_single():
    """m heads with the extra heads' fusion/gate weights zeroed must equal the
    single-head layer carrying head 0's parameters (containment)."""
    x, structs, _ = make_batch(4, 10, 6, seed=4)
    torch.manual_seed(2)
    L1 = T.TopologyLayer(6, 16, 1, 8, multiplicity=True, essential=True, node_level=False)
    Lm = T.TopologyLayer(6, 16, 1, 8, multiplicity=True, essential=True, node_level=False,
                         num_filtrations=3)
    sd1 = L1.state_dict(); sdm = Lm.state_dict()
    F1 = L1.ph.out_features
    for k in sdm:
        if k in sd1 and sd1[k].shape == sdm[k].shape:
            sdm[k] = sd1[k].clone()
    for key in ('fusion.0.weight', 'gate_conv.weight'):
        W = torch.zeros_like(sdm[key]); W[:, :6 + F1] = sd1[key][:, :6 + F1]
        sdm[key] = W
    Lm.load_state_dict(sdm); L1.eval(); Lm.eval()
    with torch.no_grad():
        return (L1(x, structs) - Lm(x, structs)).abs().max().item()


if __name__ == '__main__':
    ok = True
    for kw in [dict(ALL), dict(ALL, node_level=False), dict(essential=True),
               dict(ALL, num_filtrations=3), dict(ALL, num_filtrations=2, node_level=False),
               dict(essential=True, multiplicity=True, scale_stats=True)]:
        df, dg = batch_vs_single(**dict(kw)); e = equivariance(**dict(kw))
        good = df < 1e-5 and dg < 1e-5 and e < 1e-4; ok &= good
        print(f"{kw}: batch fwd={df:.1e} grad={dg:.1e} equiv={e:.1e} [{'OK' if good else 'FAIL'}]")
    mi = model_invariance(); ok &= mi < 1e-4
    print(f"full model invariance (all options, readout on): {mi:.1e} [{'OK' if mi < 1e-4 else 'FAIL'}]")
    il = invertible_ln(); ok &= il < 1e-4
    print(f"invertible LN reconstruction error: {il:.1e} [{'OK' if il < 1e-4 else 'FAIL'}]")
    ec = essential_counts(); good = ec == {"C6": (1, 1), "2xC4": (2, 2)}; ok &= good
    print(f"essential multiplicities (b0,b1): {ec} [{'OK' if good else 'FAIL'}]")
    sp = squash_pairing(); ok &= sp
    print(f"tanh squash preserves pairing: {sp} [{'OK' if sp else 'FAIL'}]")
    hc = heads_contain_single(); ok &= hc < 1e-5
    print(f"3 heads with extra heads zeroed == 1 head: {hc:.1e} [{'OK' if hc < 1e-5 else 'FAIL'}]")
    print("\nPASS" if ok else "\nFAIL"); sys.exit(0 if ok else 1)

