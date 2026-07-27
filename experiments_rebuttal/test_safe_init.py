"""
Verify the ReZero-style scalar gate ("safe init"):

 1. With gate_init = 0 the topology-augmented network computes EXACTLY the same
    function as the pure equivariant baseline (so it cannot start off worse).
 2. The gate still receives a nonzero gradient, so topology can switch on.
 3. Expressivity is preserved: with the gate on, the SRG pairs are still
    separated (Corollary 6 is an existence statement over parameters, so a
    change of initialization cannot affect it -- this checks it empirically).
 4. Permutation equivariance still holds.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config
from models.base_model import BaseModel
import layers.topology as T


def make_input(B, M, C, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.zeros(B, C, M, M)
    for b in range(B):
        A = (rng.random((M, M)) < 0.3)
        A = np.triu(A, 1); A = (A + A.T).astype(np.float32)
        x[b, 0] = torch.from_numpy(A)
        for i in range(M):
            x[b, rng.integers(1, C), i, i] = 1.0
    return x


def test_exact_baseline_at_init(ds='PTC'):
    """topo(gate_init=0) must equal baseline EXACTLY on the shared weights."""
    torch.manual_seed(0)
    base = BaseModel(build_config(ds, 'baseline'))
    torch.manual_seed(0)
    topo = BaseModel(build_config(ds, 'topo', {
        'topo_gate_mode': 'scalar', 'topo_gate_init': 0.0,
        'topo_apply_layers': 'last', 'topo_node_level': False,
        'topo_num_stats': 8, 'topo_hidden_dim': 16}))
    # copy the shared (equivariant + readout) weights so only the topology
    # branch differs
    sd_b = base.state_dict()
    sd_t = topo.state_dict()
    for k, v in sd_b.items():
        if k in sd_t and sd_t[k].shape == v.shape:
            sd_t[k] = v.clone()
    topo.load_state_dict(sd_t)
    base.eval(); topo.eval()

    C = build_config(ds, 'baseline').node_labels + 1
    x = make_input(4, 14, C, seed=1)
    with torch.no_grad():
        ob, ot = base(x), topo(x)
    return (ob - ot).abs().max().item()


def test_gate_gets_gradient(ds='PTC'):
    torch.manual_seed(0)
    topo = BaseModel(build_config(ds, 'topo', {
        'topo_gate_mode': 'scalar', 'topo_gate_init': 0.0,
        'topo_apply_layers': 'last', 'topo_node_level': False,
        'topo_num_stats': 8, 'topo_hidden_dim': 16}))
    C = build_config(ds, 'baseline').node_labels + 1
    x = make_input(4, 14, C, seed=2)
    out = topo(x)
    out.sum().backward()
    gs = [m.gate_scalar for m in topo.topo_layers
          if hasattr(m, 'gate_scalar') and m.gate_scalar is not None]
    return [float(g.grad.abs().item()) for g in gs]


def test_srg_still_separates():
    """Expressivity check with the scalar gate switched ON."""
    import srg_analysis as SA
    from srg_graphs import all_pairs
    ORIG = SA.make_config

    def cfg(use_topology):
        c = ORIG(use_topology)
        c.architecture.topo_gate_mode = 'scalar'
        c.architecture.topo_gate_init = 1.0      # gate on
        c.architecture.topo_apply_layers = 'last'
        c.architecture.topo_node_level = False
        c.architecture.topo_num_stats = 8
        c.architecture.topo_hidden_dim = 16
        return c

    rows = []
    for name, (G1, G2, p) in all_pairs().items():
        SA.make_config = ORIG
        b, _, _ = SA.model_separation(G1, G2, use_topology=False, n_seeds=3)
        SA.make_config = cfg
        med, mx, _ = SA.model_separation(G1, G2, use_topology=True, n_seeds=3)
        rows.append((name, b, med, mx))
    SA.make_config = ORIG
    return rows


def test_equivariance():
    M, d = 11, 6
    rng = np.random.default_rng(5)
    A = (rng.random((M, M)) < 0.3); A = np.triu(A, 1)
    A = (A + A.T).astype(np.float32)
    x = torch.randn(1, d, M, M); x[0, 0] = torch.from_numpy(A)
    x = (x + x.transpose(-1, -2)) / 2
    torch.manual_seed(4)
    L = T.TopologyLayer(d, 16, 1, 8, node_level=False,
                        gate_mode='scalar', gate_init=1.0)
    L.eval()
    perm = torch.randperm(M)
    Ap = A[np.ix_(perm.numpy(), perm.numpy())]
    xp = x[:, :, perm, :][:, :, :, perm]
    with torch.no_grad():
        o = L(x, T.build_graph_structs([A], 2))
        op = L(xp, T.build_graph_structs([Ap], 2))
    return (o[:, :, perm, :][:, :, :, perm] - op).abs().max().item()


if __name__ == '__main__':
    ok = True
    for ds in ('PTC', 'MUTAG'):
        d = test_exact_baseline_at_init(ds)
        good = d == 0.0
        ok &= good
        print(f"[{ds}] topo(gate_init=0) vs baseline output diff: {d:.3e}"
              f"   [{'EXACT' if good else 'FAIL'}]")

    g = test_gate_gets_gradient()
    good = all(v > 0 for v in g)
    ok &= good
    print(f"gate scalar |grad| at init: {g}   "
          f"[{'can learn' if good else 'FAIL: dead gate'}]")

    e = test_equivariance()
    ok &= e < 1e-4
    print(f"equivariance err (scalar gate): {e:.2e}  "
          f"[{'OK' if e < 1e-4 else 'FAIL'}]")

    print("\nExpressivity with gate ON (Corollary 6 pairs):")
    print(f"  {'pair':<26}{'baseline':<12}{'topo med/max':<24}{'sep?'}")
    for name, b, med, mx in test_srg_still_separates():
        sep = med > 1e-5
        ok &= sep
        print(f"  {name:<26}{b:<12.1e}{med:.1e} / {mx:<14.1e}{'YES' if sep else 'NO'}")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
