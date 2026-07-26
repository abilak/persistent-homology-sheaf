"""
Verify the BATCHED topology path (one gather/segment-sum/MLP + one batched
non-decreasing correction for the whole batch) is numerically identical to
processing graphs one at a time, for every option combination. Also checks
permutation equivariance and that a batch is invariant to graph ordering.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import layers.topology as T


def rand_adj(M, seed, p=0.3):
    rng = np.random.default_rng(seed)
    A = (rng.random((M, M)) < p).astype(np.float32)
    A = np.triu(A, 1)
    return (A + A.T).astype(np.float32)


def make_batch(B, M, d, seed=0):
    torch.manual_seed(seed)
    adjs = [rand_adj(M, seed * 100 + b) for b in range(B)]
    x = torch.randn(B, d, M, M)
    for b, A in enumerate(adjs):
        x[b, 0] = torch.from_numpy(A)
    x = (x + x.transpose(-1, -2)) / 2
    structs = [T.build_graph_structs([A], 2)[0] for A in adjs]
    return x, structs, adjs


def test_batch_vs_single(node_level, multiplicity, scale_stats=False, B=6, M=12, d=8):
    """Batched forward must equal per-graph forward, in values AND gradients."""
    x, structs, adjs = make_batch(B, M, d, seed=1)
    torch.manual_seed(7)
    L = T.TopologyLayer(d, 16, 1, 8, gate_bias=1.0,
                        node_level=node_level, multiplicity=multiplicity,
                        scale_stats=scale_stats)
    L.eval()

    xb = x.clone().requires_grad_(True)
    out_b = L(xb, structs)
    out_b.sum().backward()

    # per-graph: run each graph as its own batch of 1
    outs, grads = [], []
    for b in range(B):
        xs = x[b:b + 1].clone().requires_grad_(True)
        o = L(xs, [structs[b]])
        o.sum().backward()
        outs.append(o)
        grads.append(xs.grad)
    out_s = torch.cat(outs, dim=0)
    grad_s = torch.cat(grads, dim=0)

    df = (out_b - out_s).abs().max().item()
    dg = (xb.grad - grad_s).abs().max().item()
    return df, dg


def test_order_invariance(B=6, M=12, d=8):
    """Permuting the ORDER of graphs in a batch permutes outputs identically."""
    x, structs, adjs = make_batch(B, M, d, seed=2)
    torch.manual_seed(3)
    L = T.TopologyLayer(d, 16, 1, 8, gate_bias=1.0); L.eval()
    with torch.no_grad():
        o1 = L(x, structs)
        perm = list(reversed(range(B)))
        o2 = L(x[perm], [structs[i] for i in perm])
    return (o1[perm] - o2).abs().max().item()


def test_equivariance(node_level, multiplicity, scale_stats=False, M=11, d=6, trials=3):
    worst = 0.0
    for t in range(trials):
        A = rand_adj(M, 500 + t)
        x = torch.randn(1, d, M, M); x[0, 0] = torch.from_numpy(A)
        x = (x + x.transpose(-1, -2)) / 2
        torch.manual_seed(11)
        L = T.TopologyLayer(d, 16, 1, 8, gate_bias=1.0,
                            node_level=node_level, multiplicity=multiplicity,
                            scale_stats=scale_stats)
        L.eval()
        perm = torch.randperm(M)
        Ap = A[np.ix_(perm.numpy(), perm.numpy())]
        xp = x[:, :, perm, :][:, :, :, perm]
        with torch.no_grad():
            o = L(x, T.build_graph_structs([A], 2))
            op = L(xp, T.build_graph_structs([Ap], 2))
        worst = max(worst, (o[:, :, perm, :][:, :, :, perm] - op).abs().max().item())
    return worst


if __name__ == '__main__':
    ok = True
    print("batched vs per-graph (fwd / grad max abs diff):")
    for nl in (True, False):
        for mult in (False, True):
            for sc in (False, True):
                df, dg = test_batch_vs_single(nl, mult, sc)
                good = df < 1e-5 and dg < 1e-5
                ok &= good
                print(f"  node_level={nl!s:<5} mult={mult!s:<5} scale={sc!s:<5} "
                      f"fwd={df:.2e} grad={dg:.2e}  [{'OK' if good else 'FAIL'}]")

    d_order = test_order_invariance()
    ok &= d_order < 1e-5
    print(f"\nbatch order invariance: {d_order:.2e} "
          f"[{'OK' if d_order < 1e-5 else 'FAIL'}]")

    print("\npermutation equivariance:")
    for nl in (True, False):
        for mult in (False, True):
            for sc in (False, True):
                e = test_equivariance(nl, mult, sc)
                good = e < 1e-4
                ok &= good
                print(f"  node_level={nl!s:<5} mult={mult!s:<5} scale={sc!s:<5} "
                      f"err={e:.2e}  [{'OK' if good else 'FAIL'}]")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
