"""
Empirical equivariance test for TopologyLayer (v3: lifetime-only).

Verifies that for a random permutation pi of nodes:
    TopologyLayer(pi(X)) == pi(TopologyLayer(X))

v3 uses lifetime-only (1D) input — no birth info, no essentials,
no tie-merging. Graph-level should be invariant; node-level equivariance
is expected to break on tied filtration values (known v3 limitation).
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from layers.topology import TopologyLayer, build_clique_complex


def permute_tensor(X, perm):
    """Apply node permutation to B x d x M x M tensor."""
    return X[:, :, perm, :][:, :, :, perm]


def test_topology_layer_equivariance(M=8, d_eqv=4, d_filt=3,
                                      num_trials=5, atol=1e-5):
    """
    Test that TopologyLayer is permutation-equivariant.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'

    layer = TopologyLayer(
        eqv_features=d_eqv,
        hidden_dim=16,
        max_ph_dim=1,
        num_stats=32
    ).to(device)
    layer.eval()

    passed = 0
    for trial in range(num_trials):
        adj = np.random.randint(0, 2, (M, M)).astype(np.float32)
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 1.0)

        x_eqv = torch.randn(1, d_eqv, M, M, device=device)
        x_eqv[0, 0] = torch.from_numpy(adj)
        x_eqv = (x_eqv + x_eqv.transpose(-1, -2)) / 2

        simplices_orig = [build_clique_complex(adj, max_dim=2)]

        perm = torch.randperm(M)
        perm_np = perm.numpy()
        adj_perm = adj[np.ix_(perm_np, perm_np)]
        simplices_perm = [build_clique_complex(adj_perm, max_dim=2)]

        x_eqv_perm = permute_tensor(x_eqv, perm)

        with torch.no_grad():
            out_orig = layer(x_eqv, simplices_orig)
            out_perm = layer(x_eqv_perm, simplices_perm)

        out_orig_permuted = permute_tensor(out_orig, perm)
        diff = (out_orig_permuted - out_perm).abs().max().item()
        ok = diff < atol
        status = "PASS" if ok else "FAIL"
        print(f"  Trial {trial+1}: max |pi(f(X)) - f(pi(X))| = {diff:.2e}  [{status}]")
        if ok:
            passed += 1

    return passed, num_trials


def test_graph_invariance(M=8, d_eqv=4, d_filt=3, num_trials=5, atol=1e-5):
    """
    Test that the graph-level output of DifferentiablePH is
    permutation-INVARIANT (same vector regardless of node ordering).
    """
    from layers.topology import DifferentiablePH, LearnedFiltration

    torch.manual_seed(123)
    np.random.seed(123)
    device = 'cpu'

    filt = LearnedFiltration(d_filt, hidden_dim=16).to(device)
    ph = DifferentiablePH(max_ph_dim=1, vec_dim=32).to(device)
    filt.eval()
    ph.eval()

    passed = 0
    for trial in range(num_trials):
        adj = np.random.randint(0, 2, (M, M)).astype(np.float32)
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 1.0)

        x = torch.randn(1, d_filt, M, M, device=device)
        x[0, 0] = torch.from_numpy(adj)
        x = (x + x.transpose(-1, -2)) / 2

        simplices_orig = [build_clique_complex(adj, max_dim=2)]

        perm = torch.randperm(M)
        perm_np = perm.numpy()
        adj_perm = adj[np.ix_(perm_np, perm_np)]
        simplices_perm = [build_clique_complex(adj_perm, max_dim=2)]
        x_perm = permute_tensor(x, perm)

        with torch.no_grad():
            filt_orig = filt(x, simplices_orig)
            gv_orig, nv_orig = ph(filt_orig, device, num_nodes=M)

            filt_perm = filt(x_perm, simplices_perm)
            gv_perm, nv_perm = ph(filt_perm, device, num_nodes=M)

        gv_diff = (gv_orig - gv_perm).abs().max().item()
        nv_orig_permuted = nv_orig[:, :, perm]
        nv_diff = (nv_orig_permuted - nv_perm).abs().max().item()

        gv_ok = gv_diff < atol
        nv_ok = nv_diff < atol
        ok = gv_ok and nv_ok
        status = "PASS" if ok else "FAIL"
        print(f"  Trial {trial+1}: graph_vec diff={gv_diff:.2e} ({'ok' if gv_ok else 'FAIL'}), "
              f"node_vec diff={nv_diff:.2e} ({'ok' if nv_ok else 'FAIL'})  [{status}]")
        if ok:
            passed += 1

    return passed, num_trials


if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: DifferentiablePH graph-invariance + node-equivariance")
    print("  (v3: lifetime-only, no essentials, vec_dim=32)")
    print("=" * 60)
    p1, t1 = test_graph_invariance()
    print()

    print("=" * 60)
    print("Test 2: Full TopologyLayer permutation-equivariance")
    print("  (v3: filtration -> PH(lifetime) -> broadcast -> gated fusion)")
    print("=" * 60)
    p2, t2 = test_topology_layer_equivariance()
    print()

    print("=" * 60)
    total_passed = p1 + p2
    total_trials = t1 + t2
    print(f"Results: {total_passed}/{total_trials} passed")
    if total_passed == total_trials:
        print("ALL TESTS PASSED — equivariance preserved.")
    else:
        print("SOME TESTS FAILED — equivariance may be broken (see v3 notes).")
    print("=" * 60)
    sys.exit(0 if total_passed == total_trials else 1)
