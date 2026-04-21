"""
Empirical equivariance test for TopologyLayer (v5: with essential features).

Verifies that for a random permutation pi of nodes:
    TopologyLayer(pi(X)) == pi(TopologyLayer(X))

where pi acts on the M x M pair-index dimensions:
    pi(X)_{c, pi(i), pi(j)} = X_{c, i, j}

This must hold for the full pipeline: filtration -> PH (including
essential/infinite persistence features + BP normalization) ->
broadcast -> gated residual fusion.
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

    For each trial:
    1. Create a random symmetric adjacency + features
    2. Pick a random permutation
    3. Run TopologyLayer on original and permuted inputs
    4. Check that permuting the original output matches the permuted output
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'

    layer = TopologyLayer(
        eqv_features=d_eqv,
        filt_features=d_filt,
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

        x_filt = torch.randn(1, d_filt, M, M, device=device)
        x_filt[0, 0] = torch.from_numpy(adj)
        x_filt = (x_filt + x_filt.transpose(-1, -2)) / 2

        x_eqv = torch.randn(1, d_eqv, M, M, device=device)
        x_eqv = (x_eqv + x_eqv.transpose(-1, -2)) / 2

        simplices_orig = [build_clique_complex(adj, max_dim=2)]

        perm = torch.randperm(M)
        perm_np = perm.numpy()

        adj_perm = adj[np.ix_(perm_np, perm_np)]
        simplices_perm = [build_clique_complex(adj_perm, max_dim=2)]

        x_filt_perm = permute_tensor(x_filt, perm)
        x_eqv_perm = permute_tensor(x_eqv, perm)

        with torch.no_grad():
            out_orig = layer(x_eqv, x_filt, simplices_orig)

        with torch.no_grad():
            out_perm = layer(x_eqv_perm, x_filt_perm, simplices_perm)

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
    permutation-INVARIANT and node-level output is equivariant.
    Covers essential (infinite persistence) features via ess_scale.
    """
    from layers.topology import DifferentiablePH, LearnedFiltration, build_clique_complex

    torch.manual_seed(123)
    np.random.seed(123)
    device = 'cpu'

    filt = LearnedFiltration(d_filt, hidden_dim=16).to(device)
    ph = DifferentiablePH(max_ph_dim=1, vec_dim=32).to(device)
    filt.eval()
    ph.eval()

    passed = 0
    for trial in range(num_trials):
        # Use sparse adjacency to ensure essential features exist
        # (not all H0 components merge => some have infinite persistence)
        adj = np.zeros((M, M), dtype=np.float32)
        n_edges = M  # sparse: roughly M edges for M nodes
        for _ in range(n_edges):
            i, j = np.random.randint(0, M, 2)
            if i != j:
                adj[i, j] = adj[j, i] = 1.0
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

        # Graph vector should be IDENTICAL (invariant)
        gv_diff = (gv_orig - gv_perm).abs().max().item()

        # Node vector should be equivariant: nv_orig[:, :, perm] == nv_perm
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


def test_essential_features_present(M=8, d_filt=3, num_trials=3):
    """
    Verify that essential (infinite persistence) features are actually
    being generated and processed, so the test above is meaningful.
    """
    from layers.topology import DifferentiablePH, LearnedFiltration, build_clique_complex
    import gudhi

    torch.manual_seed(999)
    np.random.seed(999)
    device = 'cpu'

    filt = LearnedFiltration(d_filt, hidden_dim=16).to(device)
    filt.eval()
    found = 0
    for trial in range(num_trials):
        # Very sparse graph: likely disconnected => essential H0 features
        adj = np.zeros((M, M), dtype=np.float32)
        for _ in range(M // 2):
            i, j = np.random.randint(0, M, 2)
            if i != j:
                adj[i, j] = adj[j, i] = 1.0
        np.fill_diagonal(adj, 1.0)

        x = torch.randn(1, d_filt, M, M, device=device)
        x[0, 0] = torch.from_numpy(adj)
        x = (x + x.transpose(-1, -2)) / 2

        simplices = [build_clique_complex(adj, max_dim=2)]
        with torch.no_grad():
            filt_batch = filt(x, simplices)

        # Count essential pairs
        graph_filt = filt_batch[0]
        st = gudhi.SimplexTree()
        for sigma, val in graph_filt:
            st.insert(list(sigma), filtration=val.item())
        st.make_filtration_non_decreasing()
        st.persistence()
        pairs = st.persistence_pairs()
        n_essential = sum(1 for b, d in pairs if len(d) == 0)
        n_finite = sum(1 for b, d in pairs if len(d) > 0)
        has_essential = n_essential > 0
        print(f"  Trial {trial+1}: {n_essential} essential, {n_finite} finite pairs {'(OK)' if has_essential else '(no essential!)'}")
        if has_essential:
            found += 1

    return found, num_trials


if __name__ == '__main__':
    print("=" * 60)
    print("Test 0: Essential features are being generated")
    print("=" * 60)
    p0, t0 = test_essential_features_present()
    print()

    print("=" * 60)
    print("Test 1: DifferentiablePH graph-invariance + node-equivariance")
    print("  (with essential features, BP normalization, vec_dim=32)")
    print("=" * 60)
    p1, t1 = test_graph_invariance()
    print()

    print("=" * 60)
    print("Test 2: Full TopologyLayer permutation-equivariance")
    print("  (filtration -> PH+essential -> BP-norm -> broadcast -> fusion)")
    print("=" * 60)
    p2, t2 = test_topology_layer_equivariance()
    print()

    print("=" * 60)
    total_passed = p0 + p1 + p2
    total_trials = t0 + t1 + t2
    print(f"Results: {total_passed}/{total_trials} passed")
    if total_passed == total_trials:
        print("ALL TESTS PASSED — equivariance preserved.")
    else:
        print("SOME TESTS FAILED — equivariance broken!")
    print("=" * 60)
    sys.exit(0 if total_passed == total_trials else 1)
