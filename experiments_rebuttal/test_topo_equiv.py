"""Verify optimized topology.py is numerically identical to the original
reference, and preserves permutation equivariance. Compares TopologyLayer
outputs (fwd + grad) with shared weights on random graphs."""
import os, sys, subprocess, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

# Reference = the ORIGINAL topology.py from origin/topology-v4-wkpi. Fetch it
# from git if not already extracted, so this test is self-bootstrapping.
REF = "/tmp/topology_ref.py"
if not os.path.exists(REF):
    with open(REF, "w") as f:
        subprocess.check_call(
            ["git", "show", "origin/topology-v4-wkpi:layers/topology.py"],
            stdout=f)

spec = importlib.util.spec_from_file_location("topo_ref", REF)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
import layers.topology as fast


def rand_graph(M, seed):
    rng = np.random.default_rng(seed)
    A = (rng.random((M, M)) < 0.35).astype(np.float32)
    A = np.triu(A, 1); A = A + A.T
    return A


def test_numeric(M=10, d=8, trials=6):
    torch.manual_seed(0)
    maxdiff_f = maxdiff_g = 0.0
    for t in range(trials):
        A = rand_graph(M, t)
        x = torch.randn(1, d, M, M)
        x[0, 0] = torch.from_numpy(A)
        x = (x + x.transpose(-1, -2)) / 2
        x_ref = x.clone().requires_grad_(True)
        x_fast = x.clone().requires_grad_(True)

        L_ref = ref.TopologyLayer(d, hidden_dim=16, max_ph_dim=1, num_stats=8)
        L_fast = fast.TopologyLayer(d, hidden_dim=16, max_ph_dim=1, num_stats=8)
        L_fast.load_state_dict(L_ref.state_dict())
        L_ref.eval(); L_fast.eval()

        simplices = [ref.build_clique_complex(A, max_dim=2)]
        structs = fast.build_graph_structs([A], max_dim=2)

        o_ref = L_ref(x_ref, simplices)
        o_fast = L_fast(x_fast, structs)
        df = (o_ref - o_fast).abs().max().item()
        o_ref.sum().backward(); o_fast.sum().backward()
        dg = (x_ref.grad - x_fast.grad).abs().max().item()
        maxdiff_f = max(maxdiff_f, df); maxdiff_g = max(maxdiff_g, dg)
    print(f"[numeric] max fwd diff={maxdiff_f:.2e}  max grad diff={maxdiff_g:.2e}")
    return maxdiff_f < 1e-5 and maxdiff_g < 1e-5


def test_equivariance(M=9, d=6, trials=5):
    torch.manual_seed(1)
    L = fast.TopologyLayer(d, hidden_dim=16, max_ph_dim=1, num_stats=16); L.eval()
    worst = 0.0
    for t in range(trials):
        A = rand_graph(M, 100 + t)
        x = torch.randn(1, d, M, M); x[0, 0] = torch.from_numpy(A)
        x = (x + x.transpose(-1, -2)) / 2
        perm = torch.randperm(M)
        Ap = A[np.ix_(perm.numpy(), perm.numpy())]
        xp = x[:, :, perm, :][:, :, :, perm]
        with torch.no_grad():
            o = L(x, fast.build_graph_structs([A], 2))
            op = L(xp, fast.build_graph_structs([Ap], 2))
        o_perm = o[:, :, perm, :][:, :, :, perm]
        worst = max(worst, (o_perm - op).abs().max().item())
    print(f"[equivariance] max |pi(f(x)) - f(pi(x))| = {worst:.2e}")
    return worst < 1e-4


if __name__ == "__main__":
    ok1 = test_numeric()
    ok2 = test_equivariance()
    print("PASS" if (ok1 and ok2) else "FAIL")
    sys.exit(0 if (ok1 and ok2) else 1)
