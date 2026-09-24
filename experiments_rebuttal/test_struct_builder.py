"""The vectorized GraphStruct builder must reproduce the original per-simplex
Python construction (frozen in _topology_ref.py) exactly: same simplices in
the same order, same filtration gather indices, same face tables."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import numpy as np, torch
import layers.topology as F
import _topology_ref as R

rng = np.random.default_rng(0); ok = True; n_checked = 0
for trial in range(300):
    n = int(rng.integers(1, 40)); p = float(rng.uniform(0, 1)); D = int(rng.integers(1, 4))
    if D == 3: n = min(n, 22)
    A = (rng.random((n, n)) < p).astype(float); A = np.triu(A, 1); A = A + A.T
    a, b = F.GraphStruct(A, D), R.GraphStruct(A, D)
    same = (a.simplices_list == b.simplices_list and a.S == b.S
            and torch.equal(a.f_flat_cpu, b.f_flat_cpu) and torch.equal(a.f_seg_cpu, b.f_seg_cpu)
            and torch.equal(a.f_counts_cpu, b.f_counts_cpu)
            and sorted(a.face_tables_cpu) == sorted(b.face_tables_cpu)
            and all(torch.equal(a.face_tables_cpu[d][0], b.face_tables_cpu[d][0]) and
                    torch.equal(a.face_tables_cpu[d][1], b.face_tables_cpu[d][1])
                    for d in a.face_tables_cpu))
    n_checked += 1
    if not same:
        ok = False; print("MISMATCH", n, p, D); break
print(f"{n_checked} random graphs (n <= 40, D <= 3): vectorized builder identical to reference: {ok}")
# timing on dense graphs
A = (np.random.default_rng(1).random((80, 80)) < 0.5).astype(float); A = np.triu(A, 1); A = A + A.T
for D in (2, 3):
    t0 = time.time(); F.GraphStruct(A, D); tf = time.time() - t0
    t0 = time.time(); R.GraphStruct(A, D); tr = time.time() - t0
    print(f"dense 80-vertex graph, D={D}: reference {tr*1e3:.0f} ms, vectorized {tf*1e3:.1f} ms")
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
