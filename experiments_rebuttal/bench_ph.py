"""Micro-benchmark of the topology layer's persistence path on real NCI1
batches (forward+backward of DifferentiablePH only)."""
import os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import layers.topology as T
sys.path.insert(0, 'experiments_rebuttal')
from run_job import build_config
from data_loader.data_generator import DataGenerator
cfg = build_config(sys.argv[1] if len(sys.argv) > 1 else 'NCI1', 'topo'); cfg.num_fold = 1
data = DataGenerator(cfg); data.initialize('train')
batches = []
for _ in range(12):
    g, _l = data.next_batch()
    A = g[:, 0].cpu().numpy() if torch.is_tensor(g) else np.asarray(g)[:, 0]
    batches.append(T.build_graph_structs(A, 2))
torch.manual_seed(0)
ph = T.DifferentiablePH(1, 32, multiplicity=True, essential=True)
def step(structs, M):
    filt = [(st, torch.randn(st.S, requires_grad=True)) for st in structs]
    g, n = ph(filt, device='cpu', num_nodes=M)
    (g.sum() + n.sum()).backward()
for st in batches[:2]: step(st, st[0].M)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
for st in batches: step(st, st[0].M)
pr.disable(); dt = (time.time() - t0) / len(batches)
print(f"DifferentiablePH fwd+bwd: {dt*1000:.1f} ms/batch ({len(batches[0])} graphs)")
if '-v' in sys.argv:
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14)
    print("\n".join(s.getvalue().splitlines()[6:24]))
