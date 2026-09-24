"""Side-by-side timing of the reference (pre-optimization) and optimized
topology layer on identical real batches, forward+backward, CPU."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import numpy as np, torch
import layers.topology as F, _topology_ref as R
from run_job import build_config
from data_loader.data_generator import DataGenerator
ds = sys.argv[1] if len(sys.argv) > 1 else 'NCI1'
cfg = build_config(ds, 'topo'); cfg.num_fold = 1
data = DataGenerator(cfg); data.initialize('train')
torch.manual_seed(0)
batches = [data.next_batch()[0] for _ in range(10)]
opts = dict(multiplicity=True, norm_stats=True, essential=True, filt_squash=True)
d = 64
res = {}
for name, mod in (('reference', R), ('optimized', F)):
    torch.manual_seed(1)
    L = mod.TopologyLayer(d, 32, 1, 32, **opts)
    proj = torch.nn.Conv2d(batches[0].shape[1], d, 1)
    def step(g):
        g = torch.as_tensor(g).float()
        structs = mod.build_graph_structs(g[:, 0].numpy(), 2)
        x = torch.relu(proj(g))
        L(x, structs).square().mean().backward()
    for g in batches[:2]: step(g)
    t0 = time.time()
    for g in batches: step(g)
    res[name] = (time.time() - t0) / len(batches) * 1000
n = np.mean([b.shape[0] for b in batches])
print(f"{ds}: topology layer fwd+bwd per batch (avg {n:.0f} graphs): "
      f"reference {res['reference']:.1f} ms, optimized {res['optimized']:.1f} ms "
      f"-> {res['reference']/res['optimized']:.1f}x")
