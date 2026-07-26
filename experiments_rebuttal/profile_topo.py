"""Profile the topology branch to find the real per-forward hotspot (what still
runs on CPU once the dense ops are on GPU). Reports cumulative time by function."""
import os, sys, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config

ds = sys.argv[1] if len(sys.argv) > 1 else 'NCI1'
n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 20
torch.manual_seed(0); np.random.seed(0)
cfg = build_config(ds, 'topo'); cfg.num_fold = 1
data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
# warm cache
mw.train(); data.initialize('train')
for i, _ in zip(range(3), range(data.num_iterations_train)):
    tr.train_step()

pr = cProfile.Profile()
data.initialize('train')
pr.enable()
for i in range(n_batches):
    tr.train_step()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
out = s.getvalue()
# print lines mentioning our modules or big cumulative
for line in out.splitlines():
    if any(k in line for k in ['topology', 'base_model', 'ncalls', 'filename',
                                'method', 'persistence', 'insert', 'SimplexTree',
                                '{built-in', 'index_add', 'scatter']):
        print(line)
