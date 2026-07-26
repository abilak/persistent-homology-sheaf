"""Train a MUTAG topology model (fold 1) and save weights for Q4 interpretability."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config

torch.manual_seed(0); np.random.seed(0)
cfg = build_config('MUTAG', 'topo')
cfg.num_fold = 1
data = DataGenerator(cfg)
mw = ModelWrapper(cfg, data)
tr = Trainer(mw, data, cfg)
best, best_state = -1, None
for ep in range(cfg.num_epochs):
    tr.train_epoch(ep)
    va, _ = tr.validate(ep)
    if va > best:
        best = va
        best_state = {k: v.detach().cpu().clone() for k, v in mw.model.state_dict().items()}
    if ep % 20 == 0:
        print(f"ep{ep} val={va:.3f} best={best:.3f}", flush=True)
torch.save({'state_dict': best_state, 'best_val': best},
           'rebuttal_results/mutag_topo_model.tar')
print(f"SAVED mutag_topo_model.tar best_val={best:.4f}")
