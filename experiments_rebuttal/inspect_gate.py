"""Retrain a few safe-init topo folds and read out the FINAL scalar gate values.
If the gate stayed near 0 the safe result is trivially "topology never turned on";
if it moved, topology was actively contributing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config

SAFE = {'topo_gate_mode': 'scalar', 'topo_gate_init': 0.0,
        'topo_apply_layers': 'last', 'topo_node_level': False,
        'topo_num_stats': 8, 'topo_hidden_dim': 16}


def run(ds, seed, fold, gate_init):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = build_config(ds, 'topo', {**SAFE, 'topo_gate_init': gate_init})
    cfg.num_fold = fold
    data = DataGenerator(cfg)
    mw = ModelWrapper(cfg, data)
    tr = Trainer(mw, data, cfg)
    best_val = -1
    for ep in range(cfg.num_epochs):
        tr.train_epoch(ep)
        va, _ = tr.validate(ep)
        best_val = max(best_val, float(va))
    gates = [float(m.gate_scalar.detach()) for m in mw.model.topo_layers
             if hasattr(m, 'gate_scalar') and m.gate_scalar is not None]
    return best_val, gates


print(f"{'ds':<6}{'gate_init':<12}{'seed/fold':<12}{'final gate':<28}{'best val'}")
print("-" * 78)
for ds in ['MUTAG', 'PTC']:
    for gi in [0.0, 0.1]:
        for s in [0, 1]:
            for f in [1, 2]:
                bv, gates = run(ds, s, f, gi)
                gs = ", ".join(f"{g:+.3f}" for g in gates)
                print(f"{ds:<6}{gi:<12.2f}{f's{s}/f{f}':<12}[{gs}]{'':<3}{bv:.3f}")
