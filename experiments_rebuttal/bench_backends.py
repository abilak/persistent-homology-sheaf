"""Train-step timing (fwd+bwd+optimizer) of the full model per backend/device.
usage: python bench_backends.py DATASET DEVICE BACKEND [n_batches]"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'experiments_rebuttal')
import torch, numpy as np
ds, dev, be = sys.argv[1], sys.argv[2], sys.argv[3]
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 20
os.environ['CCD_DEVICE'] = dev
from run_job import build_config
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
FULL = dict(topo_multiplicity=True, topo_norm_stats=True, topo_essential=True,
            topo_filt_squash=True, topo_readout=True, topo_ph_backend=be)
model = 'baseline' if be == 'baseline' else 'topo'
torch.manual_seed(0); np.random.seed(0)
cfg = build_config(ds, model, None if model == 'baseline' else FULL); cfg.num_fold = 1
data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
mw.train(); data.initialize('train')


def step():
    """one train step; starts a new epoch when the current one runs out
    (small datasets such as MUTAG have fewer batches than the benchmark)"""
    try:
        tr.train_step()
    except StopIteration:
        data.initialize('train')
        tr.train_step()

sync = (lambda: torch.mps.synchronize()) if dev == 'mps' else \
       ((lambda: torch.cuda.synchronize()) if dev.startswith('cuda') else (lambda: None))
for _ in range(4): step()
sync(); t0 = time.time(); ng = 0
# SYNC_DEBUG=1 on CUDA: raise on ANY implicit host<->device synchronization
# inside the timed steps (the torch PH backend should trigger none).
if os.environ.get('SYNC_DEBUG') == '1' and dev.startswith('cuda'):
    torch.cuda.set_sync_debug_mode('error')
for _ in range(nb):
    step()
if dev.startswith('cuda'):
    torch.cuda.set_sync_debug_mode(0)
sync(); dt = time.time() - t0
print(json.dumps(dict(dataset=ds, device=dev, backend=be, ms_per_step=round(dt / nb * 1000, 1))))
