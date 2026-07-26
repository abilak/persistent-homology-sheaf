"""Calibrate per-epoch train time: baseline PPGN vs topology model, per dataset."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config

torch.manual_seed(0); np.random.seed(0)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MUTAG'
topo = (sys.argv[2] == 'topo') if len(sys.argv) > 2 else False
cfg_file = 'configs/mutag_config.json'
config = process_config(cfg_file, ds, use_topology=topo)
config.num_epochs = 2
config.num_fold = 1
print(f"ds={ds} topo={topo} epochs=2 fold=1 device={'cuda' if torch.cuda.is_available() else 'cpu'}")
data = DataGenerator(config)
print(f"train_size={data.train_size} val_size={data.val_size}")
mw = ModelWrapper(config, data)
nparams = sum(p.numel() for p in mw.model.parameters())
print(f"params={nparams}")
tr = Trainer(mw, data, config)
# time one train epoch
mw.train(); data.initialize('train')
t0 = time.time()
tot=0
for _ in range(data.num_iterations_train):
    tr.train_step(); tot+=1
dt = time.time()-t0
print(f"TRAIN_EPOCH_TIME={dt:.2f}s over {tot} batches -> {dt/max(tot,1)*1000:.1f} ms/batch")
# time val epoch
mw.eval(); data.initialize('val')
t0=time.time()
with torch.no_grad():
    for _ in range(data.num_iterations_val):
        g,l = data.next_batch(); mw.run_model_get_loss_and_results(g,l)
print(f"VAL_EPOCH_TIME={time.time()-t0:.2f}s")
