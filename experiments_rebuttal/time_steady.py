import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config

for ds in sys.argv[1:] or ['MUTAG', 'NCI1']:
    torch.manual_seed(0); np.random.seed(0)
    cfg = build_config(ds, 'topo'); cfg.num_fold = 1
    data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
    for ep in range(3):
        mw.train(); data.initialize('train'); t0 = time.time()
        for _ in range(data.num_iterations_train):
            tr.train_step()
        tag = 'cache-build' if ep == 0 else 'STEADY'
        print(f"{ds} epoch{ep} = {time.time()-t0:.2f}s  [{tag}]", flush=True)
