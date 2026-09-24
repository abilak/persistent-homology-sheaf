"""Profile train steps of a topo config (tottime), e.g.
python profile_full.py NCI1 10 '{"topo_multiplicity": true, ...}'"""
import os, sys, cProfile, pstats, io, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config
ds = sys.argv[1]; nb = int(sys.argv[2]); ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
model = 'baseline' if ov.get('baseline') else 'topo'; ov.pop('baseline', None)
torch.manual_seed(0); np.random.seed(0)
cfg = build_config(ds, model, ov); cfg.num_fold = 1
data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
mw.train(); data.initialize('train')
for _ in range(3): tr.train_step()
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
for _ in range(nb): tr.train_step()
pr.disable(); dt = time.time() - t0
print(f"{ds} {model} {ov}: {dt/nb*1000:.1f} ms/batch (batch={cfg.hyperparams.batch_size})")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(18)
print("\n".join(s.getvalue().splitlines()[6:32]))
