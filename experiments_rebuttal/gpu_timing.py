"""
Accurate GPU training-time benchmark (Q3). Uses torch.cuda.synchronize()
around every timed region (wall-clock timing of async CUDA kernels is
otherwise meaningless) plus warmup iterations. Reports, per model:
  * forward-only  ms / graph   (this is where PH lives -- the reviewer's concern)
  * forward+backward ms / graph (a full training step)
  * s / epoch
for the same-code models mlp/gcn/gin/gsn/baseline/topo, on one fold.

usage: CCD_DEVICE=cuda python experiments_rebuttal/gpu_timing.py MUTAG PTC NCI1
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments_rebuttal'))
from run_job import build_config
from utils.device import get_device

DEV = get_device()
CUDA = DEV.type == 'cuda'
MODELS = ['mlp', 'gcn', 'gin', 'gsn', 'baseline', 'topo']


def sync():
    if CUDA:
        torch.cuda.synchronize()


def bench(dataset, warmup=2, measure=5):
    torch.manual_seed(0); np.random.seed(0)
    out = {}
    for m in MODELS:
        cfg = build_config(dataset, m); cfg.num_fold = 1
        data = DataGenerator(cfg)
        mw = ModelWrapper(cfg, data)
        opt = torch.optim.Adam(mw.model.parameters(), lr=1e-4)
        n = data.train_size

        def epoch(train):
            data.initialize('train')
            mw.model.train() if train else mw.model.eval()
            for _ in range(data.num_iterations_train):
                g, l = data.next_batch()
                if train:
                    opt.zero_grad()
                    F.cross_entropy(mw.model(g), l, reduction='sum').backward()
                    opt.step()
                else:
                    with torch.no_grad():
                        mw.model(g)

        for _ in range(warmup):        # build caches + CUDA autotune
            epoch(True)
        sync()
        tf = []
        for _ in range(measure):
            t0 = time.time(); epoch(False); sync(); tf.append(time.time() - t0)
        tb = []
        for _ in range(measure):
            t0 = time.time(); epoch(True); sync(); tb.append(time.time() - t0)
        fwd, fb = float(np.median(tf)), float(np.median(tb))
        out[m] = dict(fwd_ms_per_graph=round(fwd / n * 1000, 3),
                      fwdbwd_ms_per_graph=round(fb / n * 1000, 3),
                      fwd_s_per_epoch=round(fwd, 3),
                      fwdbwd_s_per_epoch=round(fb, 3),
                      n_params=sum(p.numel() for p in mw.model.parameters()),
                      train_size=n)
        print(f"{dataset:<8}{m:<10}fwd {out[m]['fwd_ms_per_graph']:>7.3f} ms/g   "
              f"fwd+bwd {out[m]['fwdbwd_ms_per_graph']:>7.3f} ms/g   "
              f"({out[m]['fwdbwd_s_per_epoch']:.2f} s/ep)   "
              f"params={out[m]['n_params']:,}", flush=True)
    return out


if __name__ == '__main__':
    dsets = sys.argv[1:] or ['MUTAG', 'PTC']
    print(f"device={DEV}\n")
    allout = {}
    for ds in dsets:
        allout[ds] = bench(ds)
        # ratios vs GCN and vs baseline
        g = allout[ds]['gcn']['fwdbwd_ms_per_graph']
        b = allout[ds]['baseline']['fwdbwd_ms_per_graph']
        t = allout[ds]['topo']['fwdbwd_ms_per_graph']
        print(f"  --> {ds}: topo/baseline = {t/b:.2f}x,  baseline/gcn = {b/g:.1f}x,"
              f"  topo/gcn = {t/g:.1f}x\n")
    with open('rebuttal_results/gpu_timing.json', 'w') as f:
        json.dump(allout, f, indent=2)
    print("saved rebuttal_results/gpu_timing.json")
