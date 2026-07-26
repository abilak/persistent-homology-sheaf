"""
Q3 clean runtime benchmark: single process, fixed threads, identical data. For
each model {mlp,gcn,gin,baseline(PPGN),topo(PPGN+PH)} measure forward and
forward+backward time over one full training epoch (fold 1). Reports ms/graph
and s/epoch so we can state how much the persistent-homology forward pass costs
relative to standard architectures on small graphs.

Run when the machine is otherwise idle:  python runtime_bench.py MUTAG PTC NCI1
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

MODELS = ['mlp', 'gcn', 'gin', 'baseline', 'topo']


def bench(dataset, warmup=1, measure=3):
    torch.manual_seed(0); np.random.seed(0)
    out = {}
    for m in MODELS:
        cfg = build_config(dataset, m)
        cfg.num_fold = 1
        data = DataGenerator(cfg)
        mw = ModelWrapper(cfg, data)
        opt = torch.optim.Adam(mw.model.parameters(), lr=1e-4)
        n = data.train_size

        def one_epoch(train):
            data.initialize('train')
            if train:
                mw.train()
            else:
                mw.eval()
            for _ in range(data.num_iterations_train):
                g, l = data.next_batch()
                if train:
                    opt.zero_grad()
                    loss = F.cross_entropy(mw.model(g), l, reduction='sum')
                    loss.backward(); opt.step()
                else:
                    with torch.no_grad():
                        mw.model(g)

        for _ in range(warmup):
            one_epoch(True)
        # forward-only
        tf = []
        for _ in range(measure):
            t0 = time.time(); one_epoch(False); tf.append(time.time() - t0)
        # forward+backward
        tb = []
        for _ in range(measure):
            t0 = time.time(); one_epoch(True); tb.append(time.time() - t0)
        fwd, fb = float(np.median(tf)), float(np.median(tb))
        out[m] = dict(fwd_s_epoch=round(fwd, 3), fwdbwd_s_epoch=round(fb, 3),
                      fwd_ms_graph=round(fwd / n * 1000, 3),
                      fwdbwd_ms_graph=round(fb / n * 1000, 3),
                      n_params=sum(p.numel() for p in mw.model.parameters()),
                      train_size=n)
        print(f"{dataset:<8} {m:<9} fwd={fwd:.2f}s/ep ({fwd/n*1000:.2f} ms/graph)"
              f"  fwd+bwd={fb:.2f}s/ep ({fb/n*1000:.2f} ms/graph)  "
              f"params={out[m]['n_params']}", flush=True)
    return out


if __name__ == '__main__':
    dsets = sys.argv[1:] or ['MUTAG', 'PTC']
    allout = {}
    for ds in dsets:
        allout[ds] = bench(ds)
    with open('rebuttal_results/runtime_bench.json', 'w') as f:
        json.dump(allout, f, indent=2)
    print("saved rebuttal_results/runtime_bench.json")
