"""
Steady-state training cost per epoch (paper Table tab:cost).

  python experiments_rebuttal/timing.py                 # all datasets below
  python experiments_rebuttal/timing.py NCI1 ZINC       # a subset

For each dataset (fold 1 / seed 0, the paper's configuration with padded
batching) and each of
    baseline      the PPGN backbone
    ph_cpu        PPGN+PH with CPU persistence (gudhi)
    ph_gpu        PPGN+PH with accelerator persistence (layers/torch_ph.py)
one warm-up epoch builds and caches every clique complex, then --epochs
training epochs are timed (CUDA-synchronized). Reports seconds per epoch and
peak GPU memory, writes rebuttal_results/timing.json, and prints LaTeX rows.
Run it with the GPU otherwise idle: concurrent jobs distort the times.
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import run_job, runner
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer

DATASETS = ["NCI1", "MUTAG", "PROTEINS", "IMDBBINARY", "ZINC", "MOLHIV"]
VARIANTS = [("baseline", "baseline", dict(runner.FAST)),
            ("ph_cpu", "topo", dict(runner.FULL, topo_ph_backend="gudhi", **runner.FAST)),
            ("ph_gpu", "topo", dict(runner.FULL, topo_ph_backend="torch", **runner.FAST))]
cuda = torch.cuda.is_available()


def sync():
    if cuda:
        torch.cuda.synchronize()


def measure(ds, model, ov, epochs):
    torch.manual_seed(0); np.random.seed(0)
    cfg = run_job.build_config(ds, model, ov); cfg.num_fold = 1
    data = DataGenerator(cfg); mw = ModelWrapper(cfg, data); tr = Trainer(mw, data, cfg)
    steps = data.num_iterations_train
    sync(); t = time.perf_counter(); tr.train_epoch(0); sync()
    warm = time.perf_counter() - t
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for ep in range(1, epochs + 1):
        sync(); t = time.perf_counter(); tr.train_epoch(ep); sync()
        times.append(time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float("nan")
    del tr, mw, data
    if cuda:
        torch.cuda.empty_cache()
    return dict(sec_per_epoch=round(float(np.mean(times)), 3), first_epoch_sec=round(warm, 2),
                peak_gpu_gb=round(mem, 2), steps_per_epoch=int(steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    path = "rebuttal_results/timing.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for ds in args.datasets:
        for name, model, ov in VARIANTS:
            key = f"{ds}/{name}"
            try:
                res[key] = measure(ds, model, ov, 1 if ds == "MOLHIV" else args.epochs)
            except Exception as e:                      # e.g. out of memory: record and go on
                res[key] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(key, res[key], flush=True)
            json.dump(res, open(path, "w"), indent=2)
    print("\n% ---- rows for tab:cost (seconds per training epoch) ----")
    names = {"baseline": "PPGN backbone", "ph_cpu": "PPGN+PH, CPU persistence (\\textsc{gudhi})",
             "ph_gpu": "PPGN+PH, accelerator persistence"}
    for name, _, _ in VARIANTS:
        cells = []
        for ds in DATASETS:
            r = res.get(f"{ds}/{name}", {})
            cells.append(f"{r['sec_per_epoch']:.1f}" if "sec_per_epoch" in r else "\\placeholder{--}")
        print(f"{names[name]} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
