import os, sys, time, json, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch
import run_job
from runner import FULL, FAST
from data_loader.data_generator import DataGenerator
from models.base_model import build_model
from trainers.trainer import Trainer

DATASETS = ["NCI1", "MUTAG", "PROTEINS", "IMDBBINARY", "ZINC", "MOLHIV"]
VARIANTS = [("baseline", "baseline", dict(FAST)), ("ph_cpu", "topo", dict(FULL, topo_ph_backend="gudhi", **FAST)), ("ph_gpu", "topo", dict(FULL, topo_ph_backend="torch", **FAST))]
NAMES = {"baseline": "PPGN backbone", "ph_cpu": "PPGN+PH, CPU persistence (\\textsc{gudhi})", "ph_gpu": "PPGN+PH, accelerator persistence"}
cuda = torch.cuda.is_available()

def sync():
    if(cuda):
        torch.cuda.synchronize()

def timed_epoch(trainer):
    sync()
    t0 = time.perf_counter()
    trainer.train_epoch()
    sync()
    return time.perf_counter() - t0

def measure(ds, model, ovr, epochs):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = run_job.build_config(ds, model, ovr)
    cfg.num_fold = 1
    data = DataGenerator(cfg)
    net = build_model(cfg)
    trainer = Trainer(net, data, cfg)

    warm = timed_epoch(trainer)  # first epoch builds the complexes
    steps = data.num_iterations_train
    if(cuda):
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(epochs):
        times.append(timed_epoch(trainer))
    if(cuda):
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
    else:
        mem = float("nan")

    del trainer, net, data
    if(cuda):
        torch.cuda.empty_cache()
    return dict(sec_per_epoch=round(float(np.mean(times)), 3), first_epoch_sec=round(warm, 2), peak_gpu_gb=round(mem, 2), steps_per_epoch=int(steps))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    path = "results/timing.json"
    os.makedirs("results", exist_ok=True)
    if(os.path.exists(path)):
        res = json.load(open(path))
    else:
        res = {}

    for ds in args.datasets:
        for(name, model, ovr) in VARIANTS:
            key = "%s/%s" % (ds, name)
            n_epochs = args.epochs
            if(ds == "MOLHIV"):
                n_epochs = 1
            try:
                res[key] = measure(ds, model, ovr, n_epochs)
            except Exception as e:
                res[key] = dict(error=str(e))  # usually OOM
            print(key, res[key], flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=2)

    print("\n% rows for tab:cost (seconds per training epoch)")
    for(name, _, _) in VARIANTS:
        cells = []
        for ds in DATASETS:
            r = res.get("%s/%s" % (ds, name), {})
            if("sec_per_epoch" in r):
                cells.append("%.1f" % r["sec_per_epoch"])
            else:
                cells.append("\\placeholder{--}")
        print("%s & %s \\\\" % (NAMES[name], " & ".join(cells)))

if(__name__ == "__main__"):
    main()
