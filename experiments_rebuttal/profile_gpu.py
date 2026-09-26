"""
Where does a training step of the full topology model spend its time?

  python experiments_rebuttal/profile_gpu.py [DATASET] [STEPS]   (default IMDBBINARY 10)

1. Times epoch 1 (includes building + caching every graph's clique complex)
   and epoch 2 (steady state) of the full_fast configuration.
2. Profiles STEPS training steps (CPU + CUDA) and prints
   - the model's components (clique complexes, batch plan, filtration, PH,
     torch PH, its reductions, H0 closure) with CPU and GPU time,
   - the top kernels by GPU time,
   - host<->device synchronizations.
Nothing is written to rebuttal_results.
"""
import os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QUIET", "1")
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity, record_function
import run_job, runner
import layers.topology as T
import layers.torch_ph as TP
import models.base_model as BM

DS = sys.argv[1] if len(sys.argv) > 1 else "IMDBBINARY"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cuda = torch.cuda.is_available()


def sync():
    if cuda:
        torch.cuda.synchronize()


def label(owner, name, tag):
    f = getattr(owner, name)
    @functools.wraps(f)
    def g(*a, **k):
        with record_function(tag):
            return f(*a, **k)
    setattr(owner, name, g)


TAGS = ["#complexes", "#plan", "#filtration", "#ph_layer", "#torch_ph", "#reduce", "#h0_closure",
        "#h0_relax", "#topo_layer", "#model_forward"]
label(BM.BaseModel, "_build_simplicial_complexes", "#complexes")
label(T, "get_plan", "#plan")
label(T.LearnedFiltration, "forward", "#filtration")
label(T.DifferentiablePH, "forward", "#ph_layer")
label(TP, "torch_persistence", "#torch_ph")
label(TP, "_reduce_parallel", "#reduce")
label(TP, "_minmax_closure", "#h0_closure")
label(TP, "_bottleneck_relax", "#h0_relax")
label(T.TopologyLayer, "forward", "#topo_layer")
label(BM.BaseModel, "forward", "#model_forward")

job = [j for j in runner.PLANS["core_fast"] if j["dataset"] == DS and j["model"] == "topo"][0]
torch.manual_seed(0); np.random.seed(0)
cfg = run_job.build_config(DS, "topo", job["overrides"]); cfg.num_fold = 1
from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
d = DataGenerator(cfg); mw = ModelWrapper(cfg, d); tr = Trainer(mw, d, cfg)

for ep in range(2):
    sync(); t = time.perf_counter(); tr.train_epoch(ep); sync()
    print(f"epoch {ep + 1}: {time.perf_counter() - t:.2f} s  ({d.num_iterations_train} steps)"
          + ("   <- includes building every graph's complex once" if ep == 0 else "   <- steady state"))

d.initialize("train")
batches = [d.next_batch() for _ in range(min(STEPS, d.num_iterations_train))]
opt = tr.optimizer
loss_fn = torch.nn.functional.cross_entropy


def run():
    for g, y in batches:
        opt.zero_grad(set_to_none=True)
        loss_fn(mw.model(g), y).backward()
        opt.step()


run(); sync()
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
sync(); t = time.perf_counter()
with profile(activities=acts) as prof:
    run(); sync()
wall = time.perf_counter() - t
ka = prof.key_averages()
dev_attr = "device_time_total" if hasattr(ka[0], "device_time_total") else "cuda_time_total"
sdev_attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
ms = lambda us: f"{us / 1e3:9.1f}"
print(f"\n{len(batches)} profiled steps, wall {wall:.2f} s ({wall / len(batches) * 1e3:.0f} ms/step)\n")
print(f"{'component':<16}{'calls':>7}{'CPU total ms':>14}{'GPU total ms':>14}")
by = {e.key: e for e in ka}
for tag in TAGS:
    if tag in by:
        e = by[tag]
        print(f"{tag:<16}{e.count:>7}{ms(e.cpu_time_total):>14}{ms(getattr(e, dev_attr, 0)):>14}")
print("\n(nested: #model_forward contains #topo_layer contains #filtration/#ph_layer; "
      "#ph_layer contains #torch_ph contains #reduce/#h0_*; the backward pass is outside #model_forward)")
if cuda:
    print("\ntop kernels by GPU time:")
    print(ka.table(sort_by=sdev_attr, row_limit=15))
syncs = [e for e in ka if any(s in e.key for s in ("Synchronize", "item", "_local_scalar_dense", "nonzero", "MemcpyAsync"))]
print("\nhost<->device sync / copy ops:")
for e in sorted(syncs, key=lambda e: -e.count)[:8]:
    print(f"  {e.key:<40}{e.count:>7} calls {ms(e.cpu_time_total)} ms CPU")
