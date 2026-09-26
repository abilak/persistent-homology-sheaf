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


MODE = {"timer": False}
TIMES = {}


def label(owner, name, tag):
    """profiler range, or (MODE timer) synchronized wall-clock accumulation"""
    f = getattr(owner, name)
    @functools.wraps(f)
    def g(*a, **k):
        if MODE["timer"]:
            sync(); t = time.perf_counter()
            out = f(*a, **k)
            sync(); TIMES[tag] = TIMES.get(tag, 0.0) + time.perf_counter() - t
            return out
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


# ---- synchronized wall-clock per component (inclusive), topo vs baseline ----
def timed_steps(model_type):
    torch.manual_seed(0); np.random.seed(0)
    c = run_job.build_config(DS, model_type, job["overrides"] if model_type == "topo"
                             else {"padded_batching": True}); c.num_fold = 1
    dd = DataGenerator(c); w = ModelWrapper(c, dd); t_ = Trainer(w, dd, c)
    t_.train_epoch(0)                                # warm: complexes cached
    dd.initialize("train")
    n = min(STEPS, dd.num_iterations_train)
    TIMES.clear(); MODE["timer"] = True
    for _ in range(n):
        sync(); t0 = time.perf_counter(); g, y = dd.next_batch(); sync(); t1 = time.perf_counter()
        t_.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(w.model(g), y); sync(); t2 = time.perf_counter()
        loss.backward(); sync(); t3 = time.perf_counter()
        t_.optimizer.step(); sync(); t4 = time.perf_counter()
        for k, v in (("data", t1 - t0), ("forward", t2 - t1), ("backward", t3 - t2), ("optimizer", t4 - t3), ("step total", t4 - t0)):
            TIMES[k] = TIMES.get(k, 0.0) + v
    MODE["timer"] = False
    return {k: v / n * 1e3 for k, v in TIMES.items()}

print("\nsynchronized wall-clock per step (ms; inclusive, nested components are part of forward):")
tb, tt = timed_steps("baseline"), timed_steps("topo")
print(f"{'':<16}{'baseline':>10}{'full model':>12}")
for k in ["data", "forward", "backward", "optimizer", "step total"] + [t for t in TAGS if t in tt and t != "#model_forward"]:
    print(f"{k:<16}{tb.get(k, 0):>10.1f}{tt.get(k, 0):>12.1f}")
