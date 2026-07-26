# Running the rebuttal experiments on your GPU server

Everything is on branch **`rebuttal`** (pushed to origin). The dense PPGN ops run
on the GPU; the persistent-homology branch is cached per graph and uses a single
batched host sync per layer, so the GPU is not stalled. Numerically identical to
the original code (verified: fwd/grad diff 0.0) and equivariant.

## 1. Setup (once)

```bash
git clone https://github.com/abilak/persistent-homology-sheaf.git
cd persistent-homology-sheaf
git fetch origin && git checkout rebuttal

# env: a CUDA build of torch, plus:
pip install gudhi networkx numpy   # (torch installed per your CUDA version)

# data (TU benchmark graphs incl. NCI109 + the standard Xu 10-fold splits):
python utils/get_data.py           # or just the benchmark zip; see utils/get_data.py
ls data/benchmark_graphs           # should list MUTAG NCI1 NCI109 PTC PROTEINS ...
```

Sanity check the GPU path (should print "PASS"):
```bash
CCD_DEVICE=cuda python experiments_rebuttal/test_topo_equiv.py
```

## 2. Run (resumable — completed result JSONs are skipped, Ctrl-C safe)

`CCD_DEVICE` picks the GPU. `runner.py <plan> <workers> <cpu_threads>`: `workers`
= concurrent jobs sharing the GPU (small graphs underutilize the GPU, so several
concurrent jobs raise throughput); `cpu_threads` = threads for the CPU-side PH.

```bash
export CCD_DEVICE=cuda            # or cuda:1

# reviewer-focus small datasets — MUTAG/PTC × {mlp,gcn,gin,baseline,topo} × 5 seeds × 10 folds
python experiments_rebuttal/runner.py small      8 2
# subgraph-counting baseline (GSN) on MUTAG/PTC
python experiments_rebuttal/runner.py gsn_small  8 2
# light + GSN baselines on the big datasets
python experiments_rebuttal/runner.py big_light  6 2
# heavy PPGN baseline vs +PH on NCI1 then NCI109, 2 seeds × 10 folds
python experiments_rebuttal/runner.py big_heavy  4 4

# (optional, run in PARALLEL) fair hyperparameter check on NCI1: the paper's
# stated grid (lr in {1e-4,5e-4,1e-3}, topo width in {8,16,32}) on 3 folds,
# applied to BOTH baseline and topo. Pick the config best for BOTH, then rerun
# big_heavy with it via the `overrides` field. This is defensible tuning (the
# grid the paper says it used), not p-hacking.
python experiments_rebuttal/runner.py hp_check   6 3
```

Or all of it at once (sequential plans): `CCD_DEVICE=cuda bash experiments_rebuttal/run_gpu.sh`

Tips: bump `workers` on the small plans if `nvidia-smi` shows the GPU
underused; drop to 2–3 for `big_heavy` if memory is tight. If you have several
GPUs, run different plans with different `CCD_DEVICE=cuda:0/1/...` in parallel.

## 3. Aggregate + report

```bash
python experiments_rebuttal/aggregate.py          # per-model means, PAIRED tests, CIs
python experiments_rebuttal/comparison_table.py    # our numbers + published baselines
CCD_DEVICE=cuda python experiments_rebuttal/runtime_bench.py MUTAG PTC NCI1   # clean s/epoch + ms/graph
```

Results land in `rebuttal_results/<DATASET>/<model>_s<seed>_f<fold>.json`
(fields: `best_val_acc`, `best_epoch`, `sec_per_train_epoch`, `val_curve`, …).
Paste `aggregate.py` + `comparison_table.py` + `runtime_bench.json` output back to
me and I'll fold the final numbers into the rebuttal.

## What each result means
- `best_val_acc` = best-epoch validation accuracy (the reported TU metric).
- `aggregate.py` prints the **paired** topo−baseline difference per dataset with a
  permutation p-value, Wilcoxon p-value, and 95% bootstrap CI — the statistic
  that answers "are the gains within fold std?".
