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

## Fast GPU training (impl-containment branch)

The topology branch no longer needs the CPU on GPU runs:

* `topo_ph_backend: "auto"` (default) computes persistent homology on the GPU
  (`layers/torch_ph.py`): clique complexes up to tetrahedra, homology up to
  dimension 2, essential classes included, via rank-space H0 (bottleneck
  squarings) and persistent cohomology with clearing + parallel Z/11
  reduction for H1/H2. Covers every training batch of all six TU datasets;
  gudhi only for CPU runs or single graphs above a memory cap. Molecular
  batches run a provably sufficient fixed number of reduction steps (no host
  sync at all); dense batches (PROTEINS, IMDB-B) check convergence every few
  steps (one scalar sync each). Exact vs gudhi (`test_torch_ph.py`).
* `padded_batching: true` batches graphs of similar size with exact masking
  (`test_padded.py`): 2-2.5x fewer steps per epoch. PPGN / PPGN+PH only.
* Metrics accumulate on device (one sync per epoch); Adam is fused on CUDA.

Check that a step really has no host syncs, and time it:

```bash
SYNC_DEBUG=1 python experiments_rebuttal/bench_backends.py NCI1 cuda torch 20   # raises on any sync
python experiments_rebuttal/bench_backends.py NCI1 cuda baseline 50
python experiments_rebuttal/bench_backends.py NCI1 cuda gudhi 50
python experiments_rebuttal/bench_backends.py NCI1 cuda torch 50
```

Run the improvement sweep with the fast settings (baseline gets the same
batching), then score against the matched baseline:

```bash
CCD_DEVICE=cuda python experiments_rebuttal/runner.py improve_all_fast 8 1
python experiments_rebuttal/compare_variant.py NCI1 fast
```

Several jobs share one GPU well now that no job blocks on the CPU; raise the
worker count until GPU utilization saturates.
