#!/usr/bin/env bash
# Run the full rebuttal experiment matrix on a GPU server.
#
# The dense PPGN ops run on the GPU; the persistent-homology branch uses cached
# per-graph structure + a single batched host sync per layer, so the GPU is not
# stalled. Set CCD_DEVICE to pick the GPU (e.g. cuda / cuda:0 / cuda:1).
#
# Usage:
#   bash experiments_rebuttal/run_gpu.sh            # all plans, cuda
#   CCD_DEVICE=cuda:1 bash experiments_rebuttal/run_gpu.sh
#
# The runner is resumable: completed result JSONs under rebuttal_results/ are
# skipped, so you can Ctrl-C and rerun. Each job is one (dataset,model,seed,fold).
set -e
cd "$(dirname "$0")/.."
export CCD_DEVICE="${CCD_DEVICE:-cuda}"
PY="${PY:-python}"

# On a single GPU, run jobs sequentially (WORKERS=1) so they don't contend for
# GPU memory; the GPU itself parallelizes each dense batch. Bump WORKERS if the
# GPU has spare memory and you want several small-graph jobs concurrent.
WORKERS="${WORKERS:-1}"
THREADS="${THREADS:-8}"   # CPU threads for the gudhi/PH part per job

echo "device=$CCD_DEVICE  workers=$WORKERS  threads=$THREADS"

# 1) small datasets, all models, 5 seeds x 10 folds (reviewer focus: MUTAG/PTC)
$PY experiments_rebuttal/runner.py small      "$WORKERS" "$THREADS"
$PY experiments_rebuttal/runner.py gsn_small  "$WORKERS" "$THREADS"
# 2) light + subgraph-counting baselines on big datasets
$PY experiments_rebuttal/runner.py big_light  "$WORKERS" "$THREADS"
# 3) heavy PPGN base vs topo on big datasets (NCI1 then NCI109), 2 seeds
$PY experiments_rebuttal/runner.py big_heavy  "$WORKERS" "$THREADS"

# aggregate + clean runtime benchmark
$PY experiments_rebuttal/aggregate.py
CCD_DEVICE="$CCD_DEVICE" $PY experiments_rebuttal/runtime_bench.py MUTAG PTC NCI1
echo "DONE. See rebuttal_results/summary.json and REBUTTAL.md"
