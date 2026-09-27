#!/usr/bin/env bash
# Every remaining experiment for the paper, most important first. Run from the
# repo root after molhiv_fast_s10 has finished:
#     bash experiments_rebuttal/run_after_ogb.sh
# Each plan skips jobs whose result file exists, so the script can be stopped
# and restarted at any time. Worker counts are set for a 20 GB GPU shared with
# other users: memory-heavy datasets (PROTEINS, molhiv) run with fewer workers.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <plan> <workers>
    echo "=== $(date '+%F %T')  $1 ($2 workers)"
    systemd-run --user --scope -p MemoryMax=12G python experiments_rebuttal/runner.py "$1" "$2"
}

# ---- A. main tables -------------------------------------------------------
run extra_tu_fast 3              # IMDB-M, ENZYMES (skips if already done)
# parameter-matched (widened) backbone
run wide_fast 3
run wide_extra_tu_fast 3
run wide_zinc_fast_s10 3
run wide_fast_proteins 1
run wide_molhiv_fast_s10 1
# same-pipeline message-passing baselines (MLP, GCN, GIN, GSN)
run mp_fast 3
run mp_zinc_fast_s10 3
run mp_molhiv_fast_s10 2
# static control (same branch and parameters, no homology)
run static_fast 3
run static_zinc_fast_s10 3
run static_fast_proteins 1

# ---- B. appendix studies ---------------------------------------------------
run structure_fast 3             # node labels removed
run structure_fast_proteins 1
run placement_fast 3             # topology after the last block only
run placement_zinc_fast_s10 3
run ablation_min_fast 3          # ablation ladder (a), (b)
run zinc_ablation_fast 3
run depth_mutag 3                # L = 1, 3, 4 blocks
run depth_zinc 3
run depth_ptc 3
run depth_nci1 3

echo "=== $(date '+%F %T')  all done; tables: python experiments_rebuttal/paper_tables.py"
