#!/usr/bin/env bash
# Every remaining experiment for the paper: everything except ogbg-molhiv
# first, then all molhiv runs, then the timing benchmark. Run from the repo root:
#     bash experiments_rebuttal/run_remaining.sh 2>&1 | tee remaining.log
# Each plan skips jobs whose result file exists, so the script can be stopped
# and restarted at any time. Worker counts are set for a 20 GB GPU shared with
# other users: memory-heavy datasets (PROTEINS, molhiv) run with fewer workers.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <plan> <workers>
    echo "=== $(date '+%F %T')  $1 ($2 workers)"
    systemd-run --user --scope -p MemoryMax=12G python experiments_rebuttal/runner.py "$1" "$2"
}

# ---- 0. interpretability (MUTAG; ~10 minutes) --------------------------------
echo "=== $(date '+%F %T')  interpret_mutag"
[ -f rebuttal_results/interpret_mutag/stats.json ] || \
    systemd-run --user --scope -p MemoryMax=12G python experiments_rebuttal/interpret_mutag.py

# ---- A. main tables (no molhiv) ---------------------------------------------
run extra_tu_fast 3              # IMDB-M, ENZYMES (skips if already done)
run wide_fast 3                  # parameter-matched (widened) backbone
run wide_extra_tu_fast 3
run wide_zinc_fast_s10 3
run wide_fast_proteins 1
run mp_fast 3                    # MLP, GCN, GIN, GSN in the same pipeline
run mp_zinc_fast_s10 3
run static_fast 3                # static control: same branch, no homology
run static_zinc_fast_s10 3
run static_fast_proteins 1

# ---- B. appendix studies (no molhiv) ----------------------------------------
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

# ---- C. ogbg-molhiv (largest molecules: 1-2 workers) ------------------------
run molhiv_fast_s10 1            # backbone and PPGN+PH, 10 seeds (resumes)
run mp_molhiv_fast_s10 2
run wide_molhiv_fast_s10 1       # slowest plan (~2.5 days); drop seeds here if short on time

# ---- D. timing (needs the GPU otherwise idle) --------------------------------
echo "=== $(date '+%F %T')  timing"
systemd-run --user --scope -p MemoryMax=12G python experiments_rebuttal/timing.py

echo "=== $(date '+%F %T')  all done. Tables:"
python experiments_rebuttal/paper_tables.py
