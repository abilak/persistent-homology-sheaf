# BREC on the *official* harness (citeable numbers)

Two ways to get BREC results. Use both; they cross-check.

| path | script | what it is | when |
|---|---|---|---|
| **A. standalone** | `../brec_expressivity.py` | our own Reliable Paired Comparison (RPC + Hotelling T²) reusing the rebuttal model zoo (MLP/GCN/GIN/GSN/PPGN/PPGN+PH). Runs today, no external repo. | fast, self-contained, all baselines in one table |
| **B. official** | `inject_topology.py` + BREC's own `test_BREC.py` | inject our PH branch into BREC's *own* PPGN harness and run *their* evaluation, so numbers are directly comparable to the BREC leaderboard. | the citeable version |

Path A is already validated (reliability calibration fixed). Path B is below.

---

## HARD RULE (from BREC_INTEGRATION.md): reproduce stock PPGN first

Do **not** put a BREC number in the rebuttal until BREC's *unmodified* PPGN
reproduces (roughly) its published score on at least the **Regular** category.
If their env won't run clean, cite BREC and fall back to Path A + the SR
classification result, which already covers the hardest (strongly-regular)
subcategory.

---

## Step 0 — clone + data + env

```bash
git clone https://github.com/GraphPKU/BREC.git
cd BREC
unzip BREC_data_all.zip                        # the 400 pairs (graph6 in .npy)
cd ProvablyPowerfulGraphNetworks_torch
pip install torch_geometric loguru gudhi networkx easydict
```

## Step 1 — reproduce the stock PPGN baseline (the hard rule)

```bash
python main_scripts/test_BREC.py               # must run clean
python main_scripts/test_BREC_search.py        # multi-seed final score
```
Expected: PPGN clears Basic + simple-Regular, ~0 on strongly-regular (3-WL
bound). Confirm this matches the BREC paper's PPGN row before proceeding.

## Step 2 — inject our persistent-homology branch (automated + safe)

The injector defaults `--topology-src` to this repo's `layers/topology.py`
(resolved from the script's own location, so cwd doesn't matter). From inside
BREC's `ProvablyPowerfulGraphNetworks_torch/`:

```bash
PHS=/path/to/persistent-homology-sheaf   # this repo (rebuttal branch)
python $PHS/experiments_rebuttal/brec_official/inject_topology.py --dir .
#   ^ dry-run: prints what it will do, writes nothing
python $PHS/experiments_rebuttal/brec_official/inject_topology.py --dir . --apply
#   ^ writes (with .bak backups), then self-verifies a forward pass topo off/on
```

The injector makes exactly three anchored edits to `models/base_model.py`
(import, per-block `TopologyLayer` construction in `__init__`, fuse call in
`forward`) and copies `topology.py` into `layers/`. It is **idempotent**
(re-running skips), keeps timestamped `.bak` files, and `--revert` undoes
everything. If BREC's `base_model.py` has drifted and an anchor is missing, it
**aborts with the exact manual edits** rather than writing a half-patch.

`topology.py` depends only on `torch, numpy, gudhi, networkx` — nothing from
this repo — so it drops into BREC's tree unchanged.

## Step 3 — turn topology on in the config

In whatever config json BREC's PPGN test loads (see their `config/`), add to
the `architecture` block:

```json
  "use_topology": true,
  "topo_max_simplex_dim": 3,
  "topo_max_ph_dim": 2,
  "topo_num_stats": 16,
  "topo_hidden_dim": 16,
  "topo_gate_bias": 2.0,
  "topo_node_level": true,
  "topo_gate_mode": "conv"
```

`topo_max_simplex_dim: 3` includes tetrahedra (4-cliques) in the clique
complex — needed for the strongly-regular pairs (Rook/Shrikhande). For the
baseline run set `"use_topology": false` (identical to stock PPGN), or just use
the unpatched checkout.

## Step 4 — run both and report

```bash
# baseline (use_topology=false):
python main_scripts/test_BREC.py && python main_scripts/test_BREC_search.py
# topology (use_topology=true):
python main_scripts/test_BREC.py && python main_scripts/test_BREC_search.py
```

Priority if time is short: **Regular > Basic > Extension > CFI**. The Regular
(strongly-regular) block is the headline — where PH lifts PPGN above 3-WL. CFI
graphs are largest/slowest (gudhi persistence scales with graph size); run last.

## Expected result / rebuttal line

Stock PPGN: ~0/50 on strongly-regular (3-WL bound). PPGN+PH: a nonzero fraction
of strongly-regular pairs distinguished → on the recognized benchmark, the
topology branch realizes expressiveness beyond 3-WL. This is the standardized
form of our Table-2 / SR-classification result. Report per-category
distinguished counts for stock PPGN vs PPGN+PH; the strongly-regular delta is
the citeable confirmation of Corollary 6.
