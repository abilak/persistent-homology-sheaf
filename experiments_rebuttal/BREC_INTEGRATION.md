# Running PPGN+PH on BREC (gold-standard expressiveness benchmark)

BREC (Wang & Zhang, arXiv:2304.07702) = 400 non-isomorphic graph pairs in 4
categories; a model "distinguishes" a pair iff it passes the Reliable Paired
Comparison (RPC) Siamese + statistical test. The **Regular** category (50
strongly-regular + 20 four-vertex-condition + 20 distance-regular pairs) is
exactly the 3-WL-hard regime our persistent-homology branch targets, and it
cross-checks the SR separation result we already verified by hand.

**Crucial:** BREC ships `ProvablyPowerfulGraphNetworks_torch/`, which is the
SAME PPGN-torch codebase this repo is forked from (identical `models/`,
`layers/`, `data_loader/`, `main_scripts/` structure). So integrating our
topology branch is a ~10-line injection, not a rewrite.

--------------------------------------------------------------------------
## HARD RULE: reproduce the PPGN baseline BEFORE adding topology
--------------------------------------------------------------------------
Do NOT put a BREC number in the rebuttal until BREC's *unmodified* PPGN
reproduces (roughly) its published score on at least the Regular category.
If their env doesn't run clean, abandon BREC -- the SR classification result
we already have covers the same claim and is verified.

```bash
git clone https://github.com/GraphPKU/BREC.git
cd BREC
unzip BREC_data_all.zip                       # the 400 pairs (graph6 in .npy)
cd ProvablyPowerfulGraphNetworks_torch
# env: their README pins torch 1.13 / PyG 2.2, but PPGN is dense so a modern
# torch usually works. Install what's missing: pip install torch_geometric loguru gudhi
python main_scripts/test_BREC.py              # baseline PPGN -- must run clean
python main_scripts/test_BREC_search.py       # 10-seed final score
```
Expected baseline: PPGN distinguishes the Basic/simple-Regular pairs but
scores near 0 on the strongly-regular subcategory (3-WL cannot). Confirm this
matches the BREC paper's PPGN row before proceeding.

--------------------------------------------------------------------------
## STEP 1: drop in our topology branch
--------------------------------------------------------------------------
Our `layers/topology.py` is self-contained and portable (only needs torch,
numpy, gudhi, networkx). Copy it into BREC's PPGN layers dir:

```bash
curl -L -o layers/topology.py \
  https://raw.githubusercontent.com/abilak/persistent-homology-sheaf/rebuttal/layers/topology.py
```

--------------------------------------------------------------------------
## STEP 2: inject topology into their models/base_model.py
--------------------------------------------------------------------------
Their BaseModel has the same shape as ours. Make three minimal edits.

(a) at the top, add:
```python
from layers.topology import TopologyLayer, build_graph_structs
```

(b) in `__init__`, right after the loop that builds `self.reg_blocks`
(where `block_features` is iterated), mirror our construction:
```python
        self.use_topology = getattr(config.architecture, 'use_topology', False)
        self.topo_layers = nn.ModuleList() if self.use_topology else None
        if self.use_topology:
            for nf in config.architecture.block_features:
                self.topo_layers.append(TopologyLayer(
                    eqv_features=nf,
                    hidden_dim=getattr(config.architecture, 'topo_hidden_dim', 16),
                    max_ph_dim=getattr(config.architecture, 'topo_max_ph_dim', 2),
                    num_stats=getattr(config.architecture, 'topo_num_stats', 16),
                    gate_bias=getattr(config.architecture, 'topo_gate_bias', 2.0),
                    node_level=getattr(config.architecture, 'topo_node_level', True),
                ))
```
NOTE: keep their `block_features`. Set `topo_max_simplex_dim=3` in the config
so the clique complex includes tetrahedra (needed for strongly-regular pairs
like Rook/Shrikhande).

(c) in `forward(self, input)`, build the complex once and call the topo layer
after each equivariant block. Find their loop
`for i, block in enumerate(self.reg_blocks): x = block(x); ...`
and insert the topology call right after `x = block(x)`:
```python
        simplices_batch = None
        if self.use_topology:
            adj = input[:, 0, :, :].detach().cpu().numpy()
            md = getattr(self.config.architecture, 'topo_max_simplex_dim', 3)
            simplices_batch = build_graph_structs(adj, max_dim=md)
        # ... inside the block loop, after `x = block(x)`:
            if self.use_topology:
                x = self.topo_layers[i](x, simplices_batch)
```
Leave their readout (`diag_offdiag_maxpool`, the FC suffix, and the final
`self.bn(scores)`) UNCHANGED -- topology only edits the internal tensor `x`,
so RPC sees a topology-aware embedding of the same shape.

--------------------------------------------------------------------------
## STEP 3: add the topology config flags
--------------------------------------------------------------------------
In whatever config json BREC's PPGN test uses (see `configs/`), add to the
`architecture` block:
```json
  "use_topology": true,
  "topo_max_simplex_dim": 3,
  "topo_max_ph_dim": 2,
  "topo_num_stats": 16,
  "topo_hidden_dim": 16,
  "topo_gate_bias": 2.0,
  "topo_node_level": true
```
For the baseline run, set `"use_topology": false` (identical to stock PPGN).

--------------------------------------------------------------------------
## STEP 4: run + report
--------------------------------------------------------------------------
```bash
# baseline (use_topology=false in config):
python main_scripts/test_BREC.py && python main_scripts/test_BREC_search.py
# topology (use_topology=true):
python main_scripts/test_BREC.py && python main_scripts/test_BREC_search.py
```

Prioritise by value if time is short: **Regular > Basic > Extension > CFI**.
The Regular category (esp. strongly-regular) is where PH should lift PPGN
above 3-WL and is the headline. CFI graphs are largest/slowest (topology's
gudhi step scales with graph size) -- run last or skip.

--------------------------------------------------------------------------
## Expected result / rebuttal line
--------------------------------------------------------------------------
Baseline PPGN: ~0/50 on strongly-regular (3-WL bound). PPGN+PH: a nonzero
fraction of strongly-regular pairs distinguished -> directly demonstrates,
on the recognized benchmark, that the topology branch realizes expressiveness
beyond 3-WL. This is the standardized form of our Table-2 / SR-classification
result. If it reproduces, it is a very strong addition; if the harness fights
you, cite BREC and note our SR experiment already covers its hardest
(strongly-regular) subcategory.
