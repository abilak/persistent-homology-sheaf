# PPGN + PH

This is the code for our paper **Beyond Weisfeiler-Lehman: Topological Features Increase Graph Neural Network Expressivity**, under review at AISTATS 2026-27.

GNNs are limited by the k-WL/FWL tests (Weisfeiler-Lehman), meaning there are pairs of different graphs they can't tell apart (graph isomorphism). We build off the PPGN paper (Provably Powerful Graph Networks, Maron et al. 2019, https://arxiv.org/abs/1905.11136), which is provably as strong as 3-WL, and add a topology branch to it to improveits expressivity and experimental performance. After each PPGN block the model learns sa filtration function on the clique complex of the graph, computes the PH of that filtration, makes the PH diagrams to vectors with attention pooling, and adds them to the original PPGN model features with a gated residual (which lets us let in a certain amount of topology, learned by the model).

These additions let the model separate graphs that 3-WL can't, such as the 4x4 rook graph and the Shrikhande graph. The persistence computation runs on the GPU with plain tensor ops (`layers/torch_ph.py`), and on CPU it uses gudhi instead.

The PPGN part and the data loading started from a PyTorch port of the original PPGN code (Apache 2.0, see `LICENSE`). The original TensorFlow code is at https://github.com/hadarser/ProvablyPowerfulGraphNetworks.

## Setup

You need Python 3.10 or newer. Install PyTorch for your CUDA version first (https://pytorch.org), then

```bash
pip install -r requirements.txt
```

Everything also runs on CPU, however, it is much slower (especially for bigger datasets).

## Data

The TU datasets (MUTAG, PTC, NCI1, NCI109, PROTEINS, ENZYMES, IMDB-B, IMDB-M) come with the standard 10 fold splits from Xu et al. Download them with

```bash
python utils/get_data.py
```

This puts them in `data/benchmark_graphs/`. ZINC-12k and ogbg-molhiv have their own scripts

```bash
python utils/get_data.py zinc
python utils/get_data.py molhiv
```

The BREC benchmark gets downloaded the first time you run `experiments/brec.py`.

## Running things

Everything must be run from the repo root. Results go into the directory `results/`.

To train one model, pick a dataset, a model (`baseline`, `topo`, `mlp`, `gcn`, `gin` or `gsn`), a seed and a fold

```bash
python experiments/run_job.py MUTAG topo 0 1 results/MUTAG/test.json
```

This trains the vanilla topo model with the default settings. The settings used in the paper are passed as a json string of overrides, which is easier to get from the runner below.

`experiments/runner.py` has all runs from the paper (TU dataset table, ZINC, molhiv, widened backbones, static and Betti controls, message-passing baselines, ablations, depth, etc.). Running a plan launches the jobs in parallel and skips anything already with a result file, so you can stop it and start it again whenever.

```bash
python experiments/runner.py core_fast 3 1
```

The arguments are the plan name, how many jobs run at once, and how many CPU threads each job gets. You may run it with no arguments to list all the plans.

This command runs every plan along with all the expressivity and interpretability experiments from one queue, and the timing benchmark at the end. 

```bash
python experiments/runner.py all 3
```

Add `--dry` to see how much is left without running anything.

Once results are in, this command outputs the paper tables as LaTeX rows

```bash
python experiments/tables.py
```

and the next command compares every model on one dataset against the backbone fold by fold with various statistical tests: 

```bash
python experiments/tables.py NCI1
```

### Other Experiments

| What | Command |
|---|---|
| Strongly regular pairs at random init | `python experiments/srg.py` (add `--seeds 30` for the per seed version) |
| Clique counts vs. homology | `python experiments/srg.py count` |
| Cones of the rook and Shrikhande graphs | `python experiments/srg.py cone --seeds=30` |
| Classifying strongly regular graphs and CSL | `python experiments/sr_classification.py --seeds=3` |
| BREC | `python experiments/brec.py --model topo --sample-num 400 --epochs 20 --out results/brec/topo.json` (also `ppgn` and `gin`) |
| Interpretability of filtration on MUTAG | `python experiments/interpret_mutag.py` (and `--seed=1`, `--seed=2`) |
| Training cost, CPU vs GPU PH | `python experiments/timing.py NCI1 MUTAG PROTEINS IMDBBINARY ZINC` |

To pick a GPU set `DEVICE`, for example `DEVICE=cuda:1 python experiments/runner.py core_fast 3 1`.

## Tests

```bash
python -m pytest tests
```

These check that the GPU persistence gives the same diagrams as gudhi, that the model is permutation invariant, and that padded batches give the same outputs as running each graph alone.

## Where stuff is

The topology part of the model is in `layers/topology.py` (filtration, persistence diagrams, fusion) and `layers/torch_ph.py` (the GPU persistence). `models/base_model.py` puts it together with the PPGN blocks from `layers/modules.py`, and the message passing baselines are in `models/baseline_models.py`. Data loading is in `data_loader/`, the training loop in `trainers/`, and everything you'd actually run is in `experiments/`.
