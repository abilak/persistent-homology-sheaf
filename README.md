# PPGN + persistent homology

This is the code for our paper **Beyond Weisfeiler-Lehman: Topological Features Increase Graph Neural Network Expressivity**.

Graph neural networks are limited by the Weisfeiler-Lehman (WL) test, which means there are pairs of different graphs they can't tell apart. We build off the paper PPGN (Provably Powerful Graph Networks, Maron et al. 2019, https://arxiv.org/abs/1905.11136), which is already as strong as 3-WL, and add a topology branch to it. After each PPGN block the model does the following:

1. learns a filtration on the clique complex of the graph (every vertex, edge and triangle gets a value computed from the PPGN features),
2. computes persistent homology of that filtration (connected components, cycles, voids and how long they last),
3. turns the persistence diagrams into vectors with attention pooling,
4. and adds them back into the PPGN features through a gated residual.

With these additions, the model can separate graphs that 3-WL can't, such as the 4x4 rook graph and the Shrikhande graph. The persistence computation runs on the GPU with plain tensor ops (`layers/torch_ph.py`), and on CPU it uses gudhi instead.

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

Run everything from the repo root. Results go into `results/`.

To train one model, pick a dataset, a model (`baseline`, `topo`, `mlp`, `gcn`, `gin` or `gsn`), a seed and a fold

```bash
python experiments/run_job.py MUTAG topo 0 1 results/MUTAG/test.json
```

That trains the vanilla topo model with the default settings. The settings we actually used in the paper are passed as a json string of overrides, which is easier to get from the runner below.

`experiments/runner.py` has every set of runs from the paper (main TU table, ZINC, molhiv, the widened backbone, the static and Betti controls, the message passing baselines, ablations, depth and so on). Running a plan launches its jobs in parallel and skips anything that already has a result file, so you can stop it and start it again whenever.

```bash
python experiments/runner.py core_fast 3 1
```

The arguments are the plan name, how many jobs run at once, and how many CPU threads each job gets. Run it with no arguments to list all the plans.

This one runs every plan plus all the expressivity and interpretability experiments from one queue, and the timing benchmark at the end. It takes days on one GPU.

```bash
python experiments/runner.py all 3
```

Add `--dry` to see how much is left without running anything.

Once results are in, this prints the paper tables as LaTeX rows

```bash
python experiments/tables.py
```

and this compares every model on one dataset against the backbone fold by fold (paired bootstrap interval, permutation test and Wilcoxon test)

```bash
python experiments/tables.py NCI1
```

### The other experiments

| What | Command |
|---|---|
| Strongly regular pairs at random init | `python experiments/srg.py` (add `--seeds 30` for the per seed version) |
| Clique counts vs homology | `python experiments/srg.py count` |
| Cones of the rook and Shrikhande graphs | `python experiments/srg.py cone --seeds=30` |
| Classifying strongly regular graphs and CSL | `python experiments/sr_classification.py --seeds=3` |
| BREC | `python experiments/brec.py --model topo --sample-num 400 --epochs 20 --out results/brec/topo.json` (also `ppgn` and `gin`) |
| What the filtration learns on MUTAG | `python experiments/interpret_mutag.py` (and `--seed=1`, `--seed=2`) |
| Training cost, CPU vs GPU persistence | `python experiments/timing.py NCI1 MUTAG PROTEINS IMDBBINARY ZINC` |

To pick a GPU set `DEVICE`, for example `DEVICE=cuda:1 python experiments/runner.py core_fast 3 1`.

## How we evaluate

The TU datasets use the protocol from Xu et al. (2019). That's ten folds, one seed, and a single epoch picked by the mean validation accuracy over all folds, then every fold is reported at that epoch. All models share the folds, the batching, the learning rate schedule and the number of epochs. ZINC and ogbg-molhiv use their standard splits with ten seeds, and we report the test number at the epoch with the best validation number.

## Tests

```bash
python -m pytest tests
```

These check that the GPU persistence gives the same diagrams as gudhi, that the model is permutation invariant, and that padded batches give the same outputs as running each graph alone. It takes about ten seconds on a laptop.

## Where stuff is

The topology part of the model is in `layers/topology.py` (filtration, persistence diagrams, fusion) and `layers/torch_ph.py` (the GPU persistence). `models/base_model.py` puts it together with the PPGN blocks from `layers/modules.py`, and the message passing baselines are in `models/baseline_models.py`. Data loading is in `data_loader/`, the training loop in `trainers/`, and everything you'd actually run is in `experiments/`.
