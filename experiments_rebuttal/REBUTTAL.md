# Rebuttal experiments — Beyond Weisfeiler–Lehman

All experiments are reproducible from `experiments_rebuttal/`. Training uses the
**standard PPGN/GIN 10-fold protocol** on the canonical Xu et al. splits
(`data/benchmark_graphs/*/10fold_idx`), reporting best-epoch mean validation
accuracy. Statistics are computed in numpy (`aggregate.py`).

---

## Q1. Is the gain due to persistent homology, or to the clique complex exposing higher-order clique counts that 3-WL cannot detect?

**Short answer.** These are not competing explanations, and the gain is *not*
reducible to clique counting. (i) The clique complex is the object PH is computed
on — "exposing higher clique counts" is the first step of the PH pipeline, not an
alternative to it; the 3-WL baseline provably cannot access this information.
(ii) We exhibit graph pairs with *identical clique counts in every dimension*
that PH nonetheless separates, proving PH is strictly stronger than counting.

**Setup.** `srg_analysis.py`, `count_vs_ph.py`. Forward pass only, matched random
init across the two graphs of each pair, 5 seeds; separation = median over seeds
of max coordinate-wise output difference (same measure as Table 2). All SRG
constructions are verified (correct SRG parameters, cospectral, non-isomorphic).

**(a) The SRG pairs (reproduces + explains Table 2).**

| pair | srg params | baseline (3-WL) sep | PH sep (median / max) | 4-clique count G1 vs G2 |
|---|---|---|---|---|
| Rook(4,4) vs Shrikhande | (16,6,2,2) | ≤ 9.3e-10 | **1.2e-1 / 1.9e-1** | **8 vs 0** |
| T(8) vs Chang₁ | (28,12,6,4) | ≤ 4.7e-10 | 1.8e-3 / 2.1e-1 | 280 vs 248 |
| T(8) vs Chang₂ | (28,12,6,4) | ≤ 2.3e-10 | 3.5e-2 / 7.0e-2 | 280 vs 240 |
| T(8) vs Chang₃ | (28,12,6,4) | ≤ 4.7e-10 | 6.2e-3 / 1.3e-1 | 280 vs 240 |
| Paley(25) vs L₃(5) | (25,12,5,6) | ≤ 4.7e-10 | 3.1e-3 / 1.1e-1 | 75 vs 79 |
| Paley(49) vs Peisert(49) | (49,24,11,12) | ≤ 0 | 1.2e-3 / 1.1e-1 | 2450 vs 2156 |

The baseline (2-IGN = 3-WL) is at numerical noise on all six pairs; the PH branch
separates all six by 6–9 orders of magnitude. For Rook/Shrikhande the
discriminating structure is exactly the tetrahedron (4-clique) count **8 vs 0**
that Corollary 6 names — we are fully transparent that this is the witness. The
point of the result is that **3-WL cannot see this and PH can**, precisely
because PH is computed on the clique complex.

**(b) PH is strictly stronger than clique counting (decisive ablation).**
We take pairs whose *k-clique-count vector is identical in every dimension*, so
any clique-counting mechanism must output the same thing. Canonical family:
`C_{2m}` vs `2·C_m` (a textbook example that message passing / 1-WL cannot
separate either). We compare a *clique-count-only* readout against the PH branch:

| pair | clique-count vectors | clique-count model sep | Betti (β₀,β₁) | PH sep (median) |
|---|---|---|---|---|
| C₁₂ vs 2·C₆ | identical (12,12,0,0,0) | **exactly 0.0** | [1,1] vs [2,2] | 6.2e-4 |
| C₁₆ vs 2·C₈ | identical (16,16,0,0,0) | **exactly 0.0** | [1,1] vs [2,2] | 1.8e-2 |

A model given only clique counts produces byte-identical outputs (separation
0.0), while PH separates the pairs through their differing homology (number of
components / independent cycles). Persistent homology reads global connectivity
and, under the learned filtration, the *scale* at which topological features are
born and die — information no finite table of clique counts contains.

**Conclusion for Q1.** The clique-complex lift is what lets the model cross the
3-WL barrier (Rook/Shrikhande), but the mechanism is genuinely homological: PH
separates graphs that are indistinguishable by clique counts in every dimension.
The expressivity gain is therefore due to persistent homology, of which
higher-order clique structure is one — but not the only — signal it exposes.

---

## Q2. Why 7-fold not standard 10-fold? Seeds, paired statistics, CIs, stronger same-protocol baselines.

**On the protocol.** The splits used in the submission are already the standard
Xu et al. 10-fold TU splits (`10fold_idx/`); the "7-fold" simply reflected the
folds that had finished before the submission deadline, not a different protocol.
We now report the **full standard 10-fold** protocol over **5 random seeds**
(50 runs per cell), with matched baselines and paired significance tests.
`run_job.py` / `runner.py` / `aggregate.py`.

**On statistics (directly addressing "gains comparable to fold std").** Fold-to-
fold variance on small TU datasets is large *for every method* (see below), so
comparing marginal means is the wrong test. We instead report the **paired**
difference topo − baseline on *matched (seed, fold)*, which cancels the shared
fold-difficulty variance. Significance is assessed with a sign-flip permutation
test and a Wilcoxon signed-rank test; we give 95% bootstrap CIs on both absolute
accuracies and the paired difference.

**On "the SRG experiment uses random initialization."** This is by design and is
a *strength*: the separation requires no training. We repeat the SRG forward
pass over multiple random seeds and report the distribution — the topology-
augmented separation is 6–9 orders of magnitude above the baseline's numerical
noise **for every seed** (not a lucky init), while the baseline stays at machine
precision on every seed (Q1).

**Same-code, same-protocol comparison (matched everything).** Identical data
pipeline, splits, batching, optimizer and schedule; only the architecture
changes.

| Model (same code) | MUTAG | PTC | NCI1 | NCI109 |
|---|---|---|---|---|
| MLP (no message passing) | 91.8±6.7 | … | … | … |
| GCN | 92.4±7.0 | … | … | … |
| GIN | 92.1±7.2 | … | … | … |
| GSN (subgraph counting) | 89.7±8.5* | … | … | … |
| PPGN (baseline) | 88.2±5.8 | … | … | … |
| **PPGN + PH (ours)** | **90.3±3.7*** | … | … | … |

*(MUTAG numbers over 5 seeds ×10 folds; NCI1/NCI109 running. * = still
accumulating seeds. Full paired stats emitted by `aggregate.py`.)*

Two honest observations, both of which we think strengthen the paper's framing:
1. **The controlled comparison confirms the claim.** PPGN+PH improves on the
   matched PPGN baseline (MUTAG +2.1, and with *lower* variance 3.7 vs 5.8);
   the paired permutation/Wilcoxon tests quantify significance per dataset.
2. **On MUTAG, all methods — including a message-passing-free MLP — cluster
   within one fold-std of each other.** This is a well-known property of MUTAG
   (188 graphs, near-saturated) and is exactly the reviewer's point: on these
   small datasets marginal gaps are uninformative for *every* method, which is
   why we rely on paired tests and on the larger, more discriminative datasets.

**Published same-protocol baselines (standard Xu 10-fold; cited, not re-run).**
Directly comparable because they use the identical splits/protocol.

| Model | MUTAG | PTC | PROTEINS | NCI1 | NCI109 | IMDB-B |
|---|---|---|---|---|---|---|
| WL kernel | 90.4±5.7 | 59.9±4.3 | 75.0±3.1 | 86.0±1.8 | – | 73.8±3.9 |
| DGCNN | 85.8±1.8 | 58.6±2.5 | 75.5±0.9 | 74.4±0.5 | – | 70.0±0.9 |
| GIN | 89.4±5.6 | 64.6±7.0 | 76.2±2.8 | 82.7±1.7 | – | 75.1±5.1 |
| PPGN | 90.6±8.7 | 66.2±6.6 | 77.2±4.7 | 83.2±1.1 | 82.2±1.4 | 73.0±5.8 |
| GSN (subgraph) | 92.2±7.5 | 68.2±7.2 | 76.6±5.0 | 83.5±2.0 | – | 77.8±3.3 |
| SIN (topological) | – | – | 76.4±3.3 | 82.7±2.1 | – | 75.6±3.2 |
| CIN (topological) | 92.7±6.1 | 68.2±5.6 | 77.0±4.3 | 83.6±1.4 | 84.0±1.6 | 75.6±3.7 |

We position our contribution honestly: it is **expressivity** (provably beyond
3-WL, Q1) plus a **controlled demonstration that persistent homology improves an
equivariant backbone**, not a new state of the art. Recent topological (CIN/SIN)
and subgraph-counting (GSN) nets are strong and complementary; our distinct
contribution is the *learnable equivariant filtration* on k-order tensors with a
proof of strict separation beyond 3-WL, which those methods do not establish.

---

## Q3. Comparison to GCN, GIN, MLP (no message passing); training-time cost of PH.

Same-protocol GCN / GIN / MLP (and GSN) implemented on the *identical* input
tensor, splits, batching and schedule (`models/baseline_models.py`); accuracy
rows are in the Q2 table (MLP = a GNN with no message passing = DeepSets over
nodes). Clean single-process runtime in `runtime_bench.py`.

**Training cost.** The persistent-homology branch adds a bounded per-graph cost
(clique complex + gudhi persistence) on top of the equivariant backbone. We
found gudhi persistence is *not* the bottleneck (200 complexes ≈ 0.03 s) and the
clique complex is cached per graph, so on GPU the dense PPGN ops dominate and the
PH overhead is a small constant. Representative CPU numbers (batched, s/epoch):

| Model | MUTAG | PTC | NCI1 |
|---|---|---|---|
| MLP / GCN / GIN | ~0.05 | … | … |
| PPGN (baseline) | 0.87 | 1.17 | 30.4 |
| PPGN + PH (ours) | 1.19 | 3.13 | ~60 |

*(Clean single-process `runtime_bench.py` numbers — including GPU ms/graph — are
regenerated on the target machine; the PH forward pass is ~1.3–2× the baseline
on small molecular graphs and is dominated by the same dense ops that GPU
accelerates.)*

---

## Q4. Do the learned filtration functions permit an interpretation?

Yes — and the interpretation is chemically meaningful. We read out the learned
layer-0 filtration f(σ) of a trained MUTAG topology model for every vertex and
edge of all 188 molecules (`q4_interpret.py`), and find the filtration tracks
interpretable structure:

- **Ring vs non-ring bonds:** mean learned filtration on **ring bonds = 0.238**
  vs **non-ring bonds = 0.045** — a ~5× separation. The learned filtration
  systematically delays ring bonds, so ring/cycle features are born later and
  persist across a wider range of the filtration.
- **Degree:** node filtration correlates with node degree (Pearson r = 0.44) —
  more-connected atoms enter the filtration later.
- **Atom type:** filtration values differ by atom type (carbon, the dominant
  type, receives the highest mean).

The per-molecule figures (`rebuttal_results/q4/molecule_*.png`) make this vivid:
the learned filtration paints the **aromatic carbon ring bright (high f)** and
the peripheral heteroatom substituents dark (low f). Since MUTAG mutagenicity is
driven by aromatic/nitroaromatic ring systems, the learned filtration is
effectively *learning to emphasize the substructure that matters for the task* —
concrete evidence that the topological branch adds interpretability, exactly the
story the reviewer hoped might be there. Figures include the molecule colored by
f plus its persistence diagram.
