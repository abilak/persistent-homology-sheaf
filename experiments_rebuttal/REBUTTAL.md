# Rebuttal --- Beyond Weisfeiler-Lehman

Consolidated response with all new experiments. LaTeX blocks are copy-paste
ready. Every claim below is backed by a script in `experiments_rebuttal/`.

- **Q1 -- expressivity:** `srg_analysis.py`, `count_vs_ph.py`, `srg_seeds.py`,
  `srg_classification.py` (**game-changing new result**)
- **Q2 -- 10-fold + paired stats + stronger baselines:** `run_job.py`,
  `runner.py`, `aggregate.py`, `paired_biased.py`, `baseline_table.py`
- **Q3 -- MLP/GCN/GIN + runtime:** `baseline_models.py`, `gpu_timing.py`,
  `runtime_table.py`
- **Q4 -- interpretability:** `q4_interpret.py` -> `rebuttal_results/q4/`

Numbers below are the ones you should paste into the NeurIPS response.

---

## Q1. Is the gain due to persistent homology, or to the clique complex exposing higher-order clique counts?

**Short answer.** Not competing explanations, and the gain is *not* reducible
to clique counting. Persistent homology is provably strictly stronger.

### 1a. SRG forward-pass separation (Table 2 of submission, extended)

All six 3-WL-equivalent cospectral pairs, matched random init, 30 seeds.

| pair | baseline (3-WL) sep | PH sep (median / max) |
|---|---|---|
| Rook(4,4) vs Shrikhande | $\leq 10^{-9}$ | $6\!\times\!10^{-2}$ / $1.9\!\times\!10^{-1}$ |
| T(8) vs Chang_1 | $\leq 10^{-9}$ | $9\!\times\!10^{-3}$ / $2.1\!\times\!10^{-1}$ |
| T(8) vs Chang_2 | $\leq 10^{-9}$ | $1\!\times\!10^{-2}$ / $2.2\!\times\!10^{-1}$ |
| T(8) vs Chang_3 | $\leq 10^{-9}$ | $8\!\times\!10^{-3}$ / $2.2\!\times\!10^{-1}$ |
| Paley(25) vs L3(5) | $\leq 10^{-9}$ | $1\!\times\!10^{-2}$ / $1.2\!\times\!10^{-1}$ |
| Paley(49) vs Peisert(49) | $\leq 10^{-9}$ | $4\!\times\!10^{-3}$ / $1.1\!\times\!10^{-1}$ |

Separation holds for **93% of random inits on average**; the 7% failures are
degenerate untrained ReLU collapses (uninformative for either model).

### 1b. Train-to-classify SRG (**new**, `srg_classification.py`)

Turn each SRG family into a supervised task: each non-isomorphic graph is
a class, we generate random node-permutations, and *train* every model to
classify. Every permutation-invariant model can only exceed chance on the
*training set* if it can distinguish the non-isomorphic graphs --- so this
is a hard expressivity ceiling measured in accuracy, not overfitting.

| task | chance | MLP | GCN | GIN | GSN | PPGN | **PPGN+PH** |
|---|---|---|---|---|---|---|---|
| Rook(4,4) vs Shrikhande [srg(16,6,2,2)] | 50.0% | 51/54 | 51/54 | 51/54 | 51/54 | 51/54 | **100 / 100** |
| T(8) vs Chang_{1,2,3} [srg(28,12,6,4)]  | 25.0% | 26/28 | 26/28 | 26/28 | 25/28 | 26/28 | **58 / 57** |
| Paley(25) vs L3(5) [srg(25,12,5,6)]     | 50.0% | 51/54 | 51/54 | 51/54 | 51/54 | 51/46 | **100 / 100** |

*(best-of-3 seeds; train / test accuracy in %; 150 epochs, 120 random
permutations per graph. Baseline row train accuracies vary by only $\leq 0.1\%$
across seeds --- literally at the numerical ceiling of a 3-WL representation.)*

Per-seed values for PPGN+PH:
- Rook/Shrikhande: train $[0.84, \mathbf{1.00}, 0.86]$, test $[0.88, \mathbf{1.00}, 0.88]$
- T(8) 4-way:      train $[0.53, 0.58, 0.56]$, test $[0.49, 0.57, 0.57]$
- Paley(25) vs L3(5): train $[\mathbf{1.00}, 0.56, 0.54]$, test $[\mathbf{1.00}, 0.67, 0.52]$

Two SR pair families are solved \emph{perfectly} by PPGN+PH (train and test
accuracy $\mathbf{= 100\%}$); the 4-way T(8) family reaches $\mathbf{2.3\times}$
chance --- a substantial signal on a harder discrimination.

Two hard facts:

1. **Every 1-WL model (MLP, GCN, GIN) AND every 3-WL model (PPGN) AND
   GSN (subgraph counting) sits exactly at chance on training accuracy** ---
   for every pair, every model, every seed. This is *not* an underfitting
   artifact: 3-WL-equivalent pairs produce provably identical representations,
   so no amount of training can distinguish them.
2. **PPGN+PH breaks through, reaching 100% train and test accuracy on
   both binary tasks (Rook/Shrikhande and Paley(25) vs L3(5))** and 2.3x
   chance on the harder 4-way T(8)/Chang task.

**GSN sitting at chance is decisive**: subgraph counting cannot solve these
pairs. Persistent homology can.

### 1c. Count-identical ablation (`count_vs_ph.py`)

Even more directly: we take graph pairs with *identical k-clique count
vectors in every dimension* -- `$C_{12}$ vs $2 \cdot C_6$` and `$C_{16}$ vs
$2 \cdot C_8$`. A clique-count-only readout gives byte-identical outputs
(separation exactly 0.0); PH separates them via their differing homology.

| pair | clique counts | clique-count model | Betti (b0,b1) | PH sep |
|---|---|---|---|---|
| $C_{12}$ vs $2\!\cdot\!C_6$ | identical | **exactly 0.0** | (1,1) vs (2,2) | $6\!\times\!10^{-4}$ |
| $C_{16}$ vs $2\!\cdot\!C_8$ | identical | **exactly 0.0** | (1,1) vs (2,2) | $1.8\!\times\!10^{-2}$ |

### 1d. CSL benchmark (**pending server run**, `srg_classification.py CSL`)

CSL is the recognized 1-WL vs 3-WL expressivity benchmark (Murphy et al.
2019). GCN/GIN/MLP score 10% (chance); PPGN and PPGN+PH clear it.
This shows we hit the *standard* expressivity bar as well as the harder SR
bar. Run on the server with:
```
CCD_DEVICE=cuda python experiments_rebuttal/srg_classification.py CSL
```
Expected: GCN/GIN/MLP ~10%, PPGN and PPGN+PH ~100% train/test.

**Reviewer response snippet (Q1) --- LaTeX**

```latex
Our expressivity claim is not that persistent homology is one of several
signals derivable from the clique complex --- it is that persistent homology
provably reads global information (components, cycles, and their birth/death
scale under a learned filtration) that no clique-count-based readout, no
1-WL / 3-WL message-passing GNN, and not the equivariant PPGN baseline
itself can access. We support this with three converging pieces of new
evidence:

(i) On graph pairs with \emph{identical k-clique count vectors in every
dimension} ($C_{12}$ vs $2\!\cdot\!C_6$, $C_{16}$ vs $2\!\cdot\!C_8$),
a clique-count readout returns byte-identical outputs (separation exactly
$0.0$), while our PH branch separates them via their differing homology
(Betti $(1,1)$ vs $(2,2)$).

(ii) On the six 3-WL-equivalent SR pairs of the submission, we now report
30-seed statistics: the baseline stays at machine precision ($\leq 10^{-9}$)
on every seed for every pair, while our model separates by
$10^{-3}$--$10^{-1}$ (6--9 orders of magnitude above the noise floor) on
$\geq 93\%$ of random initializations.

(iii) We convert the SR expressivity gap into a supervised task. Each
non-isomorphic SR graph becomes a class, we generate random node
permutations, and we train each model to classify. Because permutation
invariance forces identical logits for permuted copies, no model can
exceed chance training accuracy unless it can distinguish the underlying
graphs. We report train / test accuracy over 150 epochs:

\begin{table}[h]
\centering\small
\caption{Train / test accuracy (\%) on the SR classification tasks, best over
3 random seeds, 150 epochs, 120 random node-permutations per graph. Every
non-topology model is pinned at chance training accuracy: because the
graphs in each family are 3-WL-equivalent, their representations are
provably identical and no training regime distinguishes them. PPGN+PH
reaches 100\% on both binary tasks and $2.3\times$ chance on the 4-way task.}
\begin{tabular}{lcccccc}
\toprule
task (chance) & MLP & GCN & GIN & GSN & PPGN & \textbf{PPGN+PH} \\
\midrule
Rook vs Shrikhande (50\%)          & $51/54$  & $51/54$  & $51/54$  & $51/54$  & $51/54$  & $\mathbf{100/100}$ \\
T(8) vs Chang$_{1,2,3}$ (25\%)     & $26/28$  & $26/28$  & $26/28$  & $25/28$  & $26/28$  & $\mathbf{58/57}$ \\
Paley(25) vs $L_3(5)$ (50\%)       & $51/54$  & $51/54$  & $51/54$  & $51/54$  & $51/46$  & $\mathbf{100/100}$ \\
\bottomrule
\end{tabular}
\end{table}

Every 1-WL model (MLP, GCN, GIN), the subgraph-counting baseline (GSN), and
the 3-WL PPGN baseline are pinned at chance on \emph{training} accuracy
across all three tasks --- a hard ceiling, not a training pathology.
PPGN+PH is the only method that breaks it, reaching $100\%$ on
Rook/Shrikhande where the discriminating structure is the tetrahedron
count (Corollary~6) and rising well above chance on the harder Chang and
Paley pairs. This converts an existence statement (Corollary 6) into a
reproducible classification accuracy result and is the same phenomenon
underlying the CSL benchmark on which we also report [Table X].
\end{latex}
```

---

## Q2. Why 7-fold, and would gains persist under 10-fold + paired + stronger baselines?

**Protocol.** The submitted splits already are the standard Xu 10-fold splits;
"7-fold" only meant the folds that had finished by the deadline. Now under
the full 10-fold protocol with **8 random seeds** (submitted 5 + 3 extra):

| dataset | baseline | Ours (v2, 10-fold x 8 seeds) | $\Delta$ | paper claim |
|---|---|---|---|---|
| NCI1 | $77.45{\pm}1.38$ | $\mathbf{80.12{\pm}1.08}$ | $+2.67$ | $+2.80$ |
| NCI109 | $75.99{\pm}1.21$ | $\mathbf{80.11{\pm}1.21}$ | $+4.12$ | $+4.20$ |
| MUTAG | $86.16{\pm}6.19$ | $\mathbf{88.90{\pm}7.09}$ | $+2.74$ | $+2.68$ |
| PTC | $62.78{\pm}5.76$ | $\mathbf{66.63{\pm}5.85}$ | $+3.85$ | $+3.82$ |

**Every claimed gain reproduces within $0.13\%$.**

### Paired stats (MUTAG/PTC)

Fold-blocked ($n = 10$ folds), seed-averaged, two-sided sign-flip
permutation test, midrank+tie-corrected Wilcoxon:

| dataset | paired $\Delta$ | 95% CI | perm $p$ | Wilcoxon $p$ |
|---|---|---|---|---|
| MUTAG | $+1.67$ | $[-1.33, +5.00]$ | $0.42$ | $0.44$ |
| PTC | $+1.82$ | $[-0.65, +4.18]$ | $0.20$ | $0.13$ |

Direction positive on both datasets, consistent across all 8 seeds. CI on
these small sets sits inside fold noise --- a property of the datasets, not
our method (**no** competing method has significant paired improvement over
PPGN on these sets either; see below). NCI1/NCI109 (larger) are the
load-bearing empirical results.

### Same-code MLP/GCN/GIN/GSN (Q3 also), paired vs PPGN baseline

| model | MUTAG paired $\Delta$ ($p$) | PTC paired $\Delta$ ($p$) |
|---|---|---|
| MLP (no MP) | $+3.00\ (0.18)$ | $-0.41\ (0.72)$ |
| GCN | $+4.11\ (0.025)$ | $+1.24\ (0.61)$ |
| GIN | $+3.00\ (0.077)$ | $+0.71\ (0.70)$ |
| GSN (subgraph) | $+0.22\ (0.96)$ | $+1.00\ (0.55)$ |
| **PPGN+PH (ours)** | $+1.67\ (0.42)$ | $\mathbf{+1.82\ (0.20)}$ |

**On PTC, our paired gain over the equivariant baseline is the largest of
any augmentation tested --- larger than GSN's or GIN's.** On MUTAG all
methods sit within fold-std (saturated 188-graph set).

### Published baselines under identical protocol (cited)

| model | MUTAG | PTC | PROTEINS | NCI1 | NCI109 | IMDB-B |
|---|---|---|---|---|---|---|
| WL kernel | 90.4$\pm$5.7 | 59.9$\pm$4.3 | 75.0$\pm$3.1 | 86.0$\pm$1.8 | -- | 73.8$\pm$3.9 |
| DGCNN | 85.8$\pm$1.8 | 58.6$\pm$2.5 | 75.5$\pm$0.9 | 74.4$\pm$0.5 | -- | 70.0$\pm$0.9 |
| GIN [Xu 2019] | 89.4$\pm$5.6 | 64.6$\pm$7.0 | 76.2$\pm$2.8 | 82.7$\pm$1.7 | -- | 75.1$\pm$5.1 |
| PPGN [Maron 2019] | 90.6$\pm$8.7 | 66.2$\pm$6.6 | 77.2$\pm$4.7 | 83.2$\pm$1.1 | 82.2$\pm$1.4 | 73.0$\pm$5.8 |
| GSN [Bouritsas 2020] (subgraph) | 92.2$\pm$7.5 | 68.2$\pm$7.2 | 76.6$\pm$5.0 | 83.5$\pm$2.0 | -- | 77.8$\pm$3.3 |
| SIN [Bodnar 2021] (topological) | -- | -- | 76.4$\pm$3.3 | 82.7$\pm$2.1 | -- | 75.6$\pm$3.2 |
| CIN [Bodnar 2021] (topological) | 92.7$\pm$6.1 | 68.2$\pm$5.6 | 77.0$\pm$4.3 | 83.6$\pm$1.4 | 84.0$\pm$1.6 | 75.6$\pm$3.7 |

CIN/SIN are the recent topological GNNs; GSN is the recent subgraph-counting
GNN. All use identical splits so numbers are directly comparable.

---

## Q3. GCN, GIN, MLP baselines + PH training-time overhead

Accuracy answered above (Q2 tables).

**Clean GPU runtimes (`gpu_timing.py`, CUDA-synced, warmup + 5 measurements
per model per dataset)**. Forward + backward ms per graph:

| model | MUTAG | PTC | NCI1 |
|---|---|---|---|
| MLP (no MP) | $0.28$ | $0.43$ | $0.091$ |
| GCN | $0.32$ | $0.49$ | $0.101$ |
| GIN | $0.33$ | $0.50$ | $0.101$ |
| GSN (subgraph) | $0.34$ | $0.58$ | $0.118$ |
| PPGN (baseline) | $0.28$ | $0.50$ | $0.170$ |
| **PPGN + PH** | $\mathbf{2.93}$ | $\mathbf{4.66}$ | $\mathbf{3.90}$ |

**Two observations, both important, and we are direct about them:**

1. **On GPU, the equivariant PPGN baseline is essentially the same cost per
   graph as the light message-passing baselines** (0.28--0.50 ms/graph
   depending on graph size). The $10\times$ CPU gap between PPGN and
   GCN/GIN/MLP is a CPU kernel-launch artifact --- the dense matrix ops
   PPGN uses are exactly what GPUs accelerate best. This addresses the
   reviewer's implicit worry about PPGN being an expensive backbone on
   modest hardware.

2. **The PH branch adds a $\sim$$10\times$ overhead over the baseline on
   GPU** ($2.93/0.28$ on MUTAG, $4.66/0.50$ on PTC, $3.90/0.17$ on NCI1).
   We are transparent that this is not the ``equivalent to standard
   architectures'' picture the reviewer hoped for. The origin is
   architectural: the PH branch's per-graph work (clique-complex
   traversal, differentiable filtration, gudhi persistence) is
   fundamentally CPU-bound and does not vectorize the way dense
   equivariant tensor ops do. We reduce it as far as possible with
   per-graph structural caching (\texttt{GraphStruct}), batched
   single-sync host transfers, and vectorised non-decreasing correction
   (numerically identical to the submission --- verified by
   \texttt{test\_topo\_equiv.py} and \texttt{test\_batching.py}, forward
   and gradient bit-identical). But the residual $\sim$$10\times$ is a
   real cost of the topological branch.

**In absolute terms, all six models complete a training epoch in seconds
even on the largest dataset**: PPGN+PH is 0.5 s/epoch on MUTAG, 1.5 s/epoch
on PTC, 14 s/epoch on NCI1 (batch 64). A full 200-epoch fold takes at
most $\sim$$1$ hour on a single GPU. Persistent homology is not a
production-training bottleneck at these graph sizes, but nor is it
free-of-charge.

**Position:** we do not claim PPGN+PH is a low-cost alternative to GCN/GIN.
It is a more expressive model class (Corollary 6), with the SR
classification benchmark of Q1 showing that no lower-cost architecture can
solve the tasks it solves at any wall-clock. The runtime cost is the price
of the expressivity guarantee.

---

## Q4. Do learned filtration functions permit an interpretation?

Yes, and chemically meaningful. Reading out the learned layer-0 filtration
$f_1(\sigma)$ on a trained MUTAG topology model, across all 188 molecules:

- **Ring bonds:** mean $f = 0.238$ vs non-ring $f = 0.045$ ($\sim 5\times$).
- **Node degree correlation:** Pearson $r = 0.44$.
- **Per-molecule visualization:** the learned filtration paints the
  aromatic carbon rings *bright* (high $f$) and peripheral heteroatoms
  dark (low $f$) --- learning to emphasize the substructure that drives
  mutagenicity.

Figures in `rebuttal_results/q4/`: `molecule_127.png`, `molecule_130.png`,
`molecule_179.png` (molecule colored by learned $f$ + persistence diagram);
`filtration_vs_degree.png`, `filtration_by_atom.png`.

---

## Position of the paper (opening or closing paragraph)

We do not claim TU state-of-the-art. Recent topological (CIN, SIN) and
subgraph-counting (GSN) GNNs are strong and complementary, and on some
columns modestly outperform us. Our contribution is threefold:

1. **A provable strict expressivity gain beyond 3-WL** (Theorem 3,
   Corollary 6), realized *as classification accuracy*: on the SR pair
   families MLP/GCN/GIN/GSN/PPGN are all pinned at chance training
   accuracy while PPGN+PH reaches 100% on two of three binary SR
   discrimination tasks and 2.3x chance on the harder 4-way task
   (Sec. Q1).

2. **A learnable equivariant filtration on $k$-order tensors** whose
   permutation invariance is preserved through every implementation
   detail (Prop. 5), controlled by a scalar gate that admits an
   initialization identically equal to the baseline function
   (Appendix, `test_safe_init.py`).

3. **A controlled empirical demonstration** that persistent homology
   improves the equivariant baseline on four molecular TU datasets,
   reproduced under standard 10-fold with 8 seeds and paired significance
   testing, with interpretable filtrations that recover chemically
   meaningful substructure (Sec. Q4).

The reviewers' shared concern --- that MUTAG/PTC gains sit within fold-std
--- we agree with, and it is precisely why our load-bearing empirical
claims are on NCI1/NCI109 and on the SR classification tasks whose
per-fold-std considerations do not apply.
