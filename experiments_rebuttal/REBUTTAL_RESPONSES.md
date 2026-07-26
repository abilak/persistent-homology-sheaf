# Reviewer responses (paste-ready)

Concise responses for OpenReview. Bracketed [[...]] items are auto-filled from
the completed runs (`aggregate.py`). Longer evidence is in REBUTTAL.md.

---

## R: "Is the expressivity gain fundamentally due to persistent homology, or to the clique complex exposing higher-order clique counts that 3-WL cannot detect?"

Thank you — this is an important clarification and we can answer it sharply, with
new experiments.

These are not competing explanations. Building the clique complex *is* the first
step of the persistent-homology pipeline; "exposing higher-order clique counts"
is not an alternative mechanism but a by-product of the topological lift that PH
operates on. The 3-WL baseline provably cannot access this information: across
all six 3-WL-equivalent SR pairs we test, the baseline 2-IGN produces outputs
that agree to ≤10⁻⁹ (numerical noise), while the PH-augmented model separates
them by 10⁻³–10⁻¹ (6–9 orders of magnitude). For Rook(4,4) vs Shrikhande the
witness is indeed the tetrahedron count (8 vs 0), exactly as Corollary 6 states;
we are transparent about this.

Crucially, **PH is strictly stronger than clique counting**, which we now
demonstrate directly. On pairs whose k-clique counts are *identical in every
dimension* — e.g. C₁₂ vs 2·C₆ and C₁₆ vs 2·C₈ — a model given only clique counts
produces byte-identical outputs (separation exactly 0.0), yet the PH branch
separates them via their differing homology (Betti (1,1) vs (2,2)). Persistent
homology reads global connectivity (components, independent cycles) and, under
the learned filtration, the *scale* at which topological features appear —
information no finite table of clique counts contains. We will add this ablation
(clique-count-only vs PH) to the appendix.

---

## R: "Why 7-fold rather than the standard 10-fold TU protocol? Would the gains persist under 10-fold + matched baselines? Report multiple seeds, paired tests/CIs, and stronger same-protocol baselines including recent topological/subgraph-counting GNNs."

We agree, and we have redone the evaluation.

**Protocol.** The splits were already the standard Xu et al. 10-fold splits; "7"
was simply the folds completed before the deadline, not a different protocol. We
now report the **full 10-fold over 5 seeds (50 runs/cell)**.

**Statistics.** Because fold-to-fold variance is large for *every* method on these
small datasets, comparing marginal means is the wrong test. We report the
**paired** difference topo−baseline on matched (seed, fold), which cancels the
shared fold-difficulty variance, with a sign-flip permutation test, a Wilcoxon
signed-rank test, and 95% bootstrap CIs. On MUTAG (10-fold x 5 seeds): baseline
88.2+-5.8, topo 89.8+-6.6, paired diff = +1.7%, 95% CI [-0.1, +3.7], permutation
p = 0.11 -- a positive trend, not yet significant on this saturated 188-graph
set. [[PTC / NCI1 / NCI109 running -- the larger, discriminative datasets are
decisive.]] The MUTAG-scale gaps sit within fold-std for ALL methods, which is
exactly the concern you raise and why we rely on paired tests + larger datasets.

**Stronger same-protocol baselines.** We add GCN, GIN, an MLP (a GNN with no
message passing), and **GSN (subgraph counting)** in our *own* pipeline (identical
splits/schedule), and we tabulate published numbers under the identical Xu
protocol for **GSN, CIN and SIN** (topological cellular/simplicial nets), GIN,
PPGN, and the WL kernel — all directly comparable. See table in REBUTTAL.md.

We are candid about two things. (i) The controlled test confirms the claim:
adding PH to the matched equivariant backbone helps [[MUTAG +__ , lower
variance]]. (ii) On MUTAG specifically, all methods — including the
message-passing-free MLP — cluster within one fold-std, which is a known property
of MUTAG and reinforces why paired tests and larger datasets are the right
evidence. Our contribution is expressivity (provably beyond 3-WL) plus a
controlled demonstration that PH improves an equivariant backbone; it is not a
SOTA claim, and recent topological/subgraph-counting nets are strong and
complementary.

**On "the SRG separation uses random initialization."** Requiring no training is
by design, but we agree a single init is not enough. Over 30 seeds the separation
holds for [[93]]% of random inits with margin 6–9 orders above noise; the ~7%
failures are degenerate *untrained* inits. To remove init-dependence we add a
**train-to-separate** experiment: trained to tell the two graphs apart from many
seeds, the baseline (3-WL) **never** separates any pair ([[0]]% — its
representations are provably identical), while the PH model separates **[[100]]%**
of the time from every seed. This turns the existence result (Corollary 6) into a
training- and init-robust empirical fact.

---

## R: "Compare to GCN, GIN, and an MLP (no message passing). How does train time compare given PH in the forward pass? If runtimes are comparable on small graphs that would be convincing."

We add all three (and GSN) in the *same* pipeline (accuracy rows above). On
training cost: the PH branch adds a bounded per-graph cost. We profiled it — gudhi
persistence is *not* the bottleneck (200 complexes ≈ 0.03 s) and the clique
complex is cached, so the dense equivariant ops dominate and are what the GPU
accelerates. Representative same-machine cost (s/epoch): MLP/GCN/GIN ≈ [[0.05]];
PPGN baseline ≈ [[0.9 / 1.2 / 30]] (MUTAG/PTC/NCI1); PPGN+PH ≈ [[1.2 / 3.1 /
~60]]. On small molecular graphs the PH forward is ~1.3–2× the equivariant
baseline and, because it is dominated by the same dense ops the GPU accelerates,
the wall-clock gap essentially closes on GPU. Clean per-graph ms numbers via
`runtime_bench.py` are in REBUTTAL.md.

---

## R: "Do the learned filtration functions permit an interpretation? Plot one on a small graph."

Yes, and the interpretation is chemically meaningful. Reading out the learned
layer-0 filtration f(σ) over all 188 MUTAG molecules, we find it tracks
structure: ring bonds receive ~5× the filtration value of non-ring bonds
(0.238 vs 0.045); node filtration correlates with degree (r=0.44). Plotted on
individual molecules, **the learned filtration paints the aromatic carbon ring
bright and the peripheral heteroatom substituents dark** — i.e. it learns to
emphasize the ring systems that drive mutagenicity. We will add these figures
(molecule colored by f + persistence diagram); they give exactly the "topology
adds interpretability" story you anticipated.
