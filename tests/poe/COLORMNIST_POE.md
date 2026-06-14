# ColorMNIST-PoE — Compositional Generalization with Cobweb

Cobweb version of **Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional Generalization in
Diffusion Models via Concept Discovery" (2026)**. The paper turns a pretrained DDPM into a
hierarchy of Gaussian "concept" modes, picks a few that cover an out-of-distribution query, and
multiplies them into a **Product-of-Experts (PoE)**. ColorMNIST in the paper is a 32×32
**pixel-space** DDPM (no autoencoder), so we run the same pipeline on **raw pixels** with Cobweb
standing in for the diffusion model.

Implementation: `tests/poe/colormnist_poe.py`.

## The method in three steps

**1. Concepts = Cobweb nodes.** Each node is a diagonal Gaussian `N(m_n, σ²_n)` over pixels
(`m_n = node.mean`, `σ²_{n,r} = sum_sq_r/count + prior_var`). Shallow nodes are broad concepts
(a background colour), deep nodes are specific ones (a digit shape) — the paper's coarse-to-fine
mode hierarchy, for free.

**2. Select concepts by a best-first candidate pool** (`select_concepts`). This is the paper's
greedy argmax (Eq. 9) over a candidate pool — gathered from the Cobweb tree instead of by
mode-ascent. The pool's score is the paper's *coverage gain*:

```
Δ(n) = Σ_r max( ℓ_{n,r}(x_q) − cur_r , 0 ),   cur_r = best per-pixel log-lik so far
```

For each concept we **best-first expand** the tree by `Δ` (priority queue: pop the max-`Δ` node,
push its children) up to a pool of **~3% of the tree's nodes**, then take the global `Δ`-maximizer
in that pool. We fold the pick into `cur_r` and **re-run the pool for the next concept** — now `Δ`
rewards only the pixels still unexplained, so the search reaches a *different* branch (the digit
after the background). This re-routing across branches is the Cobweb analog of the paper's modes
recovered from different noise levels.

> Two cheaper variants we tried and rejected: descending by Cobweb's own posterior routing
> (`log_prob_class_given_instance`) sends every concept to the same nearest cluster — one branch,
> blurry, joint ≈ 22%; a single greedy coverage-gain *descent* (one root→leaf path per concept)
> recovers cross-branch diversity (joint ≈ 42%) but misses better off-path concepts. The
> best-first **pool** finds them — the better concepts are spread across branches, so the
> per-concept argmax has to look at a real candidate set. Pool size trades quality for cost
> (joint ≈ 42% at 1%, ≈ 43% at 3%, ≈ 46% at 10% of the tree); we use **3%** as the balance.

**How many concepts — the leftover-image cutoff.** After each pick we compose `μ_T` and measure
how much of the query it explains, `R² = 1 − ‖x_q − μ_T‖² / ‖x_q − x̄_q‖²`, stopping once
`R² ≥ 0.99` (capped at `K_MAX = 6`). Easy queries reach 99% with a couple of concepts; hard ones
recruit more. This keys the cutoff off *how much image is left to explain* — earlier cutoffs based
on the coverage gain itself (a Δ-peak, a coverage knee) stopped too early because the gain is
dominated by the background.

**3. Compose with a per-pixel product of Gaussians** (paper Eqs. 7 & 10), temperature **τ = 0.1**:

```
w_n(r) = softmax_{n∈S}( ℓ_{n,r}(x_q) / τ )
μ_T[r] = ( Σ_n w_n(r)·m_{n,r}/σ²_{n,r} ) / ( Σ_n w_n(r)/σ²_{n,r} )
```

Low τ means each pixel comes mostly from its single best-fitting concept (background pixels from
the background concept, the stroke from the digit concept). We have no diffusion sampler, so the
generated image is this PoE mean `μ_T`.

## Baselines & metrics (paper §4.2)

We evaluate the leftover-image cutoff at two thresholds — **PoE 99% / 90%** — plus a fixed
**PoE K=3** ablation (the paper's choice), against **Top-1 / Top-3** nearest seen-class retrieval
baselines. Per OOD class, generations are scored against two reference sets —
**Faithfulness** (the query images) and **Generalization** (other held-out images of the class) —
with **FID**, **CLIP** cosine, k-NN (k=3) **Precision/Recall/F1**, plus an attribute-classifier
**joint accuracy**. Reported as mean ± SE over the 40 OOD classes.

## Outputs (`tests/poe/colormnist_output/`)

- `summary.csv`, `metrics.png` — all metrics per method (Faithfulness vs Generalization).
- `methods.png` — each method's reconstruction of held-out queries.
- `concepts/concepts_{99,90,k3}.png` — per variant: query · μ_T · the selected concepts in
  recovery order, labelled by tree depth and coverage-gain share ΣΔ.
- `heatmaps/heatmaps_{99,90,k3}.png` — per variant: each concept's per-pixel donation `w_n(r)`.
- `concept_depths.png` — selected-concept depth histogram (99% cutoff).
- `concept_types.png` — mean # of selected concepts that are internal (have children) vs leaves, per variant.
- `nodes_explored.png` — bar chart of mean nodes visited per query, by variant.
- `hierarchy.png`, `subtrees/` — the Cobweb concept hierarchy as mean images.

The run prints per-query stats (at the 99% cutoff): average **K**, concept **depth**, and **nodes visited**.

## Primitive discovery — pixels the PoE donates together

A follow-on analysis that asks: across all compositions, which **groups of pixels behave as reusable
parts**? We run the **PoE 90%** selection over all 640 OOD queries and record, for every selected
concept, its per-pixel donation heatmap `w_n(r)` (channel-averaged to 32×32). This gives ~1,700
donation maps — one per composed concept.

**Pixel cross-correlation.** Treat each *pixel* as a variable and its donation value across all
those maps as its samples, and compute the pixel×pixel **Pearson cross-correlation** `R`. Two
pixels correlate highly when the composition tends to hand them to the *same* concept. We then
**group the pixels with Cobweb** (fit a `CobwebContinuousTree` on the rows of `R`; clusters = nodes
at a given tree depth) — an unsupervised number of regions, no `k` to set.

**Why these correlations are effectively 0/1.** The donation value is the per-coordinate composition
weight `w_n(r) = softmax_{n∈S}( ℓ_{n,r}(x_q) / τ )` (Eq. 10) — the share of pixel `r` that concept `n`
contributes to `μ_T`. At the working temperature **τ = 0.1** this softmax is *near one-hot*: each pixel
is handed almost entirely to a **single** concept, so every donation map `w_n` is ≈ 0/1 (a pixel is
"owned" by the concept or not). Correlating these near-binary maps therefore yields a cross-correlation
`R` that is itself effectively binary — two pixels read as correlated when they are **co-owned by the
same concept** across compositions, and uncorrelated otherwise. The Cobweb regions are thus a hard
partition of the canvas into co-donated pixel groups; pushing `τ → 0` sharpens ownership toward exact
0/1, while a larger `τ` blends concepts per pixel and softens the correlations. (We deliberately keep
this binary, donation-based signal as the primitive vocabulary rather than the raw pixel-intensity
correlation, which mixes the dataset's colour/shape covariance into the regions.)

What emerges is the method's spatial vocabulary: a **background-field** primitive and a **digit**
primitive at the coarsest split, with the digit refining into concentric **stroke shells**
(interior → stroke band → outline) as the tree deepens — purely from co-donation statistics, never
told where the digit is.

Outputs (`tests/poe/colormnist_output/primitives/`):
- `pixel_correlation_regions.png` — pixels colored by Cobweb region at depths 1/2/3 (2 → 4 → 8 regions).
- `pixel_correlation_matrix.png` — the cross-correlation `R`, pixels reordered by region (shows the
  background/digit blocks).
- `correlation_hierarchy.png` — the Cobweb hierarchy of `R`, top 5 levels, each node drawn as the
  **spatial region of pixels routed to it** (bright = in-region; label = region size in pixels).
- `subtrees/subtree_*.png` — each leaf region of that hierarchy expanded 3 levels deeper.
