# ColorMNIST-PoE — Compositional Generalization with Cobweb

Cobweb adaptation of **Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional
Generalization in Diffusion Models via Concept Discovery" (2026)**.  The paper
repurposes a pretrained diffusion model as a hierarchy of density modes: for an
out-of-distribution query it *discovers* reusable concept prototypes, *selects*
relevant ones with a submodular coverage objective, and *composes* their local
Gaussians into a **Product-of-Experts (PoE)**.

Cobweb already **is** such a hierarchy of Gaussian density modes — every node `n`
is a diagonal-Gaussian expert `q_n(x)=N(m_n, Σ_n)` with `m_n = node.mean`,
`σ²_{n,r} = sum_sq_r/count + prior_var` (verified to match `node.log_prob`).  So we
run the paper's discover → select → compose pipeline **directly on the Cobweb tree,
in raw pixel space, with no diffusion model and no autoencoder**.

Implementation: `tests/moc/colormnist_poe.py`.

## Benchmark (paper §4.1)
32×32 RGB ColorMNIST with three primitive factors — digit (10) × foreground colour
(4) × background colour (4) = **160 slots**, of which **120 are SEEN** by Cobweb and
**40 held out as OOD**.  The split is a genuine *compositional* OOD split (asserted in
code): every held-out slot is a novel combination whose digit, fg-colour and bg-colour
each appear in training, just never together; no slot, combination, or grayscale
exemplar leaks between train and OOD.

## The method — two steps

The per-pixel expert log-likelihood is `ℓ_{n,r}(x) = log N(x_r; m_{n,r}, σ²_{n,r})`.

### (1) Select the concepts to compose — top-down, heuristic, **no fixed depth**
- **Discover** candidates by a **best-first DESCENT** of the tree (the analog of the
  paper's discovery of density modes at multiple abstraction scales).  Priority is the
  node posterior `φ(n) = Σ_r ℓ_{n,r}(x) + log P(n)`; a node's children are expanded
  **only while a child raises `φ`**, so each branch stops at its own `φ`-peak — the
  natural granularity, decided per-branch by the heuristic rather than a depth cap.
  (`search_candidates`.)
- **Pick K** by greedy submodular coverage `F(S) = Σ_r max_{n∈S} ℓ_{n,r}(x)`
  (paper Eqs. 8-9): best singleton, then maximum marginal gain, until `|S| = K`.
  (`submodular_select`.)

### (2) Compose them — per-dim Product-of-Experts at the **hard limit τ→0** (paper Eq. 10)
The paper weights each concept per pixel `w_n(r) = softmax_{n∈S}(ℓ_{n,r}(x)/τ)`.  We
take **τ→0**, which is *per-pixel ownership*:

```
μ_T[r] = m_{ argmax_{n∈S} ℓ_{n,r}(x) , r }
```

i.e. **every pixel is copied from the single selected concept that best explains the
query at that pixel** (`poe_compose`).  This is the crux: soft averaging (τ=1) blends
every concept on every pixel → blurry, wrong; the hard limit routes background pixels
to the background-colour concept and the stroke to the digit-shape+colour concept → a
sharp, correct held-out (digit, fg, bg).

## Metrics — exactly as in the paper (§4.2)
Generated images are scored against two reference sets per OOD class — **Faithfulness**
(the query images) and **Generalization** (other held-out images of the same class) —
using the paper's metrics:
- **FID** (Fréchet Inception Distance, Inception-V3 pool features),
- **CLIP** image–image cosine similarity (CLIP ViT-B/32),
- **Precision / Recall** via the k-NN density estimator (k=3) in Inception feature space,
- **F1** = harmonic mean of precision and recall.
Reported as mean ± SE over the 40 OOD classes.  (Small-N FID caveats per the paper's
Appendix E apply; we generate one composed image per query.)

Baselines compared (paper §4.2): **Top-k** nearest-trained-class retrieval and
**Query-only** (`N(x_q, σ²I)` sampled — the noisy memorisation reference).

## Outputs (`tests/moc/colormnist_output/`)
- `summary.csv` — every metric (FID, CLIP, Precision, Recall, F1 for both reference sets,
  plus an attribute-classifier joint accuracy) for each method.
- **`metrics.png`** — the metrics as a **grouped bar chart**: one panel per metric
  (FID↓, CLIP↑, Precision↑, Recall↑, F1↑), bars per method with **Faithfulness vs
  Generalization** side by side and **±SE** error bars — a visual of the table above.
- `methods.png` — each method's reconstruction of held-out queries.
- `concepts.png` — the per-pixel PoE decomposition (query · μ_T · the selected concepts,
  labelled by the % of pixels each owns).
- `hierarchy.png` — the Cobweb concept hierarchy, mean image at each node, top-down.

## Key finding
The composition is faithful to the paper (selection Eqs. 8-9, composition Eq. 10); the
decisive ingredient for Cobweb on raw pixels is the **hard temperature limit τ→0** of
the per-dim PoE weighting — it roughly doubles compositional accuracy over the soft
average by giving each pixel to its single best-fitting concept.
