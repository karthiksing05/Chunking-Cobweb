# α-Agnostic Basic-Level Frontier

A closed-form, sampling-free replacement for the per-leaf Monte-Carlo
`get_basic` whose ranking matches the α → ∞ limit of expected PMI under a
uniform-over-leaves marginal.  Cobweb trees stay trained at small α
(structure-friendly); the basic-level analysis happens at the α → ∞ shape
where the PMI curve is most stable.

## Formula

For a node *c* with per-attribute counts $n_v^c(a)$, totals $N_c(a)$,
alphabet sizes $K_a$, and root counts $n_v^r(a), N_r(a)$, plus the tree's
leaf count $L$:

$$
\text{score}(c) \;=\;
\sum_a r_a \left[
  \frac{1}{N_c(a)}\!\left(\sum_v (n_v^c(a))^2 \;-\; \frac{1}{L}\sum_v n_v^c(a)\,n_v^r(a)\right)
  \;-\;
  \frac{N_c(a) - N_r(a)/L}{K_a}
\right]
$$

with $r_a$ the standard attribute weight (`weight_attr × attr_weights[a]`),
matching `entropy_attr`'s convention.

### Derivation sketch

1. Empirical EPMI under leaf-mixture marginal:
   $\frac{1}{|c|}\sum_{i \in c}\bigl[\log P_c(x_i) - \log P_{\text{mix}}(x_i)\bigr]$
   with $P_{\text{mix}}(x) = \tfrac{1}{L}\sum_l P_l(x)$.
2. Take α → ∞ and expand $P_c(v|a) - \tfrac{1}{K_a}$ to first order in
   $1/\alpha$. The leading term factorises across attributes.
3. Use $\sum_l n_v^l(a) = n_v^r(a)$ (each instance is in exactly one leaf).
4. Drop the global $1/\alpha$ factor — it doesn't affect ranking.

The result is α-free, count-only, and computable in `O(|attrs| × |values per
attr|)` per node.

## API

```cpp
// CobwebDiscreteNode
double basic_level_score(int n_leaves);

// CobwebDiscreteTree
std::vector<CobwebDiscreteNode*> get_basic_frontier();
```

`get_basic_frontier()` does a single DFS:

- post-order computes `max_subtree(N)` via the score (well, in the simpler
  implementation we use the "antichain DFS": a node is added iff no child
  *strictly* outscores it);
- top-down emits nodes that dominate their subtree, pruning their
  descendants.

This guarantees an **antichain frontier** — every leaf has exactly one
frontier ancestor on its leaf → root path.

## Sampling-free guarantees

Neither `basic_level_score` nor `get_basic_frontier` calls `sample()`,
`sample_leaf_uniform()`, or anything else stochastic.  They read cached
count tables.  No `eval_alpha`, no `n_samples`, no `max_nodes`.

## Tests

- `tests/basic-level/corter_gluck_basic_frontier.py` — verifies the
  frontier matches the labelled basic level on the seven hierarchies from
  `corter_gluck_hierarchies.py` (Murphy & Smith, Begriffshierarchien
  I/II/III, Fruit, Music, Furniture).  6/7 pass; Hier III is the
  deliberately-adversarial "cross-cuts shape" hierarchy where the only
  discriminative information lives at the instance level.
- `tests/basic-level/grammar_basic_level_test.py` — builds a Cobweb-Discrete
  context tree on TEST_GRAMMAR3, runs the NEW frontier and the OLD
  Monte-Carlo `get_basic` side-by-side, and produces:
  - per-method subtree visualisations,
  - tree-with-bars figures (red border = NEW frontier, green border = OLD),
  - `score_by_depth.png`: mean α-agnostic score per tree depth,
  - `method_comparison.txt`: per-token agreement count and one-BL-per-path
    audit.

## Files

- `cobweb-private/include/cobweb_discrete_node.h`
- `cobweb-private/src/cobweb_discrete_node.cpp` — `basic_level_score`
- `cobweb-private/include/cobweb_discrete_tree.h`
- `cobweb-private/src/cobweb_discrete_tree.cpp` — `get_basic_frontier`
- `cobweb-private/src/cobweb_discrete.cpp` — Python bindings
