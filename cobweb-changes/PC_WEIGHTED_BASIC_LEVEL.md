# P(c)-Weighted Basic Level Selection (`get_basic_pc`)

A drop-in alternative to `get_basic` that swaps the empirical leaf-bias hack
(crank `eval_alpha` until the score levels out) for a structural fix:
weight each candidate's score by its prior $P(c)$ in the tree.

Implemented for both `CobwebDiscreteNode` and `CobwebContinuousNode`.

---

## Original formula (`get_basic`)

The basic level is the ancestor $c^*$ along the leaf→root path that maximises
expected pointwise mutual information:

$$
\text{EPMI}(c)
\;=\;
\mathbb{E}_{x \sim P_c}\!\left[\,\log P_c(x) \;-\; \log P_{\text{tree}}(x)\,\right]
\;=\;
H(P_c, P_{\text{tree}}) \;-\; H(P_c)
\;=\;
D_{\mathrm{KL}}\!\bigl(P_c \,\|\, P_{\text{tree}}\bigr)
$$

Estimated by Monte Carlo: sample $x$ from $P_c$ via `sample()` (or
`sample_leaf_uniform()`), score with `log_prob_instance`, subtract
`tree->log_prob` (the full-tree mixture, beam-truncated to `max_nodes`).

### Why this leaks toward leaves

KL between a peaked distribution and a flatter mixture is structurally
large — leaves win mechanically because they are the most peaked $P_c$ in
the tree. The discrete version mitigates this with three knobs:
1. `tree->log_prob` includes the leaf itself in its mixture (cancels most
   of the leaf's self-likelihood),
2. an `eval_alpha` knob that retroactively flattens leaves,
3. `uniform_leaf=true` which dampens dominant-subtree mass.

All three are empirical hacks. The user-side workflow looks like
"set `eval_alpha=10` (or whatever) until basic-level nodes look right."

---

## New formula (`get_basic_pc`)

$$
\boxed{\;\text{score}(c)
\;=\;
\frac{|c|}{|root|}\cdot
\mathbb{E}_{x \sim P_c}\!\left[\,\log P_c(x) \;-\; \log P_{\text{root}}(x)\,\right]
\;}
$$

Two changes vs. `get_basic`:

1. **Marginal is the root**, not the full-tree mixture. The root's
   distribution is fixed; no priority queue, no beam truncation.
2. **Score is multiplied by $P(c) = |c|/|\text{root}|$**, the cluster's prior.

This is exactly the per-cluster contribution to the mutual information
decomposition

$$
I(X;C) \;=\; \sum_{c}\, P(c) \cdot D_{\mathrm{KL}}\!\bigl(P_c \,\|\, P_X\bigr).
$$

### Why $P(c)$ damps the leaf bias

Leaves have small $P(c)$. A leaf with size $1/N$ and KL of magnitude
$M$ scores $M/N$. An internal node aggregating $k$ leaves with KL of
magnitude $\approx M/k$ scores $\approx M/(N)$ (roughly comparable). In
the limit, no level dominates by entropy alone — the score becomes a
genuine information-theoretic quantity instead of a peakedness measure.

Concretely, the root has score $0$ (KL$=0$ by construction), and very
small / overfit leaves have small score because their $P(c)$ shrinks faster
than their KL grows.

### What's intentionally dropped

- **No `eval_alpha`.** The $P(c)$ prefix replaces it as the
  leaf-bias control, and we always smooth with `tree->alpha`.
- **No `tree->log_prob` / `max_nodes`.** Marginal is just root.
- **In the continuous variant, no Monte Carlo.** Diagonal Gaussians have a
  closed-form KL — we use it directly.

---

## Discrete implementation

`CobwebDiscreteNode::get_basic_pc(int n_samples, bool debug, bool uniform_leaf)`
([cobweb_discrete_node.cpp](../cobweb-private/src/cobweb_discrete_node.cpp))

```cpp
double mean_pmi = 0.0;
for (i in n_samples) {
    INSTANCE x = uniform_leaf ? c->sample_leaf_uniform() : c->sample();
    mean_pmi += c->log_prob_instance(x) - root->log_prob_instance(x);
}
mean_pmi /= n_samples;
double p_c   = c->count / root->count;
double score = p_c * mean_pmi;
```

Walks leaf → root, returns argmax. Same control flow as `get_basic`.

## Continuous implementation

`CobwebContinuousNode::get_basic_pc(bool debug)`
([cobweb_continuous_node.cpp](../cobweb-private/src/cobweb_continuous_node.cpp))

For diagonal Gaussians the KL is closed form:

$$
D_{\mathrm{KL}}\!\bigl(\mathcal{N}(\mu_c,\sigma_c^2) \,\|\, \mathcal{N}(\mu_r,\sigma_r^2)\bigr)
\;=\;
\tfrac{1}{2}\sum_i \Bigl[\,
\log\!\tfrac{\sigma_{r,i}^2}{\sigma_{c,i}^2}
\,+\,\tfrac{\sigma_{c,i}^2 + (\mu_{c,i}-\mu_{r,i})^2}{\sigma_{r,i}^2}
\,-\,1
\,\Bigr],
$$

with $\sigma^2_i = \text{sum\_sq}_i / \text{count} + \text{prior\_var}$ on
both nodes (each uses its own statistics — independent of `covar_from`).

```cpp
auto root_var = (root->sum_sq.array() / root->count) + prior_var;
auto root_mean = root->mean.array();

float kl_score(node *c) {
    auto c_var = (c->sum_sq.array() / c->count) + prior_var;
    auto c_mean = c->mean.array();
    auto kl = 0.5f * (
        (root_var / c_var).log()
      + (c_var + (c_mean - root_mean).square()) / root_var
      - 1.0f
    ).sum();
    return (c->count / root->count) * kl;
}
```

No sampling, no Monte Carlo variance — the score is fully deterministic.

---

## Empirical observations

### Discrete (Corter & Gluck "ideal" hierarchy, 4-level)

On a hand-built ideal 4-level tree (`tests/basic-level/corter_gluck_test_stable.py`):

- `get_basic` (with `eval_alpha=1`) recovers all four basic categories
  (Hammer/Brick/Knife/Pizza-cutter) at 16/16 items.
- `get_basic_pc` (no eval_alpha) recovers ~9/16 — frequently *over*-corrects
  toward the **superordinate** (Pounder/Cutter, depth 1) on this idealised
  small data because $P(\text{superord}) = 0.5$ vs. $P(\text{basic}) = 0.25$
  and raw EPMI doesn't fall fast enough to pay for the prior gap.

This means the $P(c)^1$ weighting is sometimes too aggressive on toy data;
in practice you may want $P(c)^\beta$ for $\beta \in (0,1)$ as a continuous
knob between "no leaf damping" ($\beta=0$, equivalent to `get_basic` with
root marginal) and "full MI contribution" ($\beta=1$, current implementation).

### Continuous (MNIST, 5k samples)

See `tests/basic-level/mnist_basic_level_test.py`. The prototype-image
visualisation shows that `get_basic_pc` consistently picks intermediate
nodes — never the root, almost never the leaves — and the per-subtree
class histograms cluster into recognisable digit-shape families.

---

## When to use which

| | `get_basic` | `get_basic_pc` |
|---|---|---|
| Calibration knob | `eval_alpha` (training $\alpha$ vs. eval $\alpha$ decoupled) | `P(c)` weighting — no knob |
| Marginal | full-tree mixture (priority queue, `max_nodes`) | root only |
| Stochastic? | yes (Monte Carlo) | discrete: yes (MC); continuous: no (closed form) |
| Information-theoretic interpretation | $D_{\mathrm{KL}}(P_c \| P_{\text{tree}})$ | per-cluster MI contribution |
| Strength | tunable, well-tested on toy data | structurally principled, faster, no eval-alpha tuning |
| Weakness | requires `eval_alpha` sweep | can over-generalise on small / shallow trees |

---

## Files changed

- [`cobweb-private/include/cobweb_discrete_node.h`](../cobweb-private/include/cobweb_discrete_node.h)
  — declaration
- [`cobweb-private/src/cobweb_discrete_node.cpp`](../cobweb-private/src/cobweb_discrete_node.cpp)
  — implementation
- [`cobweb-private/src/cobweb_discrete.cpp`](../cobweb-private/src/cobweb_discrete.cpp)
  — Python binding
- [`cobweb-private/include/cobweb_continuous_node.h`](../cobweb-private/include/cobweb_continuous_node.h)
  — declaration
- [`cobweb-private/src/cobweb_continuous_node.cpp`](../cobweb-private/src/cobweb_continuous_node.cpp)
  — closed-form KL implementation
- [`cobweb-private/src/cobweb_continuous.cpp`](../cobweb-private/src/cobweb_continuous.cpp)
  — Python binding (also exposes `depth()`)

## Tests

- [`tests/basic-level/corter_gluck_test_stable.py`](../tests/basic-level/corter_gluck_test_stable.py)
  — side-by-side comparison vs. `get_basic`, written to
  `corter_gluck_stable_viz/basic_level_comparison.txt`
- [`tests/basic-level/basic_level_test.py`](../tests/basic-level/basic_level_test.py)
  — runs `get_basic_pc` first, then the existing `get_basic` `eval_alpha` sweep
- [`tests/basic-level/mnist_basic_level_test.py`](../tests/basic-level/mnist_basic_level_test.py)
  — continuous variant on MNIST, prototype + class-histogram + sample-digit
  visualisation per basic-level subtree
