# Entropy-Based Basic Level Selection

## Background

A **Cobweb** tree is a hierarchical concept formation model.
Each node $c$ stores a smoothed categorical distribution $P_\alpha(x \mid c)$ over
instances $x$.
Given a new instance (a leaf in the tree), we want to identify its **basic level**:
the ancestor node that is the most informative, specific concept for that instance —
neither too broad (the root) nor too narrow (the leaf itself).

---

## Ground-Truth Objective: Expected PMI

The true basic level is the ancestor $c^*$ that **maximises the expected
pointwise mutual information (EPMI)** between an instance drawn from $c$ and
the concept label $c$ itself:

$$
\text{EPMI}(c) \;=\; \mathbb{E}_{x \sim P_c}\!\left[\log P_c(x) - \log P_{\text{tree}}(x)\right]
$$

where:

- $P_c(x) = P_\alpha(x \mid c)$ — the smoothed conditional distribution at node $c$
- $P_{\text{tree}}(x)$ — the tree-wide marginal probability of $x$ (computed via a
  tree traversal weighted by node counts)

Expanding the expectation:

$$
\text{EPMI}(c) \;=\; \mathbb{E}_{x \sim P_c}[\log P_c(x)] \;-\; \mathbb{E}_{x \sim P_c}[\log P_{\text{tree}}(x)]
$$

The first term is $-H(c)$ (negative entropy of $c$).
The second term is the **cross-entropy** of the marginal as seen through the
lens of $P_c$, which we call $H_\times(P_c \| P_{\text{tree}})$.

$$
\boxed{\text{EPMI}(c) = -H(c) - H_\times(P_c \,\|\, P_{\text{tree}})}
$$

**Implementation (`get_basic`):** Evaluated by Monte Carlo — $N$ instances are
sampled from $P_c$ and the average log-ratio is accumulated.
This costs $O(\text{height} \times N)$ per query.

---

## Entropy Proxy: `get_basic_entropy`

### The Approximation

For any fixed query distribution (i.e. the leaf's own distribution, which is
common to all ancestors on the path), the cross-entropy term
$H_\times(P_c \,\|\, P_{\text{tree}})$ changes **slowly** as $c$ moves from
leaf to root.
In practice, the dominant variation along the path comes from $H(c)$, which
decreases sharply near the basic level (the concept is peaked/specific) and
increases toward the root (the concept becomes diffuse).

This motivates the approximation:

$$
\arg\max_c\;\text{EPMI}(c) \;\approx\; \arg\min_c\; H(c)
$$

**Implementation (`get_basic_entropy`):** Walk from leaf to root ($O(\text{height})$),
return the ancestor with the smallest entropy.
Entropy values are cached per-node and invalidated only when counts change, so
the walk has essentially no per-value inner loops.

---

## Node Entropy: Formal Definition

The entropy of node $c$ is the sum of per-attribute entropies over all
non-hidden attributes:

$$
H(c) \;=\; \sum_{a \,\geq\, 0} H_\alpha(c,\, a)
$$

### Per-Attribute Entropy $H_\alpha(c, a)$

Let:

| Symbol | Meaning |
|--------|---------|
| $n_a$ | total count of attribute $a$ in node $c$ |
| $\lvert V_a \rvert$ | vocabulary size of attribute $a$ (all known values) |
| $\lvert V_c \rvert$ | number of values of $a$ actually observed in $c$ |
| $n_0 = \lvert V_a \rvert - \lvert V_c \rvert$ | number of unseen values |
| $\alpha$ | Dirichlet / Laplace smoothing parameter |
| $S = \sum_{v \in V_c}(n_v + \alpha)\log(n_v + \alpha)$ | cached sufficient statistic |
| $\rho$ | attribute-frequency weight ($= 1$ unless `weight_attr` is set) |

The smoothed distribution over values of attribute $a$ at node $c$ is:

$$
P_\alpha(v \mid a, c) \;=\; \frac{n_v + \alpha}{n_a + \lvert V_a \rvert \alpha}
$$

The corresponding entropy is:

$$
H_\alpha(c, a) \;=\; -\rho \cdot \left[
  \frac{S \;+\; n_0\,\alpha\log\alpha}{n_a + \lvert V_a \rvert\alpha}
  \;-\; \log\!\bigl(n_a + \lvert V_a \rvert\alpha\bigr)
\right]
$$

This is algebraically equivalent to
$-\rho\sum_v P_\alpha(v\mid a,c)\log P_\alpha(v\mid a,c)$,
rearranged for numerical efficiency using the cached sum $S$.

> **Edge case:** if $n_a = 0$ and $\alpha = 0$, there is no information and
> $H_\alpha(c,a) = 0$.

---

## Multi-Resolution Entropy (Ref Attributes)

When a **reference hierarchy** is present (i.e. the values themselves form a
tree, e.g. parse-tree constituents), each attribute's entropy is computed at
multiple resolution levels $d = 1, \ldots, d_{\max}$:

$$
H_\alpha^{\text{ref}}(c, a) \;=\; \frac{\sum_{d=1}^{d_{\max}} w_d\, H_\alpha^{(d)}(c, a)}{\sum_{d=1}^{d_{\max}} w_d}, \qquad w_d = \frac{d}{d_{\max}}
$$

Each $H_\alpha^{(d)}$ is the standard formula applied to coarse-grained
"bucket" counts at depth $d$ of the reference tree.
Finer levels (larger $d$) receive higher weight.

---

## Complexity Summary

| Method | Time per query | Notes |
|--------|---------------|-------|
| `get_basic(n, …)` | $O(h \cdot n)$ | Ground truth via Monte Carlo; $h$ = tree height |
| `get_basic_entropy()` | $O(h)$ | Entropy proxy; ~150,000× faster in practice |

---

## Empirical Agreement

On the WEBSTER multi-hierarchy model trained on a standard test corpus:

| Hierarchy | Content agreement | Context agreement |
|-----------|------------------|------------------|
| Entropy vs. sampled EPMI | ~84–85 % | ~50–52 % |

Disagreements are concentrated at nodes where the cross-entropy term
$H_\times(P_c \,\|\, P_{\text{tree}})$ is non-negligible (e.g. shallow nodes
near the root, or nodes with many rare attributes).
The ablation studies in `tests/speedups/test_basic_level_analytical.py`
document these cases at varying smoothing levels (`ABLATION_ALPHAS`).
