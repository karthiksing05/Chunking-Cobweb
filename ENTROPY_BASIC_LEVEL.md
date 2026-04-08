# Mode-PMI Basic Level Selection

## Background

A **Cobweb** tree is a hierarchical concept formation model. Each node $c$ stores a smoothed categorical distribution $P_\alpha(x \mid c)$ over instances $x$. Given a new instance (a leaf in the tree), we want to identify its **basic level**: the ancestor node that is the most informative, specific concept for that instance — neither too broad (the root) nor too narrow (the leaf itself).

---

## Ground-Truth Objective: Expected PMI

The true basic level is the ancestor $c^*$ that **maximises the expected pointwise mutual information (EPMI)** between an instance drawn from $c$ and the concept label $c$ itself:

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

The first term is $-H(c)$ (negative entropy of $c$). The second term is the **cross-entropy** of the marginal as seen through the lens of $P_c$, which we call $H_\times(P_c \| P_{\text{tree}})$.

$$
\boxed{\text{EPMI}(c) = -H(c) - H_\times(P_c \,\|\, P_{\text{tree}})}
$$

**Implementation (`get_basic`):** Evaluated by Monte Carlo — $N$ instances are sampled from $P_c$ and the average log-ratio is accumulated. This costs $O(\text{height} \times N)$ per query.

---

## Mode-PMI Approximation: `get_basic_mode_pmi`

### Motivation

A direct evaluation of EPMI requires sampling from $P_c$ and querying the full tree marginal $P_{\text{tree}}(x)$ — both of which are expensive. The two sources of cost are:

1. **Expectation over $P_c$** — replaced by a point evaluation at the MAP (mode) instance of the leaf.
2. **Full tree marginal $P_{\text{tree}}(x)$** — replaced by the root node's distribution $P_{\text{root}}(x)$, which is a constant offset shared by all ancestors on the path.

### The Mode Instance

Let $\text{leaf}$ be the query node. Its **MAP instance** $x^*$ is the synthetic observation formed by taking the most frequent value of each attribute:

$$
x^*_a \;=\; \arg\max_{v}\; n_v^{(\text{leaf})}, \qquad \forall a \in \text{attrs}(\text{leaf})
$$

This is an $O(|V_\text{leaf}|)$ operation — one pass over the leaf's `av_count` table.

### The Score Function

For each ancestor $c$ on the path from leaf to root, define:

$$
\text{score}_{\text{mode}}(c) \;=\; \log P_\alpha(x^* \mid c) \;-\; \log P_\alpha(x^* \mid \text{root})
$$

The second term is evaluated **once** at the root and reused for all ancestors (it is a constant w.r.t. $c$). This term plays the role of the tree marginal, anchoring the score to zero at the root and positive at more specific ancestors.

The basic level is then:

$$
\boxed{c^* \;=\; \arg\max_{c \,\in\, \text{path}} \;\log P_\alpha(x^* \mid c) \;-\; \log P_\alpha(x^* \mid \text{root})}
$$

### Approximations Made

| Ground-truth term | Mode-PMI approximation | Justification |
|---|---|---|
| $\mathbb{E}_{x \sim P_c}[\log P_c(x)]$ | $\log P_\alpha(x^* \mid c)$ | Concentrate expectation at the leaf's MAP point |
| $\mathbb{E}_{x \sim P_c}[\log P_{\text{tree}}(x)]$ | $\log P_\alpha(x^* \mid \text{root})$ | Root distribution ≈ marginal; avoids full BFS traversal |

The mode approximation is reasonable when leaf distributions are peaked (low entropy), which is common for recently-formed or highly-reinforced concepts. The root-as-marginal approximation is monotone: all scores are relative to the same baseline, so argmax is preserved as long as the root is a reasonable prior.

### Complexity

| Step | Cost |
|---|---|
| Build $x^*$ from leaf `av_count` | $O(\lvert V_\text{leaf} \rvert)$ |
| Evaluate $\log P_\alpha(x^* \mid \text{root})$ once | $O(\lvert x^* \rvert)$ |
| Walk leaf → root, score each ancestor | $O(\text{height} \times \lvert x^* \rvert)$ |

Total: $O(\text{height} \times \lvert x^* \rvert)$ with no sampling, versus $O(\text{height} \times N \times \lvert V \rvert)$ for Monte Carlo EPMI.

### Smoothing Parameter (`eval_alpha`)

`log_prob_instance` accepts an `eval_alpha` parameter that overrides the tree's structural alpha solely during basic-level evaluation. This is critical: the structural alpha (e.g. `content_alpha = 1e-3`) is optimised for *learning* (sharp distributions), while the basic-level alpha (e.g. `content_bl_alpha = 1e-1`) is larger to produce *smoother* score curves along the path. Without the correct `eval_alpha`, the leaf distribution is so sharp that $x^*$ dominates and the argmax trivially collapses to the leaf itself.

---

## Node Log-Probability: Formal Definition

The log-probability of an instance $x^*$ under node $c$ with smoothing $\alpha$ is:

$$
\log P_\alpha(x^* \mid c) \;=\; \sum_{a} \log P_\alpha(x^*_a \mid a,\, c)
$$

### Per-Attribute Term $\log P_\alpha(x^*_a \mid a, c)$

Let:

| Symbol | Meaning |
|--------|---------|
| $n_a$ | total count of attribute $a$ in node $c$ |
| $n_{x^*_a}$ | count of the mode value $x^*_a$ for attribute $a$ in $c$ |
| $\lvert V_a \rvert$ | vocabulary size of attribute $a$ |
| $\alpha$ | smoothing parameter (may be overridden by `eval_alpha`) |

$$
\log P_\alpha(x^*_a \mid a,\, c) \;=\; \log(n_{x^*_a} + \alpha) \;-\; \log\!\bigl(n_a + \lvert V_a \rvert\,\alpha\bigr)
$$

> **Unseen value:** if $x^*_a$ was never observed in $c$, then $n_{x^*_a} = 0$ and the term becomes $\log\alpha - \log(n_a + |V_a|\alpha)$ — a small but finite penalty controlled by the smoothing parameter.

---

## Multi-Resolution Log-Probability (Ref Attributes)

When a **reference hierarchy** is present (i.e. the values themselves form a tree, e.g. parse-tree constituents), each attribute's log-probability is computed at multiple resolution levels $d = 1, \ldots, d_{\max}$:

$$
\log P_\alpha^{\text{ref}}(x^*_a \mid a,\, c) \;=\; \frac{\sum_{d=1}^{d_{\max}} w_d\, \log P_\alpha^{(d)}(x^*_a \mid a,\, c)}{\sum_{d=1}^{d_{\max}} w_d}, \qquad w_d = \frac{d}{d_{\max}}
$$

Each $\log P_\alpha^{(d)}$ is the standard formula applied to the coarse-grained bucket that contains $x^*_a$ at depth $d$ of the reference tree. Finer levels (larger $d$) receive higher weight, matching the entropy convention.
