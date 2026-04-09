# Instance-PMI Basic Level Selection

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

## Instance-PMI Approximation: `get_basic_instance_pmi`

### Motivation

A direct evaluation of EPMI requires sampling from $P_c$ and querying the full tree marginal $P_{\text{tree}}(x)$ — both of which are expensive. The two sources of cost are:

1. **Expectation over $P_c$** — replaced by a point evaluation at the *actual observed instance* $x$.
2. **Full tree marginal $P_{\text{tree}}(x)$** — replaced by the root node's distribution $P_{\text{root}}(x)$, which is a constant offset shared by all ancestors on the path.

### The Instance

Unlike the mode-PMI variant (which constructs a synthetic MAP instance from the leaf's `av_count`), instance-PMI uses the **actual instance** $x$ that was categorized into the leaf. When $x$ is available (e.g. in `_score_along_path` during parsing), it is passed directly. When only the leaf node is available (e.g. during `get_basic_level_nodes` or `_basic_sample`), the mode instance $x^*$ is constructed as a fallback:

$$
x^*_a \;=\; \arg\max_{v}\; n_v^{(\text{leaf})}, \qquad \forall a \in \text{attrs}(\text{leaf})
$$

The key advantage of using the actual instance: it evaluates how well each ancestor *explains this specific observation*, rather than how well it explains a synthetic summary of the leaf. This makes the basic-level selection sensitive to the particular input being processed, which is especially important when leaves contain a mixture of instance types.

### The Score Function

For each ancestor $c$ on the path from leaf to root, define:

$$
\text{score}_{\text{inst}}(c) \;=\; \log P_\alpha(x \mid c) \;-\; \log P_\alpha(x \mid \text{root})
$$

The second term is evaluated **once** at the root and reused for all ancestors (it is a constant w.r.t. $c$). This term plays the role of the tree marginal, anchoring the score to zero at the root and positive at more specific ancestors.

The basic level is then:

$$
\boxed{c^* \;=\; \arg\max_{c \,\in\, \text{path}} \;\log P_\alpha(x \mid c) \;-\; \log P_\alpha(x \mid \text{root})}
$$

### Approximations Made

| Ground-truth term | Instance-PMI approximation | Justification |
|---|---|---|
| $\mathbb{E}_{x \sim P_c}[\log P_c(x)]$ | $\log P_\alpha(x \mid c)$ | Replace expectation with the actual observed instance |
| $\mathbb{E}_{x \sim P_c}[\log P_{\text{tree}}(x)]$ | $\log P_\alpha(x \mid \text{root})$ | Root distribution ≈ marginal; avoids full BFS traversal |

The instance-specific evaluation is natural: we are asking "which concept on this path best explains *this specific observation*?" rather than "which concept has the highest average informativeness?". The root-as-marginal approximation is monotone: all scores are relative to the same baseline, so argmax is preserved as long as the root is a reasonable prior.

### Complexity

| Step | Cost |
|---|---|
| (If using fallback) Build $x^*$ from leaf `av_count` | $O(\lvert V_\text{leaf} \rvert)$ |
| Evaluate $\log P_\alpha(x \mid \text{root})$ once | $O(\lvert x \rvert)$ |
| Walk leaf → root, score each ancestor | $O(\text{height} \times \lvert x \rvert)$ |

Total: $O(\text{height} \times \lvert x \rvert)$ with no sampling, versus $O(\text{height} \times N \times \lvert V \rvert)$ for Monte Carlo EPMI.

### Smoothing Parameter (`eval_alpha`)

#### What is $\alpha$?

Every node $c$ stores raw counts $n_v^{(a,c)}$ for each attribute–value pair. To obtain a valid probability distribution, these counts are smoothed with a **Dirichlet pseudocount** $\alpha$:

$$
P_\alpha(v \mid a,\, c) \;=\; \frac{n_v^{(a,c)} + \alpha}{n_a^{(c)} + \lvert V_a \rvert \, \alpha}
$$

where $n_a^{(c)} = \sum_v n_v^{(a,c)}$ is the total count for attribute $a$, and $\lvert V_a \rvert$ is the vocabulary size of that attribute. The parameter $\alpha$ controls the **floor probability** assigned to values the node has rarely or never observed: a smaller $\alpha$ concentrates probability mass on observed values; a larger $\alpha$ spreads mass more uniformly.

#### Why the structural $\alpha$ is small

During **learning and categorization**, the tree needs to discriminate sharply between competing concepts. A small structural alpha (e.g. `content_alpha = 1e-3`) makes each node's distribution peaked around the values it has actually seen. This is desirable: a concept that has observed the value "cat" 10 times and "dog" 0 times should assign nearly all probability to "cat" so that new observations are routed to the most similar concept. Sharp distributions drive effective category utility splits.

#### Why basic-level evaluation needs a larger $\alpha$

The instance-PMI score walks from leaf to root and computes $\log P_\alpha(x \mid c)$ at each ancestor $c$. The problem is how this quantity behaves when $\alpha$ is very small.

At the **leaf**, the instance's attribute values typically have count $\geq 1$ (the instance was categorized here), so:

$$
\log P_\alpha(x_a \mid \text{leaf}) \;\approx\; \log n_{x_a}^{(\text{leaf})} - \log n_a^{(\text{leaf})} \quad\text{(close to 0 for dominant values)}
$$

At a **higher ancestor** $c$, some of the instance's values $x_a$ may be rare or completely unseen — the ancestor aggregates instances from many sub-concepts with different distributions. For those attributes:

$$
\log P_\alpha(x_a \mid c) \;\approx\; \log \alpha \;-\; \log n_a^{(c)} \quad\text{when } n_{x_a}^{(c)} \approx 0
$$

When $\alpha$ is tiny (e.g. $10^{-3}$), this term is a **massive negative penalty** (around $\log 10^{-3} = -6.9$ before the denominator). The penalty grows more severe with smaller $\alpha$ and is incurred for every attribute where the ancestor has low counts of the instance's value.

The result: the gap between the leaf score and any ancestor's score becomes enormous. The score curve is effectively a **step function** — maximal at the leaf, dramatically lower everywhere else — and the argmax trivially collapses to the leaf regardless of where the true basic level lies.

#### The effect of increasing $\alpha$ for evaluation

With a larger evaluation alpha (e.g. `content_bl_alpha = 1e-1`):

1. **The floor probability rises.** Unseen values at ancestors now receive pseudocount $0.1$ instead of $0.001$, reducing the per-attribute penalty from $\approx -6.9$ to $\approx -2.3$ (before the denominator term). Ancestors are no longer catastrophically penalised for not having observed every instance value.

2. **The leaf score is compressed.** At the leaf, the instance value's probability can no longer approach $1.0$; the smoothing denominator $n_a + |V_a| \cdot \alpha$ is now noticeably larger. The leaf's score advantage over its parent shrinks.

3. **The score curve develops a meaningful shape.** The gradual broadening of counts as you ascend the tree — each ancestor has observed more total instances — now produces a smooth trade-off: higher ancestors gain from a larger denominator (more total evidence) but lose from diluted counts of the specific instance values. An intermediate node where $x$ is still well-represented but the node covers a larger population becomes the peak — this is the basic level.

In short, $\alpha$ controls the **sensitivity of log-probability to unseen values**. A small $\alpha$ makes the score hypersensitive to any mismatch, locking the argmax to the leaf. A larger $\alpha$ dampens this sensitivity so that the score curve can reflect the genuine specificity-vs-generality trade-off that defines the basic level.

#### Implementation

`log_prob_instance` accepts an `eval_alpha` parameter that overrides the tree's structural alpha solely during basic-level scoring. In the codebase, the structural alpha used for learning (e.g. `content_alpha = 1e-3`) is never modified; `eval_alpha` is a separate parameter passed only during calls to `get_basic_instance_pmi` and `get_basic`. This separation ensures learning dynamics remain sharp while basic-level evaluation uses the smoother curves needed to identify intermediate concepts.

```
WEBSTER(
    content_alpha    = 1e-3,   # structural: sharp, for learning
    content_bl_alpha = 1e-1,   # evaluation: smooth, for basic-level
    ...
)
```

---

## Node Log-Probability: Formal Definition

The log-probability of an instance $x$ under node $c$ with smoothing $\alpha$ is:

$$
\log P_\alpha(x \mid c) \;=\; \sum_{a} \log P_\alpha(x_a \mid a,\, c)
$$

### Per-Attribute Term $\log P_\alpha(x_a \mid a, c)$

Let:

| Symbol | Meaning |
|--------|---------|
| $n_a$ | total count of attribute $a$ in node $c$ |
| $n_{x_a}$ | count of the instance's value $x_a$ for attribute $a$ in $c$ |
| $\lvert V_a \rvert$ | vocabulary size of attribute $a$ |
| $\alpha$ | smoothing parameter (may be overridden by `eval_alpha`) |

$$
\log P_\alpha(x_a \mid a,\, c) \;=\; \log(n_{x_a} + \alpha) \;-\; \log\!\bigl(n_a + \lvert V_a \rvert\,\alpha\bigr)
$$

> **Unseen value:** if $x_a$ was never observed in $c$, then $n_{x_a} = 0$ and the term becomes $\log\alpha - \log(n_a + |V_a|\alpha)$ — a small but finite penalty controlled by the smoothing parameter.

---

## Multi-Resolution Log-Probability (Ref Attributes)

When a **reference hierarchy** is present (i.e. the values themselves form a tree, e.g. parse-tree constituents), each attribute's log-probability is computed at multiple resolution levels $d = 1, \ldots, d_{\max}$:

$$
\log P_\alpha^{\text{ref}}(x_a \mid a,\, c) \;=\; \frac{\sum_{d=1}^{d_{\max}} w_d\, \log P_\alpha^{(d)}(x_a \mid a,\, c)}{\sum_{d=1}^{d_{\max}} w_d}, \qquad w_d = \frac{d}{d_{\max}}
$$

Each $\log P_\alpha^{(d)}$ is the standard formula applied to the coarse-grained bucket that contains $x_a$ at depth $d$ of the reference tree. Finer levels (larger $d$) receive higher weight, matching the entropy convention.
