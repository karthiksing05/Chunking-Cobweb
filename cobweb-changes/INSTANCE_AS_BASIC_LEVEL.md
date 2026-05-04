# Instance-PMI Basic Level Selection

## Ground Truth: Expected PMI (`get_basic`)

The basic level is the ancestor $c^*$ maximising expected pointwise mutual information:

$$
\text{EPMI}(c) = \mathbb{E}_{x \sim P_c}\!\left[\log P_c(x) - \log P_{\text{tree}}(x)\right] = -H(c) - H_\times(P_c \| P_{\text{tree}})
$$

Evaluated by Monte Carlo sampling.  Cost: $O(\text{height} \times N)$.

From code:

$$E_{x|c} [pmi(x;c)] = H(p_{x|c}, p_x) - H(p_{x|c}).$$
---

## Instance-PMI (`get_basic_instance_pmi`)

Replaces the expectation with a point evaluation on the actual observed instance $x$, and the tree marginal with the root distribution:

$$
c^* = \arg\max_{c \in \text{path}} \; \log P_\alpha(x \mid c) - \log P_\alpha(x \mid \text{root})
$$

When $x$ is unavailable (e.g. `get_basic_level_nodes`), the mode instance $x^*_a = \arg\max_v n_v^{(\text{leaf})}$ is used as fallback. Cost: $O(\text{height} \times |x|)$, no sampling.

---

## Smoothed Log-Probability

$$
\log P_\alpha(x \mid c) = \sum_a \log \frac{n_{x_a} + \alpha}{n_a + |V_a|\,\alpha}
$$

Small $\alpha$ concentrates mass on observed values (good for learning); large $\alpha$ smooths unseen-value penalties (needed for basic-level scoring so the argmax doesn't collapse to the leaf).

```
WEBSTER(
    content_alpha    = 1e-3,   # structural: sharp, for learning
    content_bl_alpha = 1e-1,   # evaluation: smooth, for basic-level
)
```

---

## Multi-Resolution Log-Probability (Ref Attributes)

When values form a reference hierarchy:

$$
\log P_\alpha^{\text{ref}}(x_a \mid a, c) = \frac{\sum_{d=1}^{d_{\max}} w_d \log P_\alpha^{(d)}(x_a \mid a, c)}{\sum_{d=1}^{d_{\max}} w_d}, \quad w_d = \tfrac{d}{d_{\max}}
$$
