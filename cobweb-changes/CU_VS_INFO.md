# CU vs default Cobweb variants in `cobweb-private`

`cobweb-private` ships two pairs of trees that share an algorithmic skeleton
but disagree on the heuristic the tree-building decisions optimise:

| Default | CU variant |
| --- | --- |
| `cobweb.cobweb_discrete.CobwebDiscreteTree` | `cobweb.cu_cobweb_discrete.CUCobwebDiscreteTree` |
| `cobweb.cobweb_continuous.CobwebContinuousTree` | `cobweb.cu_cobweb_continuous.CUCobwebContinuousTree` |

The "CU" classes were added to replicate the **Fisher / Cobweb-3 (1990)
category utility** used in the `concept_formation` library that backs
MacLellan & Thakur (2021)'s Convolutional Cobweb. The default classes use
an information-theoretic entropy-based CU that is more modern but
quantitatively different.

This document is a side-by-side of every place the two differ.

---

## 1. Heuristic — discrete tree

### Default: `CobwebDiscreteTree`

Per-attribute "info" term, computed against the running `sum_n_logn`
bookkeeping with Laplace smoothing `α`:

```
n0   = num_vals_total - num_vals_in_concept
T    = attr_count + num_vals_total · α
H    = log(T) - (sum_n_logn + n0 · α · log(α)) / T   # diag entropy w/ Laplace
info = ratio · H                                     # ratio = root.a_count[attr]/root.count
                                                     #         when weight_attr=true
```

The node's "entropy" is `Σ_attr info(attr)`, and partition utility is
`Σ_attr (parent.info_attr − Σ_child P(child) · child.info_attr) / n_children`.

Higher PU = better partition. The whole quantity is a Laplace-smoothed
information-gain CU.

### CU variant: `CUCobwebDiscreteTree`

Per-attribute Fisher ECG (sum of squared probabilities, optionally with
value-remap aggregation, optionally with the same `weight_attr` ratio):

```
ecg(attr) = ratio · Σ_v (n_v / total)²
```

Partition utility is `Σ_attr (Σ_child P(child) · ecg_child − ecg_parent) /
n_children` — the children-side now wins when *higher* than the parent
(opposite sign convention to the entropy variant; semantics same: higher PU
= better partition).

### What this changes

`pu_for_insert`, `pu_for_new_child`, `pu_for_merge`, `pu_for_split`,
`two_best_children`, `get_best_operation`, and the iterative `cobweb` call
that picks BEST / NEW / MERGE / SPLIT *all* see different scores. Two
trees built from the same data will have different leaves, different
merges, different splits, and different categorize paths.

### What this preserves

Both classes implement value-remap (`set_value_remap` /
`clear_value_remap` / `use_value_remap`) the same way — leaf values are
aggregated under their canonical form before the per-attribute squared-sum
(CU variant) or entropy (default) is computed. The remap dict API is
identical.

`predict_probs`, `prob`, `log_prob_instance`,
`log_prob_class_given_instance`, and `is_exact_match` are implemented the
same way in both classes (they don't depend on the heuristic). The CU
discrete tree omits the legacy `predict_pmi`, `predict_parallel`,
`sample`, `get_basic`, `expected_pmi`, and JSON dump/load methods that
exist on the default tree because they aren't used by the Convolutional
Cobweb pipeline.

---

## 2. Heuristic — continuous tree

### Default: `CobwebContinuousTree`

Diagonal Gaussian, scored either via "info CU" or a Mahalanobis-from-parent
quantity depending on the `covar_from` knob (default `2`):

```
covar_from = 1:  score = 0.5 · Σ_d (log var_parent[d] − log var_child[d])
covar_from = 2:  score = 0.5 · Σ_d (child_mean[d] − parent_mean[d])² / parent_var[d]
```

Where `var = sum_sq / count + prior_var` (`prior_var = 1/(2πe) ≈ 0.0585`
by default — a uniform variance floor). There is *no* online attribute
scaling: every dimension's variance is used as-is.

### CU variant: `CUCobwebContinuousTree`

Cobweb-3 Fisher continuous ECG with online attribute scaling:

```
unbiased_var      = sum_sq / (count − 1)                  # 0 if count ≤ 1
root_unbiased_var = root.sum_sq / (root.count − 1)
scale             = root.unbiased_std / scaling           # default scaling = 0.5
scaled_var        = unbiased_var · scaling² / root_unbiased_var
σ_total           = √(scaled_var + 1/(4π))
ecg_dim           = P(A)² · 1 / (2 √π · σ_total)
```

Partition utility is `Σ_d (Σ_child P(child) · ecg_dim_child − ecg_dim_parent)
/ n_children`.

### What this changes

Same propagation as the discrete case — every tree-building decision in
`ifit` is scored by the new heuristic, so the leaf structure, merges, and
splits all diverge from what `CobwebContinuousTree` would have built on
the same data stream.

The *online attribute scaling* is a structural addition: each
dimension's variance is rescaled by the root's variance for that
dimension at every CU evaluation, so a dimension with naturally high
variance can't dominate the partition decision. The default tree has no
analogous mechanism.

### What this preserves

Welford updates for `mean` / `sum_sq` are identical in both. The 4-op
ifit pattern, fringe split, exact-match short-circuit, `get_leaf` (the
Bayesian descent helper), and `clear` semantics match. The default
tree's `predict` / `predict_pmi` / `log_prob` / JSON dump helpers are
*not* ported in the CU variant — only what Convolutional Cobweb needs.

### Constructor differences

```python
# Default
CobwebContinuousTree(size, num_labels, covar_type=1, covar_from=2,
                     alpha=0.01, prior_var=0.0585..., insert_only=False,
                     depth_max=999999, branch_max=999999)

# CU
CUCobwebContinuousTree(size, num_labels=0, scaling=0.5,
                       insert_only=False,
                       depth_max=999999, branch_max=999999)
```

The CU tree drops `covar_type`, `covar_from`, `alpha`, and `prior_var`
(none apply to ECG) and adds `scaling`. `num_labels` is accepted for API
parity but ignored — the CU variant is unsupervised (Convolutional
Cobweb's filter hierarchy doesn't carry labels).

---

## 3. Descent (categorize)

Both default classes and both CU classes ship a Bayesian descent helper
that walks root-to-leaf by `log_prob_class_given_instance`:

```python
tree.categorize(instance)      # discrete
tree.get_leaf(instance, labels)  # continuous
```

Behaviour is identical between default and CU variants on the same tree
state (the formula doesn't depend on the heuristic; only the tree
structure does, and that's the upstream divergence).

Both CU classes additionally expose a **Fisher-CU descent**:

```python
tree.categorize_with_cu(instance)        # discrete
tree.categorize_with_cu(instance, labels)  # continuous
```

This walks root-to-leaf picking the child returned by
`two_best_children` (highest partition utility for inserting the
instance) at every step — matching
`concept_formation._cobweb_categorize`. This descent is *not* available
on the default classes. (`CobwebDiscreteTree` got a `categorize_with_cu`
during the same set of changes, but it's the only default class that
has it; `CobwebContinuousTree` does not.)

---

## 4. Smoothing / scaling constants

| Knob | Default discrete | CU discrete | Default continuous | CU continuous |
| --- | --- | --- | --- | --- |
| Laplace α | `α=1.0` | `α=1.0` | n/a | n/a |
| Variance floor | n/a | n/a | `prior_var = 1/(2πe)` | `1/(4π)` (built in) |
| Online attr scaling | n/a | n/a | none | `scaling = 0.5` |
| `weight_attr` ratio | yes (default true) | yes (default true) | n/a | n/a |
| Random tie-break in `two_best_children` | yes (`custom_rand`) | yes | yes | no (deterministic by `(pu, count)`) |

The CU continuous tree's variance floor of `1/(4π) ≈ 0.0796` is baked
into the ECG formula (`σ² += 1/(4π)`); it isn't user-tunable. The
default tree's `prior_var` is exposed via the constructor and defaults
to `1/(2πe) ≈ 0.0585`.

---

## 5. Quick-reference summary

```
                    DEFAULT                          CU variant
Discrete tree
  Heuristic         Laplace-smoothed entropy CU      Fisher ECG (Σ p²)
  ifit              4-op                             4-op (same shape)
  categorize        Bayesian (log p(c|x))            Bayesian (log p(c|x))
  *_with_cu         exists (added)                   exists
  value_remap       yes                              yes
  predict_probs     Laplace-smoothed                 Laplace-smoothed
  Extras            predict_pmi, parallel, dump_json (CU drops these)

Continuous tree
  Heuristic         info-CU / Mahalanobis (covar_from)  Fisher ECG continuous
  Online scaling    none                                scaling = 0.5
  ifit              4-op                                4-op (same shape)
  get_leaf          Bayesian                            Bayesian
  categorize_with_cu n/a                                exists
  Labels            num_labels-dim Eigen vec            num_labels accepted, ignored
  Extras            predict, predict_pmi, log_prob, dump_json (CU drops these)
```

---

## 6. When to use which

* If you need to replicate **concept_formation**'s tree-building decisions
  (e.g. Cobweb-3, Convolutional Cobweb from MacLellan & Thakur 2021,
  Trestle-style models), use the CU variants and pair them with
  `categorize_with_cu` for descent.
* If you want the modern, numerically-cheap entropy-CU formulation
  (often produces wider top-level partitions, easier to scale), use the
  default classes.
* Avoid mixing the two within one hierarchy. The classification tree and
  filter hierarchy in [conv-cobweb/src/conv_cobweb.py](../conv-cobweb/src/conv_cobweb.py)
  show both consistent options: `ConvolutionalCobweb` uses default
  classes throughout; `ConvolutionalCobwebECG` uses CU classes
  throughout.
