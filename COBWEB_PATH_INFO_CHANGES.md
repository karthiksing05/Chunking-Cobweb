# Cobweb C++ Changes for Methodology 4.0 — PATH INFORMATION

This document summarizes the C++ changes made to the `cobweb-private/` library to support **leaf-pointer encoding** and **LCA-based soft similarity** as part of Methodology 4.0. For full system-level details (including Python changes), see `METHODOLOGY_4_CHANGES.md`.

---

## Motivation

Previously, content instances encoded multi-attribute depth-indexed paths (e.g., 6 attributes for `content_path_depth=3`). Matching was exact per-value with no structural similarity. Methodology 4.0 replaces this with:

- **2-attribute leaf pointers**: `{0: left_leaf_vid, 1: right_leaf_vid}`
- **LCA-based soft similarity**: values sharing a deep common ancestor in a reference hierarchy are treated as more similar
- **Multi-resolution entropy**: tree-structuring decisions (MERGE/SPLIT/NEW) account for hierarchical similarity

---

## New Concepts

### Reference Tree (`ref_tree`)

A `CobwebDiscreteTree` can now hold a pointer to another tree (the **ref_tree**) whose hierarchy defines similarity between values. In WEBSTER, the content hierarchy's ref_tree is the context hierarchy.

### Ref Attrs (`ref_attrs`)

A set of attribute indices that should use soft LCA matching instead of exact matching. In WEBSTER, attributes `0` (left child) and `1` (right child) are ref attrs.

### LCA Similarity

For two values referencing ref_tree leaf concepts:

$$\text{similarity}(v_1, v_2) = \frac{\text{depth}(\text{LCA}(v_1, v_2))}{\text{max\_leaf\_depth}(\text{ref\_tree})}$$

- Sibling concepts share a deep LCA → high similarity
- Unrelated concepts share only the root → similarity ≈ 0

### Multi-Resolution Entropy

For ref attrs, entropy is computed at multiple depth levels of the ref_tree. At each depth $d$, values are grouped by their ancestor at depth $d$ (coarsening the value space). Per-depth entropies are combined with depth-proportional weights:

$$H_{\text{combined}} = \frac{\sum_{d=1}^{D} \frac{d}{D} \cdot H_d}{\sum_{d=1}^{D} \frac{d}{D}}$$

This gives partial credit for coarse-grained clustering (e.g., separating verbs from nouns) even when fine-grained leaf separation isn't perfect.

---

## File-by-File Changes

### `include/cobweb_discrete_types.h`

- Added `using REF_ATTR_SET = std::unordered_set<ATTR_TYPE>;`

### `include/cobweb_discrete_tree.h`

New members:

| Member | Type | Purpose |
|--------|------|---------|
| `ref_tree` | `CobwebDiscreteTree*` | Pointer to the reference hierarchy (`nullptr` if none) |
| `ref_attrs` | `REF_ATTR_SET` | Attribute indices using soft LCA matching |
| `concept_map` | `unordered_map<string, CobwebDiscreteNode*>` | Hash→node index for fast lookup |
| `val_to_node` | `unordered_map<VALUE_TYPE, CobwebDiscreteNode*>` | Vocab-ID → ref_tree leaf node for LCA queries |
| `ref_tree_max_depth` | `int` | Cached max depth of ref_tree (-1 = stale) |
| `structure_generation` | `uint64_t` | Bumped on every structural change (MERGE/SPLIT/NEW) |
| `val_ancestors` | `unordered_map<VALUE_TYPE, vector<CobwebDiscreteNode*>>` | Cached ancestor paths for registered values |
| `val_ancestors_generation` | `uint64_t` | Tracks when `val_ancestors` was last rebuilt |
| `num_buckets_per_depth` | `vector<int>` | Distinct ancestor count at each depth level |

New methods:

| Method | Purpose |
|--------|---------|
| `set_ref_attr(attr)` | Mark an attribute index for soft matching |
| `lca_similarity(val1, val2)` | Compute LCA-based similarity between two values |
| `rebuild_concept_map()` | BFS walk of ref_tree to populate `concept_map` |
| `register_ref_val(val, node)` | Store val→node mapping for LCA lookup |
| `invalidate_ref_cache()` | Reset `ref_tree_max_depth` to -1 |
| `bump_generation()` | Increment `structure_generation` |
| `ensure_val_ancestors()` | Lazily rebuild ancestor paths if generation is stale |
| `rebuild_all_ref_buckets()` | BFS walk to rebuild all nodes' `RefAttrBuckets` |

Constructor updated to accept optional `ref_tree` parameter.

### `include/cobweb_discrete_node.h`

New struct:

```cpp
struct RefAttrBuckets {
    // bucket_counts[d][ancestor_ptr] = aggregated count at depth d
    vector<unordered_map<CobwebDiscreteNode*, COUNT_TYPE>> bucket_counts;
    // cached sum of (count+α)*log(count+α) for non-zero buckets at depth d
    vector<double> bucket_sum_n_logn;
};
```

New member: `ref_attr_buckets` — `unordered_map<ATTR_TYPE, RefAttrBuckets>`

New methods:

| Method | Purpose |
|--------|---------|
| `rebuild_ref_buckets()` | Full rebuild of bucket caches from `av_count` |
| `entropy_attr_ref(attr)` | Multi-resolution entropy for a ref attr |
| `entropy_attr_ref_insert(attr, ...)` | Hypothetical entropy after inserting an instance |
| `entropy_attr_ref_merge(attr, ...)` | Hypothetical entropy after merging with another node |

### `src/cobweb_discrete_tree.cpp`

- **Constructor**: Stores `ref_tree`, initializes `ref_tree_max_depth = -1`
- **`set_ref_attr`**: Adds attr to `ref_attrs`
- **`rebuild_concept_map`**: BFS walk of ref_tree, populates `concept_map`
- **`lca_similarity`**: Looks up both values via `val_to_node`, walks ancestor chains to find LCA, returns `depth(LCA) / max_depth`
- **`register_ref_val`**: Stores vocab-ID → node mapping
- **`invalidate_ref_cache`**: Resets cached max depth
- **`bump_generation`**: Increments `structure_generation`
- **`ensure_val_ancestors`**: On generation mismatch, rebuilds all ancestor paths, `num_buckets_per_depth`, and calls `rebuild_all_ref_buckets()`
- **`rebuild_all_ref_buckets`**: BFS over all nodes, calls `rebuild_ref_buckets()` on each
- **`cobweb`**: Calls `bump_generation()` after every MERGE, SPLIT, NEW, fringe split, and depth-max forced leaf

### `src/cobweb_discrete_node.cpp`

#### `log_prob_instance` — Soft matching for ref attrs

For non-ref attrs: unchanged exact-match lookup.

For ref attrs:
```
soft_count = α + Σ(stored_count × lca_similarity(query_val, stored_val))
P(val | node) = soft_count / (a_count + (num_vals + 1) × α)
```

#### `entropy_attr` / `entropy_attr_insert` / `entropy_attr_merge` — Dispatch

At entry, checks `if (tree->ref_tree != nullptr && tree->ref_attrs.count(attr))` and dispatches to the `_ref` variant. Non-ref attrs and trees without a ref_tree are completely unchanged.

#### Incremental bucket maintenance

Three existing functions updated to maintain `RefAttrBuckets` for ref attrs:

| Function | Ref attr behavior |
|----------|------------------|
| `increment_counts` | Updates per-depth bucket counts and `bucket_sum_n_logn` |
| `update_counts_from_node` | Merges another node's values into bucket caches |
| `remove_counts_from_node` | Subtracts values, cleans up zero-count buckets |

Each is $O(\text{depth})$ per value for ref attrs vs $O(1)$ for non-ref attrs.

#### Other updated functions

- **`set_av_count`**: Calls `rebuild_ref_buckets()` at end
- **`compute_counts_from_children`**: Clears `ref_attr_buckets` before recomputing

### `src/cobweb_discrete.cpp` — Python Bindings

- Constructor: added `ref_tree=nullptr` parameter
- New method bindings: `set_ref_attr`, `lca_similarity`, `rebuild_concept_map`, `register_ref_val`, `invalidate_ref_cache`, `bump_generation`, `ensure_val_ancestors`, `rebuild_all_ref_buckets`
- New read-only properties: `ref_tree`, `val_to_node`, `structure_generation`

---

## Scoring Summary

| Scoring Function | Non-ref attrs | Ref attrs |
|-----------------|---------------|-----------|
| `log_prob_instance` | Exact-match lookup | Pairwise LCA soft matching |
| `entropy_attr` | Standard exact-match entropy | Multi-resolution hierarchical coarsening |
| `partition_utility` | Uses standard entropy | Uses multi-resolution entropy for ref attrs |

---

## Performance

| Operation | Non-ref attr | Ref attr |
|-----------|-------------|----------|
| `increment_counts` per value | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr` | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr_insert` | $O(v_{\text{instance}})$ | $O(v_{\text{instance}} \times \text{depth})$ |
| `entropy_attr_merge` | $O(v_{\text{other}} + v_{\text{instance}})$ | $O((v_{\text{other}} + v_{\text{instance}}) \times \text{depth})$ |
| Full bucket rebuild (on generation change) | N/A | $O(N \times V \times \text{depth})$ |

Where `depth` is `ref_tree_max_depth` (typically 5–15), $N$ is tree node count, $V$ is values per node.

---

## Key Design Decisions

1. **LCA similarity formula**: `depth(LCA) / max_depth` — globally normalized, lazily cached
2. **Generation counter caching**: `structure_generation` avoids per-call ancestor path checks; `ensure_val_ancestors()` rebuilds lazily only when stale
3. **Incremental bucket maintenance**: $O(\text{depth})$ per value within existing `increment_counts`/`update_counts_from_node`/`remove_counts_from_node`
4. **Leaf stability**: Ref_tree leaf nodes must never be deleted — the `val_to_node` pointers depend on them
5. **Cache invalidation**: `invalidate_ref_cache()` called once at the start of content fitting, after all context hierarchy structural changes are complete
6. **Zero overhead for non-ref attrs**: All dispatch is gated on `ref_attrs.count(attr)` — existing behavior is completely unchanged for standard attributes
