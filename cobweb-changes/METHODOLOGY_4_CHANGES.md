# Methodology 4.0 — Leaf-Pointer Encoding & LCA Soft Similarity

Content instances encode **2 leaf-pointer attributes** (`{0: left_leaf_vid, 1: right_leaf_vid}`) instead of the old `2 × content_path_depth` depth-indexed path. Similarity between values comes from LCA depth in a **reference hierarchy** (ref_tree). All depth-shift propagation and `chunk_context` mode are eliminated.

---

## Core Concepts

### Reference Tree & Ref Attrs

A `CobwebDiscreteTree` can hold a pointer to another tree (`ref_tree`) whose hierarchy defines value similarity. Attributes marked as **ref attrs** use soft LCA matching instead of exact matching. In WEBSTER: content hierarchy's `ref_tree` = context hierarchy; attrs `0` (left) and `1` (right) are ref attrs.

### LCA Similarity

$$\text{sim}(v_1, v_2) = \frac{\text{depth}(\text{LCA}(v_1, v_2))}{\max\text{\_leaf\_depth}(\text{ref\_tree})}$$

Siblings share a deep LCA → high similarity. Unrelated values share only root → sim ≈ 0.

### Two Soft-Matching Mechanisms

| Used by | Mechanism | Cost |
|---------|-----------|------|
| `log_prob_instance` (categorization, EPMI, generation) | Pairwise LCA: `soft_count = α + Σ(stored_count × sim(query, stored))` | $O(V)$ per query |
| Entropy / Partition Utility (MERGE/SPLIT/NEW) | Multi-resolution coarsening: values grouped by ancestor at each depth $d$, entropies combined with $w(d)=d/D$ | $O(\text{depth})$ cached |

Non-ref attrs are completely unchanged (exact-match everywhere).

---

## Multi-Resolution Entropy

Values are grouped by ancestor at each depth level of the ref_tree:

$$H_{\text{combined}} = \frac{\sum_{d=1}^{D} \frac{d}{D} \cdot H_d}{\sum_{d=1}^{D} \frac{d}{D}}$$

Depth 0 excluded (one bucket → zero entropy). Deeper levels get higher weight. Each $H_d$ uses the standard `entropy_attr` formula on depth-$d$ bucket counts.

### Caching

- **`structure_generation`** (`uint64_t` on tree): bumped on every MERGE/SPLIT/NEW/fringe-split
- **`val_ancestors`**: lazily rebuilt via `ensure_val_ancestors()` when generation is stale
- **`RefAttrBuckets`** per node per ref attr: `bucket_counts[d][ancestor_ptr]` + cached `bucket_sum_n_logn[d]`; maintained incrementally in `increment_counts`, `update_counts_from_node`, `remove_counts_from_node` — $O(\text{depth})$ per value

---

## C++ Changes (`cobweb-private/`)

### `cobweb_discrete_types.h`
- `using REF_ATTR_SET = std::unordered_set<ATTR_TYPE>;`

### `cobweb_discrete_tree.h` — new members & methods

| Member | Type | Purpose |
|--------|------|---------|
| `ref_tree` | `CobwebDiscreteTree*` | Reference hierarchy (nullptr if none) |
| `ref_attrs` | `REF_ATTR_SET` | Attrs using soft LCA matching |
| `concept_map` | `map<string, Node*>` | Hash→node index for fast lookup |
| `val_to_node` | `map<VALUE_TYPE, Node*>` | VID→ref_tree leaf for LCA |
| `ref_tree_max_depth` | `int` | Cached max depth (-1 = stale) |
| `structure_generation` | `uint64_t` | Bumped on structural changes |
| `val_ancestors` | `map<VALUE_TYPE, vector<Node*>>` | Cached ancestor paths |
| `num_buckets_per_depth` | `vector<int>` | Distinct ancestors per depth |

Methods: `set_ref_attr`, `lca_similarity`, `rebuild_concept_map`, `register_ref_val`, `invalidate_ref_cache`, `bump_generation`, `ensure_val_ancestors`, `rebuild_all_ref_buckets`

### `cobweb_discrete_node.h` — new struct & methods

```cpp
struct RefAttrBuckets {
    vector<unordered_map<CobwebDiscreteNode*, COUNT_TYPE>> bucket_counts;
    vector<double> bucket_sum_n_logn;
};
```

Member: `ref_attr_buckets` per node.  
Methods: `rebuild_ref_buckets`, `entropy_attr_ref`, `entropy_attr_ref_insert`, `entropy_attr_ref_merge`

### `cobweb_discrete_node.cpp` — scoring changes

- **`log_prob_instance`**: ref attrs use pairwise LCA soft-count; non-ref attrs unchanged
- **`entropy_attr` / `_insert` / `_merge`**: dispatch to `_ref` variant when `ref_tree && ref_attrs.count(attr)`
- **`increment_counts` / `update_counts_from_node` / `remove_counts_from_node`**: incremental bucket maintenance for ref attrs
- **`set_av_count`**: calls `rebuild_ref_buckets()` at end
- **`compute_counts_from_children`**: clears `ref_attr_buckets` before recomputing

### `cobweb_discrete_tree.cpp`

- Constructor accepts optional `ref_tree`
- `cobweb()`: calls `bump_generation()` after structural changes
- `ensure_val_ancestors()`: lazy rebuild of ancestor paths, `num_buckets_per_depth`, and all node `RefAttrBuckets`

### `cobweb_discrete.cpp` — bindings

Constructor: `ref_tree=nullptr`. Exposed all new methods/properties listed above.

---

## Python Changes (`src/parse_mh.py`)

- **~600 lines deleted**: `_build_label_path_from_ctx`, `_build_label_path_from_bfs`, `_build_chunk_context_instance`, all depth-snapshot & depth-shift propagation functions
- **New**: `_build_label_from_ctx_leaf` returns single leaf VID
- **`PrimitiveParseNode` / `CompositeParseNode`**: `label_path` changed from `list` to `int`; `create_content_instance` returns `{0: {left.label_path: 1}, 1: {right.label_path: 1}}`
- **`LongTermMemory`**: removed `content_path_depth`, `chunk_context`; context tree passed as `ref_tree` to content tree; `set_ref_attr(0)`, `set_ref_attr(1)` called
- **`add_parse_tree`**: step 2 calls `register_ref_val`; step 3 calls `invalidate_ref_cache()`; depth-shift propagation replaced by `_apply_rewrite_rules`
- **`WEBSTER`**: removed `content_length`, `chunk_context` params; generation simplified to read attrs 0/1 directly

---

## Performance

| Operation | Non-ref attr | Ref attr |
|-----------|-------------|----------|
| `increment_counts` / value | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr` | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr_insert` | $O(v)$ | $O(v \times \text{depth})$ |
| Full bucket rebuild | N/A | $O(N \times V \times \text{depth})$ |

`depth` = `ref_tree_max_depth` (typically 5–15). Zero overhead for non-ref attrs.

---

## Key Design Decisions

1. **LCA formula**: `depth(LCA) / max_depth` — globally normalized, lazily cached
2. **Two soft-matching mechanisms**: pairwise LCA for `log_prob_instance` (smooth per-query), hierarchical coarsening for entropy/PU (cached per-node)
3. **Generation counter**: avoids per-call ancestor rebuilds; ref_tree only changes during context fitting (step 1), stable by content fitting (step 3)
4. **Incremental bucket maintenance**: $O(\text{depth})$ per value in existing count-update functions
5. **Leaf stability**: ref_tree leaf nodes must never be deleted (`val_to_node` pointers depend on them)
6. **`register_ref_val`** wired at: `build_primitives`, `apply_candidate`, `_refresh_labels_bottom_up`
