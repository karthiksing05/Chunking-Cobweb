# Methodology 4.0 — PATH INFORMATION Change Summary

This document summarizes all changes made to implement the **Methodology 4.0 PATH INFORMATION** change described in `MULTIHIERARCHY.md`. The core change replaces multi-attribute depth-indexed path encoding with single leaf pointers and LCA-based soft similarity.

---

## Overview

### Before (Old Encoding)
- Content instances had `2 × content_path_depth` attributes (e.g., 6 attrs for `cpd=3`)
- Each side (left/right) stored concept IDs at every depth level from root to leaf
- Matching was exact per-value — no structural similarity between different concept values
- Depth-shift propagation maintained consistency when Cobweb restructured hierarchies
- `chunk_context` mode allowed context instances built from label paths of neighboring chunks

### After (New Encoding — Methodology 4.0)
- Content instances have exactly **2 attributes**: `{0: {left_leaf_vid: 1}, 1: {right_leaf_vid: 1}}`
- Each attribute stores a single pointer to the **leaf-level** context-hierarchy concept
- **LCA-based soft similarity** in `log_prob_instance`: values that share a deep common ancestor in the context hierarchy (the ref_tree) are treated as more similar
- Formula: `similarity = LCA_depth / max_depth_in_ref_tree`
- All depth-shift propagation eliminated — no longer needed
- `chunk_context` mode removed entirely

---

## C++ Changes (`cobweb-private/`)

### `include/cobweb_discrete_types.h`
- Added `using REF_ATTR_SET = std::unordered_set<ATTR_TYPE>;` — type alias for the set of attribute indices that use reference-tree soft matching

### `include/cobweb_discrete_tree.h`
- **New members:**
  - `CobwebDiscreteTree *ref_tree` — pointer to the reference hierarchy (context tree for content, `nullptr` for context)
  - `REF_ATTR_SET ref_attrs` — which attribute indices use soft LCA matching
  - `std::unordered_map<std::string, CobwebDiscreteNode*> concept_map` — hash→node index for fast lookup
  - `std::unordered_map<VALUE_TYPE, CobwebDiscreteNode*> val_to_node` — vocab-ID→ref-tree-node for LCA queries
  - `int ref_tree_max_depth` — cached max depth of ref_tree (-1 = invalid)
- **New method declarations:** `set_ref_attr`, `lca_similarity`, `rebuild_concept_map`, `register_ref_val`, `invalidate_ref_cache`
- **Constructor** updated to accept optional `ref_tree` parameter

### `src/cobweb_discrete_tree.cpp`
- **Constructor**: Accepts and stores `ref_tree`, initializes `ref_tree_max_depth = -1`
- **`set_ref_attr(attr)`**: Adds an attribute index to `ref_attrs`
- **`rebuild_concept_map()`**: Full BFS walk of ref_tree, populates `concept_map`
- **`lca_similarity(val1, val2)`**: Looks up both values in `val_to_node`, walks ancestor chains to find LCA, returns `LCA_depth / max_depth`. Max depth is lazily cached.
- **`register_ref_val(val, node)`**: Stores `val→node` in `val_to_node` for fast LCA lookup
- **`invalidate_ref_cache()`**: Resets `ref_tree_max_depth = -1` so it's recomputed on next use

### `src/cobweb_discrete_node.cpp` — `log_prob_instance`
- For attributes in `ref_attrs`: instead of exact value lookup, computes:
  ```
  soft_count = eff_alpha + Σ(stored_count × lca_similarity(query_val, stored_val))
  ```
  Falls back to exact match when `lca_similarity` returns 0 (unrelated values)

### `src/cobweb_discrete.cpp` — Python bindings
- Constructor binding: added `ref_tree=nullptr` parameter
- New method bindings: `set_ref_attr`, `lca_similarity`, `rebuild_concept_map`, `register_ref_val`, `invalidate_ref_cache`
- New read-only properties: `ref_tree`, `val_to_node`

---

## Python Changes (`src/parse_mh.py`)

### Deleted Functions (~600+ lines removed)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_build_label_path_from_ctx` | ~60 | Built multi-depth label path from DFS categorization |
| `_build_label_path_from_bfs` | ~40 | Built multi-depth label path from BFS categorization |
| `_build_chunk_context_instance` | ~171 | Built context instances from neighboring chunk label paths |
| `_get_content_hierarchy_depths` | ~30 | Snapshot content hierarchy concept depths |
| `_get_context_hierarchy_depths` | ~30 | Snapshot context hierarchy concept depths |
| `_apply_split_deletes_to_context` | ~60 | Propagated content splits to context (delete-only) |
| `_apply_content_depth_shifts_to_context` | ~100 | Propagated content depth changes to context refs |
| `_apply_context_depth_shifts` | ~100 | Propagated context depth changes to content attrs |
| `_apply_split_deletes_to_content` | ~60 | Propagated context splits to content (delete-only) |

### New Function
- **`_build_label_from_ctx_leaf(ctx_leaf, value_to_id) → int`**: Returns the single context-hierarchy leaf concept vocab ID (replaces the two multi-depth helpers)

### Data Structure Changes

#### `PrimitiveParseNode`
- `label_path`: Changed from `list` to `int = 0` (single leaf concept VID)

#### `CompositeParseNode`
- `label_path`: Changed from `list` to `int = 0` (single leaf concept VID)
- `create_content_instance(left, right)`: Now returns `{0: {left.label_path: 1}, 1: {right.label_path: 1}}` — just 2 attributes

### `LongTermMemory.__init__`
- **Removed parameters:** `content_path_depth`, `chunk_context`
- **Removed attributes:** `self.content_path_depth`, `self.chunk_context`
- Context hierarchy is created first and passed as `ref_tree` to the content hierarchy constructor
- Content hierarchy: `set_ref_attr(0)` and `set_ref_attr(1)` called to mark left/right attrs for soft matching
- `content_headers` simplified from dynamic depth-based list to `["Left", "Right"]`
- Chunk-context drawer branches removed

### `content_ref_attr` property
- Simplified: always returns `2 if bow else 2 * context_length` (no chunk_context branch)

### `add_parse_tree` (the main learning pipeline)
- **Step 1 (context fitting):** Removed chunk_context branching (`_build_chunk_context_instance` call), removed `_pre_ctx_depths` snapshot and `_apply_context_depth_shifts` call, removed `_apply_split_deletes_to_content` call. Now uses `_apply_rewrite_rules` for split propagation.
- **Step 2 (label refresh):** Uses `_build_label_from_ctx_leaf` instead of depth-based helpers. Calls `register_ref_val` to wire VID→context-leaf mapping for LCA.
- **Step 3 (content fitting):** Calls `invalidate_ref_cache()` at start. Removed `_pre_cnt_depths` snapshot and `_apply_content_depth_shifts_to_context` call. Uses `_apply_rewrite_rules` for split propagation.
- **Step 4 (content-ref writing):** Simplified from weighted ancestor path to single pointer: `ctx_leaf.increment_attr_value(_cref_attr, vid, 1)`.

### `LongTermMemory.save_state` / `load_state`
- Removed `content_path_depth` and `chunk_context` from meta dict
- Updated constructor call in `load_state` to match new signature

### `WEBSTER.__init__`
- **Removed parameters:** `content_length`, `chunk_context`
- **Removed attributes:** `self.content_length`, `self.chunk_context`
- Updated `LongTermMemory` constructor call to new signature

### `WEBSTER.save_state` / `load_state`
- Removed `content_length` and `chunk_context` from meta dict
- Updated attribute restoration in `load_state`

### `WEBSTER.generate_sentence`
- Removed `_cpd = self.ltm.content_path_depth`
- Removed `_deepest_attr` helper (no longer needed with 2-attr layout)
- `_expand` now reads attrs `0` (left) and `1` (right) directly instead of scanning depth-indexed attrs

### `FiniteParseTree._draw_node_to_dict`
- Composite content visualization: reads attrs `0` ("Left") and `1` ("Right") instead of iterating `range(_cpd)` depth-indexed attrs

### Module docstring
- Updated to describe leaf pointer encoding instead of multi-attribute path encoding

---

## Test/Demo File Updates

| File | Changes |
|------|---------|
| `unittests/gen_learn_test_mh.py` | Removed `CONTENT_LENGTH`, `content_length`, `chunk_context` params |
| `unittests/primitives_only_test_mh.py` | Removed `CONTENT_LENGTH`, `content_length`, `chunk_context` params |
| `gui/parse_tree_editor_mh.py` | Removed `CONTENT_LENGTH`, `content_length`, `chunk_context` params |

---

## How Scoring Works With LCA Soft Similarity

### Background: Standard CobwebDiscrete Scoring

In standard Cobweb (without ref_tree), all scoring is **exact-match**. A value either matches or it doesn't. The key scoring functions are:

| Function | What it computes | Used for |
|----------|-----------------|----------|
| `log_prob_instance` | Log-probability of an instance under a node | Categorization (DFS/BFS), basic-level evaluation (EPMI) |
| `entropy` / `entropy_attr` | Shannon entropy of a node's value distributions | Partition Utility (PU) — drives MERGE/SPLIT/NEW decisions |
| `partition_utility` | `(parent_entropy − weighted_children_entropy) / num_children` | Choosing the best Cobweb operation during `ifit` |

### What Changes: `log_prob_instance` + Entropy/PU for Ref Attrs

**`log_prob_instance` uses pairwise LCA soft matching** for ref attrs — a query value gets partial credit from structurally similar stored values.

**Entropy and PU use multi-resolution hierarchical coarsening** for ref attrs — values are grouped by their ancestor at each depth level of the ref_tree, and per-depth entropies are combined with depth-proportional weights. This means tree structure decisions (MERGE, SPLIT, NEW) are now sensitive to hierarchical similarity for ref attrs. See the "Multi-Resolution Entropy" section below for full details.

**Non-ref attributes** use standard exact-match counting everywhere (completely unchanged).

### How `log_prob_instance` Computes Similarity

For each attribute in the query instance:

#### Non-ref attributes (standard path)
```
P(val | node) = (av_count[attr][val] + α) / (a_count[attr] + (num_vals + 1) × α)

log_prob += count × log(P(val | node))
```
Exact lookup: `av_count[attr][val]` is the raw count of that specific value under this node. If the value hasn't been seen at this node, only the smoothing term `α` contributes.

#### Ref attributes (soft matching path — NEW)
```
soft_count = α + Σ_{stored_val} (av_count[attr][stored_val] × lca_similarity(query_val, stored_val))

P(val | node) = soft_count / (a_count[attr] + (num_vals + 1) × α)

log_prob += count × log(P(val | node))
```
Instead of looking up a single exact value, we **iterate over all values stored at this attribute in this node** and weight each stored count by its LCA similarity to the query value.

### How `lca_similarity(v1, v2)` Works

Given two vocab-ID values that reference context-hierarchy concepts:

1. **Lookup**: Map each value to its context-hierarchy node via `val_to_node[v]`. If either value isn't a registered ref pointer (e.g., it's a word ID for a primitive), return `0.0`.

2. **Find LCA**: Walk ancestors of `node1` into a set. Walk `node2` upward until hitting a node in that set — this is the Lowest Common Ancestor.

3. **Compute similarity**:
   ```
   similarity = depth(LCA) / max_leaf_depth(ref_tree)
   ```
   - `depth(LCA)` = how many edges from root to the LCA (0 = root itself)
   - `max_leaf_depth(ref_tree)` = depth of the deepest leaf in the entire context hierarchy (cached, lazily computed)

4. **Edge cases**:
   - Same node: LCA = the node itself → `similarity = node_depth / max_depth` (high, close to 1.0 for deep leaves)
   - Siblings: LCA = their shared parent → `similarity = parent_depth / max_depth`
   - Unrelated (different root — shouldn't happen): `similarity = 0.0`
   - Root as LCA: `similarity = 0 / max_depth = 0.0` (completely unrelated in the hierarchy)

### Concrete Example

Suppose the context hierarchy looks like:
```
         ROOT (depth 0)
        /           \
    VERB (depth 1)   NOUN (depth 1)
    /    \           /    \
  RAN    SAT      CAT    DOG
  (d=2)  (d=2)   (d=2)  (d=2)
```
Max depth = 2.

Content instance query: `{0: {CAT_vid: 1}, 1: {RAN_vid: 1}}`

At a content node that has stored `{0: {DOG_vid: 3, SAT_vid: 1}}`:

For attr 0 (left child, a ref attr):
- `lca_similarity(CAT_vid, DOG_vid)` → LCA is NOUN (depth 1) → `1/2 = 0.5`
- `lca_similarity(CAT_vid, SAT_vid)` → LCA is ROOT (depth 0) → `0/2 = 0.0`
- `soft_count = α + (3 × 0.5) + (1 × 0.0) = α + 1.5`

So CAT gets partial credit from DOG (they're both nouns), but zero credit from SAT (a verb — only related at the root).

### Why Two Different Soft-Matching Approaches?

`log_prob_instance` and entropy/PU use *different* mechanisms for incorporating path information, each suited to its purpose:

- **`log_prob_instance` (pairwise LCA)**: For a query value, iterates over all stored values and sums `count × lca_similarity(query, stored)`. This gives a smooth, continuous probability for any query value. Used for categorization, EPMI, and generation — tasks that need to score how well a *specific* observation fits a concept.

- **Entropy/PU (hierarchical coarsening)**: Groups values by their ancestor at each depth level and computes entropy over the resulting buckets. This measures *how uncertain* a concept is about what it will see, across multiple granularities. Used for MERGE/SPLIT/NEW decisions — tasks that need to evaluate whether a partition *reduces uncertainty*.

Both approaches respect the same ref_tree structure and produce results where structurally similar values are treated as more alike. The difference is computational: pairwise LCA is $O(V)$ per query (affordable for single-instance scoring), while hierarchical coarsening uses cached bucket counts for $O(\text{depth})$ entropy computation (necessary for the many entropy evaluations per `ifit` call).

---

## Multi-Resolution Entropy for Ref Attrs (Hierarchical Coarsening)

The initial Methodology 4.0 implementation only applied soft matching in `log_prob_instance` (categorization), while entropy and partition utility (PU) — which drive Cobweb's structural decisions (MERGE/SPLIT/NEW) — used exact-match counting. This meant two content values pointing to sibling context-hierarchy concepts were treated as completely unrelated for tree-building purposes.

Multi-resolution entropy extends path awareness into the entropy/PU calculations so that structurally similar values reduce measured uncertainty even when they aren't identical.

### Core Idea: Hierarchical Coarsening

Instead of computing entropy over raw values (where every distinct leaf pointer is its own category), we compute entropy at multiple **depth levels** of the reference hierarchy. At each depth $d$, values are grouped by their **ancestor at depth $d$** — this "coarsens" the value space.

```
Context hierarchy:
         ROOT (depth 0)         ← 1 bucket (everyone)
        /           \
    VERB (depth 1)   NOUN (d=1) ← 2 buckets
    /    \           /    \
  RAN    SAT      CAT    DOG   ← 4 buckets (exact match)
  (d=2)  (d=2)   (d=2)  (d=2)
```

At depth 0: all values collapse into one bucket → entropy = 0
At depth 1: values split into VERB-bucket and NOUN-bucket → measures verb/noun distinction
At depth 2: each leaf is its own bucket → exact-match entropy (same as standard Cobweb)

### Weighted Combination

The per-depth entropies are combined with depth-proportional weights:

$$H_{\text{combined}} = \frac{\sum_{d=1}^{D} w(d) \cdot H_d}{\sum_{d=1}^{D} w(d)}, \qquad w(d) = \frac{d}{D}$$

where $D$ is the max depth of the ref_tree and $H_d$ is the standard entropy formula applied to the depth-$d$ bucket counts. Depth 0 is excluded (always one bucket → zero entropy). Deeper levels get higher weight, so fine-grained distinctions matter more than coarse ones.

Each $H_d$ uses the same formula as standard `entropy_attr`:

$$H_d = -r \cdot \left(\frac{1}{N_d} \cdot \left(\sum_b (n_b + \alpha) \log(n_b + \alpha) + n_0 \cdot \alpha \log \alpha \right) - \log N_d\right)$$

where $n_b$ is the bucket count at depth $d$, $n_0$ is the number of zero-count buckets, and $N_d = \text{attr\_count} + |\text{buckets}_d| \cdot \alpha$.

### Generation Counter and Lazy Caching

The ref_tree (context hierarchy) changes structure during step 1 of `add_parse_tree`, which invalidates all ancestor paths. A generation counter avoids expensive per-call checks:

1. **`structure_generation`** on `CobwebDiscreteTree`: a `uint64_t` incremented every time `cobweb()` performs a structural change (MERGE, SPLIT, NEW, fringe split, depth-max leaf).
2. **`val_ancestors`**: maps each registered `VALUE_TYPE` to a vector of ancestor pointers `[root, ..., leaf]` in the ref_tree.
3. **`val_ancestors_generation`**: tracks which generation `val_ancestors` was last rebuilt at.
4. **`ensure_val_ancestors()`**: called lazily at the start of any ref entropy computation. If `val_ancestors_generation != ref_tree->structure_generation`, it rebuilds:
   - All ancestor paths for all registered values
   - `num_buckets_per_depth[d]` — how many distinct ancestor nodes exist at each depth
   - Every node's `RefAttrBuckets` via `rebuild_all_ref_buckets()`

In the WEBSTER pipeline, the ref_tree only changes during step 1 (context fitting). By step 3 (content fitting), the generation is stable, so all entropy calls hit the cache with zero rebuild cost.

### Per-Node Bucket Cache: `RefAttrBuckets`

Each content-hierarchy node maintains, for each ref attr, a `RefAttrBuckets` struct:

```cpp
struct RefAttrBuckets {
    // bucket_counts[d][ancestor_ptr] = aggregated count at depth d
    vector<unordered_map<CobwebDiscreteNode*, COUNT_TYPE>> bucket_counts;
    // cached sum of (count+α)*log(count+α) for non-zero buckets at depth d
    vector<double> bucket_sum_n_logn;
};
```

This mirrors the existing `sum_n_logn` cache pattern but at each depth level. The `bucket_sum_n_logn[d]` cache enables $O(1)$ per-depth entropy evaluation — the same trick standard `entropy_attr` uses.

### Incremental Updates

The bucket caches are maintained incrementally in the same three functions that maintain `sum_n_logn`:

| Function | What it does for ref attrs |
|----------|--------------------------|
| `increment_counts` | For each value in the instance, looks up its ancestor at each depth, subtracts old `tf·log(tf)`, adds to bucket count, adds new `tf·log(tf)` |
| `update_counts_from_node` | Same pattern — merges another node's values into this node's buckets |
| `remove_counts_from_node` | Reverse — subtracts values, cleans up zero-count buckets |

Each incremental update is $O(\text{depth})$ per value, versus $O(1)$ for non-ref attrs. Since ref_tree depth is typically 5–15, this is negligible.

### Dispatch Logic

The existing `entropy_attr`, `entropy_attr_insert`, and `entropy_attr_merge` functions check at entry:

```cpp
if (this->tree->ref_tree != nullptr && this->tree->ref_attrs.count(attr))
    return this->entropy_attr_ref(attr);  // multi-resolution version
```

If the attribute is not a ref attr, or if there is no ref_tree, the standard exact-match path runs unchanged. This means:
- **Context hierarchy**: no ref_tree → standard entropy everywhere (unchanged behavior)
- **Content hierarchy, non-ref attrs**: standard entropy (unchanged)
- **Content hierarchy, ref attrs (0, 1)**: multi-resolution entropy

### Concrete Example

Using the same context hierarchy as before, a content node has stored:
```
attr 0: {CAT_vid: 2, DOG_vid: 3, RAN_vid: 1}  (a_count = 6)
```

**Depth 1 buckets** (ancestor at depth 1):
- NOUN bucket: CAT(2) + DOG(3) = 5
- VERB bucket: RAN(1) = 1
- $H_1 = $ entropy over {5, 1} with 2 total buckets

**Depth 2 buckets** (exact match):
- CAT bucket: 2, DOG bucket: 3, RAN bucket: 1
- $H_2 = $ entropy over {2, 3, 1} with 4 total buckets (SAT has count 0)

**Combined**: $H = \frac{\frac{1}{2} H_1 + \frac{2}{2} H_2}{\frac{1}{2} + \frac{2}{2}} = \frac{0.5 \cdot H_1 + 1.0 \cdot H_2}{1.5}$

The depth-1 term rewards the tree for at least separating verbs from nouns (lower $H_1$), even if individual leaves aren't perfectly separated. The depth-2 term still rewards exact-match separation. The weighted combination means the tree gets partial credit for coarse-grained clustering.

### C++ Changes for Multi-Resolution Entropy

#### `include/cobweb_discrete_node.h`
- **`RefAttrBuckets` struct**: per-depth bucket counts and `bucket_sum_n_logn` vectors
- **`ref_attr_buckets`** member: `unordered_map<ATTR_TYPE, RefAttrBuckets>`
- **New methods**: `rebuild_ref_buckets()`, `entropy_attr_ref()`, `entropy_attr_ref_insert()`, `entropy_attr_ref_merge()`

#### `include/cobweb_discrete_tree.h`
- **`structure_generation`** (`uint64_t`): bumped on structural changes
- **`val_ancestors`**: `unordered_map<VALUE_TYPE, vector<CobwebDiscreteNode*>>`
- **`val_ancestors_generation`** (`uint64_t`): tracks rebuild currency
- **`num_buckets_per_depth`**: `vector<int>` — distinct ancestor count at each depth
- **New methods**: `bump_generation()`, `ensure_val_ancestors()`, `rebuild_all_ref_buckets()`

#### `src/cobweb_discrete_tree.cpp`
- **`bump_generation()`**: `structure_generation++`
- **`ensure_val_ancestors()`**: Lazily rebuilds ancestor paths, max depth, `num_buckets_per_depth`, and all node `RefAttrBuckets` when generation mismatch detected
- **`rebuild_all_ref_buckets()`**: BFS walk over all nodes, calls `rebuild_ref_buckets()` on each
- **`cobweb()`**: `bump_generation()` called after every MERGE, SPLIT, NEW, fringe split, and depth-max forced leaf

#### `src/cobweb_discrete_node.cpp`
- **`increment_counts`**: For ref attr values, updates per-depth bucket counts and `bucket_sum_n_logn` incrementally
- **`update_counts_from_node`**: Same incremental bucket updates
- **`remove_counts_from_node`**: Same incremental bucket updates (subtraction)
- **`set_av_count`**: Calls `rebuild_ref_buckets()` at end
- **`compute_counts_from_children`**: Clears `ref_attr_buckets` before recomputing
- **`rebuild_ref_buckets()`**: Full rebuild from `av_count` — iterates values, maps to ancestors, computes bucket counts and `bucket_sum_n_logn` at each depth
- **`entropy_attr_ref()`**: Multi-resolution entropy from cached bucket data
- **`entropy_attr_ref_insert()`**: Hypothetical multi-resolution entropy after inserting instance (creates hypothetical bucket adjustments inline)
- **`entropy_attr_ref_merge()`**: Hypothetical multi-resolution entropy after merging with another node + instance
- **`entropy_attr` / `entropy_attr_insert` / `entropy_attr_merge`**: Dispatch to `_ref` variant when attr is in `ref_attrs` and `ref_tree` is non-null

#### `src/cobweb_discrete.cpp` — Python bindings
- Exposed `bump_generation`, `ensure_val_ancestors`, `rebuild_all_ref_buckets`, `structure_generation`

### Performance Characteristics

| Operation | Non-ref attr | Ref attr |
|-----------|-------------|----------|
| `increment_counts` per value | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr` | $O(1)$ | $O(\text{depth})$ |
| `entropy_attr_insert` | $O(v_{\text{instance}})$ | $O(v_{\text{instance}} \times \text{depth})$ |
| `entropy_attr_merge` | $O(v_{\text{other}} + v_{\text{instance}})$ | $O((v_{\text{other}} + v_{\text{instance}}) \times \text{depth})$ |
| Full bucket rebuild (on generation change) | N/A | $O(N \times V \times \text{depth})$ |

Where $\text{depth}$ is `ref_tree_max_depth` (typically 5–15), $N$ is tree node count, $V$ is values per node.

---

## Key Design Decisions

1. **LCA formula**: `LCA_depth / max_depth_in_ref_tree` — globally normalized, lazily cached
2. **Soft matching in `log_prob_instance`**: Uses pairwise LCA similarity for ref attrs — query value gets partial credit from structurally similar stored values
3. **Multi-resolution entropy in PU**: Ref attrs use hierarchical coarsening across all depth levels of the ref_tree, with depth-proportional weighting ($w(d) = d/D$). Non-ref attrs are unchanged.
4. **Generation counter caching**: `structure_generation` on tree, lazy `ensure_val_ancestors()` rebuild — avoids per-call overhead when ref_tree hasn't changed
5. **Incremental bucket maintenance**: `RefAttrBuckets` updated in $O(\text{depth})$ per value within `increment_counts`, `update_counts_from_node`, and `remove_counts_from_node`
6. **Leaf nodes should never be deleted**: The ref_tree pointers depend on leaf stability
7. **Cache invalidation**: `invalidate_ref_cache()` called once at the start of step 3 (content fitting), after all context hierarchy structural changes from step 1 are complete
8. **`register_ref_val`** wired at three points: `build_primitives`, `apply_candidate`, and `_refresh_labels_bottom_up` (step 2 of `add_parse_tree`)
