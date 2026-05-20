# TopKPoolEncoder — incremental TopK-Disc-Cnt1 for WEBSTER

This document describes the content-hierarchy encoder that WEBSTER uses
today, and explains in detail how it monitors stale references and
keeps old data relevant as the context tree restructures under
incremental learning.

## Design rationale: store leaves, canonicalize at evaluation time

The original ``TopK-Disc-Cnt1`` representation produces a bag of the
top-K context-tree nodes at a fixed depth `d` (the "pool depth"). The
naive incremental translation — write those depth-`d` node ids
directly into the content tree's ``av_count`` — has a fundamental
problem: depth-`d` nodes are **volatile** under Cobweb restructuring.
Any split *at* depth `d` removes a pool node entirely; any merge above
pulls a different node into the band. Each of those events makes a
previously-stored id "wrong" until we rewrite the content tree.

The encoder side-steps this by **storing leaves** and **deferring the
depth-`d` aggregation to evaluation time** via Cobweb's
``CobwebDiscreteTree::set_value_remap`` mechanism:

1. Score every depth-`d` node against the query (same as the original
   ``TopK-Disc-Cnt1``).
2. For each of the top-`k` pool nodes, find its **best leaf** under it
   (highest-count descendant).
3. Write the **best leaf's** id into the bag.
4. In parallel, maintain a ``value_remap_dict`` that maps each
   stored-leaf id to its current depth-`d` ancestor id. Push that map
   to the content tree via ``set_value_remap``.

At evaluation time, Cobweb's C++ ``canonical(val)`` looks up each
stored leaf id in the remap and substitutes its current depth-`d`
ancestor before computing entropy / log-probabilities. The
aggregation that ``TopK-Disc-Cnt1`` produced at storage time is now
**recomputed lazily** every time the tree restructures.

Why leaves and not pool nodes?

- A leaf only disappears when its underlying context-tree leaf is
  itself deleted, which is rare in normal Cobweb operation.
- A pool node disappears every time there's a split at depth `d`.
- So almost every stored id stays meaningful across restructuring —
  the work is just recomputing the cheap `leaf_id → pool_id` mapping,
  not rewriting expensive `av_count` rows.

The classical Cobweb3 version of this same idea lives in
[`concept_formation/concept_formation/convo_cobweb.py`](../concept_formation/concept_formation/convo_cobweb.py):
``ConvoCobwebSubTreeNode.expected_correct_guesses`` stores actual
node references in `av_counts` and groups them at evaluation time via
``val.get_root_sub()`` (their current top-level ancestor). We do the
integer-id version of the same trick.

## Bag layout

```python
content_instance = {
    0: {leaf_id_a: 1.0, leaf_id_b: 1.0, ...},   # K ids for left child
    1: {leaf_id_c: 1.0, leaf_id_d: 1.0, ...},   # K ids for right child
}
```

Each `leaf_id` is a stable integer minted from the corresponding
context-tree leaf's `concept_hash`. The encoder maintains
``value_remap_dict[leaf_id] = depth_d_ancestor_id`` so Cobweb's
canonical lookup re-routes the leaf id to its pool ancestor during
evaluation.

## WEBSTER wiring (where the encoder lives)

[`src/parse_mh.py`](../src/parse_mh.py) routes everything content-side
through the encoder. Concretely:

| Site | What it does |
| --- | --- |
| `LongTermMemory.__init__` ([2731](../src/parse_mh.py#L2731)) | Constructs `self.content_encoder = TopKPoolEncoder(context_hierarchy, depth, k)`. The content hierarchy is a plain `CobwebDiscreteTree` — **no `ref_tree`, no `set_ref_attr`, no `register_ref_val`** anywhere on the content side. |
| `LongTermMemory._bag_for_context_inst` ([2848](../src/parse_mh.py#L2848)) | The single content-encoding entry point. Returns `self.content_encoder.bag_for(ctx_inst, self.content_hierarchy)`. |
| `CompositeParseNode.create_content_instance(left, right, ltm)` ([658](../src/parse_mh.py#L658)) | Builds the 2-attribute dict by calling `_bag_for_context_inst` on each child's `get_context_instance()`. |
| Five callers of `create_content_instance` | Pass `ltm` (or `self` when inside LTM): `FiniteParseTree` at 1266, 1427, 1680; `LongTermMemory._refresh_labels_bottom_up` at 3062; `LongTermMemory.add_parse_tree` at 3102. |
| Save / load ([3221](../src/parse_mh.py#L3221), [3283](../src/parse_mh.py#L3283)) | `meta.json` carries `content_pool_depth`, `content_top_k`, `content_value_vocab`, `content_value_remap_dict`. Load restores all of them and re-pushes `set_value_remap` if the remap dict is non-empty. |
| `LongTermMemory` content drawer ([2786](../src/parse_mh.py#L2786)) | Custom `attr_value_fn` for attrs 0/1 renders each id as `leaf_hash→anc_hash` from the encoder's vocab — so the visualization shows *both* the stored leaf and what it canonicalises to. |
| `LongTermMemory.add_parse_tree` step-1 context fit ([3009–3022](../src/parse_mh.py#L3009)) | Context-hierarchy splits **do NOT** propagate to the content hierarchy via `_apply_rewrite_rules`. The encoder's `set_value_remap` mechanism handles context-tree changes correctly; a blind `av_count` rewrite would mis-rewrite pool ids that happen to coincide numerically with LTM `CONCEPT-` vids (separate namespaces). |
| Generation ([`_read_canonical_bag`](../src/parse_mh.py#L3737) + [`_resolve_bag`](../src/parse_mh.py#L3875)) | Reads the full K-canonical bag from a content leaf's attr (after aggregating stored leaves by `value_remap_dict`), then resolves the bag jointly: build a synthetic context instance from weighted-aggregated canonical av_counts, `categorize` against the context hierarchy to find the leaf "best matched across all ancestors", read its content-ref. See "Generation" section below. |

There is **no other path** by which content-side bags get into the
content tree. Every content `ifit` goes through `create_content_instance` →
`_bag_for_context_inst` → `content_encoder.bag_for(..., content_tree)`,
which means every `ifit` is preceded by a `set_value_remap` push if
the encoder's remap dict has changed since the last push.

## How stale references are tracked

This is the core safety property of the encoder. The context tree is
written to by every `parse_sentence` call, so any node reference the
encoder holds can become stale (merged, split, removed) between one
call and the next. The encoder treats every cached node reference as
disposable and rederives identities from the *current* tree on each
generation change.

### The three structural changes Cobweb actually does

Pool depth `d` is fixed (default `d=4`). The encoder cares about three
operations the context tree can perform:

1. **Merge above the pool.** Two depth-`d-1` siblings become children
   of a new depth-`d-1` node. The two former siblings now sit at depth
   `d` (they're new pool members), and their former depth-`d` children
   now sit at depth `d+1`.
2. **Split above the pool.** A depth-`d-1` node is removed and its
   children adopted by its parent. The children, formerly at depth
   `d`, are now at depth `d-1`.
3. **Split *at* the pool.** A depth-`d` node X is removed and its
   children adopted by X's parent (at depth `d-1`). X's former
   children move from depth `d+1` to depth `d`. X itself is
   **deleted**.

Splits *below* the pool (deep tree growth under existing pool nodes)
don't move pool nodes; they just change their descendants. Under
leaf-storage these still matter — a leaf's depth-`d` ancestor can
change because of restructuring above the leaf — but the leaf itself
is fine.

### Cache invalidation

Each entry in the cache is paired with the context tree's
`structure_generation` counter at the time of caching. Every
`bag_for` and `sync_remap` call does:

```python
if self._cached_gen != self.context_tree.structure_generation:
    self._refresh_pool()
```

So the cache is lazy and `O(1)` validated. Refresh runs only when the
tree has actually changed since the last query.

### Refresh: the safe-walker

On refresh ([cobweb-private/src/cobweb/leaf_remap.py: `_refresh_pool`](../cobweb-private/src/cobweb/leaf_remap.py)):

1. **Walk the tree from root.** Build two things in one pass:
   - `pool_nodes` / `pool_hashes`: the current set of depth-`d` nodes
     (used by `bag_for` scoring this generation).
   - `hash_to_node`: a `{concept_hash → node}` index of *every node
     reachable from root*. Every entry in this dict was just freshly
     dereferenced during the walk, so it points at a live C++ node
     (no stale pointers).
2. **For every interned hash in `value_vocab`**, recompute its current
   canonical (depth-`d` ancestor):
   - **Alive (moved):** `hash_to_node[stored_hash]` exists. Walk
     parents from that node until depth ≤ `d`. If the walker is the
     stored node itself (already at depth ≤ `d`), no remap entry is
     needed. Otherwise `remap[stored_id] = anc_id`.
   - **Dead, but former pool ancestor alive (rescued):** the hash is
     gone, but the *previously-cached* depth-`d` ancestor for this
     hash is in the fresh `hash_to_node`. Recovery via
     best-leaf-of-former-pool:
     1. `parent = hash_to_node[prev_anc_hash]`.
     2. Find the best leaf under `parent` (highest-count descendant
        leaf).
     3. Walk that leaf up to depth `d` and use its `concept_hash` as
        the new canonical.
     4. `remap[stored_id] = anc_id`.

     The leaf is acting as a *selector* — its depth-`d` ancestor is
     the pool node we re-canonicalise onto, and "best by count" picks
     the intermediary that has accumulated the most data history.
   - **Dead, former ancestor also dead (orphaned):** cascading
     deletion. No rescue is possible without unsafe stale-pointer
     access. The id is left identity-mapped; old `av_count` entries
     stay as a stale canonical category and smoothing absorbs them.

The encoder reports per-refresh diagnostics in
``self.last_refresh_stats`` as ``{"moved": A, "rescued": B,
"orphaned": C}`` so callers can monitor how often each case fires.
The two chunking tests print this after their encoding loops.

The critical safety property: we **never dereference a cached node
wrapper from a previous generation.** The walker only follows
`.parent` from `hash_to_node[stored_hash]`, which is a fresh
reference acquired during the current walk. This eliminates the
class of segfaults you'd otherwise get from a freed C++ pointer.

### When does the remap reach the C++ tree?

`bag_for(ctx_inst, content_tree)` pushes the remap via
`content_tree.set_value_remap(self.value_remap_dict)` whenever
`self.dirty` is true at the end of a call. So every content ifit sees
an up-to-date remap; the call sequence is

```
create_content_instance(left, right, ltm)
  → ltm._bag_for_context_inst(ctx_inst)
      → encoder.bag_for(ctx_inst, content_tree=content_hierarchy)
          → (lazy) _refresh_pool() if generation changed
          → score depth-d pool, pick best leaf under each top-K
          → bag = {leaf_id: 1.0, ...}
          → push value_remap_dict if dirty (each entry =
            leaf_id → current depth-d ancestor id)
      → returns the leaf-id bag
  → returns the 2-attribute content instance
content_tree.ifit(content_instance)
  ← entropy / log_prob_instance computations call canonical(val) →
    aggregate av_count rows by depth-d ancestor before evaluating
    n log n
```

For batch construction (the test scripts), `bag_for(..., content_tree=None)`
defers the push, and `sync_remap(content_tree)` is called once after
the encoding loop. The semantics are otherwise identical.

## Why old data stays relevant

The promise is "we never delete information; we just retarget how it
canonicalises." Walking through what happens for each change:

- **Merge above the pool.** A pool ancestor of stored leaf X gets
  pushed to depth `d+1`. The leaf is still alive — `_refresh_pool`
  walks X up to the new depth-`d` ancestor (the freshly-merged node)
  and updates `remap[X_id]` to point there. Old counts contribute to
  the new canonical's smoothed distribution exactly the same way new
  counts do.
- **Split above the pool.** Pool ancestor gets pulled to depth `d-1`.
  Same story; the walker stops at the new shallower ancestor and the
  remap is updated.
- **Split at the pool.** The pool ancestor of stored leaf X is gone,
  but the leaf X is still alive (it's now a descendant of one of the
  new depth-`d` pool nodes — a former depth-`d+1` sibling of X). The
  walker still finds a current depth-`d` ancestor by walking X up,
  and the remap is updated. No rescue needed.
- **Leaf X itself deleted (rare).** The hash is gone. Best-leaf-of-
  former-pool rescue: enumerate descendants of the previous depth-`d`
  ancestor, pick the best-leaf-under-it as the substitute canonical.
  Old data accumulates against the new substitute id.
- **Cascading deletion (rarest).** Both X and X's former pool
  ancestor are gone. Orphaned; old counts stay as a stale category
  that smoothing washes out over time.

In none of these scenarios does the old `av_count` data have to be
rewritten or transferred. The ids in `av_count` *are* the persistent
identity — what changes is what they canonicalise to.

## Note: namespace separation between encoder and LTM vocab

The encoder's `value_vocab` (concept_hash → int) lives in its own
namespace, completely separate from the LTM's `value_to_id` mapping
(which holds words and `CONCEPT-{hash}` strings). Pool/leaf ids stored
in the content tree's attrs 0/1 are **encoder-namespace** integers.

`LongTermMemory.add_parse_tree` deliberately does **not** call
`_apply_rewrite_rules(self.content_hierarchy, rewrites)` after a
context-hierarchy ifit. That helper does a numeric rewrite on `av_count`
keys; since the two namespaces happen to share small positive integers,
a blind rewrite would corrupt pool ids that coincide with LTM concept
vids. The encoder's `set_value_remap` mechanism handles context-tree
changes correctly via concept hashes, with no namespace collision risk.

`_apply_rewrite_rules` is still used in the other direction
(content-hierarchy splits → context-hierarchy's content-ref attribute)
because the context tree's content-ref attribute does store LTM concept
vids, so the rewrite is valid there.

## Generation: leaf-restricted joint sampling

Generation has to invert what storage did: given a content leaf whose
`av_count[attr_idx]` holds a bag of canonicals (after aggregation by
`value_remap_dict`), pick **one** context-tree leaf to descend into
for the next word/concept-ref.

A side stores **K canonicals** per attribute, weighted by accumulated
counts across many storage events. Two failed approaches:

- **Sample a single canonical and descend within it**: throws away
  K-1 worth of evidence; the encoder chose K canonicals because
  together they describe this side.
- **Aggregate canonicals' av_counts and global `categorize`**: drifts
  to "central" tree-wide nodes that just happen to be popular,
  producing severe word bias (one word like "with" or "under"
  dominates 25–30% of generation output, against ~2% training
  frequency).

`_resolve_bag` instead does **leaf-restricted joint sampling**:

1. Resolve every canonical id in the bag to its **live** context-tree
   node (drop dead canonicals; `set_value_remap` already retargeted
   their data into live ancestors).
2. Enumerate **all leaves under those canonicals' subtrees** as the
   candidate pool — no candidates outside the bag's named ancestors,
   so popular-but-unrelated nodes can't win.
3. Weight each candidate leaf by
   `canonical_weight × leaf.count`. Both factors come from training
   data: the canonical's bag weight reflects how often this ancestor
   described this side, and the leaf's count reflects how popular
   this specific leaf is under that ancestor. Their product is the
   joint relevance.
4. (Optional) If a `target_complexity` is passed by the caller,
   apply a **soft** multiplier `1 / (1 + |leaf_cplx - target|)`
   — small bonus for matching the structural level the parent
   expected, never zeroed out so primitive leaves stay viable for
   natural termination.
5. **Weighted-random sample** one candidate. Diversity comes from
   here; the bag's distribution is honored.
6. Read the matched leaf's content-ref directly (no
   `prefer_concept` manipulation). If it's a word, the recursion
   terminates here; if it's a `CONCEPT-<hash>`, the caller recurses.

The structural decision (terminate vs recurse) is driven by **what
training data says** — the matched leaf's own ref distribution —
rather than a rigid `parent_cplx - 1` depth schedule. This is
crucial: real training trees are unbalanced (a 7-word sentence might
have a cplx-6 branch and a cplx-1 branch, not two cplx-6 children),
so a rigid decrement would force expansion into 2^N leaves regardless
of what training looked like. Leaf-driven termination respects the
true branching structure.

### Sentence-root sampling (from-scratch generation)

For from-scratch generation, we sample a **sentence-root context
leaf** (all-empty context slots, has a content-ref) to seed
`_expand`. The sampling weighting is:

```python
weights[i] = complexity[i]^2 * max(1.0, root_node.count[i])
```

- `complexity^2`: prefer longer / more-structured sentences.
- `× node.count`: prefer patterns we've actually seen often (without
  this factor, a single complexity-10 root sampled once dominates a
  complexity-5 root sampled 50 times).
- Filter to `complexity ≥ 3` so we don't pick one-word "sentences"
  (which produce trivial output). Falls back if filtering empties
  the pool.

## Persistence

`LongTermMemory.save_state` writes both halves of the encoder's state
to `meta.json`:

- `content_value_vocab`: the `{concept_hash → int}` mapping.
- `content_value_remap_dict`: the `{leaf_id → current canonical id}`
  bindings.
- `content_pool_depth`, `content_top_k`: encoder settings.

`load_state` restores them and calls `set_value_remap` immediately if
the remap dict is non-empty, so the content tree is canonicalisation-
ready before the first post-load query. Settings have backward-compat
fallbacks for older meta keys (`content_remap_depth`, `content_bfs_k`).

## Quick verification commands

```python
import sys; sys.path.insert(0, 'src')
from parse_mh import WEBSTER
from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
import random; random.seed(0)

w = WEBSTER(TEST_CORPUS1, context_length=2, threshold=2,
            content_alpha=1e-3, context_alpha=1e-3,
            content_bl_alpha=10, context_bl_alpha=1,
            bow=False, empty_weighting=True, chunk_context=False,
            weighting='binary', categorization_mode='dfs')

# Phase 1: primitives only.
for _ in range(15):
    w.parse_sentence(generate('S', TEST_GRAMMAR1),
                     threshold=1e9, new_vocab=True, learning=True, debug=False)

# Phase 2: low threshold → context tree restructures under merges/splits.
for _ in range(30):
    w.parse_sentence(generate('S', TEST_GRAMMAR1),
                     threshold=2, new_vocab=False, learning=True, debug=False)

enc = w.ltm.content_encoder
print(f'pool_size              = {enc.pool_size}')
print(f'value_vocab            = {len(enc.value_vocab)}')
print(f'remap non-identity     = {len(enc.value_remap_dict)}')
print(f'use_value_remap on tree= {w.ltm.content_hierarchy.use_value_remap}')
print(f'last refresh stats     = {enc.last_refresh_stats}')
```

`last_refresh_stats` reports `moved`/`rescued`/`orphaned` for the most
recent refresh, so you can watch the rescue path actually firing as
the tree restructures.
