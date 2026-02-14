# PROCESSES.md — WEBSTER Multi-Hierarchy System

This document describes the four core processes in the WEBSTER multi-hierarchy parsing system:
1. **Parse Tree Construction** — turning a sentence into a hierarchical parse tree
2. **Learning** — updating the long-term memory (LTM) with a completed parse tree
3. **Generation from Nothing** — producing a sentence from scratch
4. **Generation from Masked Input** — completing a sentence with `[mask]` tokens

All processes rely on two Cobweb hierarchies stored in `LongTermMemory`:
- **Context hierarchy**: stores sliding-window context (words before/after), node complexity, and a content-reference linking to the content hierarchy.
- **Content hierarchy**: stores *what* two children are, using multi-attribute path encoding with `content_length` depth levels per side (total of `2 × content_length` attributes).

---

## 1. Parse Tree Construction

**Entry point:** `WEBSTER.parse_sentence(sentence, threshold, new_vocab, learning, debug)`

### Step 1: Build Primitives (`FiniteParseTree.build_primitives`)

1. **Tokenize** the sentence into individual words using regex.
2. For each word token at position `i`:
   - Look up its `word_id` from the vocabulary (`value_to_id`).
   - Build a **context instance** — a dict with `2 × context_length + 1` visible attributes plus one hidden attribute:
     - Attrs `0..context_length-1`: words before (exponentially decayed weights `1/2^(j+1)`).
     - Attrs `context_length..2×context_length-1`: words after (same decay).
     - Attr `2×context_length`: complexity = 1 (primitives always have complexity 1).
     - Attr `-1` *(hidden)*: word identity (the `word_id` itself). Does not influence Cobweb categorization.
   - **Categorize** the context instance in the context hierarchy via DFS traversal (`_categorize_dfs`), which walks from the root toward the best-matching leaf.
   - Set the node's **label** to `{word_id: 1}` (discrete single-identity).
   - Build a **label_path** — a list of `content_length` vocab IDs representing the node's ancestry in the context hierarchy (leaf → parent → grandparent), used as multi-depth content attributes.
   - Create a `PrimitiveParseNode` and attach it to the global root.

### Step 2: Greedy Merging Loop (`FiniteParseTree.build`)

Repeat until no more merges are possible or the threshold is not met:

1. **Get parentless pairs**: enumerate all consecutive pairs of root-level children.
2. **Evaluate each pair** (`evaluate_pair`):
   - Build a **content instance** from the two children's `label_path` values:
     ```
     {0: {left_depth0: 1}, 1: {left_depth1: 1}, ..., 
      cpd: {right_depth0: 1}, cpd+1: {right_depth1: 1}, ...}
     ```
     Each attribute has exactly one value with count 1, ensuring `log_prob_instance` produces clean, comparable scores.
   - Categorize the content instance in the **content hierarchy** to get a content leaf. Store a reference to this leaf as `CONCEPT-<hash>`.
   - Build a **context instance** for the composite by combining the two children's context (before from left, after from right), with the composite's complexity and the content-ref.
   - Categorize the context instance in the **context hierarchy**.
   - **Score** using `content_hierarchy.log_prob(content_inst)` — the tree-wide log-probability.
3. **Select the best-scoring pair** (highest log-prob).
4. If the score exceeds the threshold:
   - **Apply the merge** (`apply_candidate`): create a `CompositeParseNode`, reparent the two children under it, and set up its `label`, `label_path`, `content_instance`, and `context_instance`.
5. If the score is below the threshold, or only one root child remains (with `end_behavior="converge"`), stop.

### Result
A `FiniteParseTree` with a global root whose children form the hierarchical parse. Primitives are leaves; composites are internal nodes representing learned chunks.

---

## 2. Learning

**Entry point:** `LongTermMemory.add_parse_tree(parse_tree, debug)`

Called when `learning=True` in `parse_sentence`. Updates both Cobweb hierarchies to incorporate the structure of a completed parse tree.

### Step 0: Collect Raw Instances
- Extract all **context instances** from parsed nodes (both primitives and composites).

### Step 1: Fit Context Instances
- For each context instance, call `ifit` on the **context hierarchy**.
- `ifit` incrementally incorporates the instance — it may trigger structural changes (splits, merges, new nodes) in the Cobweb tree.
- Collect any **rewrite rules** produced by splits (mapping deleted concept hashes to their parent replacements).
- Register any new concept nodes in the vocabulary.
- Store each node's `_ifit_leaf_id` (the leaf ID returned by `ifit`) and `_categorize_leaf_id` (the leaf ID returned by a separate `categorize` call). These two IDs may differ because `ifit` can restructure the tree, moving instances to different leaves than `categorize` would find.

### Step 1b: Cross-Hierarchy Propagation (Context → Content)
- If the context hierarchy underwent splits, the deleted concept vocab IDs may already be stored as values inside the content hierarchy's `av_count` (from prior training sentences).
- Walk the entire content hierarchy tree and **replace stale vocab IDs** with their parent replacements using the rewrite rules.
- This ensures that old and new instances remain comparable — a word that was assigned concept ID `X` before the split now maps to the same ID as a word assigned after the split.

### Step 2: Refresh Labels
- Bottom-up DFS through the parse tree to re-categorize every node through the now-updated context hierarchy:
  - **Primitives**: re-categorize their context instance, rebuild `label_path` using `_build_label_path_from_ctx`.
  - **Composites**: re-categorize to get a fresh `concept_label`, rebuild `label_path`, and reconstruct `content_instance` from the children's refreshed labels.

### Step 3: Collect Content Instances
- Gather content instances from all composite nodes in the parse tree.
- Also build content instances for **unparsed candidate pairs** (consecutive root-level children that weren't merged) — these negative examples are also learned by the content hierarchy.

### Step 4: Fit Content Instances
- For each content instance, call `ifit` on the **content hierarchy**.
- Collect rewrite rules from any splits.

### Step 4b: Cross-Hierarchy Propagation (Content → Context)
- If the content hierarchy underwent splits, propagate the rewrite rules to the **context hierarchy** (which stores content-ref vocab IDs in its `av_count`).
- This keeps the content-ref attribute consistent across both hierarchies.

### Step 5: Populate the Expansion Map

After fitting all instances, populate the `expansion_map` — a dictionary that enables deterministic sentence generation by storing direct child references for every composite node.

For every composite node in the parse tree:
1. Retrieve its `_ifit_leaf_id` and `_categorize_leaf_id` (obtained in Step 1).
2. Determine **left/right child references**:
   - If the child is a `PrimitiveParseNode`: `('word', word_id)`
   - If the child is a `CompositeParseNode`: `('comp', child._ifit_leaf_id)`
3. Build a 6-tuple entry:
   ```
   (sentence_id, content_instance, left_child_ref, right_child_ref,
    complexity, source_ifit_nid)
   ```
4. Store the entry under **both** `_ifit_leaf_id` **and** `_categorize_leaf_id` in the `expansion_map`. Dual-keying ensures maximum reachability — during generation, either ID may be used to look up a node's expansion.
5. Increment `_sentence_counter` for the next training sentence.

**Why dual-keying?** During generation, a context-hierarchy lookup may return either the `ifit` leaf or the `categorize` leaf for a given node. By storing entries under both keys, the expansion map can be reached from either path. The `source_ifit_nid` field in each entry distinguishes **primary entries** (stored under the node's own `ifit` leaf ID) from **alias entries** (stored under the node's `categorize` leaf ID, which may be shared with other nodes).

---

## 3. Generation from Nothing

**Entry point:** `WEBSTER.generate_sentence(masked_sentence="", debug)`

Produces a sentence from scratch by leveraging the `expansion_map` to reconstruct learned parse trees. Generation naturally supports both **exact recall** and **creative cross-sentence mixing**, controlled entirely by the Cobweb hierarchy's structure:

- When a leaf node is its own basic level (very specific, all entries from one sentence), expansion follows a single sentence's child references → exact recall.
- When a leaf's basic level is an ancestor (shared across sentences), `_pick_entry` randomly selects from entries across multiple sentences → creative cross-sentence mixing.

No explicit sentence-ID filtering is applied; the behavior emerges organically from the specificity of each node in the hierarchy.

### Overview

The generation algorithm has three phases:
1. **Select a sentence root** — sample a context-hierarchy leaf representing a full sentence.
2. **Look up its expansion** — retrieve the root's content, left/right child references, and complexity from the `expansion_map`.
3. **Recursively expand** — use a priority queue to expand composite children depth-first, following child references at each level.

### Step 1: Find Sentence-Level Context Leaves

Scan the context hierarchy for **sentence-level leaves** — leaves whose surrounding context slots (before/after) contain exclusively `EMPTYNULL` (vocab ID 0). These correspond to root-level composites from training, since sentence roots have no preceding or following words in their immediate context window.

Each qualifying leaf is collected with its count (weight) and dominant content-reference from `av_count[ref_attr]`.

### Step 2: Sample a Root Leaf

From the sentence-level leaves, **sample one weighted by count** using `random.choices`. This naturally biases toward sentences that were seen more frequently during training.

Extract the chosen leaf's concept hash and derive its `context_leaf_id` (the stable node-ID portion of the hash).

### Step 3: Look Up the Root Expansion

Query `expansion_map[context_leaf_id]` to find all entries stored for this leaf.

**Entry selection logic** (implemented in `_pick_entry`):
1. Remove self-referential entries (where a child ref points back to the lookup node).
2. **Prefer primary entries** — where `source_ifit_nid == context_leaf_id` (the entry was created by this exact leaf during training).
3. Randomly select from the primary pool (or alias pool as fallback).

No sentence-ID filtering is applied — if the leaf has entries from multiple training sentences, any of them may be selected. This enables **cross-sentence mixing** when a node represents a shared concept across sentences.

The selected entry provides:
- `seed_content` — the content instance for the root composite
- `root_left_ref` / `root_right_ref` — direct child references
- `root_complexity` — the root's complexity value

If no expansion_map entry exists, **fall back** to the content hierarchy: look up the content-ref concept hash, find the corresponding content node, and sample a leaf's content. (This path lacks direct child references and is less accurate.)

### Step 4: Build the Seed Node

Create a `CompositeParseNode` as the root of the generated tree:
- Content = `seed_content` from the expansion_map entry
- Context = empty context (sentence-level has no surrounding words)
- Complexity = `root_complexity`
- Attach `_left_child_ref`, `_right_child_ref`, and `_visited_hashes` (cycle detection)

### Step 5: Recursive Expansion (Priority Queue)

Place the seed node on a **max-heap** keyed by complexity. While the frontier is non-empty (up to `max_expansions=100`):

1. **Pop** the highest-complexity composite node (`current_node`).
2. **Enrich** — ensure its content instance is populated (may need a content-hierarchy categorization if missing).
3. Extract path_vids for the left and right sides of `current_node.content_instance`.
4. Retrieve `current_node._left_child_ref` and `current_node._right_child_ref`.

5. **Resolve left child** via `_resolve(path_vids, context, parent_pos, visited, child_ref=left_ref)`:

   **Fast path** (when `child_ref` is provided from the expansion_map):
   - `('word', word_id)` → create a `PrimitiveParseNode` directly. No hierarchy lookup needed.
   - `('comp', ifit_leaf_id)` → look up `expansion_map[ifit_leaf_id]`, randomly pick an entry via `_pick_entry` (may come from any sentence sharing this node), extract child's content and child references, create a new `CompositeParseNode` with those references attached. Push onto frontier.

   **Slow path** (fallback when no child_ref is available):
   - Decode `path_vids` → find a matching node in the context hierarchy (via `context_hash_index` or `context_id_index` fallback).
   - Read the context node's content-ref attribute from its `av_count`.
   - If the ref is a word → create a `PrimitiveParseNode`.
   - If the ref is a `CONCEPT-<hash>` → look up the concept in the `expansion_map` by node-ID, or fall back to the content hierarchy to sample content.

6. **Resolve right child** — same process, but first inject the left child's `word_id` into the right child's before-context for coherence.

7. **Attach children** to `current_node`, assign position indices (fractional offsets from parent), push composites onto the frontier.

8. Increment `expansion_count`.

### Step 6: Flatten and Return

- DFS-collect all `PrimitiveParseNode` leaves in position order.
- Look up each primitive's `word_id` in the vocabulary to get the word string.
- Join into a sentence and return `[generated_text, FiniteParseTree]`.

### Cycle Prevention

Each composite node carries a `_visited_hashes` set containing all ancestor node-IDs encountered during expansion. Before expanding a child as composite, check if its node-ID is already in the visited set — if so, force it to resolve as a primitive to break the cycle.

### Cross-Sentence Mixing vs. Exact Recall

Generation does **not** enforce single-sentence coherence. Instead, the behavior emerges from the Cobweb hierarchy's structure:

- **Exact recall**: When a context-hierarchy leaf is highly specific (its own basic level), all its expansion_map entries originate from one training sentence. Following child references through `('comp', ifit_leaf_id)` naturally stays within that sentence because each child leaf also has entries from only that sentence.

- **Creative mixing**: When a leaf's `get_basic()` returns an ancestor (the leaf represents a shared concept across multiple sentences), its expansion_map entries span multiple sentences. `_pick_entry` randomly selects among them, so different branches of the generated tree may draw from different training sentences — producing novel combinations of learned sub-structures.

This design lets the **data** control the creativity level: early in training (few sentences, specific leaves), generation recalls exact sentences; as more data is learned and concepts generalize (shared leaves), generation becomes increasingly creative.

---

## 4. Generation from Masked Input

**Entry point:** `WEBSTER.generate_sentence(masked_sentence="the [mask] dog [mask] the park", debug)`

Completes a sentence where `[mask]` tokens represent unknown words or phrases.

### Step 1: Tokenize and Identify Masks
- Tokenize the masked sentence, identifying `[mask]` positions.
- Resolve known tokens to vocab IDs; mask positions get `None`.

### Step 2: Expand Each Mask
For each `[mask]` position `mi`:

1. **Build a seeded context instance** using the known words surrounding the mask position (sliding window with exponential decay, same as primitive construction).
2. **Predict complexity** using the context hierarchy — how complex (deep) the masked region should be.
3. **Sample content** for the mask:
   - Use the context hierarchy to predict the content-ref for this context.
   - If a `CONCEPT-<hash>` is predicted, look up the expansion_map for the corresponding node-ID. If found, use the entry's content and child references. Otherwise, fall back to the content hierarchy.
   - If a word is predicted, create a primitive directly.
4. **Create a seed `CompositeParseNode`** with the sampled content and real surrounding context. Attach any child references from the expansion_map entry.
5. **Recursively expand** using the same priority-queue mechanism as from-scratch generation (Step 5 above), but with real context words guiding predictions (up to 50 expansions per mask).
6. **Flatten** the expanded subtree to a list of words.

### Step 3: Reassemble
- Replace each `[mask]` token with its expanded word list.
- Known tokens are kept in place.
- Join into the completed sentence.

### Step 4: Final Parse
- Run `parse_sentence` on the completed text (with `learning=False`) to build a proper parse tree for the result.
- Return `[completed_text, FiniteParseTree]`.

---

## Key Data Structures

### Expansion Map (`LongTermMemory.expansion_map`)

A `Dict[str, List[Tuple]]` mapping context-hierarchy leaf node-IDs to lists of expansion entries.

**Entry format** (6-tuple):
```
(sentence_id,           # int — which training sentence produced this entry
 content_instance,      # dict — the composite's content instance
 left_child_ref,        # ('word', word_id) | ('comp', ifit_leaf_id) | None
 right_child_ref,       # ('word', word_id) | ('comp', ifit_leaf_id) | None
 complexity,            # int — max(left_complexity, right_complexity) + 1
 source_ifit_nid)       # str — the ifit leaf ID of the node that created this entry
```

**Dual-keyed**: each entry is stored under both the node's `_ifit_leaf_id` and `_categorize_leaf_id`. Primary entries (where the storage key matches `source_ifit_nid`) are preferred over alias entries during lookup.

**Child references** form a self-contained expansion tree:
- `('word', word_id)` — terminal: the child is a word (primitive)
- `('comp', ifit_leaf_id)` — recursive: the child is another composite whose expansion can be looked up by its `ifit_leaf_id`

This design makes generation **immune to context-hierarchy restructuring** — once entries are stored, they can be followed as a linked tree without needing to re-traverse the Cobweb hierarchies.

### Content Instance (Multi-Attribute Path Encoding)
```
{
    0:   {left_depth_0_vid: 1},    # most specific (word or concept leaf)
    1:   {left_depth_1_vid: 1},    # context-hierarchy parent concept
    ...
    cpd-1: {left_depth_N_vid: 1},  # most general ancestor (up to content_length)
    cpd:   {right_depth_0_vid: 1}, # right side, most specific
    cpd+1: {right_depth_1_vid: 1},
    ...
    2*cpd-1: {right_depth_N_vid: 1},
}
```
Where `cpd = content_length`. Each attribute has exactly one value with count 1.

### Context Instance
```
{
    0..ctx_len-1:         context_before (exponentially decayed word weights),
    ctx_len..2*ctx_len-1: context_after (same),
    2*ctx_len:            complexity ({COMPLEXITY_VID: complexity_value}),
    -1:                   content-ref ({word_id: 1} for primitives,
                                       {concept_vid: 1} for composites)
                          HIDDEN — negative index means Cobweb stores it
                          but it contributes zero entropy, so it does not
                          influence category-utility or fit/categorize.
}
```

### Label Path
A list of `content_length` integer vocab IDs representing a node's ancestry in the context hierarchy, ordered from most specific (leaf) to most general (ancestor). For primitives, `label_path[0]` is the `word_id`; for composites, it's the concept leaf's vocab ID.

### `_pick_entry(entries, lookup_nid)`

Helper that selects the best expansion_map entry for a given lookup:
1. Remove self-referential entries (where `left_child_ref` or `right_child_ref` points back to `lookup_nid`) to prevent infinite loops.
2. Prefer **primary entries** (`source_ifit_nid == lookup_nid`) over alias entries.
3. Randomly select from the preferred pool.
4. Return `None` if no entries exist.

No sentence-ID filtering is applied — cross-sentence mixing occurs naturally when a node has entries from multiple training sentences.
