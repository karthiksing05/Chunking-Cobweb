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
   - Build a **context instance** — a dict with `2 × context_length + 2` attributes:
     - Attrs `0..context_length-1`: words before (exponentially decayed weights `1/2^(j+1)`).
     - Attrs `context_length..2×context_length-1`: words after (same decay).
     - Attr `2×context_length`: complexity = 1 (primitives always have complexity 1).
     - Attr `2×context_length+1`: word identity (the `word_id` itself).
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

---

## 3. Generation from Nothing

**Entry point:** `WEBSTER.generate_sentence(masked_sentence="", debug)`

Produces a complete sentence by recursively expanding a seed composite node from the content hierarchy.

### Step 1: Sample a Seed Content Node
- Call `_sample_content_node()` to categorize a partial/empty instance in the content hierarchy and find a coherent leaf node.
- Build a `label` and `label_path` from the resulting context categorization path.

### Step 2: Predict Initial Complexity
- Use the context hierarchy to predict the expected complexity with `use_max=True`, which returns approximately 2× the average observed complexity.
- This ensures from-scratch generation starts at sentence-level complexity (typically 3+) rather than word-level.
- Floor at complexity 3 for meaningful multi-level expansion.

### Step 3: Build Seed Content
- Sample a **left label** from the content leaf's depth-0 attribute (attr 0).
- Use **conditional right prediction**: categorize `{0: left_label}` back in the content hierarchy to find a leaf where this left co-occurs with a coherent right. Sample the **right label** from that conditional leaf's depth-0 right attribute (attr `content_length`).
- Fill in deeper depth attributes (1..`content_length-1`) from the sampled leaf's `av_count`.
- Create the initial `CompositeParseNode` with this content and an empty context instance.

### Step 4: Recursive Expansion (Priority Queue)
- Place the seed node on a max-heap keyed by complexity.
- While the frontier is non-empty (up to 100 expansions):
  1. **Pop** the highest-complexity composite node.
  2. **Expand** it into left and right children via `_expand_node`:
     - Categorize the node's content instance in the content hierarchy to find a coherent leaf.
     - Sample a left from depth-0, then conditionally sample a right.
     - **Derive child context from parent context** via `_derive_child_ctx`:
       - The left child inherits the parent's **before-context** (surrounding words to the left).
       - The right child inherits the parent's **after-context** (surrounding words to the right).
       - The inner boundary (between siblings) is unknown during generation and left as EMPTYNULL.
     - For each side, categorize the derived child context in the **context hierarchy** to get a context-aware leaf, then read its **content-ref** attribute:
       - If the content-ref is a **word** AND child complexity == 1 → create a `PrimitiveParseNode` (leaf).
       - If the content-ref is a **word** but child complexity > 1 → create a `CompositeParseNode` (forces deeper expansion to match the expected complexity).
       - If the content-ref is a **`CONCEPT-<hash>`** and child complexity > 1 → find the corresponding content node, build sub-content, create a `CompositeParseNode`, and push it onto the frontier for further expansion.
       - If child complexity = 1 → force a primitive by predicting a word from the context hierarchy using the inherited context.
  3. Attach children to the parent node.

### Step 5: Flatten and Return
- DFS-collect all `PrimitiveParseNode` leaves in position order.
- Look up each primitive's `word_id` in the vocabulary to get the word string.
- Join into a sentence and return `[generated_text, FiniteParseTree]`.

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
   - If a `CONCEPT-<hash>` is predicted, find the corresponding content node and sample coherent left/right pairs from it.
   - Otherwise, fall back to sampling directly from the content hierarchy root.
4. **Create a seed `CompositeParseNode`** with the sampled content and real surrounding context.
5. **Recursively expand** using the same priority-queue mechanism as from-scratch generation (`_expand_node`), but with the advantage of real context guiding predictions (up to 50 expansions per mask).
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
    0..ctx_len-1:       context_before (exponentially decayed word weights),
    ctx_len..2*ctx_len-1: context_after (same),
    2*ctx_len:          complexity ({COMPLEXITY_VID: complexity_value}),
    2*ctx_len+1:        content-ref ({word_id: 1} for primitives,
                                     {concept_vid: 1} for composites),
}
```

### Label Path
A list of `content_length` integer vocab IDs representing a node's ancestry in the context hierarchy, ordered from most specific (leaf) to most general (ancestor). For primitives, `label_path[0]` is the `word_id`; for composites, it's the concept leaf's vocab ID.
