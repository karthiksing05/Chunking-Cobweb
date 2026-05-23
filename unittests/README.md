# WEBSTER Test Suite — Evaluation Guide

## Open problems & roadmap (kept as living state)

This list tracks every concrete failure mode in WEBSTER, with status
(OPEN / FIXED) and the root cause once known. New failure modes are
appended at the bottom; FIXED entries stay in the list with their
resolution and the metric impact, so the chain of improvements is
auditable.

### FIXED

| # | Problem | Root cause | Fix | Impact |
|---|---|---|---|---|
| 1 | Parse F1 stuck at 47% with old count gate | `basic_level_count > THRESHOLD` filter used `get_basic()`'s single answer with a `-1` sentinel for root-collapse; 21.7% of test sentences got zero chunks even though plenty of well-clustered ancestors existed | Climbing-ancestor gate: walk leaf→root, admit at first ancestor with `count > THRESHOLD`. See `feedback_parse_strategy.md`. | F1 47%→60% |
| 2 | Parse F1 plateaued at 60% even with climbing gate | Content tree had no complexity signal, so VP and S chunks of similar surface shape clustered together at basic-level nodes | Added LEFT/RIGHT child cplx as attrs 2/3 of `content_instance`. Cobweb-CU now partitions on cplx — same trick the supervised probe uses. | F1 60%→82%, step-pick 70%→87%, chunk-class 97%→99% |
| 3 | Sweep results meaningless | Cobweb C++ RNG seeded with `std::random_device` at module load; same SOTA config produced F1 47-77% across runs | Added `cobweb.set_random_seed(seed)` in helper.cpp; same seed → identical tree | Deterministic benchmarks |
| 4 | Generation 0% grammatical with 2-word outputs | Sampled context-tree sentence-root leaves; `_resolve_bag` at depth-1 collapsed to most-common WORD (Det/N) instead of a CONCEPT | Sample CONTENT-tree leaves directly weighted by max_cplx; UNPACK-FROM-LEAF path; force WORD/CONCEPT by target_complexity | gen 0%→12% |
| 5 | Generation rare deep caterpillars from single training sentences (max_cplx=10, count=1) dominated sampling | `max_cplx^3 × count` weighting | Restrict seeds to sentence-root context-tree leaves + S-shape filter (`left_cplx ≥ 2`); weight by `count × max_cplx` linear | gen 12%→36% |
| 6 | Single-instance seed transitions force class-wrong children | Seed with count=1 has one L_child entry that may be an AdjP+N internal chunk from a longer training sentence | Weight seed sampling by `count^2 × max_cplx` to strongly prefer multi-instance well-clustered leaves | gen 36%→48% |
| 7 | From-scratch generation capped at ~48% grammatical | `_resolve_bag`-based UNPACK-FROM-LEAF resolved each level by re-categorizing bags; cross-instance class mixing at every recursion compounded errors. The 99% class-pure leaves were never the actual structural anchor — they were only used as a transition filter on top of bag-resampling. | **Subtree-exchange generation** (`learn_chunk_records` + `generate_via_chunk_replay`): RECORD per-leaf, the specific training chunks that landed there (L/R child leaves + word_ids). At gen time, sample a sentence-root chunk, then at every level sample a random training chunk from each child's leaf and recurse. Class-pure leaves preserve grammar by construction; cross-instance sampling gives novelty. | **gen 48%→100% grammatical** (supervised), **0%→98%** (unsupervised). Stable across reruns. ~18% of outputs are novel combinations not seen verbatim in training. |
| 8 | Mid-sentence mask completions ballooning into multi-token subtrees | `_read_content_ref` with `prefer_concept=False` still returned a CONCEPT- ref when it was the top-1 distribution; `_expand` then recursively unpacked it into "the the the small dog" garbage that obliterated POS recovery. | In `generate_sentence` masked path, for `is_mid_sentence=True` slots, walk the ctx-leaf's `content_ref_attr` distribution filtering to WORD refs only (skip `CONCEPT-` strings), then **greedy top-1** select. Word-only + deterministic kills both bugs at once. | **Mask POS 87.0% → 95.7%** (+8.7pp). Exact-token unchanged at 17.4% (remaining losses are semantically-equivalent neighbor words — see #9). |
| 9 | `generate_via_chunk_replay` returned a ROOT-only placeholder tree | The method emitted a `gen_text` string but returned a bare `FiniteParseTree(self.ltm, ...)` with no `build_primitives` or merges, so callers got a tree containing only the global root. Every `generated_parse_tree*.png` was a single ROOT box. | Re-parse the generated text via `self.parse_sentence(gen_text, threshold="converge", learning=False)` before returning. The output tree now contains primitives + composites mirroring the replayed structure. | All visualizations now show the full nested parse tree with per-node content/context attribute tables. |
| 10 | Parse F1 ranker couldn't distinguish attachment ambiguities (capped at 82%) | Greedy `argmax cnt_root_lp` had no way to express "this specific (L, R) child combination has training evidence" — it only knew "the parent leaf is well-supported". For PP-attachment, both high and low parses produced well-supported parent leaves; the ranker picked the higher-prob one regardless of training attestation. | **Subtree-exchange-as-parsing-prior** with **joint + marginal** attestation: at parse time, for each candidate merge, query `leaf_to_chunks[would-be-parent-leaf]` (joint pair match) AND `content_leaf_transitions[would-be-parent-leaf].L_children / R_children / L_words / R_words` (marginals — has L been a left child of this parent before? has R been a right child?). Combine into a single boost `λ · [log(1+joint) + log(1+L_marg) + log(1+R_marg)]`. The marginals are denser than the joint (which only fires on exact pair matches), so combinations never seen together still earn partial credit when each side is individually attested. Implemented in [`_chunk_pool_attestation`](../src/parse_mh.py) + [`_leaf_transition_attestation`](../src/parse_mh.py) + [`evaluate_pair`](../src/parse_mh.py) + [`build()`](../src/parse_mh.py). Default `chunk_pool_weight = 5.0`. | **Sup F1 82.0% → 86.2%** (+4.2pp), **Step-pick 87.4% → 93.3%** (+5.9pp — the headline), **Exact-match parses 52.2% → 60.9%** (+8.7pp). **Unsup F1 43.9% → 46.7%** (+2.8pp), **Step-pick 40.3% → 42.9%**, **EM 21.7% → 26.1%** (+4.4pp). Marginal terms add ~1pp F1 / +3.4pp step-pick over joint-only because they recover credit for never-jointly-seen but individually-attested combinations. |

### OPEN

| # | Problem | Symptom | Hypothesis | Status |
|---|---|---|---|---|
| 9 | Parse F1 capped at 82% | 18pp gap to ceiling. Most failures are attachment ambiguities ("X V NP under NP" — high vs low PP attachment) and a few clear bad merges. | Greedy `argmax cnt_root_lp` can't distinguish "what humans preferred" from "what's most common in training". Beam search or lookahead might help. | OPEN |
| 10 | Mask exact-token recovery only 17% | POS-class recovery is 95.7% — model knows the SLOT is "N" but doesn't pick "dog" specifically. Errors are uniform across semantically-equivalent neighbors (dog↔cat↔man, big↔red↔small). | Bag-of-context representations have no way to distinguish "dog" from "cat" when both appear in identical contexts in training. Inherent representational limit at this corpus size. Lift requires either (a) semantic features beyond co-occurrence, or (b) far more training data so distinguishing co-occurrence patterns emerge. | OPEN |
| 11 | Exact-match parses at 52% | Strict bracket-equality lags F1 (82%) by 30pp. Two close-but-not-equal parses both count as F1 wins but exact-match losses. | Same root cause as #9; eliminating attachment errors lifts this in lockstep. | OPEN |

---



This directory contains two end-to-end benchmarks for WEBSTER as a
chunking parser + generator. They share the same metric panel
(documented below) but differ in the training signal they receive:

- [`hollow_learn_test_mh.py`](hollow_learn_test_mh.py) — **SUPERVISED**.
  Trains on human-annotated hollow merges (`data/test_hollow_grammar_1/`).
  WEBSTER replays each gold merge via `tree.apply_candidate(...)` and
  learns the structures humans chose.
- [`gen_learn_test_mh.py`](gen_learn_test_mh.py) — **UNSUPERVISED**.
  Trains on random CFG-generated sentences via
  `parse_sentence(threshold=THRESHOLD, learning=True)`. WEBSTER's
  `build()` decides the chunks on its own (climbing-ancestor gate +
  argmax `cnt_root_lp`) and learns from its OWN parses.

Both evaluate against the SAME held-out hollow test fold so the two
tests are directly comparable: hollow_learn is the supervised ceiling,
gen_learn is the unsupervised baseline. The gap quantifies how much
the heuristics need supervision to recover human structure.

## Side-by-side at SEED=13

| Metric | hollow_learn (sup) | gen_learn (unsup) | Δ |
|---|---:|---:|---:|
| Parse F1 (λ=0, baseline) | 82.0% | 43.9% | -38.1pp |
| Parse F1 (**λ=5, joint + marginal heuristic**) | **86.2%** (+4.2pp) | **46.7%** (+2.8pp) | -39.5pp |
| Exact-match parses (λ=0) | 52.2% | 21.7% | -30.5pp |
| Exact-match parses (**λ=5**) | **60.9%** (+8.7pp) | **26.1%** (+4.4pp) | -34.8pp |
| Step-pick (λ=0) | 87.4% | 40.3% | -47.1pp |
| Step-pick (**λ=5**) | **93.3%** (+5.9pp) | **42.9%** (+2.6pp) | -50.4pp |
| Chunk-class probe | 99.2% | **95.8%** | -3.4pp |
| From-scratch grammatical | 40.0% | (TBD) | |
| From-scratch novelty | **92.0%** | (TBD) | |
| From-scratch diversity | **100.0%** | (TBD) | |
| Grammatical AND novel | **32.0%** | (TBD) | |
| Mask POS recovery | **95.7%** | 73.9% | -21.8pp |
| Mask exact-token | 17.4% | 13.0% | -4.4pp |

The novelty / diversity / gram-and-novel rows quantify whether the
subtree-exchange algorithm is GENERATING or just REPLAYING. Both
configurations keep 76% diversity (38 of 50 outputs are unique), but
novelty rate is 14-18% — meaning ~80% of outputs ALSO appear verbatim
in the training set. That's because the training corpus is small
(89 supervised / 290 unsupervised sentences) and the CFG only has
~hundreds of distinct grammatical sentences, so combinatorial overlap
is high. The 18% supervised gram-AND-novel rate maps to 9 genuinely
new grammatical sentences per 50 outputs — non-trivial composition.

The headline takeaway: **the representations remain strong even
unsupervised** (chunk-class probe: 95.8%) — Cobweb still clusters
chunks by class via the cplx attrs (see `project_content_instance_cplx_attrs.md`).
But the SELECTION RULE at parse time can't recover the human's
preferred bracketing without supervised hints — step-pick drops from
87% to 40%, F1 from 82% to 44%.

The headline GEN result: **100% grammatical** from-scratch supervised
and **98% unsupervised** via the subtree-exchange algorithm. The
unsupervised case used to score 0% because the old `_resolve_bag`
generation path compounded per-level errors. Subtree-exchange replays
chunks from per-leaf training pools, so class-pure leaves (94.5%)
preserve grammar regardless of supervision level. The 2pp unsupervised
gap is from one output containing an off-by-one "the X the Y" pattern
where the unsupervised parser had grouped chunks the supervised parser
didn't.

This is exactly the right shape: the heuristics encode "what looks
common" via `cnt_root_lp`, which works as long as TRAINING SAW that
"common" pattern. In hollow_learn, training saw gold merges so common
≈ gold. In gen_learn, training saw whatever WEBSTER's own greedy
parses produced (some right, some wrong) — so common drifts away from
the human gold.

## Running

```bash
# Supervised (hollow merges)
PYTHONHASHSEED=0 python unittests/hollow_learn_test_mh.py

# Unsupervised (no labels)
PYTHONHASHSEED=0 python unittests/gen_learn_test_mh.py

# Subtree-exchange parsing sweep (try several chunk_pool_weight values)
PYTHONHASHSEED=0 python unittests/chunk_pool_sweep_test.py            # supervised
PYTHONHASHSEED=0 python unittests/chunk_pool_sweep_gen_learn_test.py  # unsupervised
```

`PYTHONHASHSEED=0` and the explicit `random.seed(SEED) / np.random.seed(SEED)
/ cobweb_set_seed(SEED)` at the top of each file are required for
reproducible results — see [`memory/project_cobweb_determinism.md`](#)
for why.

The hollow_learn_test outputs land in `unittests/hollow_learn_test_mh/`.
The gen_learn_test outputs land in `unittests/gen_learn_test_mh/`.
Both directories contain the same set of artefacts:
- `parse_accuracy.csv` — per-test-sentence gold vs predicted brackets
- `chunk_class_accuracy.csv` — per-class P/R/F1 from the supervised probe
- `step_pick_accuracy.csv` — per-step gold-trajectory selection log
- `generation_from_scratch.csv` — 50 from-scratch generations + flags
- `generation_masked.csv` — 30 single-mask completions
- `performance_summary.png` — six-panel scoreboard
- `final_ltm_data/` — saved WEBSTER state

---

## hollow_learn_test_mh.py — supervised setup detail

### Training data

Same train/test split as `tests/met5/grammar_threshold_test.py` and
`tests/met5/grammar_decoding_test.py`:

- 89 train sentences (80%) from the hollow corpus — replayed via gold
  merges (`tree.apply_candidate(...)`). WEBSTER learns the structures
  the human annotator chose.
- 23 test sentences (20%) — same fold used for evaluation in both tests.
- `SEED=13`, `PRIMITIVES_FIRST=200`, `THRESHOLD=30`,
  `content_alpha=context_alpha=1e-4`, `content_top_k=7`, `pool_depth=4`

---

## gen_learn_test_mh.py — unsupervised setup detail

### Training data

- 200 random CFG-generated sentences for primitives-only learning
  (`threshold=1e9` so no merges fire).
- 90 random CFG-generated sentences for chunk learning
  (`threshold=THRESHOLD` — `build()` decides the chunks).
- Evaluation uses the SAME 23-sentence test fold as hollow_learn, so
  scores are directly comparable.

The chunk-class probe (metric 2) trains on the HOLLOW train fold even
in this test — the probe needs labels and the only label source is the
hollow annotations. This isolates "representation quality" from
"WEBSTER's selection rule": the probe asks *is the bag class-separable*,
not *did WEBSTER pick the right chunk*.

### What the unsupervised numbers tell you

- **Chunk-class probe at 95.8%** — WEBSTER's unsupervised content
  tree clusters chunks by class almost as well as the supervised one.
  The cplx attrs added to `content_instance` (see
  `memory/project_content_instance_cplx_attrs.md`) give Cobweb-CU a
  class-correlated axis that survives unsupervised training.
- **Step-pick at 40%** — without seeing human-preferred merges in
  training, `cnt_root_lp` ranks WEBSTER's-own-preferred chunks above
  human-preferred chunks at ~60% of steps. The gap to hollow_learn's
  87% reflects exactly how much the training trajectory matters.
- **Parse F1 at 44%** — end-to-end is a multi-step compounding of
  step-pick (0.4^6 ≈ 0.4% theoretical floor if errors were independent,
  but they're not — many WEBSTER merges still partially match gold
  brackets, hence ~44%).
- **Generation at 98% grammatical** — subtree-exchange replay (see
  `generate_via_chunk_replay`) replays per-leaf chunk pools mined from
  WEBSTER's OWN unsupervised parses. Even though `build()` occasionally
  stops short of S, enough sentence-root chunks form to seed generation,
  and class-pure content-tree leaves preserve grammar at every level of
  the recursive replay. This nearly closes the supervised/unsupervised
  gap for from-scratch generation.

---

## What each metric panel scores

## Score reference

Each score reports a different aspect of the system. They are NOT
substitutes for each other.

### (1) Parse bracket Precision / Recall / F1

**What it scores:** WEBSTER's end-to-end auto-parse against the
human-annotated hollow brackets.

**How:**
- For each test sentence, build the gold bracket set by replaying the
  hollow merges via `tree.apply_candidate(...)` and reading every
  composite's `_chunk_span()` as `(start, end)`.
- WEBSTER's `parse_sentence(threshold=30)` runs `build()` which uses
  climbing-ancestor count gate + `argmax cnt_root_lp`.
- Score the predicted bracket set against gold:
  - Precision = `|gold ∩ pred| / |pred|`
  - Recall = `|gold ∩ pred| / |gold|`
  - F1 = harmonic mean
- **Order-independent**: bracket SETS are compared. WEBSTER's merge
  order doesn't need to match the human's — only the resulting
  structure does.

**Current performance:** ~82% F1, ~78% precision, ~82% recall.

**What pulls it down:**
- **Attachment ambiguities** (~10pp): "the dog admired the big man
  with a dog" — gold attaches "with a dog" to NP; WEBSTER attaches to
  VP. Both are legitimate parses.
- **Bad merges** (~8pp): "the man admired the woman" — WEBSTER picks
  `(man admired)` before `(the man)` because `cnt_root_lp` doesn't
  perfectly distinguish gold-vs-other when both look "common" in
  bag space.

**Why not 100%?** See `memory/project_content_instance_cplx_attrs.md`
for the diagnosis. The remaining gap is mostly grammar ambiguity
that's unrecoverable without supervised attachment preference.

#### Subtree-exchange parsing prior (now baked in at λ=5)

The same `leaf_to_chunks` + `content_leaf_transitions` data that
powers `generate_via_chunk_replay` is also a **parse-time training-
attestation signal**. For a candidate `(L, R)` merge, at the would-
be-parent leaf, ask:

1. **Joint match** (`chunk_pool_match`) — # of training chunks at this
   parent whose `(L_child, R_child)` identities EXACTLY match `(L, R)`.
2. **Left marginal** (`L_trans_count`) — # of times `L`'s leaf-or-word
   was seen as the LEFT child of ANY chunk at this parent.
3. **Right marginal** (`R_trans_count`) — same for `R` as right child.

The greedy ranker combines them into one boost on top of `cnt_root_lp`:

```
score = cnt_root_lp + λ · [log(1 + joint) + log(1 + L_marg) + log(1 + R_marg)]
```

Joint is implied by both marginals (joint > 0 ⇒ both marginals > 0), so
an exact-pair match gets ~3× the boost of a marginal-only match. The
log-shape gives diminishing returns and the 0 → 1 jump is the
strongest signal (proves "this combination — or at least each
half-combination — was seen during training").

**Why marginals beat joint-only:** with only 89 supervised train
trees, most valid test-time `(L, R)` pairs were never seen jointly.
The marginal terms fire whenever EITHER side has been attested at
this parent, recovering partial credit for combinations the joint
signal can't reach.

**Implementation:**
- [`FiniteParseTree._chunk_pool_attestation`](../src/parse_mh.py) — joint pair lookup in `leaf_to_chunks`.
- [`FiniteParseTree._leaf_transition_attestation`](../src/parse_mh.py) — marginal `L_count`, `R_count`, `parent_count` from `content_leaf_transitions`.
- [`evaluate_pair`](../src/parse_mh.py) writes `chunk_pool_match`, `L_trans_count`, `R_trans_count`, `parent_trans_count` into `content_score_data`.
- [`build()`](../src/parse_mh.py) reads `self.ltm.chunk_pool_weight` and combines.

**Result at SEED=13 (λ=5, baked in default):**

| | hollow_learn (sup) | gen_learn (unsup) |
|---|---:|---:|
| Parse F1 | 82.0% → **86.2%** (+4.2pp) | 43.9% → **46.7%** (+2.8pp) |
| Step-pick | 87.4% → **93.3%** (+5.9pp) | 40.3% → **42.9%** (+2.6pp) |
| Exact-match | 52.2% → **60.9%** (+8.7pp) | 21.7% → **26.1%** (+4.4pp) |

**How to use it:** call `webster.learn_leaf_transitions(train_trees)`
and `webster.learn_chunk_records(train_trees)` after training, then
set `webster.ltm.chunk_pool_weight = 5.0`. Subsequent `parse_sentence`
calls use the boosted ranker. λ=0 disables (vanilla greedy
`argmax cnt_root_lp`).

### (1b) Step-pick accuracy

**What it scores:** Per-decision quality of WEBSTER's selection rule
on the gold trajectory.

**How:**
- For each held-out hollow sentence:
  - Compute the gold bracket set.
  - Build primitives and replay gold merges step-by-step.
  - At each step, evaluate EVERY parentless pair via
    `evaluate_pair(climb_count_threshold=THRESHOLD)`.
  - Apply WEBSTER's selection rule (climbing-ancestor gate + argmax
    `cnt_root_lp`).
  - Mark the step CORRECT if the top-ranked pair's resulting span is
    in the gold bracket set.
  - Apply the gold merge (not the selected one) and continue.

**Current performance:** ~87% step-pick.

**Interpretation:**
- This is per-decision accuracy on a CORRECT trajectory. End-to-end
  F1 (82%) is lower because errors compound.
- 100% would mean WEBSTER's selection rule is perfect on this LTM;
  the only remaining gap to F1=100% would be trajectory commitment
  (which gold step to take first when many are valid).

### (1c) Exact-match parses

**What it scores:** Fraction of test sentences whose predicted
bracket set equals the gold bracket set EXACTLY.

**Current performance:** ~52%.

**Interpretation:** Stricter than F1 — every bracket must match. A
sentence with 7 gold brackets and 6 matches counts here as 0/1, not
6/7.

### (2) Chunk-class accuracy (Cobweb-Discrete probe)

**What it scores:** Whether the representations WEBSTER builds are
linearly class-separable. THIS IS A REPRESENTATION QUALITY METRIC,
NOT A PARSE METRIC.

**How:**
- Train a separate `CobwebDiscreteTree` probe with `_CLASS_ATTR = -1000`
  on `(content_instance, chunk_class)` pairs from the train fold.
  Chunk classes are derived head-based from gold parses (NP, VP, PP,
  AdjP, S).
- Predict each test-fold chunk's class via greedy probe descent +
  walk-up-to-labeled-ancestor.
- Report per-class P / R / F1 + overall accuracy.

**Current performance:** ~99% (NP, VP, PP, AdjP, S all >= 94%).

**Interpretation:**
- **Upper bound**: If this is HIGH but step-pick is LOW, the
  representations have the info but our SELECTION RULE doesn't use
  it (e.g., `cnt_root_lp` isn't as discriminative as a supervised
  probe).
- **If this is LOW**, the representations themselves are inadequate
  — adding more attrs to `content_instance` is the right intervention
  (see [`memory/project_content_instance_cplx_attrs.md`](#) for the
  cplx-attr fix that brought this from 97.5% → 99.2%).

### (3a) From-scratch generation

**What it scores:** Can WEBSTER generate novel grammatical sentences
unsupervised?

**How (subtree-exchange via basic-level pooling):**
- `WEBSTER.learn_chunk_records(train_trees)` after training walks every
  composite chunk and records:
  - `leaf_to_chunks[parent_leaf_hash]` — chunks landing at each leaf
    (`L_leaf_hash`, `R_leaf_hash`, `L_word_id`, `R_word_id`, cplx, L_cplx, R_cplx).
  - `sentence_root_chunks` — chunks whose context_instance is all-empty.
  - `leaf_to_bl[leaf_hash]` — each parent leaf's BASIC-LEVEL ancestor
    (Cobweb-CU's preferred class-coherent cluster via `node.get_basic()`).
  - `bl_to_chunks[bl_hash]` — union of chunks under each BL ancestor (the
    cross-leaf pool that gives us novelty).
  - `leaf_to_shapes[leaf_hash]` — set of `(L_cplx, R_cplx)` topologies
    the leaf actually produced in training (used as the unsupervised
    class-purity filter).
- `webster.generate_via_chunk_replay()` samples a sentence-root chunk
  via the BL-pooled root pool (widened seed distribution), then
  recursively at each composite-child slot picks a chunk from the
  **shape-filtered BL pool**:
  ```python
  pool = [c for c in bl_to_chunks[BL(child_leaf)]
          if c["cplx"] == expected_cplx
          and (c["L_cplx"], c["R_cplx"]) in leaf_to_shapes[child_leaf]]
  ```
  Falls back to the leaf's own pool when the shape-filtered BL pool is
  empty. The shape filter is the unsupervised class-purity guard —
  `(cplx, L_cplx, R_cplx)` is a 99% class-correlated signature (per the
  supervised probe), so filtering keeps BL pooling from mixing Det+N
  chunks with Adj+N chunks at the same `(1,1)` shape that land at
  different leaves.
- Two flags scored per output:
  - **In-lexicon**: every token is a known word (vs garbage).
  - **Grammatical**: a CYK recognizer over `TEST_GRAMMAR1` accepts
    the sequence as `S`.

**Current performance** (SEED=13, supervised hollow_learn, BL-pool):
**100% in-lexicon, 40% grammatical, 92% novelty, 100% diversity,
32% gram-AND-novel** (50/50 outputs). The headline is the
**gram-AND-novel rate of 32%** — almost 2× the leaf-only baseline's
18%. Going up to the BL ancestor unlocks combinatorial novelty across
class-coherent siblings; the shape filter is what keeps grammar from
collapsing entirely (compare 40% gram with filter vs 34% gram with
plain BL pool).

**Tradeoff**: novelty exploded (18% → 92%) but absolute gram dropped
(100% → 40%). To trade some novelty back for grammar, swap
`_pool_for` back to `self.leaf_to_chunks.get(leaf_hash, [])` (returns
to the 100%-gram / 18%-novel operating point). The user-facing
fixed-knob lives at the top of `generate_via_chunk_replay`'s
`_pool_for` definition.

**Why this works** (vs the old `_resolve_bag`-based UNPACK-FROM-LEAF):
- The OLD path resolved each level by re-categorizing a bag of canonical
  contexts → content-ref → recurse. Every recursion was a re-sampling
  decision that could pick a wrong-class child. Errors compounded
  multiplicatively: a 36% per-level coherence rate → 5% sentence-level.
- The NEW path skips bag-resampling entirely. Each leaf has a list of
  SPECIFIC training chunks that landed there; we just pick one and
  follow it. The 99% class-purity of leaves (already proved by the
  decoding probe) provides the structural anchor with no extra
  inference. Novelty comes from picking DIFFERENT training chunks at
  each subtree position.
- Effectively: training data is a finite tree-bank; per-leaf chunk
  pools turn it into a context-free grammar where each non-terminal
  (leaf) has a finite list of right-hand-side productions, and we
  sample uniformly at random. The leaf-clustering does the
  POS-induction unsupervised, so this is a discovered PCFG.

**Chain of improvements:**
- 0% → 12% gram: sample CONTENT-tree leaves directly weighted by max_cplx;
  UNPACK-FROM-LEAF via `_resolve_bag`.
- 12% → 36% gram: leaf-transition filter on `_resolve_bag`.
- 36% → 48% gram: weight seed sampling by `count² × max_cplx`.
- 48% → **100% gram, 18% novelty**: subtree-exchange (`generate_via_chunk_replay`)
  with per-leaf chunk pools.
- 100% gram, 18% novelty → **40% gram, 92% novelty, 32% gram-AND-novel**:
  basic-level chunk pooling with `(cplx, L_cplx, R_cplx)` shape filter.
  The user's "go up to the highest pure node" insight — trades
  absolute grammaticality for combinatorial novelty across BL-sibling
  leaves, with the shape filter as the unsupervised purity guard.

#### Uniqueness sub-metrics (NEW)

A grammaticality-only score doesn't tell you whether the model is
GENERATING or just REPLAYING training sentences. To disambiguate, the
test now also reports three uniqueness measures:

| Metric | Formula | What it answers |
|---|---|---|
| **Novelty rate** | `|{g ∈ gens : g ∉ train_set}| / N_GEN` | Fraction of outputs that were NEVER seen verbatim during training. A pure replay scores 0%; pure hallucination scores 100%. |
| **Diversity** | `|unique({g.text for g in gens})| / N_GEN` | Fraction of outputs that are unique within the run. A model that emits the same sentence 50 times scores 0.02; one that always emits something different scores 1.0. |
| **Grammatical AND novel** | `|{g : g.gram_ok ∧ g ∉ train_set}| / N_GEN` | The headline number — fraction of outputs that are simultaneously CYK-grammatical AND not in the training corpus. This is the only one of these three that combines correctness with creativity. |

**How the train_set is computed:**
- `hollow_learn_test_mh.py`: `train_set = {h["sentence"].strip() for h in train_hollow}` (the 89-sentence supervised train fold).
- `gen_learn_test_mh.py`: `train_set = set(s.strip() for s in unsup_sentences)` (the union of all 290 unsupervised training sentences — 200 primitives-only + 90 chunk-learning).

Membership is **exact-string** match. We don't fuzzy-match: a single
word swap counts as novel, which is the right granularity for a
subtree-exchange algorithm that mixes class-pure pieces. This metric
is in [`generation_from_scratch.csv`](#) as the per-row `novel` column
in addition to the aggregate prints / panel-D bar.

**Why three numbers, not one:**
- *Novelty alone* could be high while diversity is low (model finds
  one novel string and repeats it).
- *Diversity alone* could be high while novelty is low (model picks
  50 distinct training sentences).
- *Gram-and-novel* alone could be high but with low diversity (one
  good novel string emitted 25 times — score is 50%).
The combination tells the full story. Subtree-exchange's expected
profile is HIGH on all three because per-leaf chunk pools yield a
combinatorial explosion of in-class combinations.

### (3b) Single-token masked completion

**What it scores:** Given a sentence with one token replaced by
`[mask]`, can WEBSTER fill in (i) the exact gold token and
(ii) at least the correct POS class?

**How:**
- For each held-out test sentence, take the middle token, replace
  with `[mask]`, call `webster.generate_sentence(masked_sentence=...)`.
- Mid-sentence single-token masks: walk the context-leaf's
  `content_ref_attr` distribution, filter to WORD-only refs (skip
  `CONCEPT-*` candidates whose `_expand` would emit multi-token
  subtrees), greedy top-1. End/start-of-sentence: fall back to the
  `_resolve_bag` + `_expand` path with `prefer_concept=True` and the
  adjacent parsed-chunk complexity.
- Score two things:
  - **Exact-token recovery**: filled token equals the gold token.
  - **POS-class recovery**: filled token's POS equals gold token's
    POS (much more lenient — counts any noun as a noun-class hit).

**Current performance:** **17.4% exact, 95.7% POS-class** (SEED=13).

**Chance baselines** (reported in the test output):
- Exact: 1/21 ≈ 5% (uniform over the 21-word lexicon)
- POS-class: 1/5 = 20% (5 POS classes)

**Interpretation:**
- 95.7% POS recovery vs 20% chance = the model knows POS slots
  almost perfectly (one V/P confusion at the "the X a Y" slot, where
  both verbs and prepositions are licit fillers given pure
  co-occurrence context). The +8.7pp jump from 87% → 95.7% came from
  eliminating the bug where `_expand` recursively unpacked CONCEPT
  refs into multi-token subtrees on mid-sentence masks — see fix #8.
- 17.4% exact recovery vs 5% chance = above chance but the model
  doesn't know the exact word — it picks a noun, just not the right
  one. This is expected behavior for unsupervised: the model has no
  semantic preference between "dog" and "cat" in the same slot
  (same bag-of-context). See open problem #10.

## How to interpret the scoreboard graphic

`unittests/hollow_learn_test_mh/performance_summary.png` is a
six-panel summary:

- **(A) End-to-end Parse Accuracy** — bracket P / R / F1 / exact-match
  bars.
- **(B) Step-pick (Climbing-Ancestor)** — step-pick %, gate pass-rate,
  and chance baseline (1/n_pairs avg).
- **(C) Chunk-Class Probe** — per-class P/R/F1 grouped bars + overall.
- **(D) From-scratch Generation** — in-lexicon, grammatical, novelty,
  diversity (unique/total), and gram-AND-novel rates as five bars.
- **(E) Single-token Masked Completion** — exact-token and POS-class
  recovery, with chance baselines side-by-side.
- **(F) Scorecard** — headline numbers as a text table.

## What changes between runs

WITHOUT proper seeding (no `cobweb_set_seed`, no `PYTHONHASHSEED=0`),
the same config can produce F1 from 47% to 77% across runs — see
`memory/project_cobweb_determinism.md`. Once the C++ RNG is seeded,
results are deterministic at fixed `SEED`.

Across SEEDS, the variance is real and unavoidable: different
training-data orderings produce structurally different LTMs.
Approximate spread at `SEED ∈ {1..5}` (5 runs of the same config):

- F1: mean ~69%, σ ~7-8pp
- Step-pick: mean ~85%, σ ~5pp
- Chunk-class: mean ~98%, σ ~1pp (very stable)

`SEED=13` (the test default) tends to land at the higher end:
F1 ~82%, step-pick ~87%.

## Adding a new metric

The test is sectioned by `=== (N) ... ===` headers. To add a new
score:

1. Add a new section under the relevant block in
   `unittests/hollow_learn_test_mh.py`.
2. Append to the summary block at the bottom.
3. Add a panel to the `performance_summary.png` graphic.
4. Document the score here in this README.

## Common failure modes when scores drop

| Symptom | Likely cause | Fix |
|---|---|---|
| Parse F1 drops, chunk-class stays high | Selection rule (build()) regressed | Inspect `parse_accuracy.csv`; check `build()` gate / ranker |
| Chunk-class drops | `content_instance` shape changed; Cobweb clustering broke | Diff `create_content_instance`; ensure cplx attrs (2, 3) are still there |
| Step-pick drops to ~25% | Climbing gate rejecting everything | Check that THRESHOLD ≤ `node.count` for typical chunks |
| Generation drops to 0% but lex-ok 100% | CYK can't parse output; outputs are still in-lexicon | Check CYK handles all production arities; verify outputs look right |
| Mask exact / POS stuck at chance | `_resolve_bag` not finding any candidates | Check `content_ref_attr` is populated at context-tree leaves |
| Tests vary 20pp+ run-to-run | Cobweb RNG not seeded | Confirm `cobweb_set_seed(SEED)` is called before WEBSTER init |

## See also

- `memory/feedback_parse_strategy.md` — current build() selection rule
- `memory/feedback_generation_strategy.md` — current generation algorithm
- `memory/project_content_instance_cplx_attrs.md` — the cplx-attr fix
- `memory/project_cobweb_determinism.md` — RNG seeding requirement
- `tests/met5/grammar_decoding_test.py` — the "ideas" probe (chunk-class)
- `tests/met5/grammar_threshold_test.py` — the heuristic-comparison harness
