# WEBSTER Test Suite — Evaluation Guide

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
| Parse F1 | **82.0%** | 44.3% | -37.7pp |
| Exact-match parses | **52.2%** | 21.7% | -30.5pp |
| Step-pick | **87.4%** | 40.3% | -47.1pp |
| Chunk-class probe | 99.2% | **95.8%** | -3.4pp |
| From-scratch grammatical | **36.0%** | 0.0% | -36.0pp |
| Mask POS recovery | **87.0%** | 69.6% | -17.4pp |
| Mask exact-token | 17.4% | 13.0% | -4.4pp |

The headline takeaway: **the representations remain strong even
unsupervised** (chunk-class probe: 95.8%) — Cobweb still clusters
chunks by class via the cplx attrs (see `project_content_instance_cplx_attrs.md`).
But the SELECTION RULE at parse time can't recover the human's
preferred bracketing without supervised hints — step-pick drops from
87% to 40%, F1 from 82% to 44%.

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
- **Generation at 0% grammatical** — same difficulty as hollow_learn;
  unsupervised generation can't recover top-level S structure without
  seeing it labelled.

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

**How:**
- Call `webster.generate_sentence(debug=False)` 50 times.
- Each call:
  1. Walks every context-tree leaf, traces its content-ref → content-
     tree leaf, builds `{content_leaf → max_complexity, count}`.
  2. Samples a content leaf weighted by `max_complexity³ × count`.
  3. Uses UNPACK-FROM-LEAF (read attrs 0/1/2/3 of leaf, recurse via
     `_resolve_bag` + `_expand`).
  4. Returns the resulting token sequence.
- Two flags scored per output:
  - **In-lexicon**: every token is a known word (vs garbage).
  - **Grammatical**: a CYK recognizer over `TEST_GRAMMAR1` accepts
    the sequence as `S`.

**Current performance:** ~100% in-lexicon, **~36% grammatical**
(SEED=13). Example grammatical outputs: "the telescope liked the
small cat", "the big lazy telescope liked", "a big cat admired the
dog", "the small man chased the man", "the telescope liked the park
with the lazy park" — full SVO sentences with prepositional phrases,
adjectival modifiers, and recursive AdjP structures, all generated
from scratch.

**Interpretation:**
- 36% grammatical is up from 12% (which was up from 2% which was up
  from 0%) via the **unsupervised leaf-transition filter** added
  most recently. The chain of improvements:
  - **2% → 12%**: sentence-root + S-shape filter on seed
  - **12% → 36%**: leaf-transition filter on `_resolve_bag`
- The leaf-transition trick: `WEBSTER.learn_leaf_transitions(trees)`
  walks training parse trees and records, per content-tree leaf,
  which OTHER content-tree leaves (and primitive word-ids) appeared
  as its LEFT and RIGHT children. At unpack time, `_resolve_bag`
  filters candidate refs to that set. Content-tree leaves are 99%
  class-pure (probe-verified) so leaf-identity recovers class
  info WITHOUT POS dictionaries or chunk-class labels.
- The CYK is in `_grammar_recognize` — it handles binary + ternary
  + unary productions of `TEST_GRAMMAR1` correctly.
- **Why it's not higher yet**: outputs like "a quick a liked a big a
  dog" still happen when a leaf's transitions are mixed across classes
  (e.g., a leaf saw both Det-words and Adj-words at the LEFT child).
  Larger training corpora would tighten the transition distributions.

**What would move this score up:**
- Force CONCEPT-resolution at all non-terminal levels (done — see
  the `target_complexity` filter in `_resolve_bag`)
- Match-complexity filter on candidate content-refs (tried, hurt
  performance — too restrictive)
- Cross-side context conditioning (not implemented — would
  require non-trivial generation-time inference)
- More training data so chunks separate cleanly by class

### (3b) Single-token masked completion

**What it scores:** Given a sentence with one token replaced by
`[mask]`, can WEBSTER fill in (i) the exact gold token and
(ii) at least the correct POS class?

**How:**
- For each held-out test sentence, take the middle token, replace
  with `[mask]`, call `webster.generate_sentence(masked_sentence=...)`.
- Score two things:
  - **Exact-token recovery**: filled token equals the gold token.
  - **POS-class recovery**: filled token's POS equals gold token's
    POS (much more lenient — counts any noun as a noun-class hit).

**Current performance:** ~17% exact, ~87% POS-class.

**Chance baselines** (reported in the test output):
- Exact: 1/21 ≈ 5% (uniform over the 21-word lexicon)
- POS-class: 1/5 = 20% (5 POS classes)

**Interpretation:**
- 87% POS recovery vs 20% chance = the model knows POS slots
  STRONGLY. It picks a noun where a noun should go.
- 17% exact recovery vs 5% chance = above chance but the model
  doesn't know the exact word — it picks a noun, just not the right
  one. This is expected behavior for unsupervised: the model has no
  semantic preference between "dog" and "cat" in the same slot.

## How to interpret the scoreboard graphic

`unittests/hollow_learn_test_mh/performance_summary.png` is a
six-panel summary:

- **(A) End-to-end Parse Accuracy** — bracket P / R / F1 / exact-match
  bars.
- **(B) Step-pick (Climbing-Ancestor)** — step-pick %, gate pass-rate,
  and chance baseline (1/n_pairs avg).
- **(C) Chunk-Class Probe** — per-class P/R/F1 grouped bars + overall.
- **(D) From-scratch Generation** — in-lexicon and grammatical rates.
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
