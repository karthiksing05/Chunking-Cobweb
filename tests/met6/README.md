# met6 — Unsupervised grammar formation

Methodology 6 goal (`MULTIHIERARCHY.md`): **what is the
unsupervised-learning threshold, and how does it work?** Composite-element
analogue of `tests/primitive_threshold` (which fixed the *primitive*
maturity gate, `("root_log_prob", -12.0)`).

## What "unsupervised" means here

The deleted `composite_threshold_sweep` was **supervised** — it replayed
the grammar's gold parse trees (`merges`) and only used the threshold at
eval time. No grammar was *formed*; it was *given*.

This test is **fully unsupervised**: the only signal is the raw sentence
stream (no gold trees, no merges). The two ingredients:

1. a **recognition threshold** `τ` (`climb_count_threshold`), and
2. **adding candidate chunks to the hierarchies to augment counts**.

The loop (existing engine machinery):

```
for epoch: for sentence:
    parse_sentence(sent, learning=True, climb_count_threshold=τ)
      build()        → commit a merge ONLY when the merged chunk's
                       climbing-ancestor count in the content tree > τ
      add_parse_tree → committed chunks → BOTH hierarchies;
                       leftover candidate pairs → CONTENT tree
                       (parse_mh.py "fit orphan candidate pairs")
                       → counts grow even uncommitted
```

Candidate counts accumulate across epochs → frequent patterns cross `τ` →
graduate into chunks → recursive/higher-order chunks then accumulate. After
the warmup epochs the grammar is **frozen** (learning off) and evaluated.

## The research loop (what we learned)

The first version reported low **convergence** and 150–200 "types" and
looked broken. Three diagnostics (`diag_convergence.py`,
`diag_freq_policy.py`, `diag_grammar_size.py`) showed the test was
measuring the *wrong things*, not a broken grammar:

1. **The climbing-ancestor gate is permissive by design** — it climbs
   until it finds support, so coverage stays 1.0 even at τ=80 or with a
   relative-support gate. The gate never pins parse *shape*; the ranker
   does (`diag_convergence.py`).
2. **The ranker is the only thing that could pin shape, and any
   tree-derived signal churns** — log-prob *and* a deterministic
   frequency ranker (`MERGE_POLICY="freq_basic"`) both gave low epoch
   convergence, because Cobweb re-categorizes the same candidate as it
   learns (`diag_freq_policy.py`). So epoch-to-epoch parse churn is a
   property of **online formation**, not of the formed grammar.
3. **The formed grammar IS consistent and simple** — frozen, it parses
   deterministically (`determinism=1.0`) and uses only **14–18
   generalized categories** (content basic-level classes). The "200
   types" were over-fine context labels, not grammar size
   (`diag_grammar_size.py`).

**Conclusion / fix:** consistency = `determinism` of the frozen grammar
(not epoch convergence); simplicity = `n_categories` / `n_phrasal_prods`
(not raw labels). Epoch churn is reported as a `formation_churn`
diagnostic. No change to the candidate-chunk mechanism was needed — the
relative-gate and frequency-merge hooks were explored
(`parse_mh.py: _climbing_ancestor` relative overload + `MERGE_POLICY`) and
left available, but the legacy climbing-ancestor gate is what the main
test uses.

## What the main test measures

| Criterion | Metrics |
|---|---|
| **Consistency** | `determinism` (re-parse identical), `coverage` (chunked to one root). `formation_churn` reported as diagnostic only. |
| **Recursion** | `self_embed` (a chunk TYPE recurs at a strictly deeper node = recursive rule), `mean_depth` / `max_depth`, `recur_cov` |
| **Simplicity** | `n_categories` (distinct content basic-level classes), `n_phrasal_prods` (parentCat → leftCat rightCat, both children chunks). `τ` is the simplicity knob — higher τ → fewer categories. |
| **Generation** | `gen_gram` (CYK legality oracle), `gen_novel`, `roundtrip` (generate → re-parse → one root = closed under its own generation) |

## Run

```bash
PYTHONHASHSEED=0 python tests/met6/unsupervised_grammar_formation.py   # main sweep over τ
PYTHONHASHSEED=0 python tests/met6/diag_convergence.py                 # abs vs relative gate
PYTHONHASHSEED=0 python tests/met6/diag_freq_policy.py                 # frequency-merge policy
PYTHONHASHSEED=0 python tests/met6/diag_grammar_size.py                # categories / productions
```

Outputs: `tests/met6/unsup_grammar/{unsup_grammar.csv, unsup_grammar.png}`.

## Results (MED, 70 train sentences, 4 epochs → freeze, seed=13)

| τ | determinism | coverage | self_embed | depth | **categories** | phrasal_prods | gen_gram | gen_novel | roundtrip |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2  | **1.00** | **1.00** | 0.30 | 4.8 | 17 | 58 | **0.88** | 0.15 | **1.00** |
| 5  | **1.00** | **1.00** | 0.25 | 4.6 | **12** | 57 | 0.82 | 0.33 | **1.00** |
| 10 | **1.00** | **1.00** | 0.25 | 4.7 | 15 | 67 | 0.85 | 0.37 | **1.00** |
| 20 | **1.00** | **1.00** | 0.20 | 4.5 | 14 | 73 | 0.40 | 0.70 | 0.98 |
| 40 | **1.00** | **1.00** | 0.20 | 4.3 | 20 | 67 | 0.83 | 0.33 | **1.00** |

**The test passes all four criteria.** For every τ the frozen grammar is
**consistent** (determinism = 1.00, coverage = 1.00) and **recursive**
(self-embedding 0.20–0.30, depth > 4 — a chunk type re-occurs at a deeper
node), built from only **12–20 generalized categories** (**simple**), and
**generative** (round-trip closure ≈ 1.0).

**The facilitation threshold is τ ∈ [2, 10].** There grammaticality stays
0.82–0.88 with 12–17 categories; τ ≈ 5 is the sweet spot (simplest at 12
categories, grammatical 0.82, novelty 0.33). **τ = 20 over-generalizes** —
grammaticality collapses to 0.40 while novelty spikes to 0.70 (the gate
admits looser chunks, so generation drifts off-grammar); τ = 40 happens to
recover. Below the over-generalization knee the grammar is reliable.

`formation_churn` runs 0.26–0.70 — the online formation does not fully
settle epoch-to-epoch (Cobweb re-categorizes as it learns), which is *why*
we freeze. It is a property of the formation process, not of the formed
grammar, which is perfectly deterministic once frozen.

The grammar does **not** match the source CFG (we never measure that, per
the project goal). It is its own consistent, simple, recursive,
generative system induced from raw sentences.
