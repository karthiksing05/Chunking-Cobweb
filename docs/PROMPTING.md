# Generation from partial prompts (2026-10-26)

Goal: continue or fill in a partial sentence so that the result is grammatical, often novel, and about the prompt. The continuation should be conditioned on the prompt's chunks (chunk context), not only on the last two words.

## Mechanisms (`Trellis3.generate_tree`)

- **`pattern`**: a template of words and wildcards (`*` one or more words, `?` one word). Top-down generation draws fillers without replacement and backtracks when the words so far cannot complete the template. It works for prefixes but almost never for infill, because the ending is decided last.
- **`ctx_fill`**: re-sorts each slot from its generated left neighbours. It failed (word salad). Sorting a partial context description with `categorize`, or greedily by probability (`_query_path`), lands among unrelated instances.
- **`gap=(left, right)`, `gap_mode="window"`**: generate only the missing piece. A committed instance is drawn by its posterior of standing in the gap's surroundings. The prior is the instance's count; the likelihood is P(l2 l1 r1 r2 | its context leaf), with hierarchical Dirichlet back-off along the leaf's path (`_gap_fills`). The instance is expanded if it is a chunk.
- **`gap_mode="sibling"`** (prefixes): the chunk is parameterised by its constituents. The prompt is parsed, and a continuation attaches to a unit on the parse's right edge. Committed composites are scored by P(their left part is that unit), using the part name and content with the same back-off, summed over attachment points. The gap is the drawn composite's right part (`_sibling_fills`).
- **critic**: draw 8 completions and keep the one the parser recognises best (log Z per composite).

## Measures (`experiments/v3/diag/t3_prompt.py`)

Prompts are built from held-out sentences: prefix-1 `[w1 *]`, prefix-2 `[w1 w2 *]`, and infill `[w1 * w_{n-1} w_n]`. For each, we measure:
- success;
- trigrams attested;
- tag sequence attested (NLTK, diagnostic only);
- novel;
- distinct;
- coherence: the mean co-occurrence PMI between prompt words and generated words, minus the same with prompts and outputs shuffled.

## First results (400 sentences, seed 13, 30 prompts)

| | success | attested + novel |
|---|---|---|
| prefix-2: rejection / pattern / gap (window) | 0.53 / 0.73 / 1.0 | 0.06 / 0.09 / 0.23 |
| infill: rejection / pattern / gap (window) | 0.12 / 0.12 / 1.0 | 0 / 0 / 0.19 |

Good infills include "lily played and laughed" and "anna looked at each other". Typical errors look like "it smiled and for him": the expanded chunk does not fit both sides. Full runs: TinyStories 2000 (`logs/v3-prompt-*`) and bAbI 3000 (`logs/babi-prompt-*`).

## Round 41: same-level chunk context (`neighbour_chunks`, 2026-10-26)

A context also names the units beside the element at its own level: the adjacent span of the same width (cut at the experience's edge), by the content concept of memory's best analysis of it. A word's neighbour is a word. The chart takes a second pass, because the neighbours' best analyses must exist first. In gap filling, a filler of width m is scored against the prompt's m-word chunk beside the gap (`_gap_neighbours`). It follows the 23 October rule: same level, content only.

600 TinyStories sentences, verify 700, base → neighbour chunks, seeds 13 / 17 (100 held-out prompts):

| gap filling | trigrams attested | coherence gain over shuffled | distinct |
|---|---|---|---|
| prefix-1 | 0.79 / 0.77 → 0.87 / 0.86 | 0.70 / 0.53 → 0.76 / 0.88 | 0.83 / 0.84 → 0.59 / 0.71 |
| prefix-2 | 0.70 / 0.69 → 0.59 / 0.66 | 0.94 / 1.20 → 0.88 / 0.97 | 0.93 / 0.96 → 0.92 / 0.93 |
| infill | 0.21 / 0.20 → 0.17 / 0.15 | 0.92 / 1.03 → 0.98 / 0.87 | 0.98 → 0.98 |

Grammar guard (gram / gram+new at 320; verify 700 alone: 0.871 / 0.892 / 0.846 / 0.839): 1.0 / 0.667, 1.0 / 0.73, 1.0 / 0.511, 1.0 / 0.595. **Fails.** As with learned neighbour names (round 11), a more specific context makes the learner memorise. It helps one-word prefixes only, so it stays off by default.

Sibling mode is weak in every setting (trigrams attested 0.19–0.37); bug fixed in this round: the prompt's last word was a fresh element without a context description (crashed when the word had no type name).

**Open:** infill. Gap fills satisfy each side's window separately: "the * the tree → the all the tree", "i * and ben → i am and ben".
