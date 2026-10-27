# Semantic generation (rounds 51–59, 2026-10-27)

Goal (user direction, 2026-10-27): generated text that holds together in meaning,
not only in grammar. General concepts keep their general knowledge and are
directed by context to a specific output (TRELLIS as a diffusion model whose step
is the split operator). Generation comes mostly from chunks around a given chunk
and frozen chunks below it; top-down conditioning is optional only. Words alone
may not be enough context.

Base configuration: as at the end of `TEMPORAL_INTERDEP.md` (path repair on use
700, `name_mix_k` 4). Gold POS tags and PMI are used for scoring only.

## Within a sentence (`experiments/v3/diag/t3_sem.py`, rounds 51–55)

Coherence is the mean PMI of content-word pairs (NN/VB/JJ, two or more apart),
from within-sentence co-occurrence over the whole TinyStories file. share =
(coherence − same-tag shuffled) / (real − real shuffled). Grammar is measured as
trigrams attested and, independent of any word window, tag pattern attested.

Mechanisms (all readout, `_generate_tree`):

- `sibling=k`, lift (51): weight a right part's fillers by the lift of their
  facets under the concept the frozen left part sorts to. Small novel-share gain
  at 1000 sentences, none at 2000; levels above the leaf break grammar.
- `sibling_mode="pair"` (53–55): the right part's slot comes from stored composites
  whose left parts sit under that concept (the general concept, parameterised by
  the frozen left part), with a height bound, five retries, `sibling_facets`, and
  `sibling_window` (the last n generated words and the sentence end must fit).

Round 55, 2000 sentences (seed 13 / seed 17):

| arm | tag pattern attested | tag pattern attested + novel | share | novel share |
|---|---|---|---|---|
| base | 0.894 / 0.947 | 0.261 / 0.220 | 0.767 / 0.885 | 0.818 / 0.816 |
| window 1, k4 β4 | 0.750 / 0.779 | 0.408 / 0.325 | 0.684 / 0.754 | 0.574 / 0.505 |
| window 2, k8 β4 | 0.883 / 0.912 | 0.214 / 0.244 | 0.804 / 0.988 | 0.745 / 0.818 |
| window 2, k8 β0 | 0.894 / 0.892 | 0.252 / 0.300 | 0.880 / 0.897 | 0.751 / 0.729 |

Pairing raises coherence over all sentences by recombining attested structure,
but novel sentences are no more coherent than the base's, which is already at
about 0.82 of real text within one short sentence. The within-sentence metric
leaves little room; the next test is across sentences.

## Across sentences (`experiments/v3/diag/t3_story.py`, rounds 56–59)

Stories are read in order. Cross coherence: mean PMI between the prompt's
content words and the generated sentence's; cross share = 0 for a random
held-out sentence, 1 for the story's real next sentence. Four generations per
prompt (from round 57), 400 prompts. Reference without a learner: *retrieve*, the
stored sentence whose previous sentence shares the most idf-weighted words with
the prompt.

- Round 56, `NEXT`: the previous whole experience records what followed it;
  given a prompt, the first content concept above its chunk holding NEXT gives
  the root. No gain: the content tree groups sentences by shape, not topic.
- Round 57, `story_facet`: every element of a following sentence gets the
  previous sentence's words as context facet `pv` (follows = any earlier kept
  sentence of the story, about 65% of sentences); the root is weighted by the
  geometric-mean lift of the prompt as `pv`. No gain: too flat.
- Round 58: product of lifts (the posterior). Still flat: a flat alpha swamps a
  leaf holding one record (lift about 1.2 per word).
- Round 59: hierarchical back-off along the context path (as `_left_part_post`).

Round 59, 1000 sentences (seed 13 / seed 17):

| arm | cross share | tag pattern attested | novel |
|---|---|---|---|
| unconditioned | −0.06 / 0.01 | 0.907 / 0.883 | 0.09 / 0.07 |
| pv level 0, β 0.5 | 0.04 / 0.16 | 0.928 / 0.880 | 0.02 / 0.05 |
| pv level 0, β 1 | **0.29 / 0.27** | 0.939 / 0.879 | 0.02 / 0.04 |
| pv level 0, β 2 | **0.35 / 0.39** | 0.930 / 0.874 | 0.01 / 0.04 |
| pv level 1 | −0.04 / −0.02 | 0.896 / 0.869 | 0.02 / 0.06 |
| retrieve (reference) | 0.41 / 0.42 | — | — |

The previous sentence as same-level chunk context makes the next sentence
belong with the prompt, close to retrieval, with grammar kept. It acts at the
leaf only: the generated sentences are almost all stored ones (novel 1–4%), and
one level up the context concepts group by the other facets, so the topic does
not reach a general concept yet. Next: let `pv` direct general concepts (the
slots' fills and higher concepts), so that a topic-matched structure is filled
with new words that stay on topic.
