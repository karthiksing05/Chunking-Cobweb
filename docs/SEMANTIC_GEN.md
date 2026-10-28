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

## Round 60: the prompt directs a general concept (`story_fill`)

`story_fill=b`: a slot's fillers come from a concept above the stored leaf
(`level` 1, or `backoff`), and each candidate is weighted by exp(b × lift of the
prompt as `pv`) at the context leaves of its own records under that concept
(`_pv_fills`). The root is drawn as in round 59 (leaf, β 2).

| arm (seed 13 / seed 17) | cross share 1000 | cross share 2000 | tag pattern attested 2000 | novel 2000 |
|---|---|---|---|---|
| unconditioned | −0.06 / 0.01 | −0.13 / −0.04 | 0.895 / 0.895 | 10% / 9% |
| leaf (round 59) | 0.35 / 0.39 | 0.34 / 0.51 | 0.912 / 0.909 | 2% / 3% |
| level 1, story_fill 3 | 0.39 / 0.49 | 0.43 / 0.50 | 0.877 / 0.862 | 9% / 9% |
| level 1, story_fill 4 | 0.42 / 0.44 | 0.40 / 0.54 | 0.884 / 0.875 | 8% / 7% |
| backoff, story_fill 3 | 0.40 / 0.47 | 0.39 / 0.54 | 0.858 / 0.842 | 12% / 15% |
| retrieve (reference) | 0.41 / 0.42 | 0.66 / 0.68 | — | — |

A general concept directed by the previous sentence fits the prompt as well as
or better than copying a stored sentence, with three to five times the novel
sentences; it costs 2–7 points of tag-pattern grammar against unconditioned
generation. Retrieval pulls ahead as memory grows (0.66–0.68 at 2000).

## Rounds 61–62: recovering grammar (1000 sentences, seed 13 / seed 17)

| arm | cross share | tag pattern attested | novel |
|---|---|---|---|
| leaf (round 59) | 0.35 / 0.39 | 0.930 / 0.874 | 1% / 4% |
| level 1, story_fill 3 | 0.39 / 0.49 | 0.892 / 0.827 | 9% / 13% |
| + pair window 2, β 4 (61) | 0.31 / 0.44 | 0.821 / 0.722 | 24% / 40% |
| + pair window 2, β 0 (61) | 0.30 / 0.40 | 0.734 / 0.667 | 46% / 55% |
| + story_window (62) | 0.39 / 0.45 | 0.897 / 0.841 | 8% / 10% |
| level 2 + story_window (62) | 0.43 / 0.46 | 0.760 / 0.706 | 34% / 34% |
| backoff + story_window (62) | 0.39 / 0.45 | 0.877 / 0.811 | 10% / 15% |

Round 61 combined the prompt with paired right parts (round 55): novelty rises
steeply, topic mostly holds, grammar falls. Round 62, `story_window`: a
candidate counts only through records that stood after the word just generated
(the frozen left neighbour); grammar rises by 0.5–1.4 points only, so the loss
is not mainly at the left boundary. Across rounds 60–62 the prompt keeps the
topic at every level of generality (cross share 0.39–0.49 against 0.35–0.39 for
copying), and the trade-off left is grammar against novelty.

## Round 63: the right neighbour too (`story_window="both"`)

A filler counts only through records that stood after the word just generated and
before the stored slot's right neighbour (a sentence end included). 1000
sentences, seeds 13 / 17 / 21:

| arm | cross share | tag pattern attested | novel |
|---|---|---|---|
| level 1, left | 0.39 / 0.45 / 0.39 | 0.897 / 0.841 / 0.875 | 8% / 10% / 9% |
| level 1, both | 0.34 / 0.48 / 0.36 | 0.920 / 0.839 / 0.902 | 5% / 9% / 5% |
| level 2, left | 0.43 / 0.46 / 0.32 | 0.760 / 0.706 / 0.789 | 34% / 34% / 27% |
| level 2, both | 0.35 / 0.46 / 0.41 | 0.889 / 0.825 / 0.885 | 14% / 14% / 11% |
| backoff, both | 0.43 / 0.46 / 0.36 | 0.917 / 0.837 / 0.891 | 6% / 10% / 7% |

The right neighbour restores the grammar level 2 lost (+10 to +13 points) and
halves its novelty. Level 2 + both: grammar 0.866 and cross share 0.41 (seed
means), as level 1 + left (0.871, 0.41), with 13% new sentences against 9%: the
best point so far, a small gain. Grammar against novelty remains the trade-off.

At 2000 sentences (seed 13 / 17): level 1 left 0.42/0.51, 0.882/0.877, 8%/8%;
level 1 both 0.45/0.51, 0.886/0.889, 7%/6%; level 2 left 0.41/0.57,
0.771/0.766, 29%/32%; level 2 both 0.42/0.52, 0.855/0.861, 14%/13%; backoff
both 0.43/0.58, 0.882/0.874, 8%/9% (cross share, tag pattern attested, novel).
The pattern holds.

## Round 64: how far generality reaches (story_window "both")

Seed means (13 / 17 / 21, 1000 sentences), cross share / tag pattern attested / novel:
backoff β1 (r63) 0.42 / 0.882 / 8%; backoff β4 0.43 / 0.871 / 10%; level 2 (r63)
0.41 / 0.866 / 13%; backoff β16 0.36 / 0.850 / 16%; level 3 0.40 / 0.836 / 18%;
backoff β64 0.34 / 0.821 / 20%. One frontier: about half a point of grammar per
point of novelty, and topic weakens past 13%. Tuning generality does not move it.

## Round 64, judged: acceptability of the novel sentences

The tag-pattern check is unreliable on novel sentences (it fails "sue was happy to
have new friends", passes "they decided to her mommy"). Every generation was dumped
(STORY_DUMP) and scored by a few-shot LM judge, Qwen2.5-3B-Instruct, evaluation only
(t3_judge.py). Calibration on 300 held-out sentences per seed: real 0.89 / 0.91
pass, word order scrambled 0.12 / 0.11, content words swapped by tag 0.05 / 0.06 —
so the judge measures acceptability (grammar and sense), not grammar alone.

| arm (s13 / s17) | judged acceptable | novel | novel acceptable | acceptable novel per generation |
|---|---|---|---|---|
| unconditioned | 0.849 / 0.857 | 9% / 7% | 0.78 / 0.43 | 6.9% / 3.0% |
| leaf (copy) | 0.853 / 0.862 | 1% / 4% | 0.25 / 0.11 | 0.2% / 0.4% |
| level 1, left | 0.819 / 0.826 | 8% / 10% | 0.31 / 0.14 | 2.5% / 1.4% |
| backoff, both | 0.854 / 0.841 | 6% / 10% | 0.53 / 0.29 | 3.3% / 3.0% |
| level 2, left | 0.632 / 0.627 | 34% / 34% | 0.19 / 0.13 | 6.4% / 4.6% |
| level 2, both | 0.825 / 0.832 | 14% / 14% | 0.46 / 0.42 | 6.3% / 5.9% |
| backoff β64, both | 0.762 / 0.797 | 20% / 19% | 0.33 / 0.33 | 6.6% / 6.3% |

Topic-directed generation (level 2 both, β64 both) yields 6–7% acceptable novel
sentences per draw while holding the topic, about the unconditioned rate (5% mean)
which has no topic. Under half the novel sentences are acceptable in every arm: the
open problem is the acceptability of what is new, not the amount of novelty.
