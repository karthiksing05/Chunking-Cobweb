# Round 202: generate through the parse

Code: `src/trellis2/parse_gen.py` (PrefixChart, parse_step), `experiments/v3/diag/t3_story_r202.py` (the story run with
the new opt-in arm key `"gen": "parse"`), tests `tests/trellis2/test_r202_parse_gen.py`. `trellis3.py` and
`t3_story.py` are not touched (round 200 jobs read them).

## 1. Design (written before the first run)

**What the window generator lacks.** The causal arm draws each word from the context tree's read of l1..lW
(`_gap_fills`), one back-off factor per attribute at each stored occurrence's leaf. Nothing in that read looks at
how the words already placed are analysed. In this recipe (word_cc off, word_cnt off) a word's identity is not in
its own context or content description at all; it enters memory's descriptions only as a window value of the
elements around it. So the parse can only see a new word w through the elements to its LEFT whose right window
reaches it: the pivot's context concept with r1 = w (its role), the chunks over the pivot and the words before it
(their context with r1.. = w, their content concepts), and the content of the chunks above those.

**State = the learner's own chart of the prefix.** For the words placed so far, every (span, split) of the prefix
is built exactly as `Trellis3._chart` builds it (parts = best sub-analyses, content from the parts' context and
content names, context = window + donated own content), with factor 2^(gain / tau) (`Trellis3._factor`, tau = 1,
the parser's own), and the inside score Z is the sum over all binary analyses (`Trellis3._inside_outside`). The
one change is at the right edge, read two ways:

- OPEN: the window slots past the last word are the unwritten future, left out of every description (Cobweb sorts
  partial instances; the compiled tree's categorize takes partial dicts). Every element touching the edge is an
  open constituent: its role is fixed only by what is to its left.
- CLOSED: end markers in those slots, exactly as learning describes a sentence. The closed chart of a whole
  sentence is `Trellis3.parse` of it (test: equal log Z), minus the story facet (generation is unconditioned, so
  the previous-sentence facet `pv` is left out of every description, as the window read leaves it out).

**Step rule.** The window read is kept as the proposal q(w, end) (same pair gate, same temperature), and each
candidate is reweighted by the chart of the extended prefix:

    P(w, go on) ∝ q(w, go on) · Z_open(prefix + w)
    P(w, end)   ∝ q(w, end)   · Z_closed(prefix + w)

(both raised to 1 / temp, as the plain arm raises q). Read as a product of two experts: the window's, and the
parse's evidence for the whole prefix once w stands at its end. Z_open(prefix + w) differs between candidates
only through the elements whose right window reaches w, i.e. through how w makes memory analyse the words before
it (which context concept the pivot sorts to with w as its right neighbour, whether the chunks over it are known
chunk types), so it carries constituent state beyond l1..lW without any new grammar. The end is the same read
with the edge closed: Z_closed / Z_open is how much the analysis' concepts expect the end markers where they would
stand, a closure probability in odds form; a sentence whose spanning analysis is not a complete one (its right
constituents expect a continuation) is down-weighted at its end.

**Why this and not a stack / left-corner generator over the content tree.** A shift-reduce generator would need its
own expansion rule for which right part a content concept takes, and here a part is named by its context concept,
which depends on the part's own right window (the future), so a new rule would have to be invented to read it.
The prefix chart needs none: it is the parser, run on what has been written. Every composite of every analysis is
an open or closed constituent in it, weighted by its posterior, so "the open constituents a word would extend or
close" is exactly the set of chart entries whose window or span reaches the new word.

**Novelty.** Novel words still come from the window read (the proposal), which recombines across occurrences by its
back-off; the chart then keeps the candidates whose insertion leaves a prefix memory can analyse into known
chunk types with known roles (a slot of a known constituent filled by a word that gives the pivot a role it
has with that filler), and demotes those that only a window chaining through a pivot used two ways supports.
Nothing is copied: no stored sentence or chunk is retrieved.

**Knobs.** None new in the model. One compute budget: M = 16 proposals per step (sampling-importance-resampling:
draw M candidates from q^(1/T), weight each distinct one by Z^(1/T), draw one of them in proportion to count times
weight; as M grows this samples (q Z)^(1/T) exactly, tests check the limit). M is fixed at 16 throughout.

**Cost.** An element's description depends only on the words within W of its span and on whether its window reaches
the open / closed edge, so `PrefixChart` memoises every span by that local key: prefixes shared across draws and
candidates are parsed once.

## 2. Status

(filled in as the round runs)
