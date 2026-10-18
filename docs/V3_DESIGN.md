# TRELLIS v3: candidate trees

v2 distilled a probabilistic grammar from two Cobweb hierarchies (cuts chosen by
description length, structure searched by night or incrementally). It is
archived at git tag `archive/pcfg-v2`. v3 follows Chris's direction: learn
everything, distil nothing, and let strong chunks emerge from what accumulates
in the trees.

## How text functions, and the general form

Each mechanism rests on how text works, stated so that it carries over to any
domain whose experiences are parts in relations.

* **Distribution defines category (Harris).** A word's category is the set of
  contexts it occurs in, so the context tree describes an element only by its
  surroundings. *General form:* an element is what it is by where it occurs —
  its slot in a character, its neighbourhood on a board.
* **Substitution defines constituency.** `the lazy dog` is a unit because a
  simpler unit (`it`, `dog`) can stand where it stands; `dog saw` occupies a
  place nothing simpler ever does. The context gain therefore counts the units
  in use (words and committed chunks) under a candidate's context concept.
  *General form:* a chunk is a unit when its surroundings are surroundings in
  which simpler units already function.
* **Cohesion: parts of a unit predict each other.** Inside a constituent the
  parts co-vary (determiner with noun); across a boundary they do not. The
  content gain measures how much better a concept codes the pair than the root.
  *General form:* the parts of a chunk are more predictable together than
  apart.
* **Reuse is compression.** A unit worth having recurs across positions
  (subject and object); naming it once and reusing it saves bits. The strength
  term is the concept's share. *General form:* the same.
* **Text is produced from its units.** Generation regenerates from the
  compression: it recombines the units committed parses used, each part replaced
  by material of its own type from its basic level.

## Memory

Two Cobweb hierarchies (`incremental_cobweb.IncrementalCobwebTree`, which also
supports removal), with the ACS paper's unsupervised representation:

* **context tree** — every element (primitive or candidate composite) described
  only by its context: the two tokens on either side of its span (a composite's
  window is taken over its outermost positions). A **hidden content reference**
  is stored with it at its leaf, never used to sort: the token of a primitive,
  the content leaf of a composite. A context concept's type (word-like or
  phrase-like) is read off the references under it (`node_type`).
* **content tree** — every candidate composite, naming its two parts by their
  paths in the context tree (concept ids at depths 1..D), as `content-left` and
  `content-right` do in the paper.

Nothing is read out of the trees into a grammar. Every decision sorts an
instance through a tree and uses the counts on its path, computed when needed.
After each experience every stored content instance is re-described from its
parts' current paths (leaves are stable handles; the concepts above them move),
and its context's hidden reference follows it to its new content leaf.

## Learning: greedy parsing that stores every candidate

For each experience:

1. its primitives enter the context tree;
2. every adjacent pair on the frontier is a candidate; each new candidate is
   scored against the trees as they are, then **stored in both trees, whether
   or not it is committed**;
3. the best candidate with a positive score is committed (it replaces its two
   parts on the frontier) and step 2 repeats; with no positive score the
   analysis is a forest.

Recurring candidates accumulate counts and rise in the trees; one-off candidates
stay weak leaves.

## Performance: the score of a candidate

A candidate's content instance `x` is sorted down the content tree (without
updating it). For each concept `C` on the path:

    gain(C) = log2 (n_C / n_root)  +  log2 P(x | C)  -  log2 P(x | root)
              concept strength        fit: coding x through C instead of
                                      through the root

with `P(x | C)` the product over attributes of Laplace-smoothed value
probabilities in `C`. The score is the best gain on the path and its concept is
the proposed non-terminal. At the root the gain is exactly 0, so a positive score
means the chunk compresses; at a leaf holding a repeated pair it reduces to the
pair's pointwise mutual information. Parsing held-out experiences is the same
greedy procedure without storing.

## Generation: prediction from the trees

Left to right, top-down: a context concept at the experience's boundaries is
sampled (sorting the partial context: start before, end after); under a context
concept a leaf is sampled given the tokens already generated as its left
context, and a hidden reference drawn from it: a token is emitted, a content
leaf expanded, each of its parts under the context concept its path names.

The same counts code and generate, so compressing well means being able to
regenerate the original.

## State of the evidence (2026-10-15)

* **Categories are found.** Without supervision, the context tree's concepts at
  depth 3 group words by gold part of speech with 72-100% purity (base rates
  28-40%) on all six synthetic corpora, four seeds.
* **Chunks are found on some corpora.** Held-out bracket omission, four seeds,
  best setting (code objective, content enrichment, rehearsal 2): large 0.18
  (right-branching 0.32, better on 4/4 seeds), term_high 0.32 (0.38, 3/4),
  small 0.15 (0.25); med 0.39, term_low 0.46, term_med 0.50 are worse than
  right-branching. Characters: 0.58 with the gain objective (right-branching
  0.68), at right-branching with the code objective.
* **The objective agrees with gold when memory is right.** With trees learned
  from gold commits, the gain objective prefers the gold tree over our parse on
  med (31 vs 9 items) and large (34 vs 5); self-taught, it prefers our parse
  (characters 261 vs 9). The main obstacle is self-reinforcement: what is stored
  and used follows the parser's own early choices.
* Tried and not adopted as defaults: beam over parses, whole-parse lookahead,
  all-span candidates (helps med with the code objective, hurts generation),
  enrichment on the context side, concepts-only / minimum-support non-terminals.

## trellis3: v1's hierarchies with inside-outside (2026-10-17/18)

`src/trellis2/trellis3.py` is a direct iteration on TRELLIS v1 (ACS paper):
two hierarchies, context (an element's surface window) and content (a chunk's
parts, named by their context concepts), with no grammar and no
word-to-word prediction.

* **Learning.** Each experience is parsed by inside-outside over every binary
  analysis. A composite's factor is 2^(gain / tau), where gain is
  log2[share(C) P(x|C) / P(x|root)] at the best concept on the sorting path, in
  both trees. Every composite (span, split) is stored in both trees, weighted
  by its posterior; words go into the context tree.
* **Representation.**
  * Each facet that names a concept is one bag-valued attribute, so a facet
    counts once in category utility.
  * Parts are named by their whole ancestry: the bag of every concept on the
    path below the root ("path naming").
  * Donation runs both ways. The element's own content concept goes into its
    context (cc). The parts' content concepts go into the content (LC, RC).
  * There are no height tags (v1 kept complexity hidden).
* **No re-insertion.** When a stored element's named concepts move, its
  description is relabelled in place on its own leaf's path
  (`IncrementalCobwebTree.relabel`), for a round-robin slice of 300 records per
  experience. This is v1's leaf remap and Chris's "dynamic recomputation over
  stored leaves". It cut learning from hours to minutes.
* **Analysis.** The reported tree maximises the expected number of correct
  composites (MBR over span posteriors).
* **Generation by substitution** (`generate_sub`; diagnostic
  `experiments/v3/diag/t3_gensub.py`). Take a stored sentence element and
  replace each part with a stored member of the part's concept one level
  above its own leaf. This tests whether concepts are substitutability classes,
  which is v1's premise.

Evidence (four seeds unless noted; held-out omission is a side diagnostic):

| setting | mean omission (RB 0.339) |
|---|---|
| depth naming, refit, heights (first trellis3 best) | 0.336 |
| path naming, relabel 300, heights | 0.290 |
| path naming, relabel 300, no heights | 0.239 (2 seeds) |

What the diagnostics showed:
* **Chunk context.** Children and spine facets in the context hurt. Children
  describe content, not context. The spine is only known after an analysis.
* **Donation.** The own-content donation into the context doubles the context
  hierarchy's substitutability (med: 22-25% to 49-50% grammatical). The parts'
  content concepts and height tags in the content hurt substitutability.
* **Chunk factors.**
  * CCM contrast: right when taught gold; fails self-taught from the start.
  * Code: favours right-branching.
  * v1 climbing ancestor: worse than gain.
* **Recursion.** The adjective recursion `[lazy [red dog]]` is missed under
  every setting, even gold-taught. Both bracketings are defensible.
* **Generation.** Context-concept substitution gives 30-50% grammatical, novel
  sentences on med/large/term_high (bigram reference ~20%). Content concepts
  remain weak (2-17%).

## TRELLIS v2 framing and the parse/generate commitments (2026-10-18)

The trellis3 line is TRELLIS v2. The day/night learner is no longer part of
the account.

* **The grammar is implicit.** Inside-outside replaces v1's greedy chunking,
  not the grammar. The grammar is whatever the two trees hold. It changes with
  every experience, as a competitive frontier: every candidate composite is
  stored by its posterior share, so evidence decides which chunks become
  concepts.
* **Readout by regeneration** (`readout="regen"`). Keep the analysis whose
  chunks memory unpacks most probably: the sum over composites of
  log2 P(parts | content concept). The concept is the deepest one on the sorting
  path holding at least `support` instances. Unpacking the chunks then gives
  back the experience.
* **Expansive generation** (`generate_expand`).
  * Sample a sentence-level root chunk.
  * Expand top-down, left to right. Each slot carries metadata (its context
    concept, and whether the part is a word or a chunk) and chunk context (the
    tokens generated to its left, and the end marker when it is last).
  * Draw a filler from the concept's stored members by probability matching:
    stored weight times P(surroundings | member's concept).
  * There is no word-to-word prediction.
* **Window** is a parameter (`window`, default 2).
* **Sustainable hierarchy.** The hierarchy stays current without rebuilds:
  * there is no re-parse;
  * relabelling happens in place on a stored leaf's path;
  * a bounded round-robin refresh runs per experience.
