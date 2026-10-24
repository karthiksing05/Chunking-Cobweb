# Relational representations in TRELLIS v2

Distilled 23 October 2026 from the notes "Relational Word Representations from
Cobweb" (Word2Vec interdependence; Cobweb/4L; path vectors; the diffuse-context
problem; a trust-weighted fixed point). This document says what TRELLIS v2
already has, what it lacks, and what is being tried.

## What TRELLIS already has

| Idea in the notes | TRELLIS v2 now |
|---|---|
| Path vectors, `P(c|w)` · −log P(c) (Option 1) | `ipath` naming: a path's concepts weighted by −log2 share. A word's *type name* (`type_names="sample"`) is the bag over a weighted reservoir of 8 context leaves where it stood, which is `P(c|w)` estimated by sampling. |
| Path-encoded context (Fix 1) | Only on the content side: a chunk's parts `L`/`R` are named by their context type names. **The context window `l1..r2` is raw tokens.** |
| Cross-links update stored counts on merge | `relabel`: stored descriptions are rewritten in place on their own leaf's path, for each experience's new elements plus 300 older ones in rotation. Elements are never moved. |
| Drift | Failure check C (staleness): 0.89–0.96 of stored descriptions lag the current trees. |
| Push apart without negatives | The same as Cobweb's: category utility and splits. |

## What it lacks, and how it shows

1. **The diffuse-context problem is measured, not hypothetical.** A-or recall
   one level above the leaf is about 0.01. Members of one gold category almost
   never share a context concept there, because the window holds raw strings:
   "a dog" and "the cat" share nothing until "a" ≈ "the" is used *inside* the
   description.
2. **One role per word.** A word has only an anchor role: where it stands (its
   type name). Using the anchor name for a word acting as a neighbour causes
   syntagmatic leakage (dog drifts toward "a"). That is a likely reason
   `neighbour_types` hurt chunking on `small`.
3. **No second-order propagation (Fix 2).** Similarity only reaches as far as
   concepts already shared in the tree.
4. **No trust.** Stale and fresh concepts weigh the same in names.

## The design: three roles per word, read from the trees

TRELLIS has two trees, so a word has three roles: Word2Vec's two, plus the
AND side.

| Role | Meaning | Read from |
|---|---|---|
| anchor `v_w` | where w stands (OR side; Word2Vec W) | the bag of w's sampled context leaves' paths, plus propagation |
| context role `c_u` | how u works as a neighbour (Word2Vec W′) | the anchors of words that stood next to u |
| build role `b_w` | what w builds (AND side) | content leaves of composites w is a direct part of (`word_content`) |

The fixed point of section 6.5 of the notes, over bags of concept ids instead
of dense vectors (no dot products, nothing outside the trees):

    v_w = α · P_w + (1 − α) · Σ_u Â[w,u] · c_u        Â: w's neighbours, from the window counts of w's sampled leaves
    c_u = α · P_u + (1 − α) · Σ_w Âᵀ[u,w] · v_w        Âᵀ: the words that stood next to u, from a reservoir of their leaves

- Neighbour contributions are tagged by side (left or right of w), so "left of
  a noun" and "right of a noun" stay distinct.
- Rounds K=2 and α=0.5, with each bag truncated to its top 40. Names are read
  once per experience; nothing is stored beyond leaf pointers in bounded
  reservoirs.
- Â is first-order evidence (window counts at leaves) and never goes stale,
  matching section 6.1 of the notes. P carries the tree's current interpretation.

### Where the roles are used

- `relational="names"`: a word part's `L`/`R` in a content instance is its
  anchor `v_w`.
- `relational="window"`: a context instance also gets `lc`/`rc`, the context
  roles `c_u` of the tokens just outside its span. These are same-level
  elements, so this obeys the rule that context comes only from the same level
  or below. The raw window stays as first-order evidence.
- `word_content`: the build role `b_w` in `LC`/`RC` and in a word's own `cc`.

Trust weighting t(c) (section 6.3 of the notes) is deferred. Staleness check C
gives its drift term once the propagation alone is measured.

## Tests

1. **Toy a/the test** (`experiments/v3/diag/t3_athe.py`). "a" and "the" are
   interchangeable around many nouns, but "dog" is seen only after "a" and "cat"
   only after "the". Measure the overlap of dog's and cat's names (anchor `v`)
   against a control pair, in raw and relational modes.
2. **Corpus level:** A-or recall at u1 and u2 (the same failure, measured on
   the grammars), B cross-length AUC, and content gain > 0 (`t3_cgain.py`).
3. **Generation:** grammatical and new, so a change is judged by what it
   generates.
4. Later: the perturbation test and neighbourhood trajectories (sections 7.2 and
   7.4 of the notes), and IWord2Vec as a streaming baseline.

## Results, 23 October

**a/the test, relational names alone (300 sentences, seeds 13/17):** negative.

| mode | dog~cat | dog~n0 | a~the |
|---|---|---|---|
| current setup | 0.26 / 0.15 | 0.36 / 0.45 | 0.17 / 0.27 |
| names | 0.14 / 0.11 | 0.22 / 0.34 | 0.17 / 0.21 |
| window | 0.25 / 0.06 | 0.23 / 0.25 | 0.10 / 0.07 |
| both | 0.13 / 0.06 | 0.33 / 0.26 | 0.15 / 0.12 |

Tree sharing within 3 levels stays at about 0 for every pair.

**Diagnosis** (`logs/v3_diag/t3_athe_dbg.py`):
- A word's context instance carries `cc = w:token` (`word_cc`). Category utility rewards exact matches on it, so concepts near the leaves are token-pure.
  - "a" and "the" have identical surroundings, yet never share a concept within three levels.
  - The anchor bags P are dominated by those deep token-pure concepts (−log share weighting), so even perfect substitutes overlap weakly (a~the 0.22).
- Propagation can only spread similarity that already exists. Mixing in neighbours' roles dilutes it, and dog~cat falls with each round (0.19 → 0.14 → 0.14).
- **The bottleneck is a word identified by its raw token in the context tree.**

Running: the same test without `word_cc`, with `word_content="cc"` (a word's `cc` is what it builds), with `"cc+token"`, and the relational names on top.

### Direction (user, 23 October)

The relational notes are **background for the current system, not an overhaul of it**.
"Context" here is meant in the ordinary sense: the notes inform how we think about
word representations, and they are not a mechanism to bolt onto the context tree.
The goal is a strong representation for any word, one that grows incrementally as
the trees evolve. So:
- changes stay within the existing descriptions, learning and readout;
- the `relational` modes are one experiment the notes suggested, not the plan;
- devices that restructure a tree are out.

### Results: removing token identity from the context (300 sentences, seeds 13/17)

Name overlap and tree sharing two levels up (u2):

| variant | dog~cat | dog~n0 | a~the | dog~saw (control) | dog~cat u2 | a~the u2 |
|---|---|---|---|---|---|---|
| current setup (`word_cc`) | 0.26 / 0.15 | 0.36 / 0.45 | 0.17 / 0.27 | 0.00 / 0.02 | 0.01 / 0.01 | 0.00 / 0.00 |
| `word_cc` off | **0.38 / 0.39** | 0.67 / 0.61 | **0.54 / 0.30** | 0.03 / 0.03 | **0.21 / 0.19** | 0.06 / 0.12 |
| `word_cc` off + relational names | **0.44** / 0.27 | 0.51 / 0.40 | 0.44 / 0.35 | 0.12 / 0.10 | **0.30** / 0.17 | 0.06 / 0.06 |
| `word_content="cc"` | 0.15 / 0.17 | 0.33 / 0.47 | 0.30 / 0.33 | 0.03 / 0.01 | 0.00 / 0.00 | 0.02 / 0.02 |
| `word_content="cc+token"` | 0.11 / 0.13 | 0.22 / 0.45 | 0.27 / 0.31 | 0.02 / 0.03 | 0.00 / 0.00 | 0.02 / 0.02 |

- **Turning `word_cc` off is the change that links dog and cat.** It is one flag; the token still identifies the word in the content tree. In the context tree a word is then described only by where it stands, so dog and cat share concepts two levels up (from about 0 to 0.2) and the control pair stays near 0.
- Relational names on top help on one seed and hurt on the other, and they raise the noun~verb control (0.03 → 0.11). They are not yet a clear gain.
- `word_content` is out. On `small` it makes the content tree a chain (mean leaf depth 121–148 against 12.8), takes 15–18× longer to learn (1390–1717 s against 96 s) and leaves content gain at or below the current setup (> 0 for 0–31%, mean 0.02–0.07 bits against 25%, 0.19 bits). It restructures the tree, which the direction above excludes.

Running: `word_cc` off, with and without relational names, on `small`/`med`/`large` (generation with the leaf rule, `t3_failure`, `t3_cgain`). Logs are `logs/v3-rel-{gb,fail,cg}-<variant>-<cond>-13.log`.

### Corpus results: `word_cc` off (seed 13)

| check | cond | current (`word_cc`) | `word_cc` off | off + relational names |
|---|---|---|---|---|
| grammatical / gram+new (leaf rule) | small | 1.00 / 0.50 | 1.00 / **0.54** | 1.00 / 0.51 |
| | med | 1.00 / 0.88 | **0.86** / 0.80 | 0.82 / 0.76 |
| | large | 1.00 / 0.79 | 0.94 / **0.89** | 0.96 / 0.89 |
| A-or recall u2 | small / med / large | 0.29 / 0.08 / 0.08 | 0.20 / 0.08 / 0.10 | 0.19 / 0.08 / 0.09 |
| content leaf depth | small / med / large | 12.8 / ? / ? | 10.2 / **154 / 186** | 39 / 211 / 170 |
| learn seconds | med / large | 980 / 816 | 1727 / 2100 | 2857 / 2299 |

The toy gain does not carry over cleanly: med loses 14 points of grammaticality, and on med/large the content tree turns into chains.

**Cause.** With `word_cc` off the token appears nowhere in a two-word composite's content description:
- its parts `L`/`R` are named from the context tree, now shared across a category;
- `LC`/`RC` are blank for words.

So "the dog" and "the cat" get near-identical content descriptions, and Cobweb nests them into long chains.

**Fix within the system:** `word_cnt=True` makes a word part's `LC`/`RC` the word itself (`w:token`). A word is made of itself, so its identity sits on the content side, and the context tree describes it by where it stands. That is "identified by its representation in content and context". Running: toy and small/med/large, logs `logs/v3-*-wcnt-*`.

### Results: `word_cnt` (token on the content side), seed 13 — negative

- Toy: dog~cat falls back to 0.22 / 0.23 and shares nothing at u2. With the token in every two-word chunk's content, the trees separate words again.
- Generation memorizes: grammatical 1.00 but grammatical-and-new **0.00 / 0.005 / 0.00** on small / med / large (distinct 0.69–0.72).
- Content gain rises on `small` (> 0 for 72%, mean 0.48 bits) and stays 0 on med / large.

Token identity anywhere in a chunk's description buys content gain at the price of novelty. `word_cnt` is dropped.

**Current-setup content tree on med / large, measured for the first time:** mean leaf depth **103 / 96**, content gain 0. The chaining is a weakness of the current system, not only of the variants; `word_cc` off deepens it to 154 / 186.

Running: seed 17 of the current setup and `word_cc` off on med / large (generation), to see whether the med regression (1.00 → 0.86) is real.

### Seed 17: the `word_cc` off trade-off is real (leaf rule, grammatical / gram+new)

| cond | current, s13 | current, s17 | `word_cc` off, s13 | `word_cc` off, s17 |
|---|---|---|---|---|
| med | 1.00 / 0.88 | 0.99 / 0.73 | 0.86 / 0.80 | 0.86 / 0.81 |
| large | 1.00 / 0.79 | 1.00 / 0.83 | 0.94 / 0.89 | 0.97 / 0.93 |

- On both seeds, `word_cc` off trades grammaticality (−14 points on med, −3 to −6 on large) for more new sentences (large +10 on both seeds; med −8 on seed 13, +8 on seed 17).
- The errors are whole chunks in the wrong role, not wrong words: "the dog found the telescope liked the big lazy park" (a clause where a noun phrase belongs), "a cat in the dog" (a noun phrase standing as a sentence).
- With word identity gone from the context tree, a chunk's context concept no longer separates "what stands here" by kind. That is the same weakness as the chained content tree: chunk types are not recognised.

Not adopted as the default. Next: content-tree chaining (leaf depth about 100 on med and large in the current setup), so chunk types become recognisable; then re-test `word_cc` off.
