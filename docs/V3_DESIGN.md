# TRELLIS v3: candidate trees

v2 distilled a probabilistic grammar from two Cobweb hierarchies (cuts chosen by
description length, structure searched by night or incrementally). It is
archived at git tag `archive/pcfg-v2`. v3 follows Chris's direction: learn
everything, distil nothing, and let strong chunks emerge from what accumulates
in the trees.

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
