# TRELLIS v3: candidate trees

v2 distilled a probabilistic grammar from two Cobweb hierarchies (cuts chosen by
description length, structure searched by night or incrementally). It is
archived at git tag `archive/pcfg-v2`. v3 follows Chris's direction: learn
everything, distil nothing, and let strong chunks emerge from what accumulates
in the trees.

## Memory

Two Cobweb hierarchies (`incremental_cobweb.IncrementalCobwebTree`, which also
supports removal):

* **context tree** — every element (primitive or candidate composite) described
  by how it sits in the experience: its neighbours, first and last token, kind.
* **content tree** — every candidate composite described by what it is made of:
  each part named by its token (primitives) and its path in the context tree
  (the concept ids at depths 1..D). Shared paths let `the dog` and `the cat`
  fall under one concept.

Nothing is read out of the trees into a grammar. Every decision sorts an
instance through a tree and uses the counts on its path, computed when needed.

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

A whole experience is generated top-down by prediction (Cobweb's P2):

* a top-level element: sort the partial context instance (at the start, before
  the end) through the context tree and predict its kind;
* a composite: sort its partial content instance (its own context path) through
  the content tree, predict its parts' names, and expand each part the same way
  down to primitives.

The same counts code and generate, so compressing well means being able to
regenerate the original.
