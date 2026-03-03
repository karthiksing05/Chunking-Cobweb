"""
Test 2: Log-probability probe using concept-path IDs – bigram framing.

In the real framework, content instances encode *both* tokens of a bigram as
their label_path through the context hierarchy — 3 depth levels per side:

    {0: {left_d0: 1}, 1: {left_d1: 1}, 2: {left_d2: 1},
     3: {right_d0: 1}, 4: {right_d1: 1}, 5: {right_d2: 1}}

This test simulates that encoding with a POS hierarchy that distinguishes
function words (articles/determiners) from content words (nouns/verbs):

    ROOT
    ├── FUNC_WORD (100)
    │   └── ARTICLE (110)
    │       ├── DEF_ART (111)        ← "the"
    │       └── INDEF_ART (112)      ← "a"
    └── CONTENT_WORD (200)
        ├── NOUN (210)
        │   ├── ANIM_NOUN (211)      ← cat, dog, bird
        │   └── INANIM_NOUN (212)    ← fish, mouse
        └── VERB (220)
            ├── RUNS (221)
            └── SEES (222)

Training bigrams mimic surface bigrams from S → NP VP, NP → Det N:
  - 10  Det+Noun  bigrams  (NP-internal, e.g. "the cat", "a dog")
  - 2   Noun+Verb bigrams  (NP→VP boundary crossing, e.g. "cat runs")

Expected insight over test 1:
  • A Det+unseen_noun bigram (e.g. "the horse") should score *much closer*
    to the trained Det+Noun cluster than a Noun+Verb boundary bigram,
    because depth-0 (FUNC_WORD) and depth-1 (ARTICLE) match even when
    the leaf (horse) is novel.
  • Unseen Noun+Verb (horse, runs) should still score lower than any
    Det+Noun variant because the category mismatch persists at depth-0.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree


# ── simulated concept IDs (POS hierarchy) ────────────────────────────────────
# depth 0 (coarsest / most general)
FUNC_WORD    = 100   # function words: determiners, prepositions, ...
CONTENT_WORD = 200   # content words: nouns, verbs, adjectives, ...

# depth 1
ARTICLE      = 110   # articles (subset of FUNC_WORD)
NOUN         = 210   # nouns  (subset of CONTENT_WORD)
VERB         = 220   # verbs  (subset of CONTENT_WORD)

# depth 2 (surface token type IDs)
DEF_ART      = 111   # "the"
INDEF_ART    = 112   # "a"

# Leaf-level token IDs
THE   = 1011
A     = 1012
CAT   = 2011
DOG   = 2012
BIRD  = 2013
FISH  = 2021
MOUSE = 2022
HORSE = 2031   # UNSEEN during training (same depth-0/1/2 as animate nouns)
COW   = 2032   # UNSEEN during training
RUNS  = 2211
SEES  = 2212


def inst(l0, l1, l2, r0, r1, r2):
    """3-depth-per-side content instance (6 attributes total)."""
    return {
        0: {l0: 1.0},  #  left depth 0
        1: {l1: 1.0},  #  left depth 1
        2: {l2: 1.0},  #  left depth 2
        3: {r0: 1.0},  # right depth 0
        4: {r1: 1.0},  # right depth 1
        5: {r2: 1.0},  # right depth 2
    }


# shorthand path tuples per token type
def article_def():       return (FUNC_WORD, ARTICLE, DEF_ART)    # "the"
def article_indef():     return (FUNC_WORD, ARTICLE, INDEF_ART)   # "a"
def noun_anim(leaf):     return (CONTENT_WORD, NOUN, leaf)
def noun_inanim(leaf):   return (CONTENT_WORD, NOUN, leaf)        # same POS depth levels
def verb(leaf):          return (CONTENT_WORD, VERB, leaf)


# ── training bigrams ──────────────────────────────────────────────────────────
# NP-internal bigrams: Det + Noun (10 instances, high cohesion)
DET_NOUN_BIGRAMS = [
    inst(*article_def(),    *noun_anim(CAT)),
    inst(*article_def(),    *noun_anim(DOG)),
    inst(*article_def(),    *noun_anim(BIRD)),
    inst(*article_def(),    *noun_inanim(FISH)),
    inst(*article_def(),    *noun_inanim(MOUSE)),
    inst(*article_indef(),  *noun_anim(CAT)),
    inst(*article_indef(),  *noun_anim(DOG)),
    inst(*article_indef(),  *noun_anim(BIRD)),
    inst(*article_indef(),  *noun_inanim(FISH)),
    inst(*article_indef(),  *noun_inanim(MOUSE)),
]

# Boundary-crossing bigrams: Noun + Verb (2 instances, low cohesion)
NOUN_VERB_BIGRAMS = [
    inst(*noun_anim(CAT), *verb(RUNS)),
    inst(*noun_anim(DOG), *verb(RUNS)),
]

TRAINING = DET_NOUN_BIGRAMS + NOUN_VERB_BIGRAMS


# ── helpers (same as test 1) ──────────────────────────────────────────────────
def count_concepts(node):
    """Recursively count all nodes in the tree."""
    return 1 + sum(count_concepts(c) for c in node.children)


def path_to_leaf(tree, instance):
    """Return [root, ..., leaf] via greedy DFS."""
    leaf = tree.categorize(instance)
    path = []
    node = leaf
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def print_scores(label, tree, instance):
    path = path_to_leaf(tree, instance)

    tree_lp    = tree.log_prob(instance, 100, False)
    root_lp    = path[0].log_prob_instance(instance)
    leaf_lp    = path[-1].log_prob_instance(instance)

    # basic-level via get_basic (expected-PMI walk from leaf toward root)
    basic_node = path[-1].get_basic(1000, 100)
    basic_lp   = basic_node.log_prob_instance(instance)
    basic_depth = basic_node.depth()

    print(f"\n{'='*60}")
    print(f"  Query: {label}")
    print(f"{'='*60}")
    print(f"  tree  log-prob : {tree_lp:.6f}")
    print(f"  root  log-prob : {root_lp:.6f}  (count={path[0].count})")
    print(f"  leaf  log-prob : {leaf_lp:.6f}  (count={path[-1].count})")
    print(f"  basic log-prob : {basic_lp:.6f}  (depth={basic_depth}, count={basic_node.count})")

    print(f"\n  Full path ({len(path)} nodes):")
    for i, n in enumerate(path):
        lp = n.log_prob_instance(instance)
        marker = " ← basic-level" if n is basic_node else ""
        print(f"    [{i}] depth={i}  count={n.count:5.0f}  lp={lp:.6f}{marker}")


# ── main test ─────────────────────────────────────────────────────────────────
def test_path_logprobs():
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)
    for item in TRAINING:
        tree.ifit(item)

    print(f"\nTree has {count_concepts(tree.root)} concepts, "
          f"root.count={tree.root.count}")

    # ── Frequent NP-internal bigram: Det+Noun (SEEN leaves) ──────────────
    print_scores(
        'Det+Noun  →  "the cat"  (FREQUENT, NP-internal, SEEN leaves)',
        tree,
        inst(*article_def(), *noun_anim(CAT)),
    )

    # ── Rare boundary bigram: Noun+Verb (SEEN leaves) ─────────────────
    print_scores(
        'Noun+Verb  →  "cat runs"  (RARE, boundary-crossing, SEEN leaves)',
        tree,
        inst(*noun_anim(CAT), *verb(RUNS)),
    )

    # ── Unseen leaf, matching Det+Noun pattern: should score near trained NP ─
    # depth-0 (FUNC_WORD) and depth-1 (ARTICLE) still match the NP cluster
    # even though HORSE (depth-2) was never seen during training.
    print_scores(
        'Det+UnseenNoun  →  "the horse"  (UNSEEN leaf, correct NP pattern)',
        tree,
        inst(*article_def(), *noun_anim(HORSE)),
    )

    # ── Unseen noun + seen verb: boundary pattern with novel left token ────
    # depth-0 mismatch (CONTENT_WORD vs the NP cluster's FUNC_WORD) persists.
    print_scores(
        'UnseenNoun+Verb  →  "horse runs"  (UNSEEN N, SEEN V, wrong pattern)',
        tree,
        inst(*noun_anim(HORSE), *verb(RUNS)),
    )

    # ── Fully unseen NP bigram: both leaves new but pattern matches ───────
    # Should still score better than any Noun+Verb pair.
    print_scores(
        'Det+UnseenNoun  →  "a horse"  (UNSEEN leaf, INDEF article, NP pattern)',
        tree,
        inst(*article_indef(), *noun_anim(HORSE)),
    )


if __name__ == "__main__":
    test_path_logprobs()
