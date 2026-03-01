"""
Test 2: Log-probability probe using concept-path IDs (simulated multi-hierarchy output).

In the real framework, content instances encode *both* children as their
label_path through the context hierarchy — 3 depth levels per side:

    {0: {left_d0: 1}, 1: {left_d1: 1}, 2: {left_d2: 1},
     3: {right_d0: 1}, 4: {right_d1: 1}, 5: {right_d2: 1}}

This test simulates that encoding with a small hand-crafted concept tree:

    ROOT
    ├── NOUN_ROOT (100)
    │   ├── ANIMATE_NOUN (101)       ← cat, dog, bird
    │   └── INANIMATE_NOUN (102)     ← fish, mouse
    └── VERB_ROOT (200)
        └── ACTION_VERB (201)        ← runs

Each noun/verb gets a unique leaf ID (1001-1005, 2001).

Training set and queries are identical to test 1 in spirit:
  - 10 N+N pairs  (frequent pattern)
  - 2  N+V pairs  (rare pattern)

Expected insight over test 1: an *unseen* N+N pair (horse+cow, encoded with
NOUN_ROOT at depth-0) should score much closer to the trained N+N cluster
than an N+V pair, because the shallow path attributes match even when the
leaves are new.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree


# ── simulated concept IDs ─────────────────────────────────────────────────────
# depth 0 (coarsest / most general)
NOUN_ROOT    = 100
VERB_ROOT    = 200

# depth 1
ANIMATE_NOUN   = 101
INANIMATE_NOUN = 102
ACTION_VERB    = 201

# depth 2 (leaves)
CAT   = 1001
DOG   = 1002
BIRD  = 1003
FISH  = 1004
MOUSE = 1005
HORSE = 1006   # UNSEEN during training (but same depth-0/1 path as animate nouns)
COW   = 1007   # UNSEEN during training

RUNS  = 2001


def inst(l0, l1, l2, r0, r1, r2):
    """3-depth-per-side content instance (6 attributes total)."""
    return {
        0: {l0: 1.0},  # left  depth 0
        1: {l1: 1.0},  # left  depth 1
        2: {l2: 1.0},  # left  depth 2
        3: {r0: 1.0},  # right depth 0
        4: {r1: 1.0},  # right depth 1
        5: {r2: 1.0},  # right depth 2
    }

# shorthand path tuples
def noun_animate(leaf):    return (NOUN_ROOT, ANIMATE_NOUN,   leaf)
def noun_inanimate(leaf):  return (NOUN_ROOT, INANIMATE_NOUN, leaf)
def verb_action(leaf):     return (VERB_ROOT, ACTION_VERB,    leaf)


# ── training set ──────────────────────────────────────────────────────────────
N_N_PAIRS = [
    inst(*noun_animate(CAT),   *noun_animate(DOG)),
    inst(*noun_animate(CAT),   *noun_animate(BIRD)),
    inst(*noun_animate(DOG),   *noun_animate(BIRD)),
    inst(*noun_animate(CAT),   *noun_inanimate(FISH)),
    inst(*noun_animate(DOG),   *noun_inanimate(FISH)),
    inst(*noun_animate(BIRD),  *noun_inanimate(FISH)),
    inst(*noun_animate(CAT),   *noun_inanimate(MOUSE)),
    inst(*noun_animate(DOG),   *noun_inanimate(MOUSE)),
    inst(*noun_animate(BIRD),  *noun_inanimate(MOUSE)),
    inst(*noun_inanimate(FISH),*noun_inanimate(MOUSE)),
]

N_V_PAIRS = [
    inst(*noun_animate(CAT),   *verb_action(RUNS)),
    inst(*noun_animate(DOG),   *verb_action(RUNS)),
]

TRAINING = N_N_PAIRS + N_V_PAIRS


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

    # Frequent instance: N+N (seen pattern)
    print_scores(
        "N+N  →  cat(animate) + dog(animate)  (FREQUENT, SEEN LEAVES)",
        tree,
        inst(*noun_animate(CAT), *noun_animate(DOG)),
    )

    # Rare instance: N+V  (rare pattern)
    print_scores(
        "N+V  →  cat(animate) + runs(action)  (RARE, SEEN LEAVES)",
        tree,
        inst(*noun_animate(CAT), *verb_action(RUNS)),
    )

    # Unseen leaves but correct pattern: should score better than N+V
    # because depth-0 and depth-1 attributes still match the N+N cluster
    print_scores(
        "N+N  →  horse(animate) + cow(animate)  (UNSEEN LEAVES, correct pattern)",
        tree,
        inst(*noun_animate(HORSE), *noun_animate(COW)),
    )

    # Unseen right-side category entirely: mixed unseen
    print_scores(
        "N+V  →  horse(animate) + runs(action)  (UNSEEN N, SEEN V, wrong pattern)",
        tree,
        inst(*noun_animate(HORSE), *verb_action(RUNS)),
    )


if __name__ == "__main__":
    test_path_logprobs()
