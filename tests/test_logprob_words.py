"""
Test 1: Log-probability probe using raw word IDs.

Training set:
  - 10 "N+N" pairs  (frequent pattern)
  - 2  "N+V" pairs  (rare pattern)

Each instance is a simple 2-attribute content-style record:
  {0: {left_word_id: 1}, 1: {right_word_id: 1}}

For both a frequent (N+N) and a rare (N+V) query instance we print:
  • tree  log-prob  (tree.log_prob, max_nodes=100)
  • root  log-prob  (root node, log_prob_instance)
  • leaf  log-prob  (greedy-DFS leaf, log_prob_instance)
  • basic-level    (node on root→leaf path with max log_prob_instance)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree


# ── vocabulary ────────────────────────────────────────────────────────────────
CAT, DOG, BIRD, FISH, MOUSE, HORSE, COW, FROG, BAT, SNAKE = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
RUNS = 20

def inst(left, right):
    """Build a 2-attribute content instance."""
    return {0: {left: 1.0}, 1: {right: 1.0}}


# ── training set ──────────────────────────────────────────────────────────────
N_N_PAIRS = [
    inst(CAT,   DOG),
    inst(CAT,   BIRD),
    inst(DOG,   BIRD),
    inst(CAT,   FISH),
    inst(DOG,   FISH),
    inst(BIRD,  FISH),
    inst(CAT,   MOUSE),
    inst(DOG,   MOUSE),
    inst(BIRD,  MOUSE),
    inst(FISH,  MOUSE),
]

N_V_PAIRS = [
    inst(CAT,  RUNS),
    inst(DOG,  RUNS),
]

TRAINING = N_N_PAIRS + N_V_PAIRS


# ── helpers ───────────────────────────────────────────────────────────────────
def count_concepts(node):
    """Recursively count all nodes in the tree."""
    return 1 + sum(count_concepts(c) for c in node.children)


def path_to_leaf(tree, instance):
    """Return [root, ..., leaf] by greedy DFS (categorize)."""
    leaf = tree.categorize(instance)
    path = []
    node = leaf
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def print_scores(label, tree, instance):
    """Print the four log-probability summaries for *instance*."""
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
def test_word_logprobs():
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)
    for item in TRAINING:
        tree.ifit(item)

    print(f"\nTree has {count_concepts(tree.root)} concepts, "
          f"root.count={tree.root.count}")

    # Frequent instance: N+N (seen 10 times in this pattern group)
    print_scores("N+N  →  cat + dog  (FREQUENT)", tree, inst(CAT, DOG))

    # Rare instance: N+V (only 2 examples, entirely different right-side vocab)
    print_scores("N+V  →  cat + runs (RARE)",     tree, inst(CAT, RUNS))

    # Extra: completely unseen combination of known types
    print_scores("N+N  →  horse + cow (UNSEEN N+N)", tree, inst(HORSE, COW))


if __name__ == "__main__":
    test_word_logprobs()
