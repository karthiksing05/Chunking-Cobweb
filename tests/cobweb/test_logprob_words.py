"""
Test 1: Log-probability probe using raw word IDs – bigram framing.

Modelling surface bigrams extracted from sentences of the form:
    S → NP VP,  NP → Det N,  VP → V NP

Training bigrams:
  - 10  Det+Noun  bigrams  (NP-internal, frequent and cohesive)
  - 2   Noun+Verb bigrams  (NP→VP boundary crossing, rare)

Each bigram instance is a simple 2-attribute record:
  {0: {left_token_id: 1}, 1: {right_token_id: 1}}

For a frequent (Det+Noun), a rare (Noun+Verb), and an unseen (Det+unseen_noun)
query we print:
  • tree  log-prob  (tree.log_prob, max_nodes=100)
  • root  log-prob  (root node, log_prob_instance)
  • leaf  log-prob  (greedy-DFS leaf, log_prob_instance)
  • basic-level    (node on root→leaf path with max log_prob_instance)

Expected insight:
  Det+Noun scores highest (10 training instances, tight cluster).
  Noun+Verb scores lowest (only 2 instances, mixed cluster).
  Det+unseen_noun scores close to Det+Noun because the left token
  (DET) matches perfectly even when the right noun is novel.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree


# ── vocabulary ────────────────────────────────────────────────────────────────
# Determiners (left side of NP-internal bigrams)
THE  = 1
A    = 2

# Nouns (right side of NP-internal bigrams, or left side of N+V boundary bigrams)
CAT   = 3
DOG   = 4
BIRD  = 5
FISH  = 6
MOUSE = 7

# Nouns unseen during training (but same morphological category as above)
HORSE = 8
COW   = 9

# Verbs (right side of NP→VP boundary crossing bigrams)
RUNS = 20
SEES = 21


def inst(left, right):
    """Build a 2-attribute bigram content instance."""
    return {0: {left: 1.0}, 1: {right: 1.0}}


# ── training bigrams ──────────────────────────────────────────────────────────
# Det+Noun bigrams: NP-internal, high cohesion (e.g. "the cat", "a dog").
# Represent the bigrams (the, cat), (the, dog), ... (a, fish), (a, mouse).
DET_NOUN_BIGRAMS = [
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, DOG),
    inst(THE, BIRD),
    inst(THE, FISH),
    inst(THE, MOUSE),
    inst(A,   CAT),
    inst(A,   DOG),
    inst(A,   BIRD),
    inst(A,   FISH),
    inst(A,   MOUSE),
]

# Noun+Verb bigrams: NP→VP boundary crossing, low cohesion (e.g. "cat runs").
NOUN_VERB_BIGRAMS = [
    inst(CAT, RUNS),
    inst(DOG, RUNS),
]

TRAINING = DET_NOUN_BIGRAMS + NOUN_VERB_BIGRAMS


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

    # ── Frequent bigram: Det+Noun (NP-internal) ──────────────────────────
    # (the, cat) seen during training → should score highest
    print_scores('Det+Noun  →  "the cat"  (FREQUENT, NP-internal bigram)',
                 tree, inst(THE, CAT))

    # ── Rare bigram: Noun+Verb (NP→VP boundary crossing) ────────────────
    # (cat, runs) seen only twice → should score much lower
    print_scores('Noun+Verb  →  "cat runs"  (RARE, boundary-crossing bigram)',
                 tree, inst(CAT, RUNS))

    # ── Unseen right token, matching left-token type (Det) ───────────────
    # (the, horse): horse never seen in training, but left side (DET=THE)
    # matches perfectly with the Det+Noun cluster → should score near the
    # trained Det+Noun pairs despite the novel right token.
    print_scores('Det+UnseenNoun  →  "the horse"  (UNSEEN right, DET left matches cluster)',
                 tree, inst(THE, HORSE))

    # ── Unseen in wrong pattern: novel Noun+Verb ─────────────────────────
    # (horse, runs): both tokens cross-bracket even when horse is unseen.
    print_scores('UnseenNoun+Verb  →  "horse runs"  (UNSEEN left + boundary-crossing)',
                 tree, inst(HORSE, RUNS))

    # ── Cross-type boundary (should score similarly to N+V) ──────────────
    # (cat, a): Noun followed by Det – reverse of the NP pattern; never trained.
    print_scores('Noun+Det  →  "cat a"  (UNTRAINED reversed bigram)',
                 tree, inst(CAT, A))


if __name__ == "__main__":
    test_word_logprobs()
