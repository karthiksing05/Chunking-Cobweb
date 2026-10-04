import random

from trellis2.cobweb import CobwebTree


def random_instances(n, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        yield {"a": rng.choice("xyz"), "b": rng.choice("pq"), "c": rng.choice("uvwst")}


def test_counts_stay_consistent():
    tree = CobwebTree(("a", "b", "c"), seed=1)
    for x in random_instances(400):
        tree.ifit(x, w=random.choice([1.0, 0.5, 2.0]))
    tree.check_invariants()
    assert tree.root.count > 0


def test_leaves_are_stable_handles():
    tree = CobwebTree(("a", "b", "c"), seed=2)
    stored = []
    for x in random_instances(300, seed=3):
        stored.append((x, tree.ifit(x)))
    leaves = {id(l) for l in tree.leaves()}
    for x, leaf in stored:
        assert id(leaf) in leaves, "a leaf was removed or turned internal"
        assert all(x[a] in leaf.av[a] and len(leaf.av[a]) == 1 for a in x)


def test_identical_instances_share_a_leaf():
    tree = CobwebTree(("a", "b"), seed=0)
    first = tree.ifit({"a": "x", "b": "p"})
    tree.ifit({"a": "y", "b": "q"})
    again = tree.ifit({"a": "x", "b": "p"})
    assert first is again and first.count == 2.0


def test_bag_attributes_keep_counts_consistent():
    rng = random.Random(4)
    tree = CobwebTree(("a", "bag"), seed=3)
    for _ in range(300):
        words = [rng.choice("pqrst") for _ in range(rng.randint(1, 4))]
        bag = {}
        for w_ in words:
            bag[w_] = bag.get(w_, 0.0) + 1.0 / len(words)
        tree.ifit({"a": rng.choice("xy"), "bag": bag}, w=rng.choice([1.0, 0.5]))
    tree.check_invariants()


def _shape(tree):
    return [(n.id, n.parent.id if n.parent is not None else None, [c.id for c in n.children],
             n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.nodes()]


def _instances(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        words = [rng.choice("pqrst") for _ in range(rng.randint(1, 4))]
        bag = {}
        for w_ in words:
            bag[w_] = bag.get(w_, 0.0) + 1.0 / len(words)
        yield {"a": rng.choice("xyz"), "b": rng.choice([1, 2, "q"]), "bag": bag}, rng.choice([1.0, 0.5, 2.0])


def test_compiled_tree_reproduces_the_reference_exactly():
    """Same instances, same seed: the same nodes, links, counts and leaves."""
    import sys
    import pytest
    if sys.version_info < (3, 12):
        pytest.skip("the reference sums with Python 3.12's compensated sum()")
    from reference_cobweb import CobwebTree as Reference
    data = list(_instances(2000, 5))
    ref, fast = Reference(("a", "b", "bag"), seed=7), CobwebTree(("a", "b", "bag"), seed=7)
    assert [ref.ifit(x, w).id for x, w in data] == [fast.ifit(x, w).id for x, w in data]
    assert _shape(ref) == _shape(fast)


def test_concept_codes_match_the_attribute_counts():
    import numpy as np
    from scipy.special import gammaln
    tree = CobwebTree(("a", "b", "bag"), seed=3)
    for x, w in _instances(500, 9):
        tree.ifit(x, w)
    alpha = 0.5
    n_values = {a: max(len(d), 1) for a, d in tree.root.av.items()}
    expected = []
    for node in tree.nodes():
        tot = 0.0
        for a, d in node.av.items():
            cnt = np.array(list(d.values()))
            tot += (gammaln(cnt.sum() + n_values[a] * alpha) - gammaln(n_values[a] * alpha)
                    - np.sum(gammaln(cnt + alpha) - gammaln(alpha)))
        expected.append(tot)
    assert np.allclose(tree.concept_codes(alpha), expected, rtol=1e-12, atol=1e-9)
