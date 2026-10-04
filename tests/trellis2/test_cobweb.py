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
