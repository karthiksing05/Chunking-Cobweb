"""Cobweb with insertion and removal: the incremental tree against the
reference and the compiled tree, and removal restoring the counts."""
import math
import random
import sys

import numpy as np
import pytest

from trellis2.incremental_cobweb import IncrementalCobwebTree

def _instances(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        words = [rng.choice("pqrst") for _ in range(rng.randint(1, 4))]
        bag = {}
        for w_ in words:
            bag[w_] = bag.get(w_, 0.0) + 1.0 / len(words)
        yield {"a": rng.choice("xyz"), "b": rng.choice([1, 2, "q"]), "bag": bag}, rng.choice([1.0, 0.5, 2.0])


def _plain(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        yield {"a": rng.choice("xyz"), "b": rng.choice("pq"), "c": rng.choice("uvwst")}, rng.choice([1.0, 2.0])


def _shape(tree):
    return [(n.id, n.parent.id if n.parent is not None else None, [c.id for c in n.children],
             n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.nodes()]


def _leaf_counts(tree):
    return {n.id: (n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.leaves()}


# --------------------------------------------------------------------------- #
# Cobweb with removal
# --------------------------------------------------------------------------- #
def test_insertion_reproduces_the_reference_and_the_compiled_tree():
    if sys.version_info < (3, 12):
        pytest.skip("the reference sums with Python 3.12's compensated sum()")
    from reference_cobweb import CobwebTree as Reference
    from trellis2.cobweb import CobwebTree as Compiled
    data = list(_instances(1500, 5))
    ref = Reference(("a", "b", "bag"), seed=7)
    inc = IncrementalCobwebTree(("a", "b", "bag"), seed=7)
    fast = Compiled(("a", "b", "bag"), seed=7)
    ids = [inc.ifit(x, w).id for x, w in data]
    assert ids == [ref.ifit(x, w).id for x, w in data] == [fast.ifit(x, w).id for x, w in data]
    assert _shape(inc) == _shape(ref) == _shape(fast)
    assert np.allclose(inc.concept_codes(0.5), fast.concept_codes(0.5), rtol=1e-12, atol=1e-9)


def test_insert_then_remove_restores_the_counts():
    tree = IncrementalCobwebTree(("a", "b", "c"), seed=1)
    for x, w in _plain(300, 2):
        tree.ifit(x, w)
    before_leaves = _leaf_counts(tree)
    before_root = {a: dict(d) for a, d in tree.root.av.items()}
    count = tree.root.count
    for x, w in _plain(60, 4):
        leaf = tree.ifit(x, w)
        tree.unfit(leaf, x, w)
        tree.check_invariants()
        assert tree.root.count == pytest.approx(count, abs=1e-12)
        for a, d in before_root.items():
            assert tree.root.av[a] == pytest.approx(d)
        # Every leaf that held instances before holds exactly them again.
        after = _leaf_counts(tree)
        for nid, (c, av) in before_leaves.items():
            assert nid in after and after[nid][0] == pytest.approx(c)
            for a in av:
                assert after[nid][1][a] == pytest.approx(av[a])
        assert set(after) == set(before_leaves)


def test_removing_everything_in_any_order_keeps_the_invariants():
    tree = IncrementalCobwebTree(("a", "b", "bag"), seed=3)
    stored = []
    for x, w in _instances(400, 9):
        stored.append((tree.ifit(x, w), x, w))
    rng = random.Random(0)
    rng.shuffle(stored)
    leaves_of = {}
    for leaf, x, w in stored:
        leaves_of.setdefault(id(leaf), []).append((x, w))
    for k, (leaf, x, w) in enumerate(stored):
        tree.unfit(leaf, x, w)
        if k % 25 == 0:
            tree.check_invariants(tol=1e-6)
            # Leaves are stable handles: each still holds only its instances.
            live = {id(n) for n in tree.leaves()}
            for l2, _, _ in stored[k + 1:]:
                assert id(l2) in live
    assert tree.root.count == 0.0 and not tree.root.children


def test_reinsertion_after_removal_matches_a_count_check():
    tree = IncrementalCobwebTree(("a", "b", "c"), seed=5)
    stored = [(tree.ifit(x, w), x, w) for x, w in _plain(500, 6)]
    rng = random.Random(1)
    for _ in range(300):
        k = rng.randrange(len(stored))
        leaf, x, w = stored[k]
        tree.unfit(leaf, x, w)
        x2 = dict(x, c=rng.choice("uvwst"))          # a re-description
        stored[k] = (tree.ifit(x2, w), x2, w)
    tree.check_invariants()
    assert tree.root.count == pytest.approx(sum(w for _, _, w in stored))
    for a in ("a", "b", "c"):
        expect = {}
        for _, x, w in stored:
            expect[x[a]] = expect.get(x[a], 0.0) + w
        assert tree.root.av[a] == pytest.approx(expect)
