"""TRELLIS v3 candidate trees: hidden references and the trees' consistency."""
import os

import pytest

from trellis2.candidates import REF, CandidateLearner
from trellis2.incremental_cobweb import IncrementalCobwebTree


def test_hidden_values_do_not_change_sorting_and_sum_to_the_counts():
    a = IncrementalCobwebTree(("x", "y"), seed=3)
    b = IncrementalCobwebTree(("x", "y"), seed=3)
    import random
    rng = random.Random(0)
    for k in range(300):
        inst = {"x": rng.choice("abc"), "y": rng.choice("pqrs")}
        la = a.ifit(inst)
        lb = b.ifit(inst)
        b.hide(lb, {"ref": k % 7})
        assert la.id == lb.id
    assert sum(b.hidden_counts(b.root, "ref").values()) == pytest.approx(b.root.count)
    b.check_invariants()


def test_learner_keeps_trees_and_references_consistent():
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    path = os.path.join(default_data_root(), CONDITIONS["small"])
    if not os.path.exists(path):
        pytest.skip("synthetic corpora not available")
    train, _ = v1_split(load_corpus(path), 13)
    L = CandidateLearner(seed=13)
    for e in train[:60]:
        L.learn(e.tokens)
    L.ctx.check_invariants()
    L.cnt.check_invariants()
    refs = L.ctx.hidden_counts(L.ctx.root, REF)
    assert sum(refs.values()) == pytest.approx(L.ctx.root.count)
    # every composite's reference names the content leaf that stores it
    for c in L.records:
        assert c.ref == ("cnt", c.cleaf.id) and c.cleaf.count > 0
    assert L.cnt.root.count == pytest.approx(len(L.records))
    assert sum(L.generate() is not None for _ in range(20)) > 0
