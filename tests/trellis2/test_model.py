import os

import numpy as np
import pytest

from trellis2 import Trellis2, Tree, load_corpus, tree_from_merges, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG

DATA = default_data_root()
SMALL = os.path.join(DATA, CONDITIONS["small"])
needs_data = pytest.mark.skipif(not os.path.isdir(SMALL), reason="paper corpora not found")


def test_tree_from_merges_and_brackets():
    # "the dog saw a cat": ((the dog) (saw (a cat)))
    merges = [{"left": 0, "right": 1}, {"left": 3, "right": 4},
              {"left": 2, "right": 3.5}, {"left": 0.5, "right": 2.75}]
    t = tree_from_merges(5, merges)
    assert t.brackets() == {(0, 2), (3, 5), (2, 5), (0, 5)}
    assert Tree.from_brackets(5, t.brackets()).brackets() == t.brackets()


@needs_data
def test_v1_split_sizes():
    examples = load_corpus(SMALL)
    train, test = v1_split(examples, seed=13)
    assert len(examples) == 400 and len(train) == 320 and len(test) == 40


@needs_data
def test_small_grammar_end_to_end():
    examples = load_corpus(SMALL)
    train, test = v1_split(examples, seed=13)
    model = Trellis2(seed=13)
    for ex in train[:60]:
        model.learn(ex.tokens, ex.tree)
    g = model.consolidate()
    for table in (g.S, g.U, g.Lt, g.Rt, g.E):
        assert np.allclose(table.sum(axis=-1), 1.0)
    for ex in test[:10]:
        assert model.parse(ex.tokens).is_valid()
    cfg = CFG(target_grammar("small"))
    samples, _ = model.generate(50, np.random.default_rng(0))
    for tokens, tree in samples:
        assert tree.n == len(tokens) and tree.is_valid()
    assert np.mean([cfg.recognizes(t) for t, _ in samples]) > 0.8


@needs_data
def test_unsupervised_learner_recovers_small_grammar():
    from trellis2.unsupervised import UnsupervisedLearner
    examples = load_corpus(SMALL)
    train, test = v1_split(examples, seed=13)
    learner = UnsupervisedLearner(inits=("balanced", "right"), em_iterations=4, seed=13)
    for ex in train[:80]:
        learner.observe(ex.tokens)
    learner.consolidate()
    # Description length prefers the balanced analysis over right-branching;
    # on SMALL that analysis is the gold one, so held-out parses match gold.
    from trellis2.evaluation import BracketTally
    tally = BracketTally()
    for ex in test[:10]:
        tally.add(ex.tree, learner.parse(ex.tokens))
    assert tally.omission == 0.0
