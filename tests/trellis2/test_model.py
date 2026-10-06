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
def test_unsupervised_chunking_compresses_and_generates_the_language():
    from trellis2.unsupervised import UnsupervisedLearner
    examples = load_corpus(SMALL)
    train, test = v1_split(examples, seed=13)
    learner = UnsupervisedLearner(seed=13)
    for ex in train[:60]:
        learner.observe(ex.tokens)
    learner.sleep()
    flat = learner.history[0]["bits"]
    # Chunks were formed only because they shorten the description.
    assert learner.grammar.info["chunk types"] > 0
    assert learner.grammar.info["total bits"] < 0.75 * flat
    for ex in test[:10]:
        assert learner.parse(ex.tokens).is_valid()
    cfg = CFG(target_grammar("small"))
    samples, _ = learner.generate(200, np.random.default_rng(0))
    assert np.mean([cfg.recognizes(t) for t, _ in samples]) > 0.95


@needs_data
def test_learning_by_day_and_by_night():
    from trellis2.unsupervised import UnsupervisedLearner
    examples = load_corpus(SMALL)
    train, test = v1_split(examples, seed=13)
    learner = UnsupervisedLearner(seed=13)
    for ex in train[:20]:
        assert learner.observe(ex.tokens) is None      # no grammar yet
    learner.sleep()
    for ex in train[20:60]:
        tree = learner.observe(ex.tokens)              # perceived with the grammar
        assert tree.n == len(ex.tokens) and tree.is_valid()
    assert all(a is not None for a in learner.analyses)
    learner.sleep()
    assert [h["night"] for h in learner.history][-1] == 1
    cfg = CFG(target_grammar("small"))
    samples, _ = learner.generate(200, np.random.default_rng(0))
    assert np.mean([cfg.recognizes(t) for t, _ in samples]) > 0.95


@needs_data
def test_parallel_nights_learn_the_same_grammar():
    """A night's searches and consolidations are independent of each other,
    so running them in parallel processes changes nothing (every Cobweb tree
    has its own seed), on a first night and on one that starts from what the
    day perceived."""
    from trellis2.unsupervised import UnsupervisedLearner
    train, _ = v1_split(load_corpus(SMALL), seed=13)
    grammars = []
    for workers in (1, 3):
        learner = UnsupervisedLearner(seed=13, workers=workers)
        for ex in train[:20]:
            learner.observe(ex.tokens)
        learner.sleep()
        for ex in train[20:40]:
            learner.observe(ex.tokens)
        grammars.append(learner.sleep())
    a, b = grammars
    assert a.info["total bits"] == b.info["total bits"]
    for table in ("S", "S_piece", "U", "pk", "Lt", "Rt", "E"):
        assert np.array_equal(getattr(a, table), getattr(b, table))


def test_a_forest_piece_is_described_like_a_whole_trees_part():
    from trellis2.memory import Memory, ROOT, SENTENCE
    mem = Memory(sentence_parent=True)
    mem.add(["tim", "was", "very", "happy"], Tree(4, {(0, 4): 2, (0, 2): 1, (2, 4): 3}, {}, [(0, 4)]))
    mem.add(["it", "was", "so", "fun"], Tree(4, {(0, 2): 1, (2, 4): 3}, {}, [(0, 2), (2, 4)]))
    labels = [np.zeros(len(mem), dtype=int)] * mem.granularities
    spine = lambda e: {k: v for k, v in mem.instance(e, labels).items() if k[0] in "as"}
    part = next(e for e in range(len(mem)) if mem.describe(e) == "tim was")
    piece = next(e for e in range(len(mem)) if mem.describe(e) == "it was")
    assert spine(part) == spine(piece)
    assert spine(part)["a1.0"] == SENTENCE and spine(part)["a2.0"] == ROOT
    root = next(e for e in range(len(mem)) if mem.describe(e) == "tim was very happy")
    assert spine(root)["a1.0"] == ROOT


def test_a_piece_read_afresh_starts_from_bos():
    from trellis2.memory import BOS, Memory
    forest = Tree(4, {(0, 2): 1, (2, 4): 3}, {}, [(0, 2), (2, 4)])
    whole = Tree(4, {(0, 4): 2, (0, 2): 1, (2, 4): 3}, {}, [(0, 4)])
    for fresh in (False, True):
        mem = Memory(fresh_pieces=fresh)
        mem.add(["it", "was", "so", "fun"], forest)
        mem.add(["tim", "was", "so", "happy"], whole)
        ctx = dict(zip(((mem.experience_of[e], mem.span[e]) for e in range(len(mem))), mem.contexts()))
        assert ctx[(0, (2, 4))] == ctx[(0, (2, 3))] == (BOS if fresh else "was")   # the second piece
        assert ctx[(0, (3, 4))] == "so" and ctx[(1, (2, 4))] == "was"              # inside it; a whole tree

