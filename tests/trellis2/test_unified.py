"""One framework, three domains. A domain's memory records analysed
experiences element by element; the two hierarchies and the grammar are
built from the records in the same way for every domain; the grammar codes an
experience (``log_prob``) and draws one (``generate``) in the domain's own
reading order. Sentences, characters and chess positions go through the same
calls here, and in each the grammar draws an experience as often as its code
says it should."""
import math
import os
import random
from collections import Counter

import numpy as np
import pytest

from trellis2 import Trellis2, load_corpus, v1_split
from trellis2.characters import RELATIONS, CharacterMemory, from_relational, placements
from trellis2.chess import ChessLearner, plausibility, render
from trellis2.data import CONDITIONS, default_data_root

SMALL = os.path.join(default_data_root(), CONDITIONS["small"])


def toy_characters():
    """A toy script: a left radical beside a right component, a top over a
    bottom, and a left radical beside a top-over-bottom."""
    left, right, top, bottom = ["氵", "木", "亻", "口"], ["古", "可", "主", "月"], ["艹", "日"], ["口", "月"]
    s = [("⿰", a, b) for a in left for b in right]
    s += [("⿱", a, b) for a in top for b in bottom]
    s += [("⿰", a, ("⿱", b, c)) for a in left[:2] for b in top for c in bottom]
    return s


def shielded_position(rng: random.Random) -> dict:
    """Each king behind three pawns, on a random file; two minor pieces."""
    p = {}
    for colour, back, front in (("w", 0, 1), ("b", 7, 6)):
        f = rng.randrange(1, 7)
        p[(f, back)] = colour + "K"
        for df in (-1, 0, 1):
            p[(f + df, front)] = colour + "P"
    free = [(x, y) for x in range(8) for y in range(2, 6)]
    for tok in rng.sample(["wN", "wB", "bN", "bB"], 2):
        p[rng.choice([s for s in free if s not in p])] = tok
    return p


def test_relational_characters_end_to_end():
    train = toy_characters()
    model = Trellis2(seed=0, memory=CharacterMemory())
    for c in train:
        model.learn(c)
    g = model.consolidate()
    for table in (g.S, g.U, g.Lt, g.Rt, g.E, g.Rel):
        assert np.allclose(table.sum(axis=-1), 1.0)
    # Every generated tree is a character: relations join known components.
    components = {part for c in train for _, _, part in placements(c)}
    samples, _ = model.generate(300, np.random.default_rng(0))
    for tree in samples:
        for rel, _, part in placements(from_relational(tree)):
            assert rel in RELATIONS and part in components
    # The slot keeps the sides apart: a right component on the left costs.
    assert model.log_prob(("⿰", "氵", "古")) > model.log_prob(("⿰", "古", "氵")) + 5


def test_chess_learner_end_to_end():
    rng = random.Random(0)
    positions = [shielded_position(rng) for _ in range(40)]
    learner = ChessLearner(seed=0, max_steps=40)
    for p in positions:
        learner.observe(p)
    g = learner.sleep()
    # The search shortens the plain code with a chunk of a king and its shield.
    searched = [h["bits"] for h in learner.history if h["stage"] in ("flat", "chunk")]
    assert searched[-1] < searched[0]
    chunks = {"".join(sorted((B, C))) for B, _, C, _ in learner.search.moves}
    assert chunks & {"wKwP", "bKbP"}
    for p in positions[:10]:
        tops = learner.analyse(p)
        covered = Counter(sq for t in tops for sq in _squares(t))
        assert set(covered) == set(p) and max(covered.values()) == 1
        assert any("K" in render(t, p) and "P" in render(t, p) for t in tops)
        assert np.isfinite(learner.log_prob(p))
    for table in (g.U, g.Lt, g.Rt, g.E, g.Rel, g.T):
        assert np.allclose(table.sum(axis=-1), 1.0)
    assert ((g.Q > 0) & (g.Q < 1)).all()          # the read's yes-or-no questions
    samples, _ = learner.generate(200, np.random.default_rng(0))
    assert np.mean([plausibility(p)["one king each"] for p, _ in samples]) > 0.9


def _squares(node):
    if len(node[1]) == 3:
        x, _, y = node[1]
        return _squares(x) + _squares(y)
    return [node[1]]


def _shape(node):
    """A chess analysis without its category labels (which ``log_prob``
    sums over)."""
    if len(node[1]) == 3:
        x, rel, y = node[1]
        return (_shape(x), rel, _shape(y))
    return node[1]


def _sentences():
    if not os.path.isdir(SMALL):
        pytest.skip("paper corpora not found")
    train, _ = v1_split(load_corpus(SMALL), seed=13)
    model = Trellis2(seed=13)
    for ex in train[:60]:
        model.learn(ex.tokens, ex.tree)
    return model, lambda s: tuple(s[0]), lambda s: model.log_prob(s[0])


def _characters():
    model = Trellis2(seed=0, memory=CharacterMemory())
    for c in toy_characters():
        model.learn(c)
    return model, lambda s: s, model.log_prob


def _chess():
    # A king and three pawns per side in fixed places, and a knight on one of four squares.
    fixed = {(6, 0): "wK", (5, 1): "wP", (6, 1): "wP", (7, 1): "wP",
             (6, 7): "bK", (5, 6): "bP", (6, 6): "bP", (7, 6): "bP"}
    spots = [(2, 2), (3, 3), (4, 3), (1, 2)]
    learner = ChessLearner(seed=0, max_steps=30)
    for i in range(40):
        learner.observe({**fixed, spots[i % 4]: "wN"})
    learner.sleep()
    key = lambda s: (frozenset(s[0].items()), tuple(_shape(t) for t in s[1]))  # noqa: E731
    return learner, key, lambda s: learner.model.log_prob(s[0], s[1])


@pytest.mark.parametrize("domain", ["sentences", "characters", "chess"])
def test_the_grammar_draws_what_it_codes(domain):
    """Draw twice; the experiences drawn most often the first time are drawn
    the second time as often as exp(log_prob) says, once the draws the
    domain rejects (too long, a piece off the board) are accounted for."""
    model, key, log_prob = {"sentences": _sentences, "characters": _characters, "chess": _chess}[domain]()
    n = 6000
    first, _ = model.generate(n, np.random.default_rng(0))
    second, rejected = model.generate(n, np.random.default_rng(1))
    accepted = n / (n + rejected)
    some = {}
    for s in first:
        some.setdefault(key(s), s)
    counts = Counter(key(s) for s in second)
    for k, _ in Counter(key(s) for s in first).most_common(5):
        expected = n * math.exp(log_prob(some[k])) / accepted
        assert abs(counts[k] - expected) <= 4 * math.sqrt(expected) + 2, (k, counts[k], expected)
