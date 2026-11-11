"""SubstitutionLearner: one memory read as experiences, type trees and n-gram counts."""
import math
import random

from trellis2.substitution import SubstitutionLearner

STREAM = [["a", "x", "p"], ["a", "y", "p"], ["b", "x", "q"], ["c", "z", "q"], ["c", "w", "p"],
          ["b", "y", "q"]] * 3


def _learner(stream=STREAM, **kw):
    L = SubstitutionLearner(0, **kw)
    for i, s in enumerate(stream):
        L.learn(s, follows=i > 0)
    return L


def test_generate_replaces_one_token():
    L = _learner()
    learned = {tuple(s) for s in STREAM}
    rng = random.Random(1)
    outs = [L.generate(prev=["a", "x", "p"], rng=rng) for _ in range(20)]
    assert any(o is not None for o in outs)
    for o in outs:
        if o is None:
            continue
        assert any(len(o) == len(s) and sum(x != y for x, y in zip(o, s)) == 1 for s in learned)


def test_require_new_has_unstored_pair():
    L = _learner()
    stored = {p for s in STREAM for p in zip(["<s>"] + s, s + ["</s>"])}
    rng = random.Random(2)
    got = [L.generate(rng=rng, require_new=True) for _ in range(20)]
    got = [o for o in got if o]
    assert got
    for o in got:
        pairs = set(zip(["<s>"] + o, o + ["</s>"]))
        assert pairs - stored


def test_deterministic():
    a, b = _learner(), _learner()
    ra, rb = random.Random(5), random.Random(5)
    for _ in range(10):
        assert a.generate(prev=["b", "x", "q"], rng=ra) == b.generate(prev=["b", "x", "q"], rng=rb)


def test_bits_order_and_escape():
    L = _learner()
    s = ["a", "x", "p"]
    assert math.isfinite(L.bits(s))
    assert L.bits(s) < L.bits(s[::-1])
    assert abs(L.bits(s + ["unseen", "unseen2"], escape=2.5) - L.bits(s + ["unseen", "unseen2"]) - 5.0) < 1e-9


def test_incremental():
    A, B = STREAM[:4], STREAM[4:8]
    L = SubstitutionLearner(0)
    for k, s in enumerate(A):
        L.learn(s, follows=k > 0)
        assert math.isfinite(L.bits(s))
        L.generate(rng=random.Random(0))
    for s in B:
        L.learn(s, follows=True)
    F = SubstitutionLearner(0)
    for k, s in enumerate(A):
        F.learn(s, follows=k > 0)
    for s in B:
        F.learn(s, follows=True)
    for s in STREAM:
        assert L.bits(s) == F.bits(s)
    assert L.generate(prev=A[0], rng=random.Random(3)) == F.generate(prev=A[0], rng=random.Random(3))
