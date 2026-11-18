"""Round 203: Trellis3R203 (edge counted once, Goodman readouts) is Trellis3 unless an option is set."""
import itertools
import math
import random

import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3
from trellis2.trellis3_r203 import Trellis3R203, goodman_binary, goodman_nary, span_posteriors

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts")
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]


def build(cls, **kw):
    L = cls(seed=3, **CFG, **kw)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


@pytest.fixture(scope="module")
def base():
    return build(Trellis3)


@pytest.fixture(scope="module")
def plain():
    return build(Trellis3R203)


def same(a, b):
    assert a["chosen"] == b["chosen"]
    assert set(a["post"]) == set(b["post"])
    assert all(a["post"][k] == b["post"][k] for k in a["post"])
    assert a["logZ"] == b["logZ"]


def test_no_option_is_the_default(base, plain):
    for s in ("the dog ran to the park", "they were happy to play", "he went home"):
        same(base.parse(s.split()), plain.parse(s.split()))


def test_edge_w_two_is_the_default_gain():
    """Two edge slots at weight 2/2 = 1 each: the weighted gain is the default's (up to float rounding)."""
    L2 = build(Trellis3R203, edge_w=2.0)
    B = build(Trellis3)
    for s in ("the dog ran to the park", "they were happy to play"):
        a, b = B.parse(s.split()), L2.parse(s.split())
        assert a["chosen"] == b["chosen"]
        for k in a["post"]:
            assert a["post"][k] == pytest.approx(b["post"][k], abs=1e-9)


def test_edge_once_changes_only_edge_spans():
    L1 = build(Trellis3R203, edge_w=1.0)
    toks = "the dog ran to the park".split()
    n = len(toks)
    words, comps = L1._chart(toks)
    L1.edge_w = None
    words0, comps0 = L1._chart(toks)
    L1.edge_w = 1.0
    changed = set()
    for key, e in comps.items():
        if abs(e.gain - comps0[key].gain) > 1e-9:
            changed.add(key)
        i, k, j = key
        if i > 0 and j < n:
            assert e.gain == pytest.approx(comps0[key].gain, abs=1e-12)
    assert changed and all(i == 0 or j == n for i, k, j in changed)


def test_edge_weights():
    L = Trellis3R203(seed=0, edge_w=1.0, **CFG)
    assert L._edge_weights({"l1": BOS, "l2": BOS, "r1": "x", "r2": "y"}) == {"l1": 0.5, "l2": 0.5}
    assert L._edge_weights({"l1": "a", "l2": BOS, "r1": "x", "r2": EOS}) is None
    assert L._edge_weights({"l1": BOS, "l2": BOS, "r1": EOS, "r2": EOS}) == {"l1": 0.5, "l2": 0.5, "r1": 0.5, "r2": 0.5}


def all_binary(i, j):
    """All binary trees over [i, j) as sets of spans (2+ words)."""
    if j - i == 1:
        yield frozenset()
        return
    for k in range(i + 1, j):
        for a in all_binary(i, k):
            for b in all_binary(k, j):
                yield a | b | {(i, j)}


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_goodman_binary_is_optimal(n):
    rng = random.Random(n)
    sp = {(i, j): rng.random() for i in range(n) for j in range(i + 2, n + 1)}
    split = goodman_binary(n, sp)
    got = sum(sp[s] for s in split)
    want = max(sum(sp[s] for s in t) for t in all_binary(0, n))
    assert got == pytest.approx(want)
    assert len(split) == n - 1


def laminar(a, b):
    return a[1] <= b[0] or b[1] <= a[0] or (a[0] <= b[0] and b[1] <= a[1]) or (b[0] <= a[0] and a[1] <= b[1])


@pytest.mark.parametrize("n,lam", [(4, 0.3), (5, 0.5), (6, 0.4), (6, 0.7)])
def test_goodman_nary_is_optimal(n, lam):
    rng = random.Random(n * 10 + int(lam * 10))
    sp = {(i, j): rng.random() for i in range(n) for j in range(i + 2, n + 1)}
    got = goodman_nary(n, sp, lam)
    assert (0, n) not in got and all(laminar(a, b) for a in got for b in got)
    val = sum(sp[s] - lam for s in got)
    inner = [s for s in sp if s != (0, n)]
    best = 0.0
    for r in range(len(inner) + 1):
        for sub in itertools.combinations(inner, r):
            if all(laminar(a, b) for a in sub for b in sub):
                best = max(best, sum(sp[s] - lam for s in sub))
    assert val == pytest.approx(best)


def test_goodman_nary_monotone_in_lambda():
    rng = random.Random(1)
    n = 7
    sp = {(i, j): rng.random() for i in range(n) for j in range(i + 2, n + 1)}
    prev = None
    for lam in (0.0, 0.2, 0.5, 0.8, 1.0):
        cur = goodman_nary(n, sp, lam)
        if prev is not None:
            assert cur <= prev or len(cur) <= len(prev)
        prev = cur
    assert goodman_nary(n, sp, 1.0) == set()


def test_goodman_model_parses_a_binary_tree():
    G = build(Trellis3R203, goodman=True)
    for s in ("the dog ran to the park", "they were happy to play", "he went home"):
        r = G.parse(s.split())
        n = len(s.split())
        assert len(r["chosen"]) == n - 1
        sp = span_posteriors(r["post"])
        got = sum(sp[(i, j)] for i, k, j in r["chosen"])
        want = max(sum(sp[x] for x in t) for t in all_binary(0, n))
        assert got == pytest.approx(want)
