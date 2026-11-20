"""Round 223: Trellis3R223 (ctx_bits, ctx_func_bits) is Trellis3R214 unless a flag is set; the abstract read is the same read marginalised over a class."""
import math
from collections import Counter

import pytest

from trellis2.trellis3 import BOS, FILL
from trellis2.trellis3_r214 import Trellis3R214, WordTree
from trellis2.trellis3_r223 import DROP, Trellis3R223, _Cls
from test_r214_level import QS, SENTS, VARIANTS, build, words


@pytest.fixture(scope="module")
def pair():
    a, b = build(Trellis3R214), build(Trellis3R223)
    T = WordTree([s.split() for s in SENTS], seed=3)
    cnt = dict(Counter(w for s in SENTS for w in s.split()))
    for L in (a, b):
        L.word_tree, L.emit_counts = T, cnt
    return a, b, T


def reads(L):
    return [L._gap_fills(f(q)) for q in QS for f in VARIANTS]


def test_1_default_is_byte_identical(pair):
    a, b, _ = pair
    assert b.ctx_bits is None and b.ctx_func_bits is None
    for edge in (False, True):
        a.edge_exact = b.edge_exact = edge
        assert reads(a) == reads(b)
    a.edge_exact = b.edge_exact = False
    b.emit_freq_cut = 4
    b.ctx_bits = math.inf                                    # exact
    assert reads(a) == reads(b)
    b.ctx_bits = None
    b.emit_freq_cut = 0


def test_2_singleton_classes_and_wildcard(pair):
    a, b, T = pair
    # leaf level: every class is a singleton, so the query is unchanged
    b.ctx_bits = 1e9
    assert reads(a) == reads(b)
    # root: every content slot is a wildcard = the same read with those slots dropped
    b.emit_freq_cut = 3
    top = b._top_words()
    b.ctx_bits = 0.0
    q = {"l1": "the", "l2": "dog", "l3": BOS}
    kept = {a_: v for a_, v in q.items() if v in top or v == BOS}
    got = b._gap_fills(q)
    a.emit_freq_cut = 3
    ref = a._gap_fills(kept)
    assert got == ref
    b.ctx_bits = None
    b.emit_freq_cut = 0
    a.emit_freq_cut = 0


def test_3_class_read_is_the_marginal(pair):
    a, b, T = pair
    # a two-member class on one slot: occurrence factors add over the members (checked through the first-slot-only read, no back-off ambiguity at beta 1: the
    # per-leaf factor of the class is the sum of the members' factors)
    ws = sorted(T.words)
    u, v = ws[0], ws[1]
    mem = frozenset((u, v))
    cls = _Cls("t", mem)
    qa = {"l1": cls}
    out = b._gap_fills(qa)
    ru, rv = (a._gap_fills({"l1": u}), a._gap_fills({"l1": v}))
    # per-filler mass is sum over occurrences of (factor_u + factor_v) * w; the exact reads give sum of w * factor: so out = ru + rv up to the root smoothing
    # (alpha enters each member term; the class term uses alpha * number of members present, equal to the sum of the members' alphas)
    for f in out:
        assert out[f] == pytest.approx(ru.get(f, 0.0) + rv.get(f, 0.0), rel=1e-9)


def test_4_abstract_query_mapping(pair):
    a, b, T = pair
    b.emit_freq_cut = 4
    top = b._top_words()
    b.ctx_bits = 6.0
    q = {"l1": "dog", "l2": next(iter(top)), "l3": BOS}
    out = b._abstract_query(q)
    assert out["l2"] == q["l2"] and out["l3"] == BOS             # frequent words and boundaries stay exact
    assert out["l1"] == "dog" or isinstance(out["l1"], _Cls)
    b.ctx_func_bits = 0.0
    out = b._abstract_query(q)
    assert "l2" not in out                                       # the function-word end: wildcard
    b.ctx_bits = b.ctx_func_bits = None
    b.emit_freq_cut = 0
    assert b._abstract_query(q) is q
