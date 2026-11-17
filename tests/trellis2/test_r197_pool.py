"""Round 197: Trellis3.pool = "geo" weights each occurrence by the geometric mean of its per-attribute back-off
factors, (prod_a f_a) ** (1 / A), in _gap_fills, fill_words and pivot_support; the default is unchanged."""
import math

import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3
import test_r192_both_sides as r192
from test_r192_both_sides import CFG, SENTS


def build(pool=None):
    L = Trellis3(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    if pool:
        L.pool = pool
    return L


@pytest.fixture(scope="module")
def learner():
    return build("geo")


@pytest.fixture(scope="module")
def plain():
    return build()


def backoff(L):
    """P(q_a = v | leaf node): Dirichlet back-off, beta 1, from a Laplace root (as _gap_fills)."""
    root = L.ctx.root

    def p(node, a, v):
        size = len(root.av.get(a, {})) + 1
        if node.parent is None:
            return (node.av.get(a, {}).get(v, 0.0) + L.alpha) / (node.count + L.alpha * size)
        return (node.av.get(a, {}).get(v, 0.0) + p(node.parent, a, v)) / (node.count + 1.0)
    return p


@pytest.mark.parametrize("q", [{"l1": "to", "l2": "wanted", "l3": BOS}, {"l1": "the", "l2": "to"}, {"l1": "was"},
                               {"l1": "he", "l2": BOS, "l3": BOS, "r1": "was"}])
def test_geo_weights_are_product_to_the_one_over_A(plain, learner, q):
    from trellis2 import trellis3
    p = backoff(plain)
    want = {}
    for e in plain.records:
        if e.leaf is None:
            continue
        for which, attr, value, w in e.pas:
            if which == "ctx" and attr == trellis3.FILL:
                prod = math.prod(p(e.leaf, a, v) for a, v in q.items())
                want[value] = want.get(value, 0.0) + w * prod ** (1.0 / len(q))
    got = learner._gap_fills(q)
    assert set(got) == set(want)
    for v, m in want.items():
        assert got[v] == pytest.approx(m, rel=1e-9)


def test_one_attribute_is_unchanged(plain, learner):
    q = {"l1": "was"}
    a, b = plain._gap_fills(q), learner._gap_fills(q)
    assert a.keys() == b.keys() and all(b[k] == pytest.approx(a[k]) for k in a)


def test_default_is_the_product(plain):
    q = {"l1": "to", "l2": "wanted", "l3": BOS}
    assert plain.__dict__.get("pool") is None
    a = plain._gap_fills(q)
    plain.pool = "geo"
    g = plain._gap_fills(q)
    plain.pool = None
    assert plain._gap_fills(q) == a and g != a


# the r192 agreement tests, now on the geo read (pivot_support and fill_words read the same weights)
test_stored_trigrams_are_the_streams = r192.test_stored_trigrams_are_the_streams
test_pivot_ratio_matches_gap_fills = r192.test_pivot_ratio_matches_gap_fills
test_left_is_conditional_pmi = r192.test_left_is_conditional_pmi
test_pivot_ratio_in_r2_matches_gap_fills = r192.test_pivot_ratio_in_r2_matches_gap_fills
test_fill_words_matches_gap_fills = r192.test_fill_words_matches_gap_fills
