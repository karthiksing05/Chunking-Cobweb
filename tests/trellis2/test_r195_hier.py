"""Round 195: Trellis3.hier_fills, the level-mixing causal read (Witten-Bell over exact histories / the sorting path)."""
import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3
from test_r192_both_sides import CFG, SENTS

HOWS = ["kn", "top", "ngram", "path", "post"]


@pytest.fixture(scope="module")
def learner():
    L = Trellis3(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def test_wb_escape():
    lower = {"a": 0.5, "b": 0.5}
    assert Trellis3._wb(None, lower) is lower
    out = Trellis3._wb({"a": 3.0}, lower)          # T 1, c 3: escape 1/4
    assert out["a"] == pytest.approx(3 / 4 + 0.5 / 4) and out["b"] == pytest.approx(0.5 / 4)


@pytest.mark.parametrize("how", HOWS)
def test_normalised_and_end_below_fill(learner, how):
    q = {"l1": "to", "l2": "wanted", "l3": "he"}
    fill, end = learner.hier_fills(q, how=how)
    assert sum(fill.values()) == pytest.approx(1.0)
    assert all(0.0 <= end[v] <= fill[v] + 1e-12 for v in end)


def test_kn_index_counts(learner):
    hist, ends = learner._hier_index()
    d = hist[2][("to", "wanted")]                                       # wanted to play x2, wanted to go x1
    assert set(d) == {"play", "go"} and d["play"] == pytest.approx(2 * d["go"])
    c = ends[1][("play", "to")]                                         # every "to play" ends its sentence
    assert c[1] > 0 and c[0] == pytest.approx(c[1])


@pytest.mark.parametrize("how", ["kn", "top"])
def test_kn_prefers_exact_history(learner, how):
    a, _ = learner.hier_fills({"l1": "to", "l2": "wanted", "l3": "he"}, how=how)
    b, _ = learner.hier_fills({"l1": "to", "l2": "went", "l3": "he"}, how=how)
    assert a[("w", "play")] > b[("w", "play")] and b[("w", "the")] > a[("w", "the")]


def test_kn_end_follows_memory(learner):
    # "the" never ends a sentence after "to"; "play" always does
    fill, end = learner.hier_fills({"l1": "to", "l2": "wanted", "l3": "he"}, how="kn")
    assert end[("w", "play")] / fill[("w", "play")] > 0.5
    assert end.get(("w", "the"), 0.0) / fill[("w", "the")] < 0.5


def test_default_untouched(learner):
    # the default read does not depend on hier_fills having run
    q = {"l1": "were", "l2": "they", "l3": BOS}
    before = dict(learner._gap_fills(q))
    learner.hier_fills(q, how="kn")
    learner.__dict__.pop("_gap_cache", None)
    assert learner._gap_fills(q) == pytest.approx(before)
