"""Round 194: conj=True gives the context tree joined-history attributes (lj2..ljW, rj2..rjW); the r192 reads
(pivot_support, fill_words) still agree with _gap_fills, and conj=False adds nothing."""
import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3
import test_r192_both_sides as r192
from test_r192_both_sides import CFG, SENTS

# the r192 agreement tests, rerun on a conj learner (the r2-slot pivot read covers r2 only, not the rj2 join)
test_stored_trigrams_are_the_streams = r192.test_stored_trigrams_are_the_streams
test_pivot_ratio_matches_gap_fills = r192.test_pivot_ratio_matches_gap_fills
test_left_is_conditional_pmi = r192.test_left_is_conditional_pmi
test_fill_words_matches_gap_fills = r192.test_fill_words_matches_gap_fills


@pytest.fixture(scope="module")
def learner():
    L = Trellis3(seed=3, conj=True, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def test_default_has_no_joins():
    L = Trellis3(seed=3, **CFG)
    assert not any(a.startswith(("lj", "rj")) for a in L.ctx_attrs)
    assert L._joins({"l1": "a", "l2": "b"}) == {"l1": "a", "l2": "b"}


def test_joins_and_attrs(learner):
    assert {"lj2", "lj3", "rj2", "rj3"} <= set(learner.ctx_attrs)
    x = learner._joins({"l1": "to", "l2": "wanted", "r1": "play", "r2": EOS, "r3": EOS})
    assert x["lj2"] == "wanted\x1fto" and "lj3" not in x
    assert x["rj2"] == "play\x1f" + EOS and x["rj3"] == "\x1f".join(["play", EOS, EOS])


def test_tree_holds_joined_values(learner):
    av = learner.ctx.root.av
    assert any(v == "wanted\x1fto" for v in av.get("lj2", {}))


def test_conjunction_sharpens_fill(learner):
    # "wanted to _" vs "went to _": the joined history separates them where l1=to alone cannot
    a = {v[1]: m for v, m in learner._gap_fills({"l1": "to", "l2": "wanted", "l3": "he"}).items() if v[0] == "w"}
    b = {v[1]: m for v, m in learner._gap_fills({"l1": "to", "l2": "went", "l3": "he"}).items() if v[0] == "w"}
    za, zb = sum(a.values()), sum(b.values())
    assert a["play"] / za > b["play"] / zb and b["the"] / zb > a["the"] / za
