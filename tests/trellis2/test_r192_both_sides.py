"""Round 192: Trellis3.pivot_support agrees with _gap_fills, and stored_trigrams holds the stream's trigrams."""
import math

import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]


@pytest.fixture(scope="module")
def learner():
    L = Trellis3(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def test_stored_trigrams_are_the_streams(learner):
    want = set()
    for s in SENTS:
        t = [BOS] + s.split() + [EOS]
        want.update(zip(t, t[1:], t[2:]))
    got = {t for t in learner.stored_trigrams() if t[:2] != (BOS, BOS)}   # the first word: (BOS, BOS, w), never read
    assert got == want


@pytest.mark.parametrize("left,p,x", [(("to", "wanted", BOS), "play", EOS), (("went", "he", BOS), "to", "the"),
                                      (("they", BOS, BOS), "were", "happy"), (("they", BOS, BOS), "were", "the")])
def test_pivot_ratio_matches_gap_fills(learner, left, p, x):
    q = {f"l{k + 1}": v for k, v in enumerate(left)}
    m1 = {v[1]: m for v, m in learner._gap_fills(q).items() if v[0] == "w"}
    m2 = {v[1]: m for v, m in learner._gap_fills({**q, "r1": x}).items() if v[0] == "w"}
    want = math.log(m2[p] / sum(m2.values())) - math.log(m1[p] / sum(m1.values()))
    got = learner.pivot_support(q, [p], [x], kind="pivot")[(p, x)]
    assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


def test_left_is_conditional_pmi(learner):
    q = {"l1": "to", "l2": "wanted", "l3": BOS}
    p = "play"
    # P(x | left, p) / P(x | p) over p's own occurrences, from _gap_fills restricted to p
    with_x = learner._gap_fills({**q, "r1": EOS})[("w", p)] / learner._gap_fills(q)[("w", p)]
    alone = learner._gap_fills({"r1": EOS})[("w", p)] / learner._gap_fills({})[("w", p)]
    got = learner.pivot_support(q, [p], [EOS], kind="left")[(p, EOS)]
    assert got == pytest.approx(math.log(with_x / alone), rel=1e-6, abs=1e-9)
    both = learner.pivot_support(q, [p], [EOS], kind="both")[(p, EOS)]
    piv = learner.pivot_support(q, [p], [EOS], kind="pivot")[(p, EOS)]
    assert both == pytest.approx(min(got, piv))


def test_pivot_ratio_in_r2_matches_gap_fills(learner):
    q = {"l1": "he", "l2": BOS, "l3": BOS, "r1": "to"}
    p, x = "went", "the"
    m1 = {v[1]: m for v, m in learner._gap_fills(q).items() if v[0] == "w"}
    m2 = {v[1]: m for v, m in learner._gap_fills({**q, "r2": x}).items() if v[0] == "w"}
    want = math.log(m2[p] / sum(m2.values())) - math.log(m1[p] / sum(m1.values()))
    got = learner.pivot_support(q, [p], [x], kind="pivot", slot="r2")[(p, x)]
    assert got == pytest.approx(want, rel=1e-6, abs=1e-9)
