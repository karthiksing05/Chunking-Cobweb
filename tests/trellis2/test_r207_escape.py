"""Round 207: Trellis3R207 (escape_mass, M1 of r206) is Trellis3R204 unless the flag is set; escape_split gives a novel
continuation only its history's missing mass N1 / (c + N1)."""
from collections import Counter

import pytest

from trellis2.trellis3 import BOS, Trellis3
from trellis2.trellis3_r204 import Trellis3R204
from trellis2.trellis3_r207 import Trellis3R207

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]


def build(cls):
    L = cls(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def direct():
    """c and N1 of every (l2, l1) history and of every l1, counted straight from the sentences (word fillers only)."""
    tri, uni = {}, {}
    for s in SENTS:
        t = [BOS, BOS] + s.split()
        for i in range(2, len(t)):
            tri.setdefault((t[i - 2], t[i - 1]), Counter())[t[i]] += 1
            uni.setdefault(t[i - 1], Counter())[t[i]] += 1
    return tri, uni


def mn(cnt):
    c = sum(cnt.values())
    n1 = sum(1 for x in cnt.values() if x == 1)
    return c, n1


@pytest.fixture(scope="module")
def base():
    return build(Trellis3R204)


@pytest.fixture(scope="module")
def sub():
    return build(Trellis3R207)


def test_default_is_byte_identical(base, sub):
    assert sub.escape_mass is False and sub.edge_exact is False
    for q in ({"l2": BOS, "l1": BOS, "l3": BOS}, {"l2": "the", "l1": "dog", "l3": BOS}, {"l2": "to", "l1": "the", "l3": "ran"}):
        for qq in (q, {**q, "r1": "</s>"}):
            assert base._gap_fills(qq) == sub._gap_fills(qq)
    sub.edge_exact = True
    base.edge_exact = True
    try:
        for q in ({"l2": BOS, "l1": BOS, "l3": BOS}, {"l2": "the", "l1": "dog", "l3": BOS}):
            assert base._gap_fills(q) == sub._gap_fills(q)
    finally:
        sub.edge_exact = base.edge_exact = False


def test_flag_does_not_change_the_read(sub):
    q = {"l2": "the", "l1": "dog", "l3": BOS}
    a = dict(sub._gap_fills(q))
    sub.escape_mass = True
    try:
        assert sub._gap_fills(q) == a
    finally:
        sub.escape_mass = False


def test_history_counts_match_direct_counting(sub):
    tri, uni = direct()
    for (l2, l1), cnt in tri.items():
        if l1 == BOS:
            continue
        c, n1, att = sub.history_mass(l2, l1)
        assert (c, n1) == mn(cnt)
        assert {w: int(x) for w, x in att.items()} == dict(cnt)
    assert sub.history_mass("zzz", "yyy")[0] == 0


def test_toy_missing_mass(sub):
    # (the, dog): ran once, was once -> c = 2, N1 = 2, e = 2 / 4 = .5
    c, n1, att = sub.history_mass("the", "dog")
    assert (c, n1) == (2, 2) and set(att) == {"ran", "was"}
    mass = {"ran": 3.0, "was": 3.0, "cat": 1.0, "happy": 1.0}
    f = sub.escape_split({"l2": "the", "l1": "dog"}, mass)
    new = {w: mass[w] * f[w] for w in mass}
    tot = sum(mass.values())
    assert sum(new.values()) == pytest.approx(tot)                       # the total is kept
    assert (new["cat"] + new["happy"]) / tot == pytest.approx(0.5)       # novel fillers share N1 / (c + N1)
    assert new["ran"] == pytest.approx(new["was"])                       # relative masses on each side are the read's
    assert new["cat"] == pytest.approx(new["happy"])
    # a history with N1 = 0: (to, the) saw park x2, house x2 -> no mass for anything new
    c, n1, _ = sub.history_mass("to", "the")
    assert (c, n1) == (4, 0)
    g = sub.escape_split({"l2": "to", "l1": "the"}, {"park": 1.0, "house": 3.0, "dog": 2.0})
    assert (g["dog"] * 2.0) == pytest.approx(0.0)
    assert g["park"] * 1.0 + g["house"] * 3.0 == pytest.approx(6.0)


def test_unseen_history_start_and_degenerate_cases_are_unchanged(sub):
    assert sub.escape_split({"l2": "zzz", "l1": "yyy"}, {"a": 1.0, "b": 2.0}) == {}
    assert sub.escape_split({"l2": BOS, "l1": BOS}, {"the": 1.0, "xx": 2.0}) == {}
    assert sub.escape_split({"l2": "the", "l1": "dog"}, {"ran": 1.0, "was": 1.0}) == {}      # nothing novel admitted
    assert sub.escape_split({"l2": "the", "l1": "dog"}, {"cat": 1.0, "happy": 1.0}) == {}    # nothing attested admitted


def test_pair_level_missing_mass_caps_the_licence_admitted(sub):
    tri, uni = direct()
    done = 0
    for (l2, l1), cnt in tri.items():
        if l1 == BOS or len(cnt) < 1:
            continue
        c, n1 = mn(cnt)
        c1, n11 = mn(uni[l1])
        e, e1 = n1 / (c + n1), n11 / (c1 + n11)
        att = list(cnt)
        mass = {att[0]: 2.0, "novelA": 1.0, "novelB": 1.0}
        f = sub.escape_split({"l2": l2, "l1": l1}, mass, lambda w: w != "novelB")
        if not f:
            continue
        new = {w: mass[w] * f[w] for w in mass}
        tot = sum(mass.values())
        assert sum(new.values()) == pytest.approx(tot)
        pNp = min(e * 0.5, e1)
        assert new["novelB"] / tot == pytest.approx(pNp)
        assert new["novelA"] / tot == pytest.approx(e - pNp)
        assert new[att[0]] / tot == pytest.approx(1 - e)
        done += 1
    assert done >= 3
