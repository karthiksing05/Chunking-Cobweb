"""Round 204: Trellis3R204 (edge_exact, M1) is Trellis3 unless the flag is set; with it the start / end marker slots are
counted from the occurrences that really stood at that boundary."""
import pytest

from trellis2.trellis3 import BOS, EOS, Trellis3
from trellis2.trellis3_r204 import Trellis3R204, edge_slots

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]
FIRST = {s.split()[0] for s in SENTS}
LAST = {s.split()[-1] for s in SENTS}


def build(cls, sents=SENTS):
    L = cls(seed=3, **CFG)
    for k, s in enumerate(sents):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def words(m):
    return {v[1] for v, x in m.items() if v[0] == "w" and x > 0}


QS = [{"l2": BOS, "l1": BOS, "l3": BOS},                    # the start step
      {"l2": "the", "l1": "dog", "l3": BOS},                # an inside step
      {"l2": "to", "l1": "the", "l3": "ran"}]


@pytest.fixture(scope="module")
def base():
    return build(Trellis3)


@pytest.fixture(scope="module")
def sub():
    return build(Trellis3R204)


def test_edge_slots():
    assert edge_slots({"l1": BOS, "l2": BOS}) == ["l1"]
    assert edge_slots({"l1": "a", "r1": EOS}) == ["r1"]
    assert edge_slots({"l1": BOS, "r1": EOS}) == ["l1", "r1"]
    assert edge_slots({"l1": "a", "l2": BOS, "r2": EOS}) == []


def test_default_is_byte_identical(base, sub):
    assert sub.edge_exact is False
    for q in QS:
        for qq in (q, {**q, "r1": EOS}, {**q, "r1": "to"}):
            assert base._gap_fills(qq) == sub._gap_fills(qq)


def test_exact_leaves_non_boundary_queries_alone(base, sub):
    sub.edge_exact = True
    try:
        for q in QS[1:]:
            assert base._gap_fills(q) == sub._gap_fills(q)
            assert base._gap_fills({**q, "r1": "to"}) == sub._gap_fills({**q, "r1": "to"})
    finally:
        sub.edge_exact = False


def test_start_read_counts_only_first_words(base, sub):
    q = QS[0]
    sub.edge_exact = True
    try:
        ex = words(sub._gap_fills(q))
    finally:
        sub.edge_exact = False
    assert ex and ex <= FIRST
    assert ex == FIRST                                # every first word is read
    assert words(base._gap_fills(q)) - FIRST          # the default also lets mid-sentence words start (the leak)


def test_end_read_counts_only_real_ends(base, sub):
    for q in QS[1:]:
        qe = {**q, "r1": EOS}
        sub.edge_exact = True
        try:
            ex = words(sub._gap_fills(qe))
        finally:
            sub.edge_exact = False
        assert ex <= LAST
        assert "to" not in ex and "the" not in ex     # words that never end a sentence get no end mass
        assert words(base._gap_fills(qe)) - LAST      # the default gives them some


def test_end_mass_never_exceeds_go_on_total(sub):
    q = QS[1]
    sub.edge_exact = True
    try:
        a, e = sub._gap_fills(q), sub._gap_fills({**q, "r1": EOS})
    finally:
        sub.edge_exact = False
    assert all(e[v] <= a.get(v, 0.0) * (1 + 1e-9) + 1e-300 for v in e if v in a)


def test_one_sentence_toy_counts_exactly():
    L = build(Trellis3R204, ["a b c d"] * 1)
    q = {"l2": BOS, "l1": BOS, "l3": BOS}
    L.edge_exact = True
    start = words(L._gap_fills(q))
    ends_d = words(L._gap_fills({"l2": "b", "l1": "c", "l3": "a", "r1": EOS}))
    ends_c = words(L._gap_fills({"l2": "a", "l1": "b", "l3": BOS, "r1": EOS}))
    Ld = build(Trellis3, ["a b c d"])
    assert {"a", "b", "c"} & words(Ld._gap_fills({"l2": "a", "l1": "b", "l3": BOS, "r1": EOS}))   # the default gives non-ending words end mass
    L.edge_exact = False
    assert start == {"a"}
    assert ends_d == {"d"}
    assert ends_c == {"d"}                            # only d ever ended; the unseen left context is backed off, not skipped


def test_records_and_cache_restored(sub):
    n = len(sub.records)
    recs = sub.records
    sub.edge_exact = True
    try:
        sub._gap_fills(QS[0])
        sub._gap_fills({**QS[1], "r1": EOS})
    finally:
        sub.edge_exact = False
    assert sub.records is recs and len(recs) == n
    assert sub._gap_fills(QS[0]) == build_default_cache_probe(sub)


def build_default_cache_probe(L):
    """the default read after exact reads equals a fresh default learner's (no cache leaks across the flag)"""
    return build(Trellis3)._gap_fills(QS[0])
