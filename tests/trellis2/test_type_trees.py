"""SideTrees: two Cobweb trees over types, read by types."""
import pytest

from trellis2.type_trees import SideTrees

STREAM = [["a", "x"], ["a", "y"], ["b", "x"], ["c", "z"], ["c", "w"]] * 3


def _build(stream=STREAM, seed=0):
    st = SideTrees(seed)
    for s in stream:
        st.observe(s)
    return st


def test_analogy():
    st = _build()
    assert st.pair_stored("a", "x") and not st.pair_stored("b", "y")
    assert st.licence("b", "y") > 0
    assert st.licence("b", "y") > st.licence("b", "z")


def test_incremental_equals_batch():
    one = SideTrees(3)
    for s in STREAM:
        one.observe(s)
    two = _build(STREAM, 3)
    assert one.types() == two.types() == ["a", "x", "y", "b", "c", "z", "w"]
    for a in one.types():
        for b in one.types():
            assert one.pair_stored(a, b) == two.pair_stored(a, b)
    for a, b in [("b", "y"), ("b", "z"), ("a", "x"), ("c", "w")]:
        assert one.licence(a, b) == two.licence(a, b)


def test_next_distribution():
    st = _build()
    d = st.next_distribution("b")
    assert d and abs(sum(d.values()) - 1.0) < 1e-9
    assert st.next_distribution("nope") == {}


def test_votes():
    st = _build()
    assert st.votes("b", set()) == 0.0
    # b's groups contain a (shares followers with it), so a as filler votes
    assert st.votes("b", {"a"}) > 0
    assert st.votes("nope", {"a"}) == 0.0


def test_votes_all_equals_votes():
    st = _build()
    for F in [{"a"}, {"a", "x"}, {"b", "c", "z"}, set(), {"a", "nope"}, set(st.types())]:
        va = st.votes_all(F)
        for w in st.types():
            assert abs(va.get(w, 0.0) - st.votes(w, F)) < 1e-12
        assert all(v > 0 for v in va.values())


def test_observe_occurrences_equals_observe():
    one, two = SideTrees(3), SideTrees(3)
    for s in STREAM:
        one.observe(s)
        two.observe_occurrences((w, s[i - 1] if i else None, s[i + 1] if i + 1 < len(s) else None)
                                for i, w in enumerate(s))
    assert one.types() == two.types()
    for a in one.types():
        for b in one.types():
            assert one.pair_stored(a, b) == two.pair_stored(a, b)
            assert one.licence(a, b) == two.licence(a, b)


def test_neighbour_only_type():
    st = SideTrees(0)
    st.observe_occurrences([(("u",), "p", "q"), (("v",), "p", "q")])
    st.observe_occurrences([(("u",), "p", "q")])
    assert st.types() == [("u",), ("v",)]
    assert st.pair_stored("p", ("u",)) and "p" not in st.types()
    assert st.votes_all({("u",)}) == {w: st.votes(w, {("u",)}) for w in st.votes_all({("u",)})}


def test_determinism():
    s1, s2 = _build(seed=5), _build(seed=5)
    for a in s1.types():
        for b in s1.types():
            assert s1.licence(a, b) == s2.licence(a, b)
