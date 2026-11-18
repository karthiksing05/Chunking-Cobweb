"""Round 208: Trellis3R208 (M3 of r204). Without store_same it is Trellis3R207 byte for byte; with it, word occurrences carry
s_k = [own word == l_k], and repeat_read reads each occurrence with that attribute set from the query's window words."""
import pytest

from trellis2.trellis3 import BOS
from trellis2.trellis3_r207 import Trellis3R207
from trellis2.trellis3_r208 import Trellis3R208

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "ben and ben smiled", "they were happy", "he was happy",
         "she was sad", "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home",
         "they went to the park", "the cat and the cat were friends", "he went to the house", "she was happy to play",
         "they were happy to go home", "the dog saw the cat", "the cat saw the dog"]
W = 3


def build(cls, **kw):
    L = cls(seed=3, **kw, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


@pytest.fixture(scope="module")
def r207():
    return build(Trellis3R207)


@pytest.fixture(scope="module")
def plain():
    return build(Trellis3R208)


@pytest.fixture(scope="module")
def same():
    return build(Trellis3R208, store_same=True)


QS = ({"l2": BOS, "l1": BOS, "l3": BOS}, {"l2": "the", "l1": "dog", "l3": BOS}, {"l2": "to", "l1": "the", "l3": "ran"},
      {"l2": "ben", "l1": "and", "l3": BOS})


def test_default_is_byte_identical(r207, plain):
    assert plain.ctx_attrs == r207.ctx_attrs and not any(a.startswith("s") and a[1:].isdigit() for a in plain.ctx_attrs)
    assert plain.store_same is False and plain.repeat_read is False
    for q in QS:
        for qq in (q, {**q, "r1": "</s>"}):
            assert plain._gap_fills(qq) == r207._gap_fills(qq)
    plain.repeat_read = True                       # without store_same the flag does nothing
    try:
        for q in QS:
            assert plain._gap_fills(q) == r207._gap_fills(q)
    finally:
        plain.repeat_read = False


def test_flag_off_read_is_the_parent_read(same):
    for q in QS:
        assert same._gap_fills(q) == Trellis3R207._gap_fills(same, q)


def test_same_k_counts_match_direct_counting(same):
    direct = {k: [0, 0] for k in range(1, W + 1)}            # k -> [own word == l_k, own word != l_k], real l_k only
    for s in SENTS:
        t = s.split()
        for i in range(len(t)):
            for k in range(1, W + 1):
                if i - k >= 0:
                    direct[k][0 if t[i] == t[i - k] else 1] += 1
    got = {k: [0, 0] for k in range(1, W + 1)}
    for e in same.records:
        if e.token is None or e.ctx is None:
            continue
        for k in range(1, W + 1):
            v = e.ctx.get(f"s{k}")
            if v in ("0", "1"):
                got[k][0 if v == "1" else 1] += 1
    assert direct[2][0] >= 1 and direct[3][0] >= 1                # the toy does repeat inside the window
    for k in range(1, W + 1):
        assert got[k] == direct[k], (k, got[k], direct[k])


def test_pattern_marks_only_the_matching_occurrences(same):
    q = {"l2": "ben", "l1": "and", "l3": BOS}
    ks = [1, 2]
    pats = {}
    for e in same.records:
        if e.leaf is not None and e.token is not None:
            pats.setdefault(same.same_pattern(e, q, ks), set()).add(e.token)
    assert "and" in pats[("1", "0")] and "ben" in pats[("0", "1")]
    assert "dog" in pats[("0", "0")] and "and" not in pats[("0", "0")] and "ben" not in pats[("0", "0")]


def test_repeat_read_lowers_a_repeat_and_keeps_the_support(same):
    q = {"l2": "the", "l1": "dog", "l3": BOS}
    a = same._gap_fills(q)
    same.repeat_read = True
    try:
        b = same._gap_fills(q)
        assert set(b) <= set(a) and all(m > 0 for m in b.values())
        wa = {k[1]: m for k, m in a.items() if k[0] == "w"}
        wb = {k[1]: m for k, m in b.items() if k[0] == "w"}
        assert "dog" in wa and "dog" in wb or "dog" not in wa
        if "dog" in wb and "ran" in wb:
            assert wb["dog"] / wb["ran"] < wa["dog"] / wa["ran"]  # "dog dog" is read as rarer than "dog ran"
        # a window with no real word is not touched
        q0 = {"l2": BOS, "l1": BOS, "l3": BOS}
        same.repeat_read = False
        c = same._gap_fills(q0)
        same.repeat_read = True
        assert same._gap_fills(q0) == c
    finally:
        same.repeat_read = False


def test_repeat_read_works_with_edge_exact(same):
    same.edge_exact = True
    same.repeat_read = True
    try:
        out = same._gap_fills({"l2": BOS, "l1": "the", "l3": BOS})
        assert out and all(m >= 0 for m in out.values())
        out2 = same._gap_fills({"l2": "ben", "l1": "and", "l3": BOS, "r1": "</s>"})
        assert isinstance(out2, dict)
    finally:
        same.edge_exact = same.repeat_read = False
