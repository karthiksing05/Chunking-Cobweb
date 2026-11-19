"""Round 218: Trellis3R218 (windowed, depth-limited inside-outside) is full IO by default and with a window as long as the sentence."""
import pytest

from trellis2.trellis3 import Trellis3
from trellis2.trellis3_r218 import Trellis3R218

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]
TEST = ["the dog ran to the park", "they were happy to play", "he went home", "park the to ran dog the", "she wanted to go to the park"]


def build(cls, **kw):
    L = cls(seed=3, **CFG, **kw)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


def same(a, b):
    assert a["chosen"] == b["chosen"]
    assert a["logZ"] == b["logZ"]
    assert set(a["post"]) == set(b["post"]) and all(a["post"][k] == b["post"][k] for k in a["post"])
    assert set(a["comps"]) == set(b["comps"]) and all(a["comps"][k].gain == b["comps"][k].gain for k in a["comps"])


@pytest.fixture(scope="module")
def base():
    return build(Trellis3)


@pytest.fixture(scope="module")
def sub():
    return build(Trellis3R218)


def test_default_is_full_io(base, sub):
    assert len(base.records) == len(sub.records)
    for s in TEST:
        same(base.parse(s.split()), sub.parse(s.split()))


@pytest.mark.parametrize("kw", [dict(win_W=50), dict(win_W=50, win_stride=3), dict(io_depth_D=50), dict(win_W=50, io_depth_D=50)])
def test_window_as_long_as_the_sentence_is_full_io(base, sub, kw):
    for s in TEST:
        a = base.parse(s.split())
        sub.win_W = sub.win_stride = sub.io_depth_D = None
        for k, v in kw.items():
            setattr(sub, k, v)
        try:
            b = sub.parse(s.split())
        finally:
            sub.win_W = sub.win_stride = sub.io_depth_D = None
        same(a, b)
        assert b["stats"]["windows"] == 1 and b["stats"]["passes"] == 1


def test_learning_with_a_full_window_is_full_io(base):
    L = build(Trellis3R218, win_W=50, win_learn=True)
    assert len(L.records) == len(base.records)
    for s in TEST:
        same(base.parse(s.split()), L.parse(s.split()))


def test_eval_only_window_does_not_change_learning(base):
    L = build(Trellis3R218, win_W=2, win_learn=False)
    assert len(L.records) == len(base.records)
    assert L.stats == base.stats


@pytest.mark.parametrize("kw", [dict(win_W=2), dict(win_W=2, win_stride=1), dict(win_W=3), dict(win_W=3, win_stride=2),
                                dict(win_W=4, win_stride=2), dict(io_depth_D=1), dict(io_depth_D=2), dict(win_W=4, io_depth_D=1)])
def test_every_setting_gives_a_binary_tree(sub, kw):
    for s in TEST:
        toks = s.split()
        sub.win_W = sub.win_stride = sub.io_depth_D = None
        for k, v in kw.items():
            setattr(sub, k, v)
        try:
            r = sub.parse(toks)
        finally:
            sub.win_W = sub.win_stride = sub.io_depth_D = None
        n = len(toks)
        assert len(r["chosen"]) == n - 1 and len(set(r["chosen"])) == n - 1
        assert r["tree"].n == n and (0, n) in r["tree"].split
        assert sorted(k for _, k, _ in r["chosen"]) == list(range(1, n))
        assert all(c in r["comps"] and c in r["post"] for c in r["chosen"])
        assert r["stats"]["forced"] == 0


def test_pairs_and_right_branching(sub):
    toks = "the dog ran to".split()
    sub.win_W, sub.win_stride = 2, 2
    try:
        r = sub.parse(toks)
    finally:
        sub.win_W = sub.win_stride = None
    assert set(r["chosen"]) == {(0, 1, 2), (2, 3, 4), (0, 2, 4)}
    sub.win_W, sub.win_stride = 2, 1
    try:
        r = sub.parse(toks)
    finally:
        sub.win_W = sub.win_stride = None
    assert set(r["chosen"]) == {(2, 3, 4), (1, 2, 4), (0, 1, 4)}


def test_cost_falls_with_the_window(sub):
    toks = "she wanted to go to the park".split()
    full = sub.parse(toks)["stats"]
    assert full["factors"] == full["triples"] == (len(toks) ** 3 - len(toks)) // 6
    sub.win_W = 3
    try:
        part = sub.parse(toks)["stats"]
    finally:
        sub.win_W = None
    assert part["factors"] < full["factors"]


def _with(sub, **kw):
    sub.win_W = sub.win_stride = sub.io_depth_D = None
    sub.io_depth_mode = "freeze"
    for k, v in kw.items():
        setattr(sub, k, v)


def _reset(sub):
    _with(sub)


@pytest.mark.parametrize("kw", [dict(io_depth_D=1), dict(io_depth_D=2), dict(win_W=6, io_depth_D=2), dict(win_W=5, win_stride=2, io_depth_D=1),
                                dict(io_depth_D=3)])
def test_chart_mode_gives_a_binary_tree(sub, kw):
    for s in TEST + ["one two three four five six seven eight"]:
        toks = s.split()
        _with(sub, io_depth_mode="chart", **kw)
        try:
            r = sub.parse(toks)
        finally:
            _reset(sub)
        n = len(toks)
        assert len(r["chosen"]) == n - 1 and sorted(k for _, k, _ in r["chosen"]) == list(range(1, n))
        assert r["stats"]["forced"] == 0


def test_chart_mode_with_a_cap_beyond_the_sentence_is_full_io(base, sub):
    for s in TEST:
        _with(sub, io_depth_D=5, io_depth_mode="chart")
        try:
            b = sub.parse(s.split())
        finally:
            _reset(sub)
        same(base.parse(s.split()), b)


def test_chart_mode_is_cheaper(sub):
    toks = "one two three four five six seven eight".split()
    full = sub.parse(toks)["stats"]["factors"]
    _with(sub, io_depth_D=1, io_depth_mode="chart")
    try:
        d1 = sub.parse(toks)["stats"]
        sub.io_depth_D = 2
        d2 = sub.parse(toks)["stats"]
    finally:
        _reset(sub)
    assert d1["factors"] < d2["factors"] < full


def test_forest_io_against_enumeration(sub):
    """m = 4 symbols, trees of at most 3 symbols: enumerate the forests."""
    import math
    from types import SimpleNamespace
    gains = {(0, 1, 2): 1.0, (1, 2, 3): 0.5, (2, 3, 4): 2.0, (0, 1, 3): 0.3, (0, 2, 3): 0.7, (1, 2, 4): 0.1, (1, 3, 4): 0.4}
    comps = {k: SimpleNamespace(gain=g) for k, g in gains.items()}
    c = sub.tau / math.log(2)
    w = {k: 2 ** (g / sub.tau) for k, g in gains.items()}
    tree = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0}
    tree[(0, 2)], tree[(1, 3)], tree[(2, 4)] = w[(0, 1, 2)], w[(1, 2, 3)], w[(2, 3, 4)]
    tree[(0, 3)] = w[(0, 1, 3)] * tree[(1, 3)] + w[(0, 2, 3)] * tree[(0, 2)]
    tree[(1, 4)] = w[(1, 2, 4)] * tree[(2, 4)] + w[(1, 3, 4)] * tree[(1, 3)]

    def forests(i):
        if i == 4:
            yield [], 1.0
            return
        for j in range(i + 1, min(4, i + 3) + 1):
            for rest, wt in forests(j):
                yield [(i, j)] + rest, tree[(i, j)] * wt
    fs = list(forests(0))
    Z = sum(wt for _, wt in fs)
    post, logZ = sub._forest_io(4, comps, 3)
    assert logZ == pytest.approx(math.log(Z))
    # P(composite (0,1,2) in the forest) = forests with the tree (0,2) or a tree (0,3) whose analysis holds (0,2) ...: checked on the top (1,4)
    p_top14 = sum(wt for f_, wt in fs if (1, 4) in f_) / Z
    assert p_top14 == pytest.approx(post[(1, 2, 4)] + post[(1, 3, 4)])
    p_012 = (sum(wt for f_, wt in fs if (0, 2) in f_) + sum(wt / tree[(0, 3)] * w[(0, 2, 3)] * tree[(0, 2)] for f_, wt in fs if (0, 3) in f_)) / Z
    assert p_012 == pytest.approx(post[(0, 1, 2)])
