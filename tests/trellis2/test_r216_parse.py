"""Round 216: Trellis3R216 (the parser with the generator's conditionals) is Trellis3 unless r216 is set."""
import math

import pytest

from trellis2.trellis3 import BOS, LN2, Trellis3
from trellis2.trellis3_r216 import Trellis3R216

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home"]
TEST = ["the dog ran to the park", "they were happy to play", "he went home", "park the to ran dog the"]


def build(cls, **kw):
    L = cls(seed=3, **CFG, **kw)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


@pytest.fixture(scope="module")
def base():
    return build(Trellis3)


@pytest.fixture(scope="module")
def plain():
    return build(Trellis3R216)


def same(a, b):
    assert a["chosen"] == b["chosen"]
    assert a["logZ"] == b["logZ"]
    assert set(a["post"]) == set(b["post"]) and all(a["post"][k] == b["post"][k] for k in a["post"])
    assert all(a["comps"][k].gain == b["comps"][k].gain for k in a["comps"])


def test_default_is_byte_identical(base, plain):
    assert len(base.records) == len(plain.records)
    for s in TEST:
        same(base.parse(s.split()), plain.parse(s.split()))


def test_default_with_edge_exact_is_the_parent_parser(base):
    L = build(Trellis3R216)
    L.edge_exact = True                       # parse never calls _gap_fills, so edge_exact does not enter
    for s in TEST:
        same(base.parse(s.split()), L.parse(s.split()))


@pytest.mark.parametrize("mode", ["split", "chunk"])
def test_mu_zero_is_the_default(base, mode):
    L = build(Trellis3R216, r216=mode, r216_mu=0.0)
    L.edge_exact = True
    for s in TEST:
        same(base.parse(s.split()), L.parse(s.split()))


@pytest.fixture(scope="module")
def gen():
    L = build(Trellis3R216)
    L.edge_exact = True
    return L


def test_distribution_is_normalised_and_matches_gap_fills(gen):
    for left in [(BOS, BOS, BOS), (BOS, "the", "dog"), ("the", "dog"), ("dog",), ()]:
        d = gen.r216_dist(left)
        assert sum(d.values()) == pytest.approx(1.0)
    gen.r216_gate = False
    gen.__dict__.pop("_r216_gate", None)
    gen.__dict__.pop("_r216_cache", None)
    q = {"l1": "the", "l2": "dog"}
    raw = {v[1]: m for v, m in gen._gap_fills(q).items() if v[0] == "w"}
    Z = sum(raw.values())
    d = gen.r216_dist(("dog", "the"))
    assert all(d[w] == pytest.approx(m / Z) for w, m in raw.items())
    gen.r216_gate = True
    gen.__dict__.pop("_r216_gate", None)
    gen.__dict__.pop("_r216_cache", None)


def test_table_whole_window_row_is_zero_term(gen):
    toks = "the dog ran to the park".split()
    D = gen.r216_table(toks)
    assert len(D) == 6 and all(D[i][t] <= 0 for t in range(6) for i in range(t + 1))
    # a context that starts at word 0 is the generation's: i = 0 rows are the sentence's own conditionals
    left = tuple([BOS, BOS, BOS][:max(0, 3 - 2)] + toks[:2])[-3:]
    assert D[0][2] == gen.r216_logp(left, toks[2])
    # a context as long as the window equals the whole-window one
    assert D[1][4] == D[0][4] and D[2][5] == D[0][5]


@pytest.mark.parametrize("mode", ["split", "chunk"])
def test_factor_adds_the_term(base, mode):
    L = build(Trellis3R216, r216=mode)
    L.edge_exact = True
    for s in TEST[:2]:
        toks = s.split()
        a, b = base.parse(toks), L.parse(toks)
        D = L.r216_table(toks)
        for (i, k, j), e in b["comps"].items():
            term = D[i][k] - D[0][k] if mode == "split" else sum(D[i][t] - D[0][t] for t in range(i, j))
            assert e.gain == pytest.approx(a["comps"][(i, k, j)].gain + term / LN2 * L.tau)
            if i == 0 and mode == "split":
                assert term == 0.0
        assert math.isfinite(b["logZ"])


def test_split_terms_over_a_tree_cover_every_word_once(gen):
    """The split points of a binary tree are the words 1..n-1: one conditional per word."""
    toks = "the dog ran to the park".split()
    r = Trellis3R216.parse(gen, toks)
    assert sorted(k for _, k, _ in r["chosen"]) == list(range(1, len(toks)))
