"""Round 202: parse_gen.PrefixChart reads the learner's own chart (closed edge = Trellis3.parse exactly), leaves the
future out with the edge open, memoises without changing results, and never changes the learner; parse_step
samples in proportion to window weight times Z (sampling-importance-resampling)."""
import math
import random

import pytest

from trellis2.parse_gen import PrefixChart, parse_step
from trellis2.trellis3 import BOS, EOS, Trellis3
from test_r192_both_sides import CFG, SENTS

PROBES = ["the dog ran to the park", "they were happy to play", "she wanted to the park", "he was sad",
          "the cat went to the house and", "tom and tom were", "they was happy"]


@pytest.fixture(scope="module")
def learner():
    L = Trellis3(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


@pytest.mark.parametrize("s", PROBES)
def test_closed_chart_is_the_parse(learner, s):
    toks = s.split()
    PC = PrefixChart(learner, drop=())          # keep the story facet as parse() reads it
    want = learner.parse(toks)["logZ"]
    assert PC.logZ(toks, closed=True) == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_story_facet_dropped_by_default(learner):
    PC = PrefixChart(learner)
    e = PC.span("he was sad".split(), 0, 3, True)[1]
    assert "pv" not in e.ctx and "pv" not in e.left.ctx


@pytest.mark.parametrize("s", PROBES)
def test_open_edge_leaves_out_the_future(learner, s):
    toks = s.split()
    W = learner.window
    PC = PrefixChart(learner)
    n = len(toks)
    for i in range(n):
        e = PC.span(toks, i, i + 1, False)[1]
        for k in range(1, W + 1):
            if i + k >= n:
                assert f"r{k}" not in e.ctx
            else:
                assert e.ctx[f"r{k}"] == toks[i + k]
        c = PC.span(toks, i, i + 1, True)[1]
        for k in range(1, W + 1):
            assert c.ctx[f"r{k}"] == (toks[i + k] if i + k < n else EOS)
        for k in range(1, W + 1):
            assert e.ctx[f"l{k}"] == (toks[i - k] if i - k >= 0 else BOS)
    # elements whose window ends before the edge are the same open or closed
    if n > W + 1:
        a = PC.span(toks, 0, 1, False)
        b = PC.span(toks, 0, 1, True)
        assert a[1] is b[1]


def test_memo_does_not_change_results(learner):
    prefixes = [s.split()[:k] for s in PROBES for k in range(2, len(s.split()) + 1)]
    fresh = {}
    for p in prefixes:
        for closed in (False, True):
            fresh[(tuple(p), closed)] = PrefixChart(learner).logZ(p, closed)
    warm = PrefixChart(learner)
    order = list(fresh)
    random.Random(1).shuffle(order)
    for p, closed in order:
        assert warm.logZ(list(p), closed) == fresh[(p, closed)]
    small = PrefixChart(learner, cap=5)           # clearing the memo mid-way changes nothing either
    for p, closed in order:
        assert small.logZ(list(p), closed) == fresh[(p, closed)]


def test_learner_unchanged(learner):
    before = [learner.parse(s.split())["logZ"] for s in PROBES]
    n_rec, n_ctx, n_cnt = len(learner.records), learner.ctx.root.count, learner.cnt.root.count
    PC = PrefixChart(learner)
    for s in PROBES:
        toks = s.split()
        for k in range(1, len(toks) + 1):
            PC.logZ(toks[:k], False)
            PC.logZ(toks[:k], True)
    assert (len(learner.records), learner.ctx.root.count, learner.cnt.root.count) == (n_rec, n_ctx, n_cnt)
    assert [learner.parse(s.split())["logZ"] for s in PROBES] == before


def test_single_word_has_no_composite(learner):
    PC = PrefixChart(learner)
    assert PC.logZ(["the"], False) == 0.0 and PC.logZ(["the"], True) == 0.0


class _StubChart:
    def __init__(self, z):
        self.z = z

    def logZ(self, toks, closed):
        return self.z[(toks[-1], closed)]


@pytest.mark.parametrize("temp", [1.0, 2.0])
def test_parse_step_samples_window_times_Z(temp):
    vs = [(("w", "a"), False), (("w", "b"), False), (("w", "a"), True), (("w", "c"), False)]
    q = [0.4, 0.3, 0.2, 0.1]
    z = {("a", False): 0.0, ("b", False): 1.5, ("a", True): -1.0, ("c", False): 2.0}
    pc = _StubChart(z)
    target = [(x ** (1 / temp)) * math.exp(z[(v[0][1], v[1])] / temp) for v, x in zip(vs, q)]
    tot = sum(target)
    target = [t / tot for t in target]
    rng = random.Random(0)
    ws = [x ** (1 / temp) for x in q]
    counts = [0] * len(vs)
    N = 3000
    for _ in range(N):
        v = parse_step(pc, ["x"], vs, ws, lambda v: [v[1]], rng, temp=temp, M=400)
        counts[vs.index(v)] += 1
    for c, t in zip(counts, target):
        assert c / N == pytest.approx(t, abs=0.03)


class _StubChart2:
    def __init__(self, z):
        self.z = z

    def logZ(self, toks, closed):
        return self.z[(toks[-1], closed)]


def test_close_mode_reweights_ends_only():
    vs = [(("w", "a"), False), (("w", "a"), True), (("w", "b"), True)]
    q = [0.5, 0.25, 0.25]
    z = {("a", False): 5.0, ("a", True): 4.0, ("b", False): -3.0, ("b", True): -2.0}
    target = [0.5, 0.25 * math.exp(-1.0), 0.25 * math.exp(1.0)]
    tot = sum(target)
    target = [t / tot for t in target]
    rng = random.Random(2)
    counts = [0] * 3
    N = 3000
    for _ in range(N):
        v = parse_step(_StubChart2(z), ["x"], vs, q, lambda v: [v[1]], rng, M=400, mode="close")
        counts[vs.index(v)] += 1
    for c, t in zip(counts, target):
        assert c / N == pytest.approx(t, abs=0.03)


def test_closure_is_a_share(learner):
    PC = PrefixChart(learner)
    root = learner.cnt.root
    assert PC.sent_share(root) == pytest.approx(len(SENTS) / root.count)
    for s in PROBES:
        toks = s.split()
        for k in range(2, len(toks) + 1):
            c = PC.closure(toks[:k])
            assert 0.0 <= c <= 1.0
    small = PrefixChart(learner, cap=3)           # the memo cap never loses a closure read
    for s in PROBES:
        assert small.closure(s.split()) == pytest.approx(PC.closure(s.split()))


def test_parse3_reads_the_open_edge_only():
    vs = [(("w", "a"), False), (("w", "a"), True), (("w", "b"), False)]
    z = {("a", False): 1.0, ("a", True): 50.0, ("b", False): 0.0}
    target = [0.4 * math.e, 0.2 * math.e, 0.4]
    target = [t / sum(target) for t in target]
    rng = random.Random(3)
    counts = [0] * 3
    N = 3000
    for _ in range(N):
        v = parse_step(_StubChart2(z), ["x"], vs, [0.4, 0.2, 0.4], lambda v: [v[1]], rng, M=400, mode="parse3")
        counts[vs.index(v)] += 1
    for c, t in zip(counts, target):
        assert c / N == pytest.approx(t, abs=0.03)


def test_unsupported_recipe_refused():
    L = Trellis3(seed=1, **{**CFG, "spine": True})
    with pytest.raises(NotImplementedError):
        PrefixChart(L)
