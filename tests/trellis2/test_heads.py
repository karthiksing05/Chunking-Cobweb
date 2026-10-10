"""Heads (experiments/v2/heads.py): the head-conditioned grammar's chart
checked against brute force, the head code against an explicit sequential
coder, and the no-heads option against the current grammar."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "v2"))

from trellis2.chart import Chart  # noqa: E402
from trellis2.data import Tree  # noqa: E402
from trellis2.grammar import UNK, Grammar  # noqa: E402
from trellis2.mdl import dm_code  # noqa: E402

import heads as H  # noqa: E402


def dist(rng, *shape):
    x = rng.random(shape) + 0.05
    return x / x.sum(axis=-1, keepdims=True)


def head_grammar(K=3, M=4, V=3, seed=0, factored=False, forests=False):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(V)] + [UNK]
    Lt, Rt = dist(rng, M, K), dist(rng, M, K)
    if factored:
        P2 = Lt[:, :, None] * Rt[:, None, :]
    else:
        side = dist(rng, M, 3)
        Hh, D = dist(rng, M, K), dist(rng, M, K, K)
        left = Hh[:, :, None] * D                                   # [c, B = head, C = dep]
        right = (Hh[:, :, None] * D).transpose(0, 2, 1)             # [c, B = dep, C = head]
        none = Lt[:, :, None] * Rt[:, None, :]
        P2 = side[:, 0, None, None] * left + side[:, 1, None, None] * right + side[:, 2, None, None] * none
    kw = dict(p_whole=0.6, p_stop=0.5, S_piece=dist(rng, K)) if forests else {}
    return H.HeadGrammar(vocab=vocab, S=dist(rng, K), U=dist(rng, K, M), pk=rng.uniform(0.2, 0.8, M),
                         Lt=Lt, Rt=Rt, E=dist(rng, M, V + 1), alpha=0.0, P2=P2, **kw)


def all_trees(i, j):
    if j - i == 1:
        yield {}
        return
    for k in range(i + 1, j):
        for left in all_trees(i, k):
            for right in all_trees(k, j):
                yield {**left, **right, (i, j): k}


def brute_force(g, tokens):
    """Whole trees only: total probability, span and label marginals, and the
    most probable labelled tree."""
    ids = g.token_ids(tokens)
    n = len(tokens)
    rule = np.einsum("ac,c,cbd->abd", g.U, g.qk, g.P2)
    lex = g.U @ (g.pk[:, None] * g.E)
    total, span_mass, label_mass, best = 0.0, {}, {}, (-1.0, None)
    for split in all_trees(0, n):
        tree = Tree(n, split)
        spans = [(i, i + 1) for i in range(n)] + tree.composite_spans()
        for labels in itertools.product(range(g.K), repeat=len(spans)):
            lab = dict(zip(spans, labels))
            p = g.S[lab[(0, n)]]
            for (i, j) in spans:
                if j - i == 1:
                    p *= lex[lab[(i, j)], ids[i]]
                else:
                    k = split[(i, j)]
                    p *= rule[lab[(i, j)], lab[(i, k)], lab[(k, j)]]
            total += p
            if p > best[0]:
                best = (p, (frozenset(tree.brackets()), tuple(sorted(lab.items()))))
            for s in spans:
                span_mass[s] = span_mass.get(s, 0.0) + p
                label_mass[s + (lab[s],)] = label_mass.get(s + (lab[s],), 0.0) + p
    return total, span_mass, label_mass, best


@pytest.mark.parametrize("n,seed", [(2, 1), (3, 2), (4, 3)])
def test_head_inside_and_posteriors_match_enumeration(n, seed):
    g = head_grammar(seed=seed)
    tokens = [f"w{(seed + i) % 3}" for i in range(n)]
    total, span_mass, label_mass, _ = brute_force(g, tokens)
    chart = H.HeadChart(g, tokens)
    assert math.isclose(chart.log_prob, math.log(total), rel_tol=1e-9)
    mu = chart.label_posteriors()
    for i in range(n):
        for j in range(i + 1, n + 1):
            for a in range(g.K):
                assert mu[i, j, a] == pytest.approx(label_mass.get((i, j, a), 0.0) / total, abs=1e-10)


@pytest.mark.parametrize("seed", [4, 5])
def test_head_viterbi_finds_the_most_probable_labelled_tree(seed):
    g = head_grammar(seed=seed)
    tokens = ["w0", "w2", "w1", "w0"]
    _, _, _, (p, (brackets, lab)) = brute_force(g, tokens)
    tree = H.HeadChart(g, tokens).viterbi_tree()
    assert frozenset(tree.brackets()) == brackets
    assert tuple(sorted(tree.label.items())) == lab


def test_head_samples_follow_the_posterior():
    g = head_grammar(seed=7)
    tokens = ["w1", "w0", "w2", "w1"]
    chart = H.HeadChart(g, tokens)
    post = chart.span_posteriors()
    rng = np.random.default_rng(0)
    counts = {}
    n_draws = 4000
    for _ in range(n_draws):
        t = chart.sample_tree(rng)
        assert t.is_valid()
        for s in t.brackets():
            counts[s] = counts.get(s, 0) + 1
    for (i, j), c in counts.items():
        assert c / n_draws == pytest.approx(post[i, j], abs=0.03)


@pytest.mark.parametrize("forests", [False, True])
def test_factored_head_grammar_equals_the_current_chart(forests):
    """With P2 = Lt x Rt the head chart is the current chart (forests included)."""
    g = head_grammar(seed=9, factored=True, forests=forests)
    plain = Grammar(vocab=g.vocab, S=g.S, U=g.U, pk=g.pk, Lt=g.Lt, Rt=g.Rt, E=g.E, alpha=0.0,
                    p_whole=g.p_whole, p_stop=g.p_stop, S_piece=g.S_piece)
    tokens = ["w0", "w1", "w2", "w0", "w1"]
    a, b = H.HeadChart(g, tokens), Chart(plain, tokens)
    assert a.log_prob == pytest.approx(b.log_prob, abs=1e-10)
    assert np.allclose(a.label_posteriors(), b.label_posteriors(), atol=1e-10)
    ta, tb = a.viterbi_tree(), b.viterbi_tree()
    assert ta.brackets() == tb.brackets() and ta.roots == tb.roots


def sequential_code(c, left, right, sides, K, alpha, beta):
    """The head code event by event, with explicit counts (an independent
    implementation of ``heads.side_codes``): -sum log P(event | events before)."""
    n_c, n_ch, n_cd, n_chd = {}, {}, {}, {}
    nats = 0.0
    for cc, l, r, sd in zip(c, left, right, sides):
        g = (cc, sd)
        if sd == H.NONE:
            pl = (n_ch.get((g, "L", l), 0) + alpha) / (n_c.get(g, 0) + K * alpha)
            pr = (n_cd.get((g, "R", r), 0) + alpha) / (n_c.get(g, 0) + K * alpha)
            nats -= math.log(pl) + math.log(pr)
            n_ch[(g, "L", l)] = n_ch.get((g, "L", l), 0) + 1
            n_cd[(g, "R", r)] = n_cd.get((g, "R", r), 0) + 1
        else:
            h, d = (l, r) if sd == H.LEFT else (r, l)
            ph = (n_ch.get((g, h), 0) + alpha) / (n_c.get(g, 0) + K * alpha)
            pb = (n_cd.get((g, d), 0) + alpha) / (n_c.get(g, 0) + K * alpha)
            pd = (n_chd.get((g, h, d), 0) + beta * pb) / (n_ch.get((g, h), 0) + beta)
            nats -= math.log(ph) + math.log(pd)
            n_ch[(g, h)] = n_ch.get((g, h), 0) + 1
            n_cd[(g, d)] = n_cd.get((g, d), 0) + 1
            n_chd[(g, h, d)] = n_chd.get((g, h, d), 0) + 1
        n_c[g] = n_c.get(g, 0) + 1
    return nats


@pytest.mark.parametrize("side", [H.LEFT, H.RIGHT, H.NONE])
def test_head_code_is_the_sequential_log_probability(side):
    rng = np.random.default_rng(3)
    n, K, M, alpha, beta = 300, 5, 4, 0.01, 4.0
    c = rng.integers(0, M, n)
    left, right = rng.integers(0, K, n), rng.integers(0, K, n)
    w = np.ones(n)
    codes = H.side_codes(c, M, left, right, w, K, alpha, beta)
    assert codes[:, side].sum() == pytest.approx(
        sequential_code(c, left, right, [side] * n, K, alpha, beta), rel=1e-12)
    if side == H.NONE:   # no heads: exactly the current code of the parts
        assert codes[:, side].sum() == pytest.approx(
            dm_code(c, left, w, K, alpha) + dm_code(c, right, w, K, alpha), rel=1e-12)


def test_head_grammar_samples_whole_trees_with_joint_parts():
    g = head_grammar(seed=11)
    rng = np.random.default_rng(1)
    for _ in range(50):
        out = g.sample(rng, max_len=30)
        if out is None:
            continue
        tokens, tree = out
        assert tree.n == len(tokens) and tree.is_valid()
