"""Inside-outside, posteriors and MBR decoding checked against brute force."""
import itertools
import math

import numpy as np
import pytest

from trellis2.chart import Chart
from trellis2.data import Tree
from trellis2.grammar import UNK, Grammar


def random_grammar(K=3, M=4, V=3, seed=0):
    rng = np.random.default_rng(seed)

    def dist(*shape):
        x = rng.random(shape) + 0.05
        return x / x.sum(axis=-1, keepdims=True)

    vocab = [f"w{i}" for i in range(V)] + [UNK]
    E = dist(M, V + 1)
    return Grammar(vocab=vocab, S=dist(K), U=dist(K, M), pk=rng.uniform(0.2, 0.8, M),
                   Lt=dist(M, K), Rt=dist(M, K), E=E, alpha=0.0)


def all_trees(i, j):
    """Every binary bracketing of [i, j) as a dict span -> split."""
    if j - i == 1:
        yield {}
        return
    for k in range(i + 1, j):
        for left in all_trees(i, k):
            for right in all_trees(k, j):
                yield {**left, **right, (i, j): k}


def brute_force(g, tokens):
    """Total probability, span marginals and label marginals by enumeration."""
    ids = g.token_ids(tokens)
    n = len(tokens)
    rule = np.einsum("ac,c,cb,cd->abd", g.U, g.qk, g.Lt, g.Rt)      # P(A -> B C)
    lex = g.U @ (g.pk[:, None] * g.E)                                # P(A -> w)
    total = 0.0
    span_mass, label_mass = {}, {}
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
            for s in spans:
                span_mass[s] = span_mass.get(s, 0.0) + p
                label_mass[s + (lab[s],)] = label_mass.get(s + (lab[s],), 0.0) + p
    return total, span_mass, label_mass


@pytest.mark.parametrize("n,seed", [(2, 1), (3, 2), (4, 3)])
def test_inside_and_posteriors_match_enumeration(n, seed):
    g = random_grammar(seed=seed)
    tokens = [f"w{(seed + i) % 3}" for i in range(n)]
    total, span_mass, label_mass = brute_force(g, tokens)
    chart = Chart(g, tokens)
    assert math.isclose(chart.log_prob, math.log(total), rel_tol=1e-9)
    mu = chart.label_posteriors()
    post = chart.span_posteriors()
    for i in range(n):
        for j in range(i + 1, n + 1):
            assert post[i, j] == pytest.approx(span_mass.get((i, j), 0.0) / total, abs=1e-10)
            for a in range(g.K):
                expected = label_mass.get((i, j, a), 0.0) / total
                assert mu[i, j, a] == pytest.approx(expected, abs=1e-10)


def test_mbr_maximises_expected_correct_spans():
    g = random_grammar(seed=7)
    tokens = ["w0", "w1", "w2", "w0", "w1"]
    chart = Chart(g, tokens)
    post = chart.span_posteriors()
    best = max(sum(post[s] for s in Tree(5, t).composite_spans())
               for t in all_trees(0, 5))
    got = chart.mbr_tree()
    assert got.is_valid()
    assert sum(post[s] for s in got.composite_spans()) == pytest.approx(best)


def test_viterbi_finds_the_most_probable_labelled_tree():
    g = random_grammar(seed=9)
    tokens = ["w1", "w0", "w2", "w2"]
    ids = g.token_ids(tokens)
    rule = np.einsum("ac,c,cb,cd->abd", g.U, g.qk, g.Lt, g.Rt)
    lex = g.U @ (g.pk[:, None] * g.E)
    best_p, best = -1.0, None
    for split in all_trees(0, 4):
        tree = Tree(4, split)
        spans = [(i, i + 1) for i in range(4)] + tree.composite_spans()
        for labels in itertools.product(range(g.K), repeat=len(spans)):
            lab = dict(zip(spans, labels))
            p = g.S[lab[(0, 4)]]
            for (i, j) in spans:
                if j - i == 1:
                    p *= lex[lab[(i, j)], ids[i]]
                else:
                    k = split[(i, j)]
                    p *= rule[lab[(i, j)], lab[(i, k)], lab[(k, j)]]
            if p > best_p:
                best_p, best = p, (tree.brackets(), lab)
    got = Chart(g, tokens).viterbi_tree()
    assert got.brackets() == best[0]
    assert all(got.label[s] == best[1][s] for s in got.label)


def test_confident_spans_never_cross():
    g = random_grammar(seed=11)
    chart = Chart(g, ["w0", "w1", "w2", "w1", "w0", "w2"])
    spans = chart.confident_spans(0.5)
    for (a, b), (c, d) in itertools.combinations(spans, 2):
        assert not (a < c < b < d or c < a < d < b)


def test_long_sentence_does_not_underflow():
    g = random_grammar(K=4, M=5, seed=5)
    tokens = [f"w{i % 3}" for i in range(80)]
    chart = Chart(g, tokens)
    assert np.isfinite(chart.log_prob)
    post = chart.span_posteriors()
    assert post[0, 80] == pytest.approx(1.0)
    for i in range(80):
        assert post[i, i + 1] == pytest.approx(1.0)


def test_posterior_samples_are_valid_trees():
    g = random_grammar(seed=3)
    chart = Chart(g, ["w0", "w2", "w1", "w1"])
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(4000):
        t = chart.sample_tree(rng)
        assert t.is_valid()
        for s in t.composite_spans():
            counts[s] = counts.get(s, 0) + 1
    post = chart.span_posteriors()
    for s, c in counts.items():
        assert c / 4000 == pytest.approx(post[s], abs=0.03)


# --------------------------------------------------------------------------- #
# Partial analyses: a sentence is a sequence of top-level chunks.
# --------------------------------------------------------------------------- #
def all_forests(n):
    """Every (roots, split) covering [0, n) with top-level chunks."""
    def segmentations(i):
        if i == n:
            yield []
            return
        for j in range(i + 1, n + 1):
            for rest in segmentations(j):
                yield [(i, j)] + rest

    for roots in segmentations(0):
        def trees(idx):
            if idx == len(roots):
                yield {}
                return
            for t in all_trees(*roots[idx]):
                for rest in trees(idx + 1):
                    yield {**t, **rest}
        for split in trees(0):
            yield roots, split


def forest_grammar(seed, p_stop=0.6):
    g = random_grammar(seed=seed)
    return Grammar(vocab=g.vocab, S=g.S, U=g.U, pk=g.pk, Lt=g.Lt, Rt=g.Rt, E=g.E,
                   alpha=0.0, p_stop=p_stop)


def brute_force_forests(g, tokens):
    ids = g.token_ids(tokens)
    n = len(tokens)
    rule = np.einsum("ac,c,cb,cd->abd", g.U, g.qk, g.Lt, g.Rt)
    lex = g.U @ (g.pk[:, None] * g.E)
    total, span_mass, label_mass, best = 0.0, {}, {}, (-1.0, None)
    for roots, split in all_forests(n):
        tree = Tree(n, split, roots=roots)
        spans = [(i, i + 1) for i in range(n)] + tree.composite_spans()
        for labels in itertools.product(range(g.K), repeat=len(spans)):
            lab = dict(zip(spans, labels))
            p = (1 - g.p_stop) ** (len(roots) - 1) * g.p_stop
            for r in roots:
                p *= g.S[lab[r]]
            for (i, j) in spans:
                if j - i == 1:
                    p *= lex[lab[(i, j)], ids[i]]
                else:
                    k = split[(i, j)]
                    p *= rule[lab[(i, j)], lab[(i, k)], lab[(k, j)]]
            total += p
            if p > best[0]:
                best = (p, (tuple(roots), tree.brackets(), lab))
            for s in spans:
                span_mass[s] = span_mass.get(s, 0.0) + p
                label_mass[s + (lab[s],)] = label_mass.get(s + (lab[s],), 0.0) + p
    return total, span_mass, label_mass, best


@pytest.mark.parametrize("n,seed", [(2, 4), (3, 5), (4, 6)])
def test_forest_inside_and_posteriors_match_enumeration(n, seed):
    g = forest_grammar(seed)
    tokens = [f"w{(seed + i) % 3}" for i in range(n)]
    total, span_mass, label_mass, _ = brute_force_forests(g, tokens)
    chart = Chart(g, tokens)
    assert math.isclose(chart.log_prob, math.log(total), rel_tol=1e-9)
    mu = chart.label_posteriors()
    for i in range(n):
        for j in range(i + 1, n + 1):
            for a in range(g.K):
                expected = label_mass.get((i, j, a), 0.0) / total
                assert mu[i, j, a] == pytest.approx(expected, abs=1e-10)


def test_forest_viterbi_finds_the_most_probable_analysis():
    g = forest_grammar(8, p_stop=0.5)
    tokens = ["w1", "w0", "w2", "w2"]
    _, _, _, (_, (roots, brackets, lab)) = brute_force_forests(g, tokens)
    got = Chart(g, tokens).viterbi_tree()
    assert tuple(got.roots) == roots and got.brackets() == brackets
    assert all(got.label[s] == lab[s] for s in got.label)


def test_forest_samples_match_posteriors():
    g = forest_grammar(2, p_stop=0.5)
    chart = Chart(g, ["w0", "w2", "w1", "w1"])
    rng = np.random.default_rng(1)
    counts = {}
    for _ in range(4000):
        t = chart.sample_tree(rng)
        assert t.is_valid()
        for s in t.composite_spans():
            counts[s] = counts.get(s, 0) + 1
    post = chart.span_posteriors()
    for s, c in counts.items():
        assert c / 4000 == pytest.approx(post[s], abs=0.03)
