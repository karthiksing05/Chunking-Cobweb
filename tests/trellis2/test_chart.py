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
# Partial analyses: a sentence is one tree or a forest of two or more pieces.
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


def forest_grammar(seed, p_stop=0.6, p_whole=0.4):
    """A grammar whose sentences are one tree (p_whole, root symbol from S) or
    a forest of two or more pieces (symbols from S_piece)."""
    g = random_grammar(seed=seed)
    piece = np.random.default_rng(seed + 100).random(g.K) + 0.05
    return Grammar(vocab=g.vocab, S=g.S, U=g.U, pk=g.pk, Lt=g.Lt, Rt=g.Rt, E=g.E,
                   alpha=0.0, p_stop=p_stop, p_whole=p_whole, S_piece=piece / piece.sum())


def brute_force_forests(g, tokens):
    ids = g.token_ids(tokens)
    n = len(tokens)
    # A span's rule choice is made in the light of the word before it, or of
    # BOS at the start of a piece read afresh.
    U = [g.rules(g.read(tokens, i)) for i in range(n)]
    rule = [np.einsum("ac,c,cb,cd->abd", u, g.qk, g.Lt, g.Rt) for u in U]
    lex = [u @ (g.pk[:, None] * g.E) for u in U]
    U0 = g.rules(g.read([], 0))
    rule0 = np.einsum("ac,c,cb,cd->abd", U0, g.qk, g.Lt, g.Rt)
    lex0 = U0 @ (g.pk[:, None] * g.E)
    total, span_mass, label_mass, best = 0.0, {}, {}, (-1.0, None)
    top_mass = {}
    for roots, split in all_forests(n):
        tree = Tree(n, split, roots=roots)
        spans = [(i, i + 1) for i in range(n)] + tree.composite_spans()
        fresh = {i for i, _ in roots} if g.fresh_pieces and len(roots) > 1 else set()
        for labels in itertools.product(range(g.K), repeat=len(spans)):
            lab = dict(zip(spans, labels))
            if len(roots) == 1:
                p = g.p_whole * g.S[lab[roots[0]]]
            else:
                p = (1 - g.p_whole) * (1 - g.p_stop) ** (len(roots) - 2) * g.p_stop
                for r in roots:
                    p *= g.S_piece[lab[r]]
            for (i, j) in spans:
                if j - i == 1:
                    p *= (lex0 if i in fresh else lex[i])[lab[(i, j)], ids[i]]
                else:
                    k = split[(i, j)]
                    p *= (rule0 if i in fresh else rule[i])[lab[(i, j)], lab[(i, k)], lab[(k, j)]]
            total += p
            if p > best[0]:
                best = (p, (tuple(roots), tree.brackets(), lab))
            for s in spans:
                span_mass[s] = span_mass.get(s, 0.0) + p
                label_mass[s + (lab[s],)] = label_mass.get(s + (lab[s],), 0.0) + p
            for r in roots:
                top_mass[r] = top_mass.get(r, 0.0) + p
    return total, span_mass, label_mass, best, top_mass


def context_grammar(seed, p_stop=0.6, p_whole=0.4, fresh_pieces=False):
    """A forest grammar whose rule choices depend on the word before the
    element; "w2" is a context it has never seen, which falls back to U. With
    ``fresh_pieces`` a forest's pieces are each read from BOS."""
    g = forest_grammar(seed, p_stop, p_whole)
    Uc = np.random.default_rng(seed + 200).random((g.K, 3, g.M)) + 0.05
    return Grammar(vocab=g.vocab, S=g.S, U=g.U, pk=g.pk, Lt=g.Lt, Rt=g.Rt, E=g.E, alpha=0.0,
                   p_stop=p_stop, p_whole=p_whole, S_piece=g.S_piece, contexts=["<s>", "w0", "w1"],
                   Uc=Uc / Uc.sum(axis=-1, keepdims=True), fresh_pieces=fresh_pieces)


def pair_grammar(seed, p_stop=0.6, p_whole=0.4):
    """A context grammar that also reads two words: some pairs, for some
    symbols, have rows of their own; every other pair reads as its last
    word."""
    g = context_grammar(seed, p_stop, p_whole)
    rng = np.random.default_rng(seed + 300)
    Uc2 = {}
    for pair in [("<s>", "<s>"), ("<s>", "w0"), ("w0", "w1"), ("w1", "w1"), ("w2", "w0")]:
        A = np.sort(rng.choice(g.K, size=2, replace=False))
        r = rng.random((2, g.M)) + 0.05
        Uc2[pair] = (A, r / r.sum(axis=1, keepdims=True))
    return Grammar(vocab=g.vocab, S=g.S, U=g.U, pk=g.pk, Lt=g.Lt, Rt=g.Rt, E=g.E, alpha=0.0,
                   p_stop=p_stop, p_whole=p_whole, S_piece=g.S_piece, contexts=g.contexts, Uc=g.Uc, Uc2=Uc2)


def make_grammar(make, seed, **kw):
    if make == "forest":
        return forest_grammar(seed, **kw)
    if make == "pairs":
        return pair_grammar(seed, **kw)
    return context_grammar(seed, fresh_pieces=make == "fresh", **kw)


@pytest.mark.parametrize("n,seed,make", [(2, 4, "forest"), (3, 5, "forest"), (4, 6, "forest"),
                                         (3, 9, "context"), (4, 10, "context"),
                                         (3, 11, "fresh"), (4, 12, "fresh"), (3, 13, "pairs"), (4, 14, "pairs")])
def test_forest_inside_and_posteriors_match_enumeration(n, seed, make):
    g = make_grammar(make, seed)
    tokens = [f"w{(seed + i) % 3}" for i in range(n)]
    total, span_mass, label_mass, _, top_mass = brute_force_forests(g, tokens)
    chart = Chart(g, tokens)
    assert math.isclose(chart.log_prob, math.log(total), rel_tol=1e-9)
    mu = chart.label_posteriors()
    tops = chart.top_level_posteriors()
    for i in range(n):
        for j in range(i + 1, n + 1):
            assert tops[i, j] == pytest.approx(top_mass.get((i, j), 0.0) / total, abs=1e-10)
            for a in range(g.K):
                expected = label_mass.get((i, j, a), 0.0) / total
                assert mu[i, j, a] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("make", ["forest", "context", "fresh", "pairs"])
def test_forest_viterbi_finds_the_most_probable_analysis(make):
    g = make_grammar(make, 8, p_stop=0.5)
    tokens = ["w1", "w0", "w2", "w2"]
    _, _, _, (_, (roots, brackets, lab)), _ = brute_force_forests(g, tokens)
    got = Chart(g, tokens).viterbi_tree()
    assert tuple(got.roots) == roots and got.brackets() == brackets
    assert all(got.label[s] == lab[s] for s in got.label)


@pytest.mark.parametrize("make", ["forest", "context", "fresh", "pairs"])
def test_forest_samples_match_posteriors(make):
    g = make_grammar(make, 2, p_stop=0.5)
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


def test_whole_only_samples_are_single_trees():
    g = forest_grammar(3, p_stop=0.5, p_whole=0.3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        out = g.sample(rng, max_len=60, whole_only=True)
        if out is not None:
            assert len(out[1].roots) == 1
    tops = [len(o[1].roots) for o in (g.sample(rng, max_len=60) for _ in range(2000)) if o is not None]
    assert np.mean([t == 1 for t in tops]) == pytest.approx(0.3, abs=0.04)
    assert min(t for t in tops if t > 1) == 2


@pytest.mark.parametrize("make", ["context", "fresh", "pairs"])
def test_the_grammar_draws_sentences_as_often_as_the_chart_codes_them(make):
    """Each piece of a forest read in the light of the word (or two) before
    it, or afresh from BOS: either way the sampler and the chart agree."""
    g = make_grammar(make, 5, p_stop=0.6, p_whole=0.3)
    rng = np.random.default_rng(0)
    n, counts = 20000, {}
    for _ in range(n):
        out = g.sample(rng, max_len=8)
        if out is not None:
            counts[tuple(out[0])] = counts.get(tuple(out[0]), 0) + 1
    for tokens, c in sorted(counts.items(), key=lambda x: -x[1])[:6]:
        expected = n * math.exp(Chart(g, list(tokens)).log_prob)
        assert abs(c - expected) <= 4 * math.sqrt(expected) + 2, (tokens, c, expected)
