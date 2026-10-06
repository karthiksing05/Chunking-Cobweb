import itertools
import random

from trellis2.mdl_search import (_State, chunk_and_merge, class_bigram_bits, code_bits, from_tree,
                                  to_tree, word_classes)


def random_corpus(seed, n=80):
    rng = random.Random(seed)
    words = "abcdefghij"
    cls = {w: rng.choice("xyzu") for w in words}
    return [[(("w", cls[w]), w) for w in
             [rng.choice(words[:rng.randint(3, 10)]) for _ in range(rng.randint(1, 9))]]
            for _ in range(n)]


def test_scored_moves_are_exact():
    """The code after each move, computed from the changed rows only, equals
    the code recomputed from scratch."""
    for seed in range(3):
        rng = random.Random(seed)
        state = _State(random_corpus(seed), 11, 0.001)
        for _ in range(15):
            moves = list(state.scored_moves())
            if not moves:
                break
            for nats, move in moves:
                child = state.apply(move)
                assert abs(nats / 0.6931471805599453 - code_bits(child.analyses, 11, 0.001)) < 1e-6
                # The incrementally updated counts equal those rebuilt from scratch.
                ref = _State(child.analyses, 11, 0.001)
                assert child.rows == ref.rows and child.start == ref.start
                assert ({k: v for k, v in child.parents.items() if v}
                        == {k: v for k, v in ref.parents.items() if v})
            state = state.apply(rng.choice(moves)[1])


def test_beam_search_shortens_the_code():
    corpus = random_corpus(7, n=120)
    start = code_bits(corpus, 11, 0.001)
    for beam, patience in [(1, 0), (4, 3)]:
        analyses, bits = chunk_and_merge(corpus, 11, 0.001, beam=beam, patience=patience)
        assert abs(bits - code_bits(analyses, 11, 0.001)) < 1e-6
        assert bits < start


def test_symbolic_analyses_round_trip_through_trees():
    analyses, _ = chunk_and_merge(random_corpus(3), 11, 0.001)
    for tops in analyses:
        tree = to_tree(tops)
        tokens = [str(i) for i in range(tree.n)]
        back = from_tree(tokens, tree, tree.label.__getitem__)
        assert to_tree(back).brackets() == tree.brackets() and to_tree(back).roots == tree.roots


def test_each_word_class_merge_is_the_best_by_the_full_code():
    """Every step of the merge path takes the pair whose merge gives the
    shortest class-bigram code, computed from scratch."""
    rng = random.Random(5)
    words = "abcdefgh"
    sentences = [[rng.choice(words[:rng.randint(2, 8)]) for _ in range(rng.randint(1, 7))]
                 for _ in range(60)]
    path = word_classes(sentences, 0.001)
    assert len(path) > 2
    for before, after in zip(path, path[1:]):
        classes = sorted(set(before.values()))
        best = min(class_bigram_bits(sentences, {w: (a if c == b else c) for w, c in before.items()}, 0.001)
                   for a, b in itertools.combinations(classes, 2))
        assert abs(class_bigram_bits(sentences, after, 0.001) - best) < 1e-6
        assert class_bigram_bits(sentences, after, 0.001) < class_bigram_bits(sentences, before, 0.001)


def test_backoff_code_is_the_sequential_code():
    import math
    from collections import Counter
    import numpy as np
    from trellis2.mdl import backoff_code
    rng = np.random.default_rng(0)
    n, G, X, A, alpha, beta = 300, 3, 4, 5, 0.01, 1.5
    g, x, k = rng.integers(0, G, n), rng.integers(0, X, n), rng.integers(0, A, n)
    n_gk, n_g, n_gxk, n_gx = Counter(), Counter(), Counter(), Counter()
    nats = 0.0
    for gi, xi, ki in zip(g, x, k):
        p_g = (n_gk[gi, ki] + alpha) / (n_g[gi] + A * alpha)
        nats -= math.log((n_gxk[gi, xi, ki] + beta * p_g) / (n_gx[gi, xi] + beta))
        n_gk[gi, ki] += 1
        n_g[gi] += 1
        n_gxk[gi, xi, ki] += 1
        n_gx[gi, xi] += 1
    assert math.isclose(backoff_code(g, x, k, np.ones(n), A, alpha, beta), nats, rel_tol=1e-12)


def test_backoff_chain_is_the_sequential_code():
    """Two words of context backing off to one, then to none: the vectorised
    prequential code equals coding the events one by one, and one level of
    the chain is exactly ``backoff_coder``."""
    import math
    from collections import Counter
    import numpy as np
    from trellis2.mdl import backoff_chain_coder, backoff_coder
    rng = np.random.default_rng(3)
    n, A, alpha = 300, 5, 0.01
    g, x1 = rng.integers(0, 3, n), rng.integers(0, 4, n)
    x2 = x1 * 3 + rng.integers(0, 3, n)            # a pair refines its last word
    k, w = rng.integers(0, A, n), np.ones(n)
    assert backoff_chain_coder(g, [x1], w)(k, A, alpha, [4.0]) == backoff_coder(g, x1, w)(k, A, alpha, 4.0)
    seen = Counter()
    nats = 0.0
    for gi, a, b, ki in zip(g, x1, x2, k):
        p0 = (seen[(gi, ki)] + alpha) / (seen[gi] + A * alpha)
        p1 = (seen[(gi, "1", a, ki)] + 4.0 * p0) / (seen[(gi, "1", a)] + 4.0)
        p2 = (seen[(gi, "2", b, ki)] + 16.0 * p1) / (seen[(gi, "2", b)] + 16.0)
        nats -= math.log(p2)
        seen.update([(gi, ki), gi, (gi, "1", a, ki), (gi, "1", a), (gi, "2", b, ki), (gi, "2", b)])
    assert math.isclose(backoff_chain_coder(g, [x1, x2], w)(k, A, alpha, [4.0, 16.0]), nats, rel_tol=1e-12)
