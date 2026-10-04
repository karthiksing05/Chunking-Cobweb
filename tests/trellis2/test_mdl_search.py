import random

from trellis2.mdl_search import _State, chunk_and_merge, code_bits, from_tree, to_tree


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
