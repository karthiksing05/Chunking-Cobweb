"""The joint chunk move in the shared search (``mdl_search.CompositeState``)
and its place in the night (``UnsupervisedLearner(joint_chunks=...)``)."""
import math
import random

from trellis2.mdl_search import (COMPOSITE_CONFIG, CompositeState, _State, chunk_and_merge, code_bits,
                                 match_positions, shapes)
from trellis2.unsupervised import UnsupervisedLearner

LN2 = math.log(2)


def corpus(seed, n=120, words="abcdefgh", classes="xyz"):
    """Sentences with recurring multi-word units, so that windows repeat."""
    rng = random.Random(seed)
    cls = {w: rng.choice(classes) for w in words}
    units = [[rng.choice(words) for _ in range(rng.randint(2, 4))] for _ in range(4)]
    out = []
    for _ in range(n):
        s = []
        for _ in range(rng.randint(1, 3)):
            s += rng.choice(units) if rng.random() < 0.7 else [rng.choice(words)]
        out.append([(("w", cls[w]), w) for w in s])
    return out


def state(c, **kw):
    st = CompositeState(c, 9, 0.001)
    st.config = dict(COMPOSITE_CONFIG, min_count=2, top_windows=30, **kw)
    return st


def test_shapes_are_every_binary_bracketing():
    assert [len(shapes(0, m)) for m in range(1, 7)] == [1, 1, 2, 5, 14, 42]


def test_composite_moves_are_exact():
    """The code after each composite move, from the changed rows only, equals
    the code recomputed from scratch, and so do the counts; chunk and merge
    moves stay exact after one."""
    for seed in range(3):
        for reuse in (True, False):
            rng = random.Random(seed)
            st = state(corpus(seed), reuse=reuse)
            checked = 0
            for _ in range(5):
                moves = list(st.scored_moves())
                for nats, move in [m for m in moves if m[1][0] == "composite"][:12]:
                    child = st.apply(move)
                    assert abs(nats / LN2 - code_bits(child.analyses, 9, 0.001)) < 1e-6
                    ref = _State(child.analyses, 9, 0.001)
                    assert child.rows == ref.rows and child.start == ref.start and child.cont == ref.cont
                    assert ({k: v for k, v in child.parents.items() if v}
                            == {k: v for k, v in ref.parents.items() if v})
                    for nats2, move2 in list(child.scored_moves())[::25]:
                        assert abs(nats2 / LN2 - code_bits(child.apply(move2).analyses, 9, 0.001)) < 1e-6
                    checked += 1
                st = st.apply(rng.choice(moves)[1])
            assert checked > 0


def test_a_composite_replaces_each_occurrence_by_one_node():
    st = state(corpus(1), reuse=False)
    _, move = next(x for x in st.scored_moves() if x[1][0] == "composite")
    window = move[1]
    n = sum(len(match_positions(st._labels(t), window)) for t in st.tops)
    child = st.apply(move)
    assert n >= 2
    assert sum(map(len, st.tops)) - sum(map(len, child.tops)) == n * (len(window) - 1)


def test_search_with_composites_is_exact():
    for seed in range(2):
        c = corpus(seed)
        analyses, bits = chunk_and_merge(c, 9, 0.001, composites=True,
                                         composite_config={"min_count": 2})
        assert abs(bits - code_bits(analyses, 9, 0.001)) < 1e-6
        assert bits < code_bits(c, 9, 0.001)
        # Without composites the search is the chunk-and-merge search.
        assert chunk_and_merge(c, 9, 0.001, composites=False) == chunk_and_merge(c, 9, 0.001)


def test_the_night_keeps_both_kinds_as_candidates():
    """With joint chunks the night's winner codes no longer than without
    (both kinds are candidates of the same full-code choice)."""
    sents = [[w for _, w in s] for s in corpus(5, n=60)]
    codes = {}
    for joint in (False, True):
        learner = UnsupervisedLearner(seed=0, levels=2, beam=1, patience=0, consolidations=1,
                                      reanalysis_steps=1, context_free_steps=0, sampling=(),
                                      joint_chunks=joint, joint_config={"min_count": 2})
        for s in sents:
            learner.observe(s)
        codes[joint] = learner.sleep().info["total bits"]
    assert codes[True] <= codes[False] + 1e-6
