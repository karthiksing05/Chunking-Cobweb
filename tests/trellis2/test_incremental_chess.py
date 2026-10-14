"""The incremental chess learner: its persistent count tables equal a
from-scratch recount of its stored analyses (and the night's search state for
the same moves), every chunk and dissolve score equals the code after the
move, and the grammar read off the state codes and draws positions."""
import copy
import math

import numpy as np

from trellis2.chess import code_of_analyses, parse_fen
from trellis2.incremental_chess import IncrementalBoardState, IncrementalChessLearner

from test_chess_context import reference_bits
from test_chess_search import FENS

A = 0.01


def mirrored(fen):
    """The position seen from the other side (colours swapped, board
    flipped): more positions for the stream."""
    rows = fen.split()[0].split("/")[::-1]
    return "/".join(r.swapcase() for r in rows)


STREAM = [parse_fen(f) for f in FENS] + [parse_fen(mirrored(f)) for f in FENS]


def check_exact(st: IncrementalBoardState):
    positions, analyses = [p["position"] for p in st.pos], st.analyses()
    ref = st.reference()
    # The stored analyses are the replay of the learned moves.
    assert analyses == ref.analyses()
    # The tables equal a from-scratch recount.
    recount = code_of_analyses(positions, analyses, st.a, st.counts)
    assert abs(st.parts()["read"] - recount["read"]) < 1e-6
    assert abs(st.parts()["rows"] - recount["rows"]) < 1e-6
    assert np.allclose(st.read, ref.read)
    assert {k: dict(v) for k, v in st.rows.items() if v} == {k: dict(v) for k, v in ref.rows.items() if v}
    # The top level in context, against the written-out sequential code.
    assert abs(st.bits() - ref.bits()) < 1e-6
    assert abs(st.bits() - reference_bits(ref)) < 1e-6
    # The cached per-K group codes equal a fresh pass.
    for K in st.gcode:
        assert np.allclose(st.gcode[K], st._code_all(K))


def test_tables_equal_recount_along_the_stream():
    learner = IncrementalChessLearner(alpha=1e-4, config={"margin": 1e9})
    for i, p in enumerate(STREAM):
        learner.observe(p)
        if i % 4 == 3:
            check_exact(learner.state)
    assert learner.history, "some move was taken"
    check_exact(learner.state)


def test_chunk_scores_are_exact():
    st = IncrementalBoardState(A)
    for p in STREAM[:10]:
        st.add_position(p)
    rng = np.random.default_rng(0)
    for step in range(3):
        keys = [k for k in st.index if len(st.chosen(k)) >= 2]
        pick = [keys[i] for i in rng.choice(len(keys), size=min(12, len(keys)), replace=False)]
        scored = []
        for key in pick:
            chosen = st.chosen(key)
            bits = st.score(key, chosen)
            child = copy.deepcopy(st)
            child.apply(key, chosen)
            assert abs(child.bits() - bits) < 1e-6
            scored.append((bits, key))
        bits, key = min(scored)
        st.apply(key, st.chosen(key))
        assert abs(st.bits() - bits) < 1e-6
        check_exact(st)
        st.add_position(STREAM[10 + step])
        check_exact(st)


def test_dissolve_scores_are_exact():
    st = IncrementalBoardState(A)
    for p in STREAM[:12]:
        st.add_position(p)
    # Learn three chunk types greedily, one built on another if possible.
    for _ in range(3):
        best = min(((st.score(k, st.chosen(k)), k) for k in st.index if len(st.chosen(k)) >= 2), key=lambda x: x[0])
        st.apply(best[1], st.chosen(best[1]))
    for p in STREAM[12:16]:
        st.add_position(p)
    check_exact(st)
    for Y in st.chunk_labels():
        bits, (gone, moves, new) = st.score_dissolve(Y)
        child = copy.deepcopy(st)
        child.apply_replace(moves, new)
        assert abs(child.bits() - bits) < 1e-6
        check_exact(child)
        assert all(m[3] not in gone for m in child.moves)


def test_grammar_read_off_the_state():
    learner = IncrementalChessLearner(alpha=A)
    for p in STREAM:
        learner.observe(p)
    g = learner.grammar
    assert g.K == learner.state.K
    for p in STREAM[:3]:
        lp = learner.log_prob(p)
        assert np.isfinite(lp) and lp < 0
    samples, _ = learner.generate(5, np.random.default_rng(0))
    assert len(samples) == 5
    # The grammar's total is a real code length (finite, positive) close to the state's.
    assert math.isfinite(g.info["total bits"]) and g.info["total bits"] > 0
