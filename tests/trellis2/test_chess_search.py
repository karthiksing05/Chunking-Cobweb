"""The board search with merge moves: every move's score equals the code
recomputed from scratch on the analyses it leaves."""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments", "v2"))

from chess_search import HierBoardSearch, code_of_analyses  # noqa: E402
from trellis2.chess import BoardSearch, parse_fen  # noqa: E402

FENS = [
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R",
    "r1bq1rk1/pppnbppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R",
    "rnbq1rk1/ppp2ppp/4pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R",
    "r1bq1rk1/ppp1npbp/3p2p1/3Pp3/2P1P3/2N2N2/PP2BPPP/R2QK2R",
    "rnbqk2r/ppp1ppbp/3p1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQK2R",
    "r1bq1rk1/ppp1ppbp/2np1np1/8/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1",
    "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R",
]


def _check_all_moves(search, limit=None):
    """Every scored move: its score equals the recomputed code after it."""
    moves = list(search.scored_moves())
    assert moves
    rng = np.random.default_rng(0)
    if limit is not None and len(moves) > limit:
        moves = [moves[i] for i in rng.choice(len(moves), size=limit, replace=False)]
    for bits, move in moves:
        child = copy.deepcopy(search)
        child.do(move)
        ref = code_of_analyses([p["position"] for p in child.pos], child.analyses(), child.a, child.counts)
        assert abs(child.bits() - ref["total"]) < 1e-6
        assert abs(bits - ref["total"]) < 1e-6, (move[0], move[1] if move[0] == "merge" else move[1])
    return moves


@pytest.mark.parametrize("counts", [True, False])
def test_flat_code_matches_board_search(counts):
    positions = [parse_fen(f) for f in FENS]
    s = HierBoardSearch(positions, 0.01, counts)
    assert abs(s.bits() - BoardSearch(positions, 0.01, counts).bits()) < 1e-9
    ref = code_of_analyses(positions, s.analyses(), 0.01, counts)
    assert abs(s.bits() - ref["total"]) < 1e-6
    parts = s.parts()
    assert abs(sum(parts.values()) - ref["total"]) < 1e-6


@pytest.mark.parametrize("counts", [True, False])
def test_chunk_and_merge_scores_are_exact_along_a_search(counts):
    positions = [parse_fen(f) for f in FENS]
    s = HierBoardSearch(positions, 0.01, counts)
    kinds = set()
    for _ in range(6):
        moves = _check_all_moves(s, limit=60)
        kinds |= {m[0] for _, m in moves}
        bits, move = min(moves, key=lambda x: x[0])
        # Take a merge whenever one is scored, so that later chunk moves
        # see merged labels (and merges see chunks over merged labels).
        merges = [m for m in moves if m[1][0] == "merge"]
        if merges and len(s.moves) % 2 == 1:
            bits, move = min(merges, key=lambda x: x[0])
        s.do(move)
        assert abs(s.bits() - bits) < 1e-6
    assert kinds == {"chunk", "merge"}
    assert any(m[0] == "merge" for m in s.moves)


def test_search_without_merges_is_board_search():
    positions = [parse_fen(f) for f in FENS]
    a = BoardSearch(positions, 0.01)
    b = HierBoardSearch(positions, 0.01, merges=False)
    assert abs(a.run(20) - b.run(20)) < 1e-9
    assert a.analyses() == b.analyses()


def test_replay_reproduces_the_search_and_held_out_code():
    positions = [parse_fen(f) for f in FENS]
    s = HierBoardSearch(positions, 0.01)
    s.run(15, patience=2)
    r = HierBoardSearch(positions, 0.01)
    r.replay(s.moves)
    assert abs(r.bits() - s.bits()) < 1e-9
    assert r.analyses() == s.analyses()
    held, flat = s.held_out_bits(positions[:4])
    assert np.isfinite(held) and np.isfinite(flat)


def test_merge_pools_chunks_over_a_category():
    # A chunk over a merged category pools the occurrences of both labels.
    positions = [parse_fen(f) for f in FENS]
    s = HierBoardSearch(positions, 0.01)
    s.apply_merge("wB", "wN")
    pairs = s.candidates()
    assert not any("wN" in (k[0], k[2]) for k in pairs)
    assert any(k[0] == "wB" or k[2] == "wB" for k in pairs)
    ref = code_of_analyses(positions, s.analyses(), 0.01)
    assert abs(s.bits() - ref["total"]) < 1e-6
