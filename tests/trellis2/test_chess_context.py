"""The board's top level read in context: the code of an analysis equals the
sum of its events' sequential log-probabilities (an independent loop), the
search's move scores equal the recomputed code, and the grammar's top level
is a normalised predictive that the code and log_prob agree on."""
import copy
import math
from collections import Counter, defaultdict

import numpy as np
import pytest

from trellis2.chess import (CHOICE_NATS, CONTEXT_OPTIONS as OPTIONS, KIND_INDEX, BoardSearch, ChessLearner,
                            ContextBoardSearch, anchor_of, chain_values, code_of_analyses, parse_fen)

from test_chess_search import FENS


def HierBoardSearch(positions, alpha):
    return BoardSearch(positions, alpha, merges=True)


def ContextChessLearner(**kw):
    kw.pop("log", None)
    return ChessLearner(context=True, **kw)

A = 0.01


def sequential_top_nats(positions, analyses, K, alpha, option):
    """-sum log P(label | anchor kind and square, context), each label
    predicted from the labels before it in reading order (written out)."""
    d, beta = option
    n_g, n_gx = defaultdict(Counter), [defaultdict(Counter) for _ in range(d)]
    nats = 0.0
    for position, tops in zip(positions, analyses):
        for node in sorted(tops, key=lambda n: (anchor_of(n)[1], anchor_of(n)[0])):
            sq = anchor_of(node)
            g = (KIND_INDEX[position[sq]], sq)
            x = chain_values(position, sq)
            lab = node[0]
            p = (n_g[g][lab] + alpha) / (sum(n_g[g].values()) + K * alpha)
            for l in range(d):
                row = n_gx[l][(g, x[l])]
                p = (row[lab] + beta * p) / (sum(row.values()) + beta)
            nats -= math.log(p)
            n_g[g][lab] += 1
            for l in range(d):
                n_gx[l][(g, x[l])][lab] += 1
    return nats


def reference_bits(search):
    positions = [p["position"] for p in search.pos]
    analyses = search.analyses()
    ref = code_of_analyses(positions, analyses, search.a, search.counts)
    K = len(search.rows)
    top = min(sequential_top_nats(positions, analyses, K, search.a, o) for o in OPTIONS)
    return ref["read"] + ref["rows"] + (top + CHOICE_NATS) / math.log(2)


def test_depth_zero_is_the_plain_code():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A)
    h = HierBoardSearch(positions, A)
    h.run(3)
    s.replay(h.moves)
    K = len(s.rows)
    plain = h.parts()["top"] * math.log(2)
    assert abs(s.top_nats(OPTIONS[0])[0] - plain) < 1e-8
    positions_, analyses = [p["position"] for p in s.pos], s.analyses()
    assert abs(sequential_top_nats(positions_, analyses, K, A, OPTIONS[0]) - plain) < 1e-8


def test_code_equals_sequential_log_probabilities():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A)
    h = HierBoardSearch(positions, A)
    h.run(4)
    s.replay(h.moves)
    positions_, analyses, K = [p["position"] for p in s.pos], s.analyses(), len(s.rows)
    for opt in OPTIONS:
        assert abs(s.top_nats(opt)[0] - sequential_top_nats(positions_, analyses, K, A, opt)) < 1e-8
    assert abs(s.bits() - reference_bits(s)) < 1e-6


def test_chunk_and_merge_scores_are_exact():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A, merges=True)
    rng = np.random.default_rng(1)
    seen = set()
    for step in range(4):
        moves = list(s.scored_moves())
        pick = [moves[i] for i in rng.choice(len(moves), size=min(25, len(moves)), replace=False)]
        for bits, move in pick:
            seen.add(move[0])
            child = copy.deepcopy(s)
            child.do(move)
            assert abs(child.bits() - bits) < 1e-6, move[0]
            assert abs(reference_bits(child) - bits) < 1e-6, move[0]
        merges = [m for m in moves if m[1][0] == "merge"]
        bits, move = min(merges if (merges and step % 2) else moves, key=lambda x: x[0])
        s.do(move)
        assert abs(s.bits() - bits) < 1e-6
    assert seen == {"chunk", "merge"}


def test_memory_code_and_predictive():
    positions = [parse_fen(f) for f in FENS]
    learner = ContextChessLearner(seed=0, alpha=A, max_steps=3, merges=True, log=None)
    for p in positions:
        learner.observe(p)
    g = learner.sleep()
    mem = learner.model.memory
    root = np.flatnonzero(mem.is_root)
    s = np.asarray(g.elem_symbol)
    # The memory's top-level code: the sequential sum over the grammar's symbols.
    analyses, by_pos = [], defaultdict(list)
    for e in root:
        by_pos[mem.experience_of[e]].append(e)

    def node(e):
        if mem.kind[e] == mem.PRIMITIVE:
            return (int(s[e]), mem.anchor[e])
        return (int(s[e]), (node(mem.left[e]), mem.relation[e], node(mem.right[e])))
    for pid in range(len(positions)):
        analyses.append([node(e) for e in by_pos[pid]])
    nats, opt = mem.symbols_nats(s[root], g.K, A)
    assert abs(nats - sequential_top_nats(positions, analyses, g.K, A, opt)) < 1e-8
    # The contextual predictive is a distribution, and log_prob is finite.
    for position in positions[:3]:
        for sq, t in position.items():
            T = mem.top_row(g, KIND_INDEX[t], sq[1] * 8 + sq[0], position)
            assert abs(T.sum() - 1) < 1e-9 and (T > 0).all()
        assert np.isfinite(learner.log_prob(position))
    samples, _ = learner.generate(5, np.random.default_rng(0))
    assert len(samples) == 5
