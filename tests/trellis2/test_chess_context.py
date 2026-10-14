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


# --------------------------------------------------------------------------- #
# The read's answers in context
# --------------------------------------------------------------------------- #
from trellis2.chess import COUNTS, KINDS, backoff_terms, counts_before  # noqa: E402
from trellis2.mdl import backoff_chain_coder  # noqa: E402


def sequential_read_nats(positions, analyses, alpha, option):
    """-sum log P(answer | kind, square, count, context) over every question
    the read asks, each predicted from the answers before it (written out)."""
    d, beta = option
    n_g, n_gx = defaultdict(Counter), [defaultdict(Counter) for _ in range(d)]
    nats = 0.0
    for position, tops in zip(positions, analyses):
        anchored = {anchor_of(n): n for n in tops}
        before = counts_before(position)
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in position and sq not in anchored:
                continue                                   # covered by a chunk: not read
            stop = KIND_INDEX[position[sq]] if sq in anchored else len(KINDS)
            x = chain_values(position, sq)
            for k in range(min(stop + 1, len(KINDS))):
                g = (k, i, min(int(before[i][k]), COUNTS - 1))
                ans = int(k == stop)
                p = (n_g[g][ans] + alpha) / (sum(n_g[g].values()) + 2 * alpha)
                for l in range(d):
                    row = n_gx[l][(g, x[l])]
                    p = (row[ans] + beta * p) / (sum(row.values()) + beta)
                nats -= math.log(p)
                n_g[g][ans] += 1
                for l in range(d):
                    n_gx[l][(g, x[l])][ans] += 1
    return nats


def reference_bits_read(search):
    positions = [p["position"] for p in search.pos]
    analyses = search.analyses()
    ref = code_of_analyses(positions, analyses, search.a, search.counts)
    K = len(search.rows)
    top = min(sequential_top_nats(positions, analyses, K, search.a, o) for o in OPTIONS)
    read = min(sequential_read_nats(positions, analyses, search.a, o) for o in OPTIONS)
    return ref["rows"] + (top + read + 2 * CHOICE_NATS) / math.log(2)


def test_backoff_terms_are_the_shared_coder():
    rng = np.random.default_rng(0)
    n = 500
    g, k = rng.integers(0, 7, n), rng.integers(0, 3, n)
    ctx = [rng.integers(0, 4, n), rng.integers(0, 9, n)]
    w = rng.random(n) + 0.5
    for beta in (1.0, 4.0):
        t = backoff_terms(g, ctx, k, w, 3, 0.01, beta)
        for d in range(3):
            assert abs(t[d].sum() - backoff_chain_coder(g, ctx[:d], w)(k, 3, 0.01, [beta] * d)) < 1e-9


def test_read_code_equals_sequential_log_probabilities():
    positions = [parse_fen(f) for f in FENS]
    h = HierBoardSearch(positions, A)
    h.run(4)
    s = ContextBoardSearch(positions, A, read_context=True)
    s.replay(h.moves)
    analyses = s.analyses()
    for opt in OPTIONS:
        assert abs(s._read_nats(opt) - sequential_read_nats(positions, analyses, A, opt)) < 1e-8
    # Depth 0 is the counted read.
    assert abs(s._read_nats((0, 0.0)) - h._read_nats()) < 1e-8
    assert abs(s.bits() - reference_bits_read(s)) < 1e-6


def test_read_context_move_scores_are_exact():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A, merges=True, read_context=True)
    rng = np.random.default_rng(2)
    seen = set()
    for step in range(4):
        moves = list(s.scored_moves())
        pick = [moves[i] for i in rng.choice(len(moves), size=min(20, len(moves)), replace=False)]
        for bits, move in pick:
            seen.add(move[0])
            child = copy.deepcopy(s)
            child.do(move)
            assert abs(child.bits() - bits) < 1e-6, move[0]
            assert abs(reference_bits_read(child) - bits) < 1e-6, move[0]
        merges = [m for m in moves if m[1][0] == "merge"]
        bits, move = min(merges if (merges and step % 2) else moves, key=lambda x: x[0])
        s.do(move)
        assert abs(s.bits() - bits) < 1e-6
        # The per-row cache after the move equals a fresh computation.
        fresh = s._read_codes(np.arange(len(s.r_g)))
        for opt in OPTIONS:
            assert np.allclose(fresh[opt], s.r_code[opt])
    assert seen == {"chunk", "merge"}


def test_memory_read_in_context():
    positions = [parse_fen(f) for f in FENS]
    learner = ChessLearner(seed=0, alpha=A, max_steps=3, merges=True, read_context=True)
    for p in positions:
        learner.observe(p)
    learner.sleep()
    mem = learner.model.memory
    analyses = [[] for _ in positions]
    for e in np.flatnonzero(mem.is_root):
        analyses[mem.experience_of[e]].append((None, mem.anchor[e]) if mem.kind[e] == mem.PRIMITIVE
                                              else _node(mem, e))
    nats, opt = mem.answers_nats(A)
    assert abs(nats - sequential_read_nats(positions, analyses, A, opt)) < 1e-8
    for position in positions[:3]:
        assert np.isfinite(learner.log_prob(position))
    samples, _ = learner.generate(5, np.random.default_rng(0))
    assert len(samples) == 5


def _node(mem, e):
    if mem.kind[e] == mem.PRIMITIVE:
        return (None, mem.anchor[e])
    return (None, (_node(mem, mem.left[e]), mem.relation[e], _node(mem, mem.right[e])))


# --------------------------------------------------------------------------- #
# A context made of elements: the neighbours' categories, a chunk one value
# --------------------------------------------------------------------------- #
def sequential_element_top_nats(positions, analyses, K, alpha, option):
    """As sequential_top_nats, the context being the labels of the top-level
    elements holding the pieces seen (written out from the analyses)."""
    d, beta = option
    n_g, n_gx = defaultdict(Counter), [defaultdict(Counter) for _ in range(d)]
    nats = 0.0

    def squares(n):
        return squares(n[1][0]) + squares(n[1][2]) if len(n[1]) == 3 else [n[1]]
    for position, tops in zip(positions, analyses):
        holder = {m: n[0] for n in tops for m in squares(n)}
        for node in sorted(tops, key=lambda n: (anchor_of(n)[1], anchor_of(n)[0])):
            sq = anchor_of(node)
            g = (KIND_INDEX[position[sq]], sq)
            x = chain_values(position, sq, value_of=lambda q: holder[q])
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


def reference_bits_elements(search):
    positions = [p["position"] for p in search.pos]
    analyses = search.analyses()
    ref = code_of_analyses(positions, analyses, search.a, search.counts)
    K = len(search.rows)
    top = min(sequential_element_top_nats(positions, analyses, K, search.a, o) for o in OPTIONS)
    return ref["read"] + ref["rows"] + (top + CHOICE_NATS) / math.log(2)


def test_element_context_starts_as_the_piece_context():
    positions = [parse_fen(f) for f in FENS]
    a = ContextBoardSearch(positions, A)
    b = ContextBoardSearch(positions, A, context_of="elements")
    assert abs(a.bits() - b.bits()) < 1e-9


def test_element_context_moves_are_exact():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A, merges=True, context_of="elements")
    rng = np.random.default_rng(3)
    seen = set()
    for step in range(5):
        moves = list(s.scored_moves())
        pick = [moves[i] for i in rng.choice(len(moves), size=min(20, len(moves)), replace=False)]
        for bits, move in pick:
            seen.add(move[0])
            child = copy.deepcopy(s)
            child.do(move)
            assert abs(child.bits() - bits) < 1e-6, move[0]
            assert abs(reference_bits_elements(child) - bits) < 1e-6, move[0]
        merges = [m for m in moves if m[1][0] == "merge"]
        bits, move = min(merges if (merges and step % 2) else moves, key=lambda x: x[0])
        s.do(move)
        assert abs(s.bits() - bits) < 1e-6
        # The contexts kept up to date equal those of a search rebuilt from scratch.
        fresh = ContextBoardSearch(positions, A, merges=True, context_of="elements")
        fresh.replay(s.moves)
        assert abs(fresh.bits() - s.bits()) < 1e-6
    assert seen == {"chunk", "merge"}


def _element_learner(read_context=False):
    positions = [parse_fen(f) for f in FENS]
    learner = ChessLearner(seed=0, alpha=A, max_steps=4, context_of="elements", read_context=read_context)
    for p in positions:
        learner.observe(p)
    learner.sleep()
    return positions, learner


def test_memory_element_context_code():
    positions, learner = _element_learner()
    g, mem = learner.grammar, learner.model.memory
    s = np.asarray(g.elem_symbol)
    root = np.flatnonzero(mem.is_root)
    analyses = [[] for _ in positions]
    for e in root:
        analyses[mem.experience_of[e]].append(_sym_node(mem, s, e))
    nats, opt = mem.symbols_nats(s[root], g.K, A, s_all=s)
    assert abs(nats - sequential_element_top_nats(positions, analyses, g.K, A, opt)) < 1e-8
    for position in positions[:3]:
        assert np.isfinite(learner.log_prob(position))


def _sym_node(mem, s, e):
    if mem.kind[e] == mem.PRIMITIVE:
        return (int(s[e]), mem.anchor[e])
    return (int(s[e]), (_sym_node(mem, s, mem.left[e]), mem.relation[e], _sym_node(mem, s, mem.right[e])))


def test_element_context_draws_what_it_codes():
    """With the top-level symbols explicit, a position, its shapes and its
    top-level symbols are drawn as often as exp(log_prob) says."""
    rng_positions = {(6, 0): "wK", (5, 1): "wP", (6, 1): "wP", (7, 1): "wP",
                     (6, 7): "bK", (5, 6): "bP", (6, 6): "bP", (7, 6): "bP"}
    spots = [(2, 2), (3, 3), (4, 3), (1, 2)]
    learner = ChessLearner(seed=0, max_steps=30, context_of="elements")
    for i in range(40):
        learner.observe({**rng_positions, spots[i % 4]: "wN"})
    learner.sleep()
    model = learner.model

    def shape(n):
        return (shape(n[1][0]), n[1][1], shape(n[1][2])) if len(n[1]) == 3 else n[1]

    def key(smp):
        return frozenset(smp[0].items()), tuple((int(t[0]), shape(t)) for t in smp[1])
    n = 6000
    first, _ = model.generate(n, np.random.default_rng(0))
    second, rejected = model.generate(n, np.random.default_rng(1))
    accepted = n / (n + rejected)
    some = {}
    for smp in first:
        some.setdefault(key(smp), smp)
    counts = Counter(key(smp) for smp in second)
    for k, _ in Counter(key(smp) for smp in first).most_common(5):
        expected = n * math.exp(model.log_prob(some[k][0], some[k][1])) / accepted
        assert abs(counts[k] - expected) <= 4 * math.sqrt(expected) + 2, (counts[k], expected)


def test_typed_moves_with_element_context_are_exact():
    positions = [parse_fen(f) for f in FENS]
    s = ContextBoardSearch(positions, A, context_of="elements", typed=True)
    for step in range(3):
        moves = list(s.scored_moves())
        typed = [m for m in moves if m[1][1][1] in ("attacks", "defends")]
        assert typed
        for bits, move in sorted(typed, key=lambda x: x[0])[:8]:
            child = copy.deepcopy(s)
            child.do(move)
            assert abs(child.bits() - bits) < 1e-6
            assert abs(reference_bits_elements(child) - bits) < 1e-6
        bits, move = min(typed, key=lambda x: x[0])
        s.do(move)
        fresh = ContextBoardSearch(positions, A, context_of="elements", typed=True)
        fresh.replay(s.moves)
        assert abs(fresh.bits() - s.bits()) < 1e-6


def test_element_context_held_out_symbols():
    """A new position's analysis names the search's labels; each top-level
    symbol is then the one that codes its element shortest given the symbols
    chosen before it, and the code equals that of the analysis with those
    symbols named (the regression: a label's commonest training symbol could
    be nearly impossible at a square and cost tens of bits)."""
    from trellis2.chess import inside_of_analysis, is_chunk, left_corner
    positions, learner = _element_learner()
    g, mem = learner.grammar, learner.model.memory
    lc = left_corner(g)
    for position in positions[:6]:
        analysis = sorted(learner.analyse(position), key=lambda n: (anchor_of(n)[1], anchor_of(n)[0]))
        chosen = []
        lp = mem.log_prob(g, position, analysis, chosen)
        named = [(sym, n[1]) for sym, n in zip(chosen, analysis)]
        assert abs(mem.log_prob(g, position, named) - lp) < 1e-9
        held = {}
        for sym, n in zip(chosen, analysis):
            sq = anchor_of(n)
            k = KIND_INDEX[position[sq]]
            T = mem.top_row(g, k, sq[1] * 8 + sq[0], position, held)
            _, v = inside_of_analysis(g, n, parts=lambda m: (m[1][0], m[1][2]) if is_chunk(m) else None,
                                      token=lambda m: position[m[1]], relation=lambda m: m[1][1])
            joint = T * v / lc[:, k]
            assert joint[sym] >= joint.max() - 1e-15
            for m in (lambda f: f(f, n))(lambda f, x: f(f, x[1][0]) + f(f, x[1][2]) if len(x[1]) == 3 else [x[1]]):
                held[m] = sym
