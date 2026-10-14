"""Split moves (trellis2/splits.py): stage B on given symbols reproduces
compile_grammar's code exactly (sequences and relational characters); the
plain code against an independent count; refinement keeps only shorter codes,
whose grammars recompute exactly; with the option off, nothing changes."""
import math
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.special import gammaln

from trellis2 import Trellis2
from trellis2.characters import RELATIONS, CharacterMemory
from trellis2.data import Tree
from trellis2.grammar import UNK, compile_with_symbols
from trellis2.splits import _Context, propagate, propose, prune, refine
from trellis2.unsupervised import UnsupervisedLearner

LN2 = math.log(2)

# A tiny grammar with number agreement: S -> NP_x VP_x, NP_x -> D_x N_x.
DET = {"s": ["a", "this"], "p": ["these", "many"]}
NOUN = {"s": ["dog", "cat", "bird"], "p": ["dogs", "cats", "birds"]}
VERB = {"s": ["runs", "sleeps"], "p": ["run", "sleep"]}


def agreement_corpus(seed, n=160):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.choice("sp")
        toks = [rng.choice(DET[x]), rng.choice(NOUN[x]), rng.choice(VERB[x])]
        if rng.random() < 0.4:
            y = rng.choice("sp")
            toks += ["near", rng.choice(DET[y]), rng.choice(NOUN[y])]
        n_t = len(toks)
        split = {(0, n_t): 2, (0, 2): 1}
        if n_t == 6:
            split.update({(2, 6): 3, (3, 6): 4, (4, 6): 5})
        out.append((toks, Tree(n_t, split)))
    return out


def characters(seed, n=60):
    rng = random.Random(seed)
    parts = ["口", "木", "日", "月", "人", "女"]

    def make(depth):
        if depth == 0 or rng.random() < 0.4:
            return rng.choice(parts)
        return (rng.choice(RELATIONS[:4]), make(depth - 1), make(depth - 1))
    return [make(3) for _ in range(n)]


def test_stage_b_reproduces_the_chosen_grammars_code():
    m = Trellis2(seed=3)
    for toks, tree in agreement_corpus(3):
        m.learn(toks, tree)
    g = m.consolidate()
    assert compile_with_symbols(m.memory, g.elem_symbol, seed=3).info["total bits"] == g.info["total bits"]
    c = Trellis2(seed=5, memory=CharacterMemory())
    for ch in characters(5):
        c.learn(ch)
    gc = c.consolidate()
    assert compile_with_symbols(c.memory, gc.elem_symbol, seed=5).info["total bits"] == gc.info["total bits"]


def test_plain_code_is_the_count_code():
    """``_Context.plain_code``: one Dirichlet-multinomial row per symbol over
    its outcomes plus the top level, against counts made here."""
    m = Trellis2(seed=1)
    for toks, tree in agreement_corpus(1, 80):
        m.learn(toks, tree)
    g = m.consolidate()
    ctx = _Context(m.memory)
    rng = np.random.default_rng(0)
    for _ in range(3):
        s = rng.integers(0, 5, len(m.memory))
        s[0] = 4
        K, alpha = 5, 0.001
        rows = defaultdict(Counter)
        for e in range(len(s)):
            out = ("w", m.memory.token[e]) if m.memory.kind[e] == m.memory.PRIMITIVE else \
                ("p", int(s[m.memory.left[e]]), int(s[m.memory.right[e]]))
            rows[int(s[e])][out] += 1
        A = ctx.V + K * K
        nats = sum(gammaln(sum(r.values()) + A * alpha) - gammaln(A * alpha)
                   - sum(gammaln(c + alpha) - gammaln(alpha) for c in r.values()) for r in rows.values())
        nats += m.memory.top_level_nats(s, K, alpha)
        assert abs(ctx.plain_code(s, alpha=alpha) - nats) < 1e-6


def test_moves_give_valid_symbols():
    m = Trellis2(seed=2)
    for toks, tree in agreement_corpus(2):
        m.learn(toks, tree)
    g = m.consolidate()
    ctx = _Context(m.memory)
    s = g.elem_symbol.copy()
    key = ctx.keys(s)
    cfg = {"min_units": 2, "components": 3, "joint_rounds": 8, "joint_separation": 0.5, "joint_minority": 0.05}
    seen = 0
    for A in range(int(s.max()) + 1):
        for mask in propose(ctx, A, s, key, cfg):
            assert mask.any() and (mask <= (s == A)).all() and ((s == A) & ~mask).any()
            sj = propagate(ctx, A, mask, s, key, cfg)
            if sj is None:
                continue
            seen += 1
            for x in (sj, prune(ctx, s, sj, 0.001, keep=A)):
                assert set(np.unique(x)) == set(range(int(x.max()) + 1))     # contiguous
                assert (x[s != x] > s.max()).all()                         # moved elements get new symbols
    assert seen > 0


def test_refinement_keeps_only_shorter_codes_and_recomputes_exactly():
    for seed in (4, 6):
        base = Trellis2(seed=seed)
        split = Trellis2(seed=seed, joint_splits=True)
        for toks, tree in agreement_corpus(seed):
            base.learn(toks, tree)
            split.learn(toks, tree)
        g0, g1 = base.consolidate(), split.consolidate()
        assert g1.info["total bits"] <= g0.info["total bits"] + 1e-9
        assert g1.info["splits: unsplit total bits (recomputed)"] == g0.info["total bits"]
        assert abs(compile_with_symbols(split.memory, g1.elem_symbol, seed=seed).info["total bits"]
                   - g1.info["total bits"]) < 1e-9
        # The refined grammar parses and generates.
        lp = split.log_prob(["these", "dogs", "run"])
        assert np.isfinite(lp)
        samples, _ = split.generate(5, np.random.default_rng(0))
        assert len(samples) == 5


def test_option_off_is_the_old_learner():
    data = [t for t, _ in agreement_corpus(9, 60)]
    out = []
    for kw in ({"joint_splits": False}, {"joint_splits": True, "split_config": {"max_splits": 0}}):
        u = UnsupervisedLearner(seed=9, sampling=(1.0,), **kw)
        for s in data:
            u.observe(s)
        g = u.sleep()
        out.append((g.info["total bits"], [t.brackets() for t in u.trees]))
    assert out[0] == out[1]
    u = UnsupervisedLearner(seed=9, sampling=(1.0,))          # the default: split moves on
    for s in data:
        u.observe(s)
    assert u.sleep().info["total bits"] <= out[0][0] + 1e-9
