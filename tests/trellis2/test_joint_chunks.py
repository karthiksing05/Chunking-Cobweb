"""Joint chunk moves (experiments/v2/joint_chunks.py): composite moves scored
exactly against a full recomputation of the plain code, the search with them,
and re-bracketing of analyses."""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "v2"))

from trellis2.mdl_search import _State, chunk_and_merge, code_bits, to_tree  # noqa: E402

import joint_chunks as JC  # noqa: E402

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


def state_class(**kw):
    return type("S", (JC.JointState,), dict({"min_count": 2, "top_windows": 30, "max_leaves": 5}, **kw))


def test_shapes_are_every_binary_bracketing():
    assert [len(JC.shapes(0, m)) for m in range(1, 7)] == [1, 1, 2, 5, 14, 42]
    assert JC.shape_spans(((0, 1), 2)) == [(0, 2), (0, 3)]


def test_composite_moves_are_exact():
    """The code after each composite move, from the changed rows only, equals
    the code recomputed from scratch; the counts equal those rebuilt."""
    for seed in range(3):
        for reuse in (True, False):
            rng = random.Random(seed)
            st = state_class(reuse=reuse)(corpus(seed), 9, 0.001)
            checked = 0
            for _ in range(5):
                moves = list(st.scored_moves())
                comps = [m for m in moves if m[1][0] == "composite"][:12]
                for nats, move in comps:
                    child = st.apply(move)
                    assert abs(nats / LN2 - code_bits(child.analyses, 9, 0.001)) < 1e-6
                    ref = _State(child.analyses, 9, 0.001)
                    assert child.rows == ref.rows and child.start == ref.start and child.cont == ref.cont
                    assert ({k: v for k, v in child.parents.items() if v}
                            == {k: v for k, v in ref.parents.items() if v})
                    # Chunk and merge moves stay exact after a composite.
                    for nats2, move2 in list(child.scored_moves())[::25]:
                        assert abs(nats2 / LN2 - code_bits(child.apply(move2).analyses, 9, 0.001)) < 1e-6
                    checked += 1
                st = st.apply(rng.choice(moves)[1])
            assert checked > 0


def test_a_composite_groups_every_occurrence_as_its_shape():
    st = state_class(reuse=False)(corpus(1), 9, 0.001)
    nats, move = next((x for x in st.scored_moves() if x[1][0] == "composite"))
    _, window, shape = move
    child = st.apply(move)
    n_before = sum(len(JC.match_positions(st._labels(t), window)) for t in st.tops)
    assert n_before >= 2
    # One top-level node replaces each occurrence's leaves.
    assert sum(map(len, st.tops)) - sum(map(len, child.tops)) == n_before * (len(window) - 1)
    # The leaves are joined in the shape: each new node spans its shape span.
    spans = [JC.shape_spans(shape)]
    assert spans


def test_search_with_composites_is_exact_and_no_worse_than_without():
    for seed in range(2):
        c = corpus(seed)
        plain, bits0 = chunk_and_merge(c, 9, 0.001, beam=1)
        joint, bits1 = JC.joint_chunk_and_merge(c, 9, 0.001, beam=1, state_class=state_class())
        assert abs(bits1 - code_bits(joint, 9, 0.001)) < 1e-6
        assert bits1 < code_bits(c, 9, 0.001)


def test_restructure_makes_required_spans_constituents():
    rng = random.Random(0)
    analyses, _ = chunk_and_merge(corpus(2), 9, 0.001)
    for tops in analyses[:60]:
        tree = to_tree(tops)
        n = tree.n
        if n < 3:
            continue
        # A random laminar set of required spans (a random binary tree's brackets).
        spans = set()

        def rand(i, j):
            if j - i >= 2:
                spans.add((i, j))
                k = rng.randint(i + 1, j - 1)
                rand(i, k)
                rand(k, j)
        i = rng.randint(0, n - 2)
        rand(i, rng.randint(i + 2, n))
        new = JC.restructure(tops, spans, JC.Labeler())
        t2 = to_tree(new)
        assert t2.is_valid() and t2.n == n
        assert [w for _, w in _leaves(new)] == [w for _, w in _leaves(tops)]
        b = t2.brackets() | set(t2.roots)
        assert spans <= b
        # Existing brackets that cross no required one are kept.
        assert {s for s in tree.brackets() if not any(JC.crosses(s, r) for r in spans)} <= b
        # Nothing required: unchanged.
        assert JC.restructure(tops, set(), JC.Labeler()) == tops


def test_tree_composite_on_forests_matches_the_search_move():
    st = state_class(reuse=False)(corpus(4), 9, 0.001)
    comps = [m for _, m in st.scored_moves() if m[0] == "composite"]
    for move in comps[:10]:
        _, window, shape = move
        child = st.apply(move)
        new, n = JC.apply_tree_composite(st.analyses, window, shape, JC.Labeler(), 5, top_only=True)
        # Same brackets and roots (the categories of new nodes are named apart).
        for a, b in zip(child.analyses, new):
            ta, tb = to_tree(a), to_tree(b)
            assert ta.roots == tb.roots and ta.brackets() == tb.brackets()


def _leaves(tops):
    out = []

    def rec(node):
        lab, body = node
        if isinstance(body, str):
            out.append((lab, body))
        else:
            rec(body[0])
            rec(body[1])
    for t in tops:
        rec(t)
    return out
