"""The continuously incremental learner: Cobweb removal, exact online codes,
and the learner's state against recomputation from scratch."""
import math
import random
import sys

import numpy as np
import pytest

from trellis2.incremental import CONTEXT_OPTION_BITS, CodeTables, IncrementalLearner, Patch
from trellis2.incremental_cobweb import IncrementalCobwebTree

LN2 = math.log(2.0)


def _instances(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        words = [rng.choice("pqrst") for _ in range(rng.randint(1, 4))]
        bag = {}
        for w_ in words:
            bag[w_] = bag.get(w_, 0.0) + 1.0 / len(words)
        yield {"a": rng.choice("xyz"), "b": rng.choice([1, 2, "q"]), "bag": bag}, rng.choice([1.0, 0.5, 2.0])


def _plain(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        yield {"a": rng.choice("xyz"), "b": rng.choice("pq"), "c": rng.choice("uvwst")}, rng.choice([1.0, 2.0])


def _shape(tree):
    return [(n.id, n.parent.id if n.parent is not None else None, [c.id for c in n.children],
             n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.nodes()]


def _leaf_counts(tree):
    return {n.id: (n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.leaves()}


# --------------------------------------------------------------------------- #
# Cobweb with removal
# --------------------------------------------------------------------------- #
def test_insertion_reproduces_the_reference_and_the_compiled_tree():
    if sys.version_info < (3, 12):
        pytest.skip("the reference sums with Python 3.12's compensated sum()")
    from reference_cobweb import CobwebTree as Reference
    from trellis2.cobweb import CobwebTree as Compiled
    data = list(_instances(1500, 5))
    ref = Reference(("a", "b", "bag"), seed=7)
    inc = IncrementalCobwebTree(("a", "b", "bag"), seed=7)
    fast = Compiled(("a", "b", "bag"), seed=7)
    ids = [inc.ifit(x, w).id for x, w in data]
    assert ids == [ref.ifit(x, w).id for x, w in data] == [fast.ifit(x, w).id for x, w in data]
    assert _shape(inc) == _shape(ref) == _shape(fast)
    assert np.allclose(inc.concept_codes(0.5), fast.concept_codes(0.5), rtol=1e-12, atol=1e-9)


def test_insert_then_remove_restores_the_counts():
    tree = IncrementalCobwebTree(("a", "b", "c"), seed=1)
    for x, w in _plain(300, 2):
        tree.ifit(x, w)
    before_leaves = _leaf_counts(tree)
    before_root = {a: dict(d) for a, d in tree.root.av.items()}
    count = tree.root.count
    for x, w in _plain(60, 4):
        leaf = tree.ifit(x, w)
        tree.unfit(leaf, x, w)
        tree.check_invariants()
        assert tree.root.count == pytest.approx(count, abs=1e-12)
        for a, d in before_root.items():
            assert tree.root.av[a] == pytest.approx(d)
        # Every leaf that held instances before holds exactly them again.
        after = _leaf_counts(tree)
        for nid, (c, av) in before_leaves.items():
            assert nid in after and after[nid][0] == pytest.approx(c)
            for a in av:
                assert after[nid][1][a] == pytest.approx(av[a])
        assert set(after) == set(before_leaves)


def test_removing_everything_in_any_order_keeps_the_invariants():
    tree = IncrementalCobwebTree(("a", "b", "bag"), seed=3)
    stored = []
    for x, w in _instances(400, 9):
        stored.append((tree.ifit(x, w), x, w))
    rng = random.Random(0)
    rng.shuffle(stored)
    leaves_of = {}
    for leaf, x, w in stored:
        leaves_of.setdefault(id(leaf), []).append((x, w))
    for k, (leaf, x, w) in enumerate(stored):
        tree.unfit(leaf, x, w)
        if k % 25 == 0:
            tree.check_invariants(tol=1e-6)
            # Leaves are stable handles: each still holds only its instances.
            live = {id(n) for n in tree.leaves()}
            for l2, _, _ in stored[k + 1:]:
                assert id(l2) in live
    assert tree.root.count == 0.0 and not tree.root.children


def test_reinsertion_after_removal_matches_a_count_check():
    tree = IncrementalCobwebTree(("a", "b", "c"), seed=5)
    stored = [(tree.ifit(x, w), x, w) for x, w in _plain(500, 6)]
    rng = random.Random(1)
    for _ in range(300):
        k = rng.randrange(len(stored))
        leaf, x, w = stored[k]
        tree.unfit(leaf, x, w)
        x2 = dict(x, c=rng.choice("uvwst"))          # a re-description
        stored[k] = (tree.ifit(x2, w), x2, w)
    tree.check_invariants()
    assert tree.root.count == pytest.approx(sum(w for _, _, w in stored))
    for a in ("a", "b", "c"):
        expect = {}
        for _, x, w in stored:
            expect[x[a]] = expect.get(x[a], 0.0) + w
        assert tree.root.av[a] == pytest.approx(expect)


# --------------------------------------------------------------------------- #
# Exact online code
# --------------------------------------------------------------------------- #
def _random_patch(rng, sign=1.0):
    p = Patch()
    for _ in range(rng.randint(1, 6)):
        s, c = rng.randrange(5), rng.randrange(4)
        key = ("P", rng.choice("abcdef")) if rng.random() < 0.5 else ("C", rng.randrange(5), rng.randrange(5))
        p.record(s, c, key, sign * rng.choice([1.0, 0.5, 2.0]))
    p.top([rng.randrange(5) for _ in range(rng.randint(1, 3))], sign)
    return p


def test_deltas_equal_the_code_after_committing():
    rng = random.Random(11)
    t = CodeTables(0.001)
    history = []
    for step in range(300):
        if history and rng.random() < 0.3:
            p = Patch().add(history.pop(rng.randrange(len(history))), -1.0)
        else:
            p = _random_patch(rng)
            history.append(p)
        before = t.bits()
        d = t.delta(p)
        t.commit(p)
        assert t.bits() - before == pytest.approx(d, abs=1e-7)
        assert t.bits() == pytest.approx(t.recompute_bits(), abs=1e-7)
    for p in history:
        t.commit(Patch().add(p, -1.0))
    assert t.K == 0 and t.bits() == pytest.approx(0.0, abs=1e-7)


@pytest.fixture(scope="module")
def small_learner():
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    import os
    root = default_data_root()
    path = os.path.join(root, CONDITIONS["small"])
    if not os.path.exists(path):
        pytest.skip("synthetic corpora not available")
    train, _ = v1_split(load_corpus(path), 13)
    learner = IncrementalLearner(seed=13, check=True)     # verifies after every commit
    for e in train[:45]:
        learner.observe(e.tokens)
    return learner


def test_the_learner_code_is_the_factored_grammar_code(small_learner):
    """The count tables' code equals ``grammar.py``'s code of the stored
    analyses with the learner's symbols and rule classes."""
    from trellis2.grammar import UNK, _compact, _elements, _factored_code
    from trellis2.mdl import elias_delta_bits
    from trellis2.memory import Memory
    L = small_learner
    mem = Memory(granularities=1)
    syms, keys = [], []
    for tokens, tree in zip(L.experiences, L.stored_trees()):
        start = len(mem)
        mem.add(tokens, tree)
        for e in range(start, len(mem)):
            syms.append(tree.label[mem.span[e]])
    for e in range(len(mem)):
        if mem.kind[e] == Memory.PRIMITIVE:
            keys.append(("P", mem.token[e]))
        else:
            keys.append(("C", syms[mem.left[e]], syms[mem.right[e]]))
    vocab = mem.vocabulary() + [UNK]
    el = _elements(mem, None, None, {t: i for i, t in enumerate(vocab)})
    _, s = _compact(np.array(syms))
    _, c = _compact(np.array([L.class_of[k] for k in keys]))
    K, M, alpha = int(s.max()) + 1, int(c.max()) + 1, L.alpha
    nats = (_factored_code(el, s, K, len(vocab), alpha, c)(np.arange(M))
            + mem.top_level_nats(s, K, alpha) + mem.layout_nats(alpha))
    bits = nats / LN2 + elias_delta_bits(K) + elias_delta_bits(M) + CONTEXT_OPTION_BITS
    assert L.tables.bits() == pytest.approx(bits, abs=1e-6)
    assert L.grammar.info["total bits"] == pytest.approx(bits, abs=1e-6)


def test_merges_scored_from_records_equal_relabelling_every_element(small_learner):
    L = small_learner
    syms = sorted(L.members)
    for a in syms:
        for b in syms:
            if a >= b:
                continue
            p1, nc1 = L._merge_patch(a, b)
            p2, nc2 = L._relabel_patch({e: a for e in L.members[b]})
            assert L.tables.delta(p1) == pytest.approx(L.tables.delta(p2), abs=1e-8)
            assert nc1 == nc2


def test_every_candidate_move_is_scored_exactly(small_learner):
    """Apply a sample of candidate moves to copies of the learner and compare
    the scored change with the code recomputed after the move."""
    import copy
    L = small_learner
    rng = random.Random(2)
    cands = list(L._candidates(len(L.experiences) - 1))
    assert cands
    kinds = set()
    for d, move in rng.sample(cands, min(40, len(cands))):
        L2 = copy.deepcopy(L)
        before = L2.tables.bits()
        L2._apply(move)                 # check=True: verify() recomputes from the analyses
        assert L2.tables.bits() - before == pytest.approx(d, abs=1e-6)
        kinds.add(move[0])
    assert {"merge", "relabel", "classes"} <= kinds


def test_reanalysis_keeps_the_tables_exact(small_learner):
    import copy
    L = copy.deepcopy(small_learner)
    for i in range(len(L.experiences)):
        L.reanalyse(i)                  # verified after every accepted replacement
    L.verify()


def test_the_grammar_reads_the_counts(small_learner):
    g = small_learner.grammar
    assert np.allclose(g.U.sum(axis=1), 1) and np.allclose(g.E.sum(axis=1), 1)
    assert g.K == small_learner.tables.K and g.M == small_learner.tables.M
    for tokens in small_learner.experiences[:5]:
        assert np.isfinite(small_learner.log_prob(tokens))


@pytest.fixture(scope="module")
def med_forests():
    """MED after a few sentences: forests of pieces, where compounds apply."""
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    import os
    path = os.path.join(default_data_root(), CONDITIONS["med"])
    if not os.path.exists(path):
        pytest.skip("synthetic corpora not available")
    train, _ = v1_split(load_corpus(path), 13)
    learner = IncrementalLearner(seed=13, check=True, config={"compound": 0, "lookahead": 0, "chunk_window": 2})
    for e in train[:25]:
        learner.observe(e.tokens)
    return learner


def test_trials_are_undone_exactly_and_compounds_scored_exactly(med_forests):
    """A split tried on the counts (to score its follow-up chunk) leaves the
    state as it was; the compound's score is the code change of applying it."""
    import copy
    L = copy.deepcopy(med_forests)
    i = len(L.experiences) - 1
    splits = [(d, mv) for d, mv in L._candidates(i) if mv[0] == "relabel" and len(mv) > 4]
    assert splits
    bits, syms = L.tables.bits(), list(L.sym)
    classes, pairs = dict(L.class_of), {k: set(v) for k, v in L.pair_exps.items()}
    checked = 0
    for d, mv in splits:
        if checked >= 4:
            break
        undo = L._trial(mv)
        follow = min(L._followups(mv[1]), key=lambda x: x[0], default=None)
        L._undo(undo)
        L.verify()
        assert L.tables.bits() == pytest.approx(bits, abs=1e-9)
        assert L.sym == syms and L.class_of == classes
        assert {k: v for k, v in L.pair_exps.items() if v} == pairs
        if follow is None:
            continue
        L2 = copy.deepcopy(L)
        L2._apply(("compound", mv, follow[1]))
        assert L2.tables.bits() - bits == pytest.approx(d + follow[0], abs=1e-6)
        checked += 1
    assert checked


def test_profiled_moves_equal_element_relabelling(small_learner, med_forests):
    for L in (small_learner, med_forests):
        rng = random.Random(5)
        units = sorted(L.units, key=str)
        for (A, key) in rng.sample(units, min(15, len(units))):
            els = L.units[(A, key)]
            prof = L._profile(els)
            for B in sorted(L.members)[:6] + [L.next_sym]:
                if B == A:
                    continue
                p1, nc1 = L._profile_patch(prof, B)
                p2, nc2 = L._relabel_patch({e: B for e in els})
                assert L.tables.delta(p1) == pytest.approx(L.tables.delta(p2), abs=1e-8)
                assert nc1 == nc2


def test_joint_splits_rotations_and_inner_chunks_are_scored_exactly(small_learner):
    import copy
    L = small_learner
    joint, rot, inner = [], [], []
    for i in range(len(L.experiences) - 1, -1, -1):
        for d, mv in L._candidates(i):
            if mv[0] == "relabel" and len(mv) > 4 and len({L.sym[e] for e in mv[1]}) > 1:
                joint.append((d, mv))
            elif mv[0] == "rotate":
                rot.append((d, mv))
            elif mv[0] == "inner chunk":
                inner.append((d, mv))
        if len(joint) >= 4 and len(rot) >= 4 and len(inner) >= 4:
            break
    assert joint and rot and inner
    for d, mv in joint[:4] + rot[:4] + inner[:4]:
        L2 = copy.deepcopy(L)
        before = L2.tables.bits()
        L2._apply(mv)
        assert L2.tables.bits() - before == pytest.approx(d, abs=1e-6)


def test_online_word_classes_equal_the_nights():
    from trellis2.incremental import OnlineWordClasses
    from trellis2.mdl_search import word_classes
    rng = random.Random(4)
    words = "the a dog cat saw chased in on big red".split()
    sentences = [[rng.choice(words) for _ in range(rng.randint(2, 7))] for _ in range(60)]
    wc = OnlineWordClasses(0.001)
    for k, s in enumerate(sentences, 1):
        wc.add(s)
        if k % 15 == 0:
            assert wc.path() == word_classes(sentences[:k], 0.001)


def _state(L):
    return (L.tables.bits(), list(L.sym), list(L.alive), dict(L.class_of), dict(L.rec),
            {k: set(v) for k, v in L.members.items()}, {k: set(v) for k, v in L.units.items()},
            {k: set(v) for k, v in L.pair_exps.items() if v}, {k: set(v) for k, v in L.top_exps.items() if v},
            [list(x) for x in L.elems], [list(x) for x in L.roots], len(L.memory), L.next_sym, L.next_cls,
            {k: v for k, v in L.tables.ref["sym"].items()})


def test_every_kind_of_trial_is_rolled_back_exactly(small_learner, med_forests):
    import copy
    kinds = set()
    for base in (small_learner, med_forests):
        L = copy.deepcopy(base)
        before = _state(L)
        for i in range(len(L.experiences) - 1, max(-1, len(L.experiences) - 6), -1):
            for d, mv in list(L._candidates(i))[::3]:
                tok = L._trial(mv)
                assert L.tables.bits() - before[0] == pytest.approx(d, abs=1e-6)
                L._undo(tok)
                assert _state(L) == before
                kinds.add(mv[0])
        L.verify()
    assert {"merge", "relabel", "classes", "chunk"} <= kinds


def test_hedged_learner_keeps_exact_tables_over_weighted_samples():
    """Each experience stored as k posterior samples of weight 1/k: the count
    tables equal the samples' recounted counts after every commit."""
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    import os
    path = os.path.join(default_data_root(), CONDITIONS["small"])
    if not os.path.exists(path):
        pytest.skip("synthetic corpora not available")
    train, _ = v1_split(load_corpus(path), 13)
    L = IncrementalLearner(seed=13, check=True, config={"samples": 4, "lookahead": 2})
    for e in train[:20]:
        L.observe(e.tokens)
    assert len(L.slots) == 20 and len(L.elems) == 80
    assert len(L.stored_trees()) == 20 and len(L.stored_trees(every=True)) == 80
    assert all(abs(w - 0.25) < 1e-12 for w in L.weight)
    assert L.hard_code() > 0


def test_the_contextual_full_code_is_the_nights(small_learner):
    """``full_code`` equals ``grammar._factored_code`` with the best of the
    night's context options, on the stored analyses with the learner's
    symbols and classes (learning order = experience order)."""
    from trellis2.grammar import CONTEXT_WEIGHTS, UNK, _compact, _elements, _factored_code
    from trellis2.mdl import elias_delta_bits
    from trellis2.memory import Memory
    L = small_learner
    mem = Memory(granularities=1)
    syms = []
    for tokens, tree in zip([L.experiences[x[0]] for x in L.slots], L.stored_trees()):
        start = len(mem)
        mem.add(tokens, tree)
        syms += [tree.label[mem.span[e]] for e in range(start, len(mem))]
    keys = [("P", mem.token[e]) if mem.kind[e] == Memory.PRIMITIVE else ("C", syms[mem.left[e]], syms[mem.right[e]])
            for e in range(len(mem))]
    vocab = mem.vocabulary() + [UNK]
    el = _elements(mem, None, None, {t: i for i, t in enumerate(vocab)})
    _, s = _compact(np.array(syms))
    _, c = _compact(np.array([L.class_of[k] for k in keys]))
    K, M, alpha = int(s.max()) + 1, int(c.max()) + 1, L.alpha
    options = [None] + list(CONTEXT_WEIGHTS) + [(a, b) for a in CONTEXT_WEIGHTS for b in CONTEXT_WEIGHTS]
    rest = mem.top_level_nats(s, K, alpha) + mem.layout_nats(alpha)
    best = min(_factored_code(el, s, K, len(vocab), alpha, c, w)(np.arange(M)) for w in options)
    bits = (best + rest) / LN2 + elias_delta_bits(K) + elias_delta_bits(M) + CONTEXT_OPTION_BITS
    assert L.full_code()[0] == pytest.approx(bits, abs=1e-6)


def test_joining_forests_is_scored_exactly(med_forests):
    """The join move (every forest's pieces joined right-branching under one
    new symbol) changes the code by exactly its score, and leaves no forest."""
    import copy
    L = copy.deepcopy(med_forests)
    if not any(len(r) > 1 for r in L.roots):
        pytest.skip("no forests")
    moves = list(L._join_moves())
    assert len(moves) == 1
    d, move = moves[0]
    before = L.tables.bits()
    L._apply(move)                      # check=True: verified against a recount
    assert L.tables.bits() - before == pytest.approx(d, abs=1e-6)
    assert all(len(L.roots[i]) == 1 for i in range(len(L.experiences)) if L.weight[i] > 1e-9)


def test_word_class_categories_keep_tables_exact_and_classes_apart():
    """With word-class categories every refresh is an exact relabelling, and no
    symbol holds the primitives of two word classes after a refresh."""
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    from trellis2.memory import Memory
    import os
    path = os.path.join(default_data_root(), CONDITIONS["med"])
    if not os.path.exists(path):
        pytest.skip("synthetic corpora not available")
    train, _ = v1_split(load_corpus(path), 13)
    L = IncrementalLearner(seed=13, check=True, config={"wc_categories": True, "wc_every": 5,
                                                        "lookahead": 0, "compound": 0})
    for k, e in enumerate(train[:30], 1):
        L.observe(e.tokens)
        if k % 5 == 0:
            L.verify()
            L._wc_at = -10 ** 9
            L._wc_refresh()
            L.verify()
            for A, els in L.members.items():
                classes = {L._word_class(L.memory.token[x]) for x in els if L.memory.kind[x] == Memory.PRIMITIVE}
                assert len(classes) <= 1
