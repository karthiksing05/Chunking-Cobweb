"""Round 214: Trellis3R214 (read_bits, read_reach, parse_bits) is Trellis3R208 unless a flag is set."""
import math

import pytest

from trellis2.trellis3 import BOS, EOS, FILL, Trellis3
from trellis2.trellis3_r208 import Trellis3R208
from trellis2.trellis3_r214 import Trellis3R214

CFG = dict(children=False, spine=False, heights=False, naming="bands", tree="cu", refresh="verify", refresh_budget=0,
           store="posterior", restructure="sample", type_names=False, word_cc=False, repair_path=True, repair_use=700,
           exact_alone="content", story_facet=True, consist="parts", window=3)
SENTS = ["the dog ran to the park", "the cat ran to the house", "they were happy", "he was happy", "she was sad",
         "they were sad", "the dog was happy", "he wanted to play", "she wanted to go home", "they went to the park",
         "the cat wanted to play", "he went to the house", "she was happy to play", "they were happy to go home",
         "the dog saw the cat", "the cat saw the dog", "she saw the park", "he ate the apple", "the girl ate the red apple"]
W = 3


def build(cls):
    L = cls(seed=3, **CFG)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    return L


QS = [{"l2": BOS, "l1": BOS, "l3": BOS},
      {"l2": "the", "l1": "dog", "l3": BOS},
      {"l2": "to", "l1": "the", "l3": "ran"},
      {"l2": "ate", "l1": "the", "l3": "he"}]
VARIANTS = [lambda q: q, lambda q: {**q, "r1": EOS}, lambda q: {**q, "r1": "to"}]


@pytest.fixture(scope="module")
def base():
    return build(Trellis3)


@pytest.fixture(scope="module")
def r208():
    return build(Trellis3R208)


@pytest.fixture(scope="module")
def sub():
    L = build(Trellis3R214)
    assert (L.read_bits, L.read_reach, L.parse_bits) == (None, None, None)
    return L


def reset(L):
    L.read_bits = L.read_reach = L.parse_bits = None
    L.edge_exact = False


def words(m):
    return {v[1]: x for v, x in m.items() if v[0] == "w"}


def all_reads(L):
    return [L._gap_fills(f(q)) for q in QS for f in VARIANTS]


def occurrences(L):
    return [e for e in L.records if e.leaf is not None]


def test_1_default_is_byte_identical(base, r208, sub):
    reset(sub)
    assert all_reads(base) == all_reads(r208) == all_reads(sub)
    for L in (r208, sub):
        L.edge_exact = True
    try:
        assert all_reads(r208) == all_reads(sub)
    finally:
        for L in (r208, sub):
            L.edge_exact = False
    # the parse side: _gain, _code and the parse of a sentence
    for e in occurrences(sub)[:60]:
        assert sub._gain(sub.ctx, e.ctx) == r208._gain(r208.ctx, e.ctx) == base._gain(base.ctx, e.ctx)
        assert sub._code(sub.ctx, e.ctx) == r208._code(r208.ctx, e.ctx)
    for s in ("the dog ran to the house", "she was happy to go home"):
        ps, pr = sub.parse(s.split()), r208.parse(s.split())
        assert ps["logZ"] == pr["logZ"] and ps["chosen"] == pr["chosen"]
    # hier_fills (the n-gram / hierarchical reads) are unchanged
    for how in ("kn", "ngram"):
        q = QS[1]
        assert sub.hier_fills(q, how=how) == r208.hier_fills(q, how=how)


def test_2_infinite_bits_and_full_reach_equal_default(sub):
    reset(sub)
    for edge in (False, True):
        sub.edge_exact = edge
        ref = all_reads(sub)
        sub.read_bits = math.inf
        assert all_reads(sub) == ref
        sub.read_bits = None
        sub.read_reach = W
        assert all_reads(sub) == ref
        sub.read_reach = 99
        assert all_reads(sub) == ref
        reset(sub)


def test_3_zero_bits_is_the_unigram(sub):
    reset(sub)
    tot = {}
    for e in occurrences(sub):
        for which, attr, value, w in e.pas:
            if which == "ctx" and attr == FILL and value[0] == "w":
                tot[value[1]] = tot.get(value[1], 0.0) + w
    sub.read_bits = 0
    for q in QS[1:]:
        m = words(sub._gap_fills(q))
        assert set(m) == set(tot)
        r = {w: m[w] / tot[w] for w in m}
        assert max(r.values()) == pytest.approx(min(r.values()), rel=1e-9)
    # the end read keeps the same unigram shape (only the end factor, equal for every occurrence at the root)
    me = words(sub._gap_fills({**QS[1], "r1": EOS}))
    r = {w: me[w] / tot[w] for w in me}
    assert max(r.values()) == pytest.approx(min(r.values()), rel=1e-9)
    # edge_exact: the start step reads only occurrences that stood at the start; the end read only real ends
    sub.edge_exact = True
    first = {s.split()[0] for s in SENTS}
    last = {s.split()[-1] for s in SENTS}
    assert set(words(sub._gap_fills(QS[0]))) == first
    assert set(words(sub._gap_fills({**QS[1], "r1": EOS}))) <= last
    reset(sub)


def test_4_lift_is_monotone_and_bounded(sub):
    reset(sub)
    root = sub.ctx.root
    grid = [0, 1, 2, 4, 6, 8, 12]

    def anc(n, a):
        while n is not None:
            if n is a:
                return True
            n = n.parent
        return False
    seen = set()
    for e in occurrences(sub):
        if e.leaf.id in seen:
            continue
        seen.add(e.leaf.id)
        prev = None
        for b in grid:
            c = sub.lift(e.leaf, b)
            assert anc(e.leaf, c)
            assert sub.level_bits(c) <= b + 1e-9
            if prev is not None:
                assert anc(c, prev)                    # a coarser read is an ancestor of the finer one
            prev = c
        assert sub.lift(e.leaf, 0) is root or sub.level_bits(sub.lift(e.leaf, 0)) <= 1e-9
        assert sub.lift(e.leaf, math.inf) is e.leaf
        assert anc(e.leaf, sub.lift(e.leaf, "basic"))
    # a deeper node is never read above a shallower allowance: the lifted level is the deepest allowed
    for e in occurrences(sub)[:50]:
        c = sub.lift(e.leaf, 4)
        for ch in (e.leaf,):
            n = ch
            while n is not None and n is not c:
                assert sub.level_bits(n) > 4 - 1e-9
                n = n.parent
    cls = sub.word_classes(4)
    assert cls and all(isinstance(v, int) for v in cls.values())


def test_5_reach_one_equals_a_one_slot_query(sub):
    reset(sub)
    ref = {}
    for q in QS[1:]:
        for f in VARIANTS:
            ref[(tuple(sorted(f(q).items())))] = sub._gap_fills({"l1": q["l1"], **{k: v for k, v in f(q).items() if k[0] == "r"}})
    sub.read_reach = 1
    for q in QS[1:]:
        for f in VARIANTS:
            assert sub._gap_fills(f(q)) == ref[tuple(sorted(f(q).items()))]
    # a query that holds only l1 is unchanged
    assert sub._gap_fills({"l1": "the"}) == build_ref(sub, {"l1": "the"})
    reset(sub)


def build_ref(L, q):
    saved = L.read_reach
    L.read_reach = None
    try:
        return L._gap_fills(q)
    finally:
        L.read_reach = saved


def test_6_parse_bits_zero_and_restoration(sub, r208):
    reset(sub)
    xs = [e.ctx for e in occurrences(sub)[:80]]
    comps = [e.ctx for e in occurrences(sub) if e.span[1] - e.span[0] > 1][:20]
    assert any(sub._gain(sub.ctx, x) > 0 for x in xs)
    sub.parse_bits = 0
    assert all(sub._gain(sub.ctx, x) == 0.0 for x in xs + comps)
    sub.parse_bits = 6
    for x in xs:
        assert sub._gain(sub.ctx, x) <= r208._gain(r208.ctx, x) + 1e-12
    reset(sub)
    # records and caches are restored after a lifted read; the default read is not poisoned by it
    recs, n = sub.records, len(sub.records)
    sub._gap_fills(QS[1])
    gc, ec = sub.__dict__.get("_gap_cache"), sub.__dict__.get("_edge_cache")
    ref = all_reads(sub)
    for edge in (False, True):
        sub.edge_exact = edge
        for b in (0, 2, 6, "basic"):
            sub.read_bits = b
            sub._gap_fills(QS[1])
            sub._gap_fills({**QS[1], "r1": EOS})
            sub._gap_fills(QS[0])
            assert sub.records is recs and len(sub.records) == n
            assert sub.__dict__.get("_gap_cache") is gc
            assert sub.__dict__.get("_edge_cache") is ec
        sub.read_bits = None
    reset(sub)
    assert all_reads(sub) == ref
    assert all_reads(sub) == all_reads(r208)


def test_6b_coarse_reads_change_the_mass(sub):
    """The dial does something: a coarse read differs from the leaf read, and fillers stay the committed words."""
    reset(sub)
    leaf = words(sub._gap_fills(QS[3]))
    sub.read_bits = 2
    coarse = words(sub._gap_fills(QS[3]))
    reset(sub)
    assert set(coarse) == set(leaf)
    assert any(abs(coarse[w] / sum(coarse.values()) - leaf[w] / sum(leaf.values())) > 1e-6 for w in leaf)


def test_7_window_names_query_is_named():
    """With window_names the read names the query's words as the stored contexts are named; without it nothing changes."""
    kw = dict(CFG, type_names="sample", window_names="all", nb_bits=4)
    L = Trellis3R214(seed=3, **kw)
    R = Trellis3R208(seed=3, **kw)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
        R.learn(s.split(), follows=k % 3 != 0)
    L._quiet = R._quiet = True
    for q in QS[1:]:
        named = {**q, **R._named_window({a: v for a, v in q.items() if a[0] == "l"})}
        assert L._gap_fills(q) == R._gap_fills(named)
    L.read_reach = 1
    q = QS[2]
    one = {"l1": q["l1"]}
    assert L._gap_fills(q) == R._gap_fills({**one, **R._named_window(one)})


def test_8_word_tree_names_are_shared_by_storage_and_query():
    from trellis2.trellis3_r214 import WordTree
    T = WordTree([s.split() for s in SENTS], seed=3)
    assert set(T.words) == {w for s in SENTS for w in s.split()}
    root = T.names(b=0)
    assert len(set(root.values())) == 1                       # the root holds every type
    leaf = T.names(b=math.inf)
    assert len(set(leaf.values())) <= len(T.words)
    coarse, fine = T.names(b=2), T.names(b=6)
    assert len(set(coarse.values())) <= len(set(fine.values())) <= len(set(leaf.values()))
    kw = dict(CFG, window_names="all")
    L = Trellis3R214(seed=3, **kw)
    L.word_names = T.names(b=2)
    for k, s in enumerate(SENTS):
        L.learn(s.split(), follows=k % 3 != 0)
    L._quiet = True
    stored = {v for e in L.records if e.ctx for a, v in e.ctx.items() if a[0] == "l" and a[1:].isdigit()}
    assert stored <= set(L.word_names.values()) | {BOS, EOS}   # every stored window slot is a class name
    q = {"l2": "the", "l1": "dog", "l3": BOS}
    named = {"l2": L.word_names["the"], "l1": L.word_names["dog"], "l3": BOS}
    assert L._gap_fills(q) == Trellis3R208._gap_fills(L, named)


def test_9_class_emission_identities():
    from collections import Counter
    from trellis2.trellis3_r214 import WordTree
    sents = [s.split() for s in SENTS]
    T = WordTree(sents, seed=3)
    cnt = dict(Counter(w for s in sents for w in s))
    L = build(Trellis3R214)
    ref = all_reads(L)
    L.word_tree, L.emit_counts = T, cnt
    # off / leaf: the read exactly
    L.emit_bits = None
    assert all_reads(L) == ref
    L.emit_bits = math.inf
    assert all_reads(L) == ref
    L.emit_names = {w: w for w in cnt}                        # one class per word: the read exactly
    assert all_reads(L) == ref
    # root: one class -> P(C | ctx) = 1 and the mass of each word is the unigram share of the read's total
    L.emit_bits = 0
    assert len(set(L.emit_names.values())) == 1
    for q in QS[1:]:
        out = L._gap_fills(q)
        ws = {v[1]: m for v, m in out.items() if v[0] == "w"}
        r = [ws[w] / cnt[w] for w in ws]
        assert max(r) == pytest.approx(min(r), rel=1e-9)
        tot = sum(m for v, m in L._gap_fills_read(q).items() if v[0] == "w")
        assert sum(ws.values()) == pytest.approx(tot, rel=1e-9)
    # a middle level keeps each class's mass and splits it by the training counts
    L.emit_bits = 2
    names = L.emit_names
    q = QS[2]
    base = {v[1]: m for v, m in L._gap_fills_read(q).items() if v[0] == "w"}
    out = {v[1]: m for v, m in L._gap_fills(q).items() if v[0] == "w"}
    for c in set(names.values()):
        mem = [w for w in out if names[w] == c]
        assert sum(out[w] for w in mem) == pytest.approx(sum(base[w] for w in mem), rel=1e-9)
        assert all(out[w] / cnt[w] == pytest.approx(out[mem[0]] / cnt[mem[0]], rel=1e-9) for w in mem)
    L.emit_bits = None
    assert all_reads(L) == ref


def test_10_frequency_cut_emission():
    from collections import Counter
    from trellis2.trellis3_r214 import WordTree
    sents = [s.split() for s in SENTS]
    T = WordTree(sents, seed=3)
    cnt = dict(Counter(w for s in sents for w in s))
    L = build(Trellis3R214)
    ref = all_reads(L)
    L.word_tree, L.emit_counts = T, cnt
    L.emit_bits = 2
    plain = all_reads(L)
    L.emit_freq_cut = 0
    assert all_reads(L) == plain                              # K = 0 is the current emit_bits, byte for byte
    L.emit_freq_cut = len(cnt)
    assert all_reads(L) == ref                                # K >= vocabulary: the leaf token model
    L.emit_freq_cut = 10 ** 6
    assert all_reads(L) == ref
    K = 5
    L.emit_freq_cut = K
    top = L._top_words()
    assert len(top) == K and all(cnt[w] >= max(cnt[u] for u in cnt if u not in top) for w in top)
    for q in QS[1:]:
        base = {v[1]: m for v, m in L._gap_fills_read(q).items() if v[0] == "w"}
        out = {v[1]: m for v, m in L._gap_fills(q).items() if v[0] == "w"}
        assert sum(out.values()) == pytest.approx(sum(base.values()), rel=1e-9)     # mass conserved
        for w in top:
            assert out[w] == base[w]                                                 # the K most frequent keep their own mass
        rest = {w for w in out if w not in top}
        names = L.emit_names
        for c in {names[w] for w in rest}:
            mem = [w for w in rest if names[w] == c]
            assert sum(out[w] for w in mem) == pytest.approx(sum(base[w] for w in mem), rel=1e-9)   # class mass excludes the top-K words
    L.emit_freq_cut = 0
    L.emit_bits = None
    assert all_reads(L) == ref


def test_11_function_word_feature_tree():
    from collections import Counter
    from trellis2.trellis3_r214 import WordTree
    sents = [s.split() for s in SENTS]
    A = WordTree(sents, seed=3)
    B = WordTree(sents, seed=3, feats="all")
    assert [A.class_of(w, b=2) for w in A.words] == [B.class_of(w, b=2) for w in B.words]       # 'all' is the current tree
    assert A.names(b=4) == B.names(b=4) and A.names(depth=3) == B.names(depth=3)
    F = WordTree(sents, seed=3, feats="func:6")
    top = {w for w, _ in sorted(Counter(w for s in sents for w in s).items(), key=lambda kv: (-kv[1], kv[0]))[:6]} | {BOS, EOS}
    for w in F.words:
        for att in ("L", "R"):
            feats = set(F.leaf[w].av.get(att, {}))
            assert feats <= top | {"-"}, (w, att, feats)               # only top-K neighbours (and markers) are features
    assert set(F.words) == set(A.words)
    # P(w | C) uniform: every member of a class gets the same share
    L = build(Trellis3R214)
    L.word_tree, L.emit_counts = F, dict(Counter(w for s in sents for w in s))
    L.emit_uniform = True
    L.emit_bits = 0
    out = {v[1]: m for v, m in L._gap_fills(QS[2]).items() if v[0] == "w"}
    assert max(out.values()) == pytest.approx(min(out.values()), rel=1e-9)
