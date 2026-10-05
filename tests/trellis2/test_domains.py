from trellis2.characters import gold_tree, parse_prefix, placements
from trellis2.treebank import _base_phrases, _brackets, _clean, _parse_sexpr, _right_binarized, evaluable


def test_character_structure_round_trips():
    # 湖 = ⿰ 氵 ⿰ 古 月: the operator groups with its first part.
    tokens = ["⿰", "氵", "⿰", "古", "月"]
    assert parse_prefix(tokens) == ("⿰", "氵", ("⿰", "古", "月"))
    tree = gold_tree(tokens)
    assert tree.is_valid()
    assert tree.to_string(tokens) == "[[⿰ 氵] [[⿰ 古] 月]]"
    assert placements(parse_prefix(tokens)) == {("⿰", 0, "氵"), ("⿰", 0, "古"), ("⿰", 1, "月")}


def test_ternary_operators_and_ill_formed_sequences():
    tokens = ["⿳", "亠", "口", "小"]
    assert gold_tree(tokens).to_string(tokens) == "[[[⿳ 亠] 口] 小]"
    assert parse_prefix(["⿰", "氵"]) is None              # missing a part
    assert parse_prefix(["⿰", "氵", "可", "口"]) is None   # a part too many


def test_treebank_cleaning_drops_punctuation_and_collapses_unaries():
    text = "( (S (NP-SBJ (DT The) (NN dog) ) (VP (VBD barked) ) (. .) ))"
    block = _parse_sexpr(text)[0]
    tags, words = [], []
    node = _clean(block[1], tags, words)
    assert tags == ["DT", "NN", "VBD"] and words == ["The", "dog", "barked"]
    assert node == [[0, 1], 2]                      # VP -> VBD collapses to the tag
    assert _brackets(node, set()) == {(0, 3), (0, 2)}
    assert _base_phrases(node, set()) == {(0, 2)}
    assert _right_binarized(node, {}) == {(0, 3): 2, (0, 2): 1}
    assert evaluable({(0, 3), (0, 2)}, 3) == {(0, 2)}


def test_chess_star_and_forward_relations():
    from trellis2.chess import EDGE, EMPTY, forward_neighbours, parse_fen, star
    # White: Kg1, Rf1, pawns f2 g2 h2; black: Kg8, Nf6.
    pos = parse_fen("6k1/8/5n2/8/8/8/5PPP/5RK1")
    g1 = (6, 0)
    x = star(pos, g1)
    assert x["N"] == "wP" and x["NW"] == "wP" and x["NE"] == "wP"   # the pawn shield
    assert x["W"] == "wR" and x["E"] == EDGE and x["S"] == EDGE
    assert x["NNW"] == EMPTY and x["WNW"] == EMPTY and x["ESE"] == EDGE
    # The king's own members are invisible to its context.
    assert star(pos, g1, own=frozenset({(6, 0), (6, 1)}))["N"] == "bK"
    # Forward relations from f1: the pawns in front, the king to the east, a
    # knight's jump to h2; nothing behind f1 in the scan.
    rel = {r: t for r, t in forward_neighbours(pos, (5, 0))}
    assert rel == {"N1": (5, 1), "NE1": (6, 1), "E1": (6, 0), "ENE": (7, 1)}


def test_chess_scan_code_covers_every_piece_once():
    import numpy as np
    from trellis2.chess import BoardMemory, parse_fen
    pos = parse_fen("6k1/8/5n2/8/8/8/5PPP/5RK1")
    # f1 rook with the g1 king to its east as one chunk; everything else alone.
    tops = [("c", (("wR", (5, 0)), "E1", ("wK", (6, 0))))]
    tops += [(pos[sq], sq) for sq in pos if sq not in ((5, 0), (6, 0))]
    mem = BoardMemory()
    mem.add(pos, tops)
    q, el, w = mem._scan_arrays()
    # 64 squares, minus the king's square, which the chunk anchored at f1 covers.
    assert len(q) == 63 and (q == 6).sum() == 0
    assert sorted(mem.describe(e) for e in range(len(mem.kind)) if mem.is_root[e])[0] == "[wR E1 wK]"
    s = np.zeros(len(mem.kind), dtype=np.int64)
    counts = mem.scan_counts(s, 1)
    assert counts.sum() == 63 and counts[:, 0].sum() == len(tops)


def test_characters_as_relational_trees():
    from trellis2.characters import CharacterMemory, canonical, from_relational, to_relational
    # 湖 = [氵 ⿰ [古 ⿰ 月]]; a three-part operator becomes two joins.
    structure = parse_prefix(["⿰", "氵", "⿰", "古", "月"])
    assert to_relational(structure) == ("⿰", "氵", ("⿰", "古", "月"))
    assert from_relational(to_relational(structure)) == structure
    assert canonical(("⿲", "彳", "山", "攵")) == ("⿰", "彳", ("⿰", "山", "攵"))
    mem = CharacterMemory()
    mem.add(to_relational(structure))
    slots = {mem.describe(e): mem.surface(e)["slot"] for e in range(len(mem.kind))}
    assert slots == {"氵": "⿰:0", "古": "⿰:0", "月": "⿰:1", "⿰古月": "⿰:1", "⿰氵⿰古月": "<root>"}


def test_relational_tree_probability_matches_enumeration():
    import itertools
    import math
    import numpy as np
    from trellis2.characters import CharacterMemory
    from trellis2.grammar import UNK, Grammar
    rng = np.random.default_rng(0)

    def dist(*shape):
        x = rng.random(shape) + 0.05
        return x / x.sum(axis=-1, keepdims=True)
    K, M, vocab, rels = 2, 3, ["a", "b", UNK], ["⿰", "⿱"]
    g = Grammar(vocab=vocab, S=dist(K), U=dist(K, M), pk=rng.uniform(0.2, 0.8, M), Lt=dist(M, K),
                Rt=dist(M, K), E=dist(M, 3), alpha=0.0, relations=rels, Rel=dist(M, 2))
    tree = ("⿱", "a", ("⿰", "b", "a"))
    nodes = [tree, "a", ("⿰", "b", "a"), "b", "a"]          # pre-order
    children = {0: (1, 2), 2: (3, 4)}
    total = 0.0
    for syms in itertools.product(range(K), repeat=5):
        for rules in itertools.product(range(M), repeat=5):
            p = g.S[syms[0]]
            for i, node in enumerate(nodes):
                c = rules[i]
                p *= g.U[syms[i], c]
                if isinstance(node, str):
                    p *= g.pk[c] * g.E[c, vocab.index(node)]
                else:
                    l, r = children[i]
                    p *= (1 - g.pk[c]) * g.Rel[c, rels.index(node[0])] * g.Lt[c, syms[l]] * g.Rt[c, syms[r]]
            total += p
    assert math.isclose(CharacterMemory().log_prob(g, tree), math.log(total), rel_tol=1e-9)


def test_chess_read_context_counts_earlier_pieces():
    from trellis2.chess import parse_fen, scan_rows
    pos = parse_fen("6k1/8/5n2/8/8/8/5PPP/5RK1")
    rows = scan_rows(pos, [("wK", 1), ("wP", 2)])
    # a1..g1 have no white king before them; from h1 on, one king (bit 2).
    assert rows[6] == 6 * 4 and rows[7] == 7 * 4 + 2
    # Two white pawns stand before h2 (f2, g2): both bits set there.
    assert rows[15] == 15 * 4 + 3 and rows[13] == 13 * 4 + 2


def test_chess_chunk_sees_past_its_own_pieces():
    from trellis2.chess import BoardSearch, forward_neighbours, parse_fen
    # A pawn chain b2-c3-d4: from b2 the first piece to the north-east is c3;
    # once c3 belongs to the chunk anchored at b2, the chunk sees d4 behind it.
    pos = parse_fen("8/8/8/8/3P4/2P5/1P6/8")
    b2, c3, d4 = (1, 1), (2, 2), (3, 3)
    assert dict(forward_neighbours(pos, b2))["NE1"] == c3
    assert dict(forward_neighbours(pos, b2, own=frozenset({b2, c3})))["NE2"] == d4
    search = BoardSearch([pos])
    search.apply(("wP", "NE1", "wP"), [(0, b2, c3)])
    assert (("chunk", 0), "NE2", "wP") in search.candidates()
