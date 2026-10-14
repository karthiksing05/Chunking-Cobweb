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








def test_chess_read_counts_earlier_pieces():
    from trellis2.chess import KIND_INDEX, counts_before, parse_fen
    pos = parse_fen("6k1/8/5n2/8/8/8/5PPP/5RK1")
    before = counts_before(pos)
    # a1..g1 have no white king before them; from h1 on, one.
    assert before[6, KIND_INDEX["wK"]] == 0 and before[7, KIND_INDEX["wK"]] == 1
    # Two white pawns stand before h2 (f2, g2), none before f2.
    assert before[15, KIND_INDEX["wP"]] == 2 and before[13, KIND_INDEX["wP"]] == 0




