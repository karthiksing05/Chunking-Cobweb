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
