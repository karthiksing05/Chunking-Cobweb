"""
Test: Ref-tree bigram content hierarchy (Methodology 4.0).

Builds a POS reference hierarchy from single-word instances so that each
token (the, cat, runs …) lands at its own leaf inside the appropriate POS
cluster.  Then builds a content hierarchy with ref_tree=pos_tree using the
Methodology 4.0 single-leaf-pointer encoding:

    content instance = {0: {left_leaf_val: 1}, 1: {right_leaf_val: 1},
                        -1: {pair_str_id: 1}}   ← word-pair label (hidden)

Each POS instance carries a unique noise attribute (attr 4) so that every
occurrence of the same word is slightly different and lands at its own
POS-tree leaf.  The content instances reference the per-occurrence leaf vals
(not the raw word token IDs).  register_ref_val(leaf_val, leaf_node) wires
each leaf val directly to its node so the C++ LCA machinery can find the
node without an extra token-ID indirection.

The hidden attribute (-1) is excluded from Cobweb entropy / PU but is still
stored in av_count, so it shows up in the visualiser.

Sentence exercised:  "the cat runs the dog"  (Det N V Det N)
Consecutive bigrams:
  (the, cat)  — NP-internal  Det+N
  (cat, runs) — boundary     N+V
  (runs, the) — boundary     V+Det
  (the, dog)  — NP-internal  Det+N

Run directly:
    python tests/cobweb/test_ref_hierarchy_pairs.py
Run via pytest:
    pytest tests/cobweb/test_ref_hierarchy_pairs.py -s
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree
from viz import HTMLCobwebDrawer


# ── POS hierarchy concept IDs ─────────────────────────────────────────────────
# depth 0
FUNC_WORD    = 100
CONTENT_WORD = 200
# depth 1
ARTICLE      = 110
NOUN         = 210
VERB         = 220
# depth 2
DEF_ART      = 111
INDEF_ART    = 112
ANIM_NOUN    = 211
INANIM_NOUN  = 212
# (re-use VERB=220 as depth-2 for verbs — two verbs share the class)

# leaf token IDs
THE   = 1011
A     = 1012
CAT   = 2011
DOG   = 2012
BIRD  = 2013
FISH  = 2021
MOUSE = 2022
RUNS  = 2211
SEES  = 2212

# ── human-readable names ──────────────────────────────────────────────────────
VALUE_NAMES = {
    FUNC_WORD:    "FUNC_WORD",
    CONTENT_WORD: "CONTENT_WORD",
    ARTICLE:      "ARTICLE",
    NOUN:         "NOUN",
    VERB:         "VERB",
    DEF_ART:      "DEF_ART",
    INDEF_ART:    "INDEF_ART",
    ANIM_NOUN:    "ANIM_NOUN",
    INANIM_NOUN:  "INANIM_NOUN",
    THE:          "the",
    A:            "a",
    CAT:          "cat",
    DOG:          "dog",
    BIRD:         "bird",
    FISH:         "fish",
    MOUSE:        "mouse",
    RUNS:         "runs",
    SEES:         "sees",
}

WORD_NAMES = {
    THE: "the", A: "a", CAT: "cat", DOG: "dog",
    BIRD: "bird", FISH: "fish", MOUSE: "mouse",
    RUNS: "runs", SEES: "sees",
}


# ── helpers ───────────────────────────────────────────────────────────────────

_next_noise_id = [9000]

def pos_inst(d0, d1, d2, leaf_id):
    """Single-word POS instance: 5 attributes (depth-0..2 + leaf token + noise).

    Attribute 4 is a unique noise value so that every occurrence of the same
    word is slightly different and gets its own POS-tree leaf."""
    nid = _next_noise_id[0]
    _next_noise_id[0] += 1
    return {
        0: {d0: 1.0},
        1: {d1: 1.0},
        2: {d2: 1.0},
        3: {leaf_id: 1.0},
        4: {nid: 1.0},
    }


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)


# ── POS training data ─────────────────────────────────────────────────────────
# Each tuple: (leaf_token_id, d0, d1, d2) — instances are built on the fly so
# that every occurrence gets a fresh noise attribute.
POS_WORDS = [
    (THE,   FUNC_WORD,    ARTICLE,     DEF_ART),
    (A,     FUNC_WORD,    ARTICLE,     INDEF_ART),
    (CAT,   CONTENT_WORD, NOUN,        ANIM_NOUN),
    (DOG,   CONTENT_WORD, NOUN,        ANIM_NOUN),
    (BIRD,  CONTENT_WORD, NOUN,        ANIM_NOUN),
    (FISH,  CONTENT_WORD, NOUN,        INANIM_NOUN),
    (MOUSE, CONTENT_WORD, NOUN,        INANIM_NOUN),
    (RUNS,  CONTENT_WORD, VERB,        VERB),
    (SEES,  CONTENT_WORD, VERB,        VERB),
]

# Quick lookup: wid → (d0, d1, d2)
POS_PARAMS = {wid: (d0, d1, d2) for wid, d0, d1, d2 in POS_WORDS}


# ── word-pair string IDs for the hidden attribute ─────────────────────────────
_PAIR_IDS: dict  = {}
_PAIR_NAMES: dict = {}
_next_pair_id = [5000]


def get_pair_id(left_wid: int, right_wid: int) -> int:
    key = (left_wid, right_wid)
    if key not in _PAIR_IDS:
        pid = _next_pair_id[0]
        _next_pair_id[0] += 1
        _PAIR_IDS[key] = pid
        _PAIR_NAMES[pid] = f"{WORD_NAMES[left_wid]}+{WORD_NAMES[right_wid]}"
    return _PAIR_IDS[key]


# ── main test function ────────────────────────────────────────────────────────

def test_ref_hierarchy_pairs():

    # ── Step 1: Build POS reference hierarchy ──────────────────────────────
    pos_tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)

    # Repeat 10× so clustering is robust (same cadence as test_logprob_paths.py)
    for _ in range(10):
        for wid, d0, d1, d2 in POS_WORDS:
            pos_tree.ifit(pos_inst(d0, d1, d2, wid))

    print(f"\nPOS tree: {count_nodes(pos_tree.root)} concepts, "
          f"root.count={pos_tree.root.count:.0f}")

    # ── Step 2: Per-occurrence leaf categorization ─────────────────────────
    # Because each POS instance carries a unique noise attribute, every
    # occurrence of the same word may land at a different POS-tree leaf.
    # We categorize each word *fresh* for every bigram occurrence.

    def categorize_fresh(wid):
        """Categorize a word with fresh noise → unique POS-tree leaf."""
        d0, d1, d2 = POS_PARAMS[wid]
        return pos_tree.categorize(pos_inst(d0, d1, d2, wid))

    # Print one sample categorization per word for reference
    print("\n── POS-tree leaf lookup (sample per word) ──")
    for wid, d0, d1, d2 in POS_WORDS:
        leaf = categorize_fresh(wid)
        print(f"  {WORD_NAMES[wid]:8s}  token={wid}  "
              f"→ POS-leaf {leaf.concept_hash()[:12]}  (count={leaf.count:.0f})")

    # ── Step 3: Create content hierarchy linked to the POS tree ────────────
    content_tree = CobwebDiscreteTree(
        alpha=1e-3, weight_attr=False, ref_tree=pos_tree
    )
    content_tree.set_ref_attr(0)   # left-word attribute uses ref-tree LCA
    content_tree.set_ref_attr(1)   # right-word attribute uses ref-tree LCA

    # Leaf-val bookkeeping — vals are assigned per unique POS-tree leaf node
    leaf_val_names: dict = {}       # stable int val → human-readable word name
    _leaf_obj_to_val: dict = {}     # python id(leaf) → stable int val
    _next_leaf_val = [10000]

    def get_leaf_val(leaf_node, wid):
        """Get (or create) a stable int val for a specific POS-tree leaf."""
        obj_key = id(leaf_node)
        if obj_key not in _leaf_obj_to_val:
            val = _next_leaf_val[0]
            _next_leaf_val[0] += 1
            _leaf_obj_to_val[obj_key] = val
            leaf_val_names[val] = WORD_NAMES[wid]
            content_tree.register_ref_val(val, leaf_node)
        return _leaf_obj_to_val[obj_key]

    def content_inst(left_wid: int, right_wid: int) -> dict:
        """Methodology 4.0 bigram instance using per-occurrence POS-leaf vals.

        Each call categorizes both words afresh (with new noise), so two
        bigrams sharing the same word can point to different POS leaves."""
        left_leaf = categorize_fresh(left_wid)
        right_leaf = categorize_fresh(right_wid)
        left_val = get_leaf_val(left_leaf, left_wid)
        right_val = get_leaf_val(right_leaf, right_wid)
        pair_id = get_pair_id(left_wid, right_wid)
        return {
            0:  {left_val: 1.0},
            1:  {right_val: 1.0},
            -1: {pair_id: 1.0},
        }

    # ── Step 4: Train on bigrams ────────────────────────────────────────────
    # Target sentence  "the cat runs the dog"
    SENTENCE = [THE, CAT, RUNS, THE, DOG]
    sentence_bigrams = list(zip(SENTENCE, SENTENCE[1:]))

    # Additional context bigrams to build richer Det+N and N+V clusters
    extra_det_noun = [
        (THE, DOG), (THE, BIRD), (THE, FISH), (THE, MOUSE),
        (A,   CAT), (A,   DOG),  (A,   BIRD), (A,   FISH), (A, MOUSE),
    ]
    extra_noun_verb = [
        (DOG, RUNS), (CAT, SEES),
    ]

    all_bigrams = sentence_bigrams + extra_det_noun + extra_noun_verb

    import random
    random.shuffle(all_bigrams)

    for pair in all_bigrams:
        print(content_inst(*pair))
        content_tree.ifit(content_inst(*pair))

    print(f"\nContent tree: {count_nodes(content_tree.root)} concepts, "
          f"root.count={content_tree.root.count:.0f}")

    # ── Step 5: Collect ALL basic-level nodes in the content tree ──────────
    print("\n── Basic-level node for every content-tree leaf ──")

    def _all_leaves(node):
        if not node.children:
            return [node]
        leaves = []
        for ch in node.children:
            leaves.extend(_all_leaves(ch))
        return leaves

    all_content_leaves = _all_leaves(content_tree.root)
    seen_basics: dict = {}

    for leaf in all_content_leaves:
        basic = leaf.get_basic(1000, 100, eval_alpha=1)
        bh = basic.concept_hash()
        if bh not in seen_basics:
            seen_basics[bh] = basic
        pair_label = _leaf_pair_label(leaf, leaf_val_names)
        print(f"  leaf({pair_label:18s}) depth={leaf.depth()}  "
              f"→ basic depth={basic.depth()}  count={basic.count:.0f}  "
              f"hash={bh[:12]}")

    print(f"\nDistinct basic-level nodes: {len(seen_basics)}")
    for bh, bn in seen_basics.items():
        print(f"  {bh[:12]}  depth={bn.depth()}  count={bn.count:.0f}")

    # ── Step 6: Visualise both hierarchies ─────────────────────────────────
    _val_fn      = lambda vid: VALUE_NAMES.get(vid, str(vid))
    _pair_fn     = lambda vid: _PAIR_NAMES.get(vid, str(vid))
    # For content attrs: vals are now leaf_val ints, resolve via leaf_val_names
    _leaf_val_fn = lambda vid: leaf_val_names.get(vid, str(vid))

    _noise_fn = lambda vid: f"n{vid}"  # noqa: E731

    pos_drawer = HTMLCobwebDrawer(
        attributes=["Depth0", "Depth1", "Depth2", "Token", "Noise"],
        id_to_value=[],
        value_to_id={},
        attr_value_fn={i: _val_fn for i in range(4)} | {4: _noise_fn},
    )

    content_drawer = HTMLCobwebDrawer(
        attributes=["Left-Word", "Right-Word"],
        id_to_value=[],
        value_to_id={},
        attr_value_fn={0: _leaf_val_fn, 1: _leaf_val_fn, -1: _pair_fn},
        attr_name_overrides={-1: "WordPair"},
    )

    out_base = os.path.join(os.path.dirname(__file__), "output")
    pairs_to_viz = [
        ("POS",     pos_drawer,     pos_tree,     os.path.join(out_base, "ref_hierarchy_pairs_pos")),
        ("content", content_drawer, content_tree, os.path.join(out_base, "ref_hierarchy_pairs_content")),
    ]

    for name, drawer, tree, out_path in pairs_to_viz:
        try:
            html_file, _ = drawer.draw_tree(tree.root, out_path)
            print(f"\n{name} tree saved to: {html_file}")
        except Exception as exc:
            html_file = out_path + ".html"
            d3_json  = json.dumps(drawer._node_to_dict(tree.root))
            html_str = drawer._build_html(d3_json)
            os.makedirs(os.path.dirname(html_file), exist_ok=True)
            with open(html_file, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            print(f"\n{name} tree (HTML only) saved to: {html_file}")
            print(f"  (PNG skipped: {exc})")


def _leaf_pair_label(leaf, leaf_val_names: dict) -> str:
    """Return a human-readable label from the av_count of a content leaf."""
    label_parts = []
    for attr_id in (0, 1):
        vc = (leaf.av_count or {}).get(attr_id, {})
        if vc:
            top_vid = max(vc, key=vc.get)
            label_parts.append(leaf_val_names.get(top_vid, str(top_vid)))
        else:
            label_parts.append("?")
    return "+".join(label_parts)


if __name__ == "__main__":
    test_ref_hierarchy_pairs()
