"""
Test: Log-probability probe using concept-path IDs – larger bigram framing.

Instances encode *both* tokens of a bigram as a 4-level path through a richer
POS/semantic hierarchy — 4 depth levels per side (8 attributes total):

    {0: {l_d0: 1}, 1: {l_d1: 1}, 2: {l_d2: 1}, 3: {l_d3: 1},
     4: {r_d0: 1}, 5: {r_d1: 1}, 6: {r_d2: 1}, 7: {r_d3: 1}}

Simulated POS/semantic hierarchy (depth-4, root implied):

    ROOT
    ├── FUNC_WORD (100)
    │   ├── ARTICLE (110)
    │   │   ├── DEF_ART (111)
    │   │   │   └── THE (1011)
    │   │   └── INDEF_ART (112)
    │   │       ├── A (1012)
    │   │       └── AN (1013)
    │   └── PREP (120)
    │       ├── SPATIAL_PREP (121)
    │       │   ├── IN (1021)
    │       │   ├── ON (1022)
    │       │   └── UNDER (1023)
    │       └── TEMPORAL_PREP (122)
    │           ├── BEFORE (1031)
    │           └── AFTER (1032)
    └── CONTENT_WORD (200)
        ├── NOUN (210)
        │   ├── ANIM_NOUN (211)
        │   │   ├── CAT (2011)
        │   │   ├── DOG (2012)
        │   │   ├── BIRD (2013)
        │   │   └── WOLF (2014)      ← UNSEEN
        │   ├── INANIM_NOUN (212)
        │   │   ├── FISH (2021)
        │   │   ├── STONE (2022)
        │   │   ├── BOOK (2023)
        │   │   └── COIN (2024)      ← UNSEEN
        │   └── PLACE_NOUN (213)
        │       ├── PARK (2031)
        │       ├── RIVER (2032)
        │       ├── FOREST (2033)
        │       └── CAVE (2034)      ← UNSEEN
        ├── VERB (220)
        │   ├── MOTION_VERB (221)
        │   │   ├── RUNS (2211)
        │   │   ├── SWIMS (2212)
        │   │   ├── FLIES (2213)
        │   │   └── LEAPS (2214)     ← UNSEEN
        │   ├── PERCEPTION_VERB (222)
        │   │   ├── SEES (2221)
        │   │   ├── HEARS (2222)
        │   │   └── SMELLS (2223)    ← UNSEEN
        │   └── STATIVE_VERB (223)
        │       ├── LIKES (2231)
        │       ├── KNOWS (2232)
        │       └── FEARS (2233)     ← UNSEEN
        └── ADJ (230)
            ├── SIZE_ADJ (231)
            │   ├── BIG (2311)
            │   ├── SMALL (2312)
            │   └── TINY (2313)      ← UNSEEN
            └── COLOR_ADJ (232)
                ├── RED (2321)
                ├── BLUE (2322)
                ├── GREEN (2323)
                └── BLACK (2324)     ← UNSEEN

Training corpus covers:
  - NP-internal bigrams:  Det + AnimNoun  (×10, high frequency)
  - NP-internal bigrams:  Det + InanimNoun (×8)
  - NP-internal bigrams:  Det + PlaceNoun  (×6)
  - Adj+Noun bigrams:     SizeAdj/ColorAdj + Noun  (×12)
  - VP bigrams:           AnimNoun + MotionVerb     (×8)
  - VP bigrams:           AnimNoun + PerceptionVerb (×6)
  - VP bigrams:           AnimNoun + StatVerb       (×4)
  - PP-internal bigrams:  SpatialPrep + PlaceNoun   (×6)
  - PP-internal bigrams:  TemporalPrep + Verb       (×4)
  - Boundary crossings:   Noun + SpatialPrep        (×4)
  - Noisy/rare:           InanimNoun + MotionVerb   (×2)

Expected probe insights:
  • Det + unseen animate noun should outscore any boundary pattern
    because d0/d1/d2 (FUNC_WORD, ARTICLE, DEF_ART / INDEF_ART) all match.
  • SizeAdj + unseen noun should score closer to trained Adj+Noun bigrams
    than to VP bigrams.
  • An unseen MotionVerb after an AnimNoun should still score well because
    d0 (CONTENT_WORD) and d1 (VERB) and d2 (MOTION_VERB) all match.
  • Temporally odd combos (PrepTemporal + AnimNoun) should score low.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree
from viz import HTMLCobwebDrawer

import random


# ═══════════════════════════════════════════════════════════════════════════
# Concept IDs  (4-level hierarchy; root is implied)
# ═══════════════════════════════════════════════════════════════════════════

# ── depth 0 (coarsest) ───────────────────────────────────────────────────────
FUNC_WORD    = 100
CONTENT_WORD = 200

# ── depth 1 ──────────────────────────────────────────────────────────────────
ARTICLE      = 110   # subset of FUNC_WORD
PREP         = 120   # subset of FUNC_WORD

NOUN         = 210   # subset of CONTENT_WORD
VERB         = 220   # subset of CONTENT_WORD
ADJ          = 230   # subset of CONTENT_WORD

# ── depth 2 ──────────────────────────────────────────────────────────────────
DEF_ART      = 111   # definite article
INDEF_ART    = 112   # indefinite article

SPATIAL_PREP   = 121
TEMPORAL_PREP  = 122

ANIM_NOUN    = 211   # animate nouns
INANIM_NOUN  = 212   # inanimate nouns
PLACE_NOUN   = 213   # place nouns

MOTION_VERB      = 221
PERCEPTION_VERB  = 222
STATIVE_VERB     = 223

SIZE_ADJ   = 231
COLOR_ADJ  = 232

# ── depth 3 (leaf token IDs) ─────────────────────────────────────────────────
#   Articles
THE   = 1011
A     = 1012
AN    = 1013

#   Spatial prepositions
IN    = 1021
ON    = 1022
UNDER = 1023

#   Temporal prepositions
BEFORE = 1031
AFTER  = 1032

#   Animate nouns  (SEEN)
CAT   = 2011
DOG   = 2012
BIRD  = 2013
#   Animate nouns  (UNSEEN)
WOLF  = 2014

#   Inanimate nouns  (SEEN)
FISH  = 2021
STONE = 2022
BOOK  = 2023
#   Inanimate nouns  (UNSEEN)
COIN  = 2024

#   Place nouns  (SEEN)
PARK   = 2031
RIVER  = 2032
FOREST = 2033
#   Place nouns  (UNSEEN)
CAVE   = 2034

#   Motion verbs  (SEEN)
RUNS   = 2211
SWIMS  = 2212
FLIES  = 2213
#   Motion verbs  (UNSEEN)
LEAPS  = 2214

#   Perception verbs  (SEEN)
SEES   = 2221
HEARS  = 2222
#   Perception verbs  (UNSEEN)
SMELLS = 2223

#   Stative verbs  (SEEN)
LIKES  = 2231
KNOWS  = 2232
#   Stative verbs  (UNSEEN)
FEARS  = 2233

#   Size adjectives  (SEEN)
BIG   = 2311
SMALL = 2312
#   Size adjectives  (UNSEEN)
TINY  = 2313

#   Color adjectives  (SEEN)
RED   = 2321
BLUE  = 2322
GREEN = 2323
#   Color adjectives  (UNSEEN)
BLACK = 2324


# ═══════════════════════════════════════════════════════════════════════════
# Human-readable label map
# ═══════════════════════════════════════════════════════════════════════════
VALUE_NAMES = {
    FUNC_WORD:    "FUNC_WORD",
    CONTENT_WORD: "CONTENT_WORD",
    ARTICLE:      "ARTICLE",
    PREP:         "PREP",
    NOUN:         "NOUN",
    VERB:         "VERB",
    ADJ:          "ADJ",
    DEF_ART:      "DEF_ART",
    INDEF_ART:    "INDEF_ART",
    SPATIAL_PREP:   "SPATIAL_PREP",
    TEMPORAL_PREP:  "TEMPORAL_PREP",
    ANIM_NOUN:    "ANIM_NOUN",
    INANIM_NOUN:  "INANIM_NOUN",
    PLACE_NOUN:   "PLACE_NOUN",
    MOTION_VERB:     "MOTION_VERB",
    PERCEPTION_VERB: "PERCEPTION_VERB",
    STATIVE_VERB:    "STATIVE_VERB",
    SIZE_ADJ:  "SIZE_ADJ",
    COLOR_ADJ: "COLOR_ADJ",
    THE:   "the",
    A:     "a",
    AN:    "an",
    IN:    "in",
    ON:    "on",
    UNDER: "under",
    BEFORE: "before",
    AFTER:  "after",
    CAT:   "cat",
    DOG:   "dog",
    BIRD:  "bird",
    WOLF:  "wolf (unseen)",
    FISH:  "fish",
    STONE: "stone",
    BOOK:  "book",
    COIN:  "coin (unseen)",
    PARK:   "park",
    RIVER:  "river",
    FOREST: "forest",
    CAVE:   "cave (unseen)",
    RUNS:   "runs",
    SWIMS:  "swims",
    FLIES:  "flies",
    LEAPS:  "leaps (unseen)",
    SEES:   "sees",
    HEARS:  "hears",
    SMELLS: "smells (unseen)",
    LIKES:  "likes",
    KNOWS:  "knows",
    FEARS:  "fears (unseen)",
    BIG:   "big",
    SMALL: "small",
    TINY:  "tiny (unseen)",
    RED:   "red",
    BLUE:  "blue",
    GREEN: "green",
    BLACK: "black (unseen)",
}


# ═══════════════════════════════════════════════════════════════════════════
# Instance constructor  (4 depths per side = 8 attributes)
# ═══════════════════════════════════════════════════════════════════════════
def inst(l0, l1, l2, l3, r0, r1, r2, r3):
    """4-depth-per-side content instance (8 attributes total)."""
    return {
        0: {l0: 1.0},  # left  depth 0
        1: {l1: 1.0},  # left  depth 1
        2: {l2: 1.0},  # left  depth 2
        3: {l3: 1.0},  # left  depth 3  (leaf token ID)
        4: {r0: 1.0},  # right depth 0
        5: {r1: 1.0},  # right depth 1
        6: {r2: 1.0},  # right depth 2
        7: {r3: 1.0},  # right depth 3  (leaf token ID)
    }


# ── path-tuple shorthands ────────────────────────────────────────────────────
def p_the():         return (FUNC_WORD, ARTICLE, DEF_ART,     THE)
def p_a():           return (FUNC_WORD, ARTICLE, INDEF_ART,   A)
def p_an():          return (FUNC_WORD, ARTICLE, INDEF_ART,   AN)

def p_in():          return (FUNC_WORD, PREP, SPATIAL_PREP,   IN)
def p_on():          return (FUNC_WORD, PREP, SPATIAL_PREP,   ON)
def p_under():       return (FUNC_WORD, PREP, SPATIAL_PREP,   UNDER)
def p_before():      return (FUNC_WORD, PREP, TEMPORAL_PREP,  BEFORE)
def p_after():       return (FUNC_WORD, PREP, TEMPORAL_PREP,  AFTER)

def p_anim(leaf):    return (CONTENT_WORD, NOUN, ANIM_NOUN,   leaf)
def p_inanim(leaf):  return (CONTENT_WORD, NOUN, INANIM_NOUN, leaf)
def p_place(leaf):   return (CONTENT_WORD, NOUN, PLACE_NOUN,  leaf)

def p_motion(leaf):  return (CONTENT_WORD, VERB, MOTION_VERB,     leaf)
def p_percep(leaf):  return (CONTENT_WORD, VERB, PERCEPTION_VERB, leaf)
def p_stativ(leaf):  return (CONTENT_WORD, VERB, STATIVE_VERB,    leaf)

def p_size(leaf):    return (CONTENT_WORD, ADJ, SIZE_ADJ,  leaf)
def p_color(leaf):   return (CONTENT_WORD, ADJ, COLOR_ADJ, leaf)


# ═══════════════════════════════════════════════════════════════════════════
# Training corpus
# ═══════════════════════════════════════════════════════════════════════════

# ── NP-internal: Det + AnimNoun  (×10 – high frequency) ─────────────────────
DET_ANIM_BIGRAMS = [
    inst(*p_the(),  *p_anim(CAT)),
    inst(*p_the(),  *p_anim(DOG)),
    inst(*p_the(),  *p_anim(BIRD)),
    inst(*p_the(),  *p_anim(CAT)),   # repeat to boost "the cat"
    inst(*p_the(),  *p_anim(DOG)),
    inst(*p_a(),    *p_anim(CAT)),
    inst(*p_a(),    *p_anim(DOG)),
    inst(*p_a(),    *p_anim(BIRD)),
    inst(*p_a(),    *p_anim(CAT)),
    inst(*p_an(),   *p_anim(BIRD)),
]

# ── NP-internal: Det + InanimNoun  (×8) ──────────────────────────────────────
DET_INANIM_BIGRAMS = [
    inst(*p_the(),  *p_inanim(FISH)),
    inst(*p_the(),  *p_inanim(STONE)),
    inst(*p_the(),  *p_inanim(BOOK)),
    inst(*p_the(),  *p_inanim(FISH)),
    inst(*p_a(),    *p_inanim(FISH)),
    inst(*p_a(),    *p_inanim(STONE)),
    inst(*p_a(),    *p_inanim(BOOK)),
    inst(*p_an(),   *p_inanim(STONE)),
]

# ── NP-internal: Det + PlaceNoun  (×6) ───────────────────────────────────────
DET_PLACE_BIGRAMS = [
    inst(*p_the(),  *p_place(PARK)),
    inst(*p_the(),  *p_place(RIVER)),
    inst(*p_the(),  *p_place(FOREST)),
    inst(*p_a(),    *p_place(PARK)),
    inst(*p_a(),    *p_place(RIVER)),
    inst(*p_a(),    *p_place(FOREST)),
]

# ── NP-internal: Adj + Noun  (×12) ───────────────────────────────────────────
ADJ_NOUN_BIGRAMS = [
    # size + animate
    inst(*p_size(BIG),    *p_anim(CAT)),
    inst(*p_size(BIG),    *p_anim(DOG)),
    inst(*p_size(SMALL),  *p_anim(BIRD)),
    inst(*p_size(SMALL),  *p_anim(CAT)),
    # size + inanimate
    inst(*p_size(BIG),    *p_inanim(STONE)),
    inst(*p_size(SMALL),  *p_inanim(FISH)),
    # color + animate
    inst(*p_color(RED),   *p_anim(BIRD)),
    inst(*p_color(BLUE),  *p_anim(FISH)),   # fish used as animate loosely
    inst(*p_color(GREEN), *p_anim(BIRD)),
    # color + inanimate
    inst(*p_color(RED),   *p_inanim(STONE)),
    inst(*p_color(BLUE),  *p_inanim(BOOK)),
    inst(*p_color(GREEN), *p_inanim(FISH)),
]

# ── VP: AnimNoun + MotionVerb  (×8) ──────────────────────────────────────────
ANIM_MOTION_BIGRAMS = [
    inst(*p_anim(CAT),   *p_motion(RUNS)),
    inst(*p_anim(DOG),   *p_motion(RUNS)),
    inst(*p_anim(BIRD),  *p_motion(FLIES)),
    inst(*p_anim(FISH),  *p_motion(SWIMS)),
    inst(*p_anim(CAT),   *p_motion(SWIMS)),
    inst(*p_anim(DOG),   *p_motion(FLIES)),
    inst(*p_anim(CAT),   *p_motion(RUNS)),   # repeat
    inst(*p_anim(BIRD),  *p_motion(FLIES)),  # repeat
]

# ── VP: AnimNoun + PerceptionVerb  (×6) ──────────────────────────────────────
ANIM_PERCEP_BIGRAMS = [
    inst(*p_anim(CAT),   *p_percep(SEES)),
    inst(*p_anim(DOG),   *p_percep(SEES)),
    inst(*p_anim(BIRD),  *p_percep(HEARS)),
    inst(*p_anim(CAT),   *p_percep(HEARS)),
    inst(*p_anim(DOG),   *p_percep(HEARS)),
    inst(*p_anim(BIRD),  *p_percep(SEES)),
]

# ── VP: AnimNoun + StativeVerb  (×4) ─────────────────────────────────────────
ANIM_STATIV_BIGRAMS = [
    inst(*p_anim(CAT),  *p_stativ(LIKES)),
    inst(*p_anim(DOG),  *p_stativ(KNOWS)),
    inst(*p_anim(CAT),  *p_stativ(KNOWS)),
    inst(*p_anim(BIRD), *p_stativ(LIKES)),
]

# ── PP-internal: SpatialPrep + PlaceNoun  (×6) ───────────────────────────────
SPATPREP_PLACE_BIGRAMS = [
    inst(*p_in(),    *p_place(PARK)),
    inst(*p_in(),    *p_place(RIVER)),
    inst(*p_on(),    *p_place(RIVER)),
    inst(*p_on(),    *p_place(FOREST)),
    inst(*p_under(),  *p_place(FOREST)),
    inst(*p_under(),  *p_place(PARK)),
]

# ── PP-internal: TemporalPrep + Verb  (×4) ───────────────────────────────────
TEMPPREP_VERB_BIGRAMS = [
    inst(*p_before(), *p_motion(RUNS)),
    inst(*p_before(), *p_motion(FLIES)),
    inst(*p_after(),  *p_percep(SEES)),
    inst(*p_after(),  *p_stativ(LIKES)),
]

# ── Boundary crossing: AnimNoun + SpatialPrep  (×4, rare) ────────────────────
ANIM_PREP_BOUNDARY = [
    inst(*p_anim(CAT),  *p_in()),
    inst(*p_anim(DOG),  *p_on()),
    inst(*p_anim(BIRD), *p_in()),
    inst(*p_anim(CAT),  *p_under()),
]

# ── Noisy / rare: InanimNoun + MotionVerb  (×2) ──────────────────────────────
INANIM_MOTION_NOISE = [
    inst(*p_inanim(STONE), *p_motion(RUNS)),   # weird but seen once
    inst(*p_inanim(BOOK),  *p_motion(FLIES)),  # metaphorical
]

# ── Full training set ─────────────────────────────────────────────────────────
TRAINING = (
    DET_ANIM_BIGRAMS
    + DET_INANIM_BIGRAMS
    + DET_PLACE_BIGRAMS
    + ADJ_NOUN_BIGRAMS
    + ANIM_MOTION_BIGRAMS
    + ANIM_PERCEP_BIGRAMS
    + ANIM_STATIV_BIGRAMS
    + SPATPREP_PLACE_BIGRAMS
    + TEMPPREP_VERB_BIGRAMS
    + ANIM_PREP_BOUNDARY
    + INANIM_MOTION_NOISE
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def count_concepts(node):
    return 1 + sum(count_concepts(c) for c in node.children)


def path_to_leaf(tree, instance):
    leaf = tree.categorize(instance)
    path = []
    node = leaf
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def print_scores(label, tree, instance):
    path = path_to_leaf(tree, instance)

    left_instance  = {k: instance[k] for k in range(4)}
    right_instance = {k: instance[k] for k in range(4, 8)}

    tree_lp         = tree.log_prob(instance, 100, False)
    tree_class_lp   = tree.log_prob_class_given_instance(instance, 100, False)
    tree_left_lp    = tree.log_prob(left_instance, 100, False)
    tree_right_lp   = tree.log_prob(right_instance, 100, False)
    root_lp         = path[0].log_prob_instance(instance)
    leaf_lp         = path[-1].log_prob_instance(instance)

    basic_node  = path[-1].get_basic(1000, 100, True, eval_alpha=10)
    basic_lp    = basic_node.log_prob_instance(instance)
    basic_class_lp = basic_node.log_prob_class_given_instance(instance)
    basic_depth = basic_node.depth()

    best_node   = path[-1].get_best(instance)
    best_lp     = best_node.log_prob_instance(instance)
    best_class_lp  = best_node.log_prob_class_given_instance(instance)
    best_depth  = best_node.depth()

    print(f"\n{'='*66}")
    print(f"  Query: {label}")
    print(f"{'='*66}")
    print(f"         tree log-prob : {tree_lp:.6f}")
    print(f"   tree class log-prob : {tree_class_lp:.6f}")
    print(f"    tree left log-prob : {tree_left_lp:.6f}")
    print(f"   tree right log-prob : {tree_right_lp:.6f}")
    print(f"         root log-prob : {root_lp:.6f}  (count={path[0].count})")
    print(f"         leaf log-prob : {leaf_lp:.6f}  (count={path[-1].count})")
    print(f"        basic log-prob : {basic_lp:.6f}  (depth={basic_depth}, count={basic_node.count})")
    print(f"  basic-class log-prob : {basic_class_lp:.6f}")
    print(f"         best log-prob : {best_lp:.6f}  (depth={best_depth}, count={best_node.count})")
    print(f"   best-class log-prob : {best_class_lp:.6f}")
    print(f"\n  Full path ({len(path)} nodes):")
    for i, n in enumerate(path):
        lp = n.log_prob_instance(instance)
        markers = []
        if n is basic_node:
            markers.append("basic-level")
        if n is best_node:
            markers.append("best")
        marker = (" ← " + ", ".join(markers)) if markers else ""
        print(f"    [{i}] depth={i}  count={n.count:5.0f}  lp={lp:.6f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════
def test_path_logprobs_large():
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)

    random.shuffle(TRAINING)
    for item in TRAINING:
        tree.ifit(item)

    print(f"\nTree has {count_concepts(tree.root)} concepts, "
          f"root.count={tree.root.count}, training size={len(TRAINING)}")

    # ── [A] NP-internal bigrams – SEEN leaves ────────────────────────────────
    print("\n" + "─"*66)
    print("  [A] NP-internal bigrams – SEEN leaves")
    print("─"*66)

    print_scores(
        'Det(def)+AnimNoun  →  "the cat"  (FREQUENT, NP-internal)',
        tree,
        inst(*p_the(), *p_anim(CAT)),
    )
    print_scores(
        'Det(indef)+InanimNoun  →  "a stone"  (NP-internal)',
        tree,
        inst(*p_a(), *p_inanim(STONE)),
    )
    print_scores(
        'Det+PlaceNoun  →  "the park"  (NP-internal)',
        tree,
        inst(*p_the(), *p_place(PARK)),
    )

    # ── [B] Adj+Noun bigrams – SEEN leaves ───────────────────────────────────
    print("\n" + "─"*66)
    print("  [B] Adj+Noun bigrams – SEEN leaves")
    print("─"*66)

    print_scores(
        'SizeAdj+AnimNoun  →  "big cat"  (NP modifier, SEEN)',
        tree,
        inst(*p_size(BIG), *p_anim(CAT)),
    )
    print_scores(
        'ColorAdj+InanimNoun  →  "red stone"  (NP modifier, SEEN)',
        tree,
        inst(*p_color(RED), *p_inanim(STONE)),
    )

    # ── [C] VP bigrams – SEEN leaves ─────────────────────────────────────────
    print("\n" + "─"*66)
    print("  [C] VP bigrams – SEEN leaves")
    print("─"*66)

    print_scores(
        'AnimNoun+MotionVerb  →  "cat runs"  (FREQUENT, VP)',
        tree,
        inst(*p_anim(CAT), *p_motion(RUNS)),
    )
    print_scores(
        'AnimNoun+PerceptionVerb  →  "dog sees"  (VP)',
        tree,
        inst(*p_anim(DOG), *p_percep(SEES)),
    )
    print_scores(
        'AnimNoun+StatVerb  →  "cat likes"  (VP, stative)',
        tree,
        inst(*p_anim(CAT), *p_stativ(LIKES)),
    )

    # ── [D] PP-internal bigrams ───────────────────────────────────────────────
    print("\n" + "─"*66)
    print("  [D] PP-internal bigrams – SEEN leaves")
    print("─"*66)

    print_scores(
        'SpatialPrep+PlaceNoun  →  "in park"  (PP-internal, SEEN)',
        tree,
        inst(*p_in(), *p_place(PARK)),
    )
    print_scores(
        'TemporalPrep+MotionVerb  →  "before runs"  (PP-internal, SEEN)',
        tree,
        inst(*p_before(), *p_motion(RUNS)),
    )

    # ── [E] Boundary crossings (rare) ────────────────────────────────────────
    print("\n" + "─"*66)
    print("  [E] Boundary-crossing bigrams (rare, trained)")
    print("─"*66)

    print_scores(
        'AnimNoun+SpatialPrep  →  "cat in"  (RARE, boundary crossing)',
        tree,
        inst(*p_anim(CAT), *p_in()),
    )
    print_scores(
        'InanimNoun+MotionVerb  →  "stone runs"  (NOISY, low cohesion)',
        tree,
        inst(*p_inanim(STONE), *p_motion(RUNS)),
    )

    # ── [F] UNSEEN leaf – same d0/d1/d2 as frequent NP cluster ───────────────
    print("\n" + "─"*66)
    print("  [F] UNSEEN leaf tokens – well-matched category context")
    print("─"*66)

    print_scores(
        'Det+UnseenAnimNoun  →  "the wolf"  (UNSEEN leaf, NP pattern matches d0-d2)',
        tree,
        inst(*p_the(), *p_anim(WOLF)),
    )
    print_scores(
        'Det+UnseenInanimNoun  →  "a coin"  (UNSEEN leaf, NP-inanim pattern)',
        tree,
        inst(*p_a(), *p_inanim(COIN)),
    )
    print_scores(
        'Det+UnseenPlaceNoun  →  "the cave"  (UNSEEN leaf, NP-place pattern)',
        tree,
        inst(*p_the(), *p_place(CAVE)),
    )
    print_scores(
        'SizeAdj+UnseenAnimNoun  →  "tiny wolf"  (UNSEEN adj+noun, d0-d1 match ADJ+NOUN)',
        tree,
        inst(*p_size(TINY), *p_anim(WOLF)),
    )
    print_scores(
        'ColorAdj+UnseenNoun  →  "black wolf"  (UNSEEN color+noun)',
        tree,
        inst(*p_color(BLACK), *p_anim(WOLF)),
    )

    # ── [G] UNSEEN verb – VP pattern partially matched ───────────────────────
    print("\n" + "─"*66)
    print("  [G] UNSEEN verb tokens – VP pattern partially matched")
    print("─"*66)

    print_scores(
        'AnimNoun+UnseenMotionVerb  →  "cat leaps"  (UNSEEN verb, d0-d2 match MOTION_VERB)',
        tree,
        inst(*p_anim(CAT), *p_motion(LEAPS)),
    )
    print_scores(
        'AnimNoun+UnseenPerceptionVerb  →  "dog smells"  (UNSEEN perception)',
        tree,
        inst(*p_anim(DOG), *p_percep(SMELLS)),
    )
    print_scores(
        'AnimNoun+UnseenStatVerb  →  "bird fears"  (UNSEEN stative)',
        tree,
        inst(*p_anim(BIRD), *p_stativ(FEARS)),
    )

    # ── [H] Fully UNSEEN bigrams – both sides novel ──────────────────────────
    print("\n" + "─"*66)
    print("  [H] Both sides UNSEEN – structural pattern only")
    print("─"*66)

    print_scores(
        'UnseenDet+UnseenAnimNoun  →  "an wolf"  (UNSEEN leaf+token, NP structure)',
        tree,
        inst(*p_an(), *p_anim(WOLF)),
    )
    print_scores(
        'UnseenAnimNoun+UnseenMotionVerb  →  "wolf leaps"  (UNSEEN N+V, VP structure)',
        tree,
        inst(*p_anim(WOLF), *p_motion(LEAPS)),
    )
    print_scores(
        'UnseenSpatialPrep+UnseenPlaceNoun  →  "under cave"  (UNSEEN PP, structure intact)',
        tree,
        inst(*p_under(), *p_place(CAVE)),
    )

    # ── [I] Category-mismatch probes  ────────────────────────────────────────
    # These should score LOWER than their correct-pattern counterparts.
    print("\n" + "─"*66)
    print("  [I] Category-mismatch probes (should score poorly)")
    print("─"*66)

    print_scores(
        'AnimNoun+Article  →  "cat the"  (WRONG: N followed by Det)',
        tree,
        inst(*p_anim(CAT), *p_the()),
    )
    print_scores(
        'TemporalPrep+AnimNoun  →  "before cat"  (WRONG: temp prep before noun)',
        tree,
        inst(*p_before(), *p_anim(CAT)),
    )
    print_scores(
        'MotionVerb+SpatialPrep  →  "runs in"  (WRONG: verb+prep without NP context)',
        tree,
        inst(*p_motion(RUNS), *p_in()),
    )
    print_scores(
        'ColorAdj+MotionVerb  →  "blue runs"  (WRONG: adj before verb)',
        tree,
        inst(*p_color(BLUE), *p_motion(RUNS)),
    )
    print_scores(
        'PlaceNoun+AnimNoun  →  "park cat"  (WRONG: NP+NP without determiner)',
        tree,
        inst(*p_place(PARK), *p_anim(CAT)),
    )

    # ── HTML tree visualization ───────────────────────────────────────────────
    _val_fn = lambda vid: VALUE_NAMES.get(vid, str(vid))
    drawer = HTMLCobwebDrawer(
        attributes=[
            "Left-D0", "Left-D1", "Left-D2", "Left-D3",
            "Right-D0", "Right-D1", "Right-D2", "Right-D3",
        ],
        id_to_value=[],
        value_to_id={},
        attr_value_fn={i: _val_fn for i in range(8)},
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "output", "logprob_paths_large_tree"
    )
    try:
        html_file, png_file = drawer.draw_tree(tree.root, output_path)
        print(f"\nTree visualization saved to: {html_file}")
    except Exception as exc:
        html_file = output_path + ".html"
        import json
        d3_json = json.dumps(drawer._node_to_dict(tree.root))
        html_str = drawer._build_html(d3_json)
        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"\nTree visualization (HTML only) saved to: {html_file}")
        print(f"  (PNG skipped: {exc})")


if __name__ == "__main__":
    test_path_logprobs_large()
