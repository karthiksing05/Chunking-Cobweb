"""
HDP Test 2: Recognition score probe using concept-path IDs – bigram framing.

Mirrors tests/cobweb/test_logprob_paths.py, replacing CobwebDiscreteTree
scoring with HDP's recognition score.

In the real framework, content instances encode *both* tokens of a bigram
as their label_path through the context hierarchy — 3 depth levels per side:

    {0: {left_d0: 1}, 1: {left_d1: 1}, 2: {left_d2: 1},
     3: {right_d0: 1}, 4: {right_d1: 1}, 5: {right_d2: 1}}

POS concept hierarchy used here:

    ROOT
    ├── FUNC_WORD (100)
    │   └── ARTICLE (110)
    │       ├── DEF_ART (111)        ← "the"
    │       └── INDEF_ART (112)      ← "a"
    └── CONTENT_WORD (200)
        ├── NOUN (210)
        │   ├── ANIM_NOUN (211)      ← cat, dog, bird
        │   └── INANIM_NOUN (212)    ← fish, mouse
        └── VERB (220)
            ├── RUNS (221)
            └── SEES (222)

Training bigrams:
  - 10  Det+Noun  bigrams  (NP-internal, high cohesion)
  - 2   Noun+Verb bigrams  (NP→VP boundary crossing, low cohesion)

Key insight replicated for HDP:
  • Det+UnseenNoun ("the horse") should outscore UnseenNoun+Verb ("horse runs")
    because depth-0 (FUNC_WORD) and depth-1 (ARTICLE) overlap with the
    Det+Noun training cluster even when the leaf concept (HORSE) is novel.
  • The HDP's ``basic_level`` shifts toward shallower nodes for unseen
    leaves, increasing log_frequency at the cost of log_likelihood — the
    recognition score naturally balances the two.

HDP-specific quantities printed:
  • marginal log p(x)      — log Σ_k p(k)p(x|k) over all leaves
  • root / basic / leaf    — recognition score at each path position
  • full per-node path table
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdp import HDP


# ── simulated concept IDs (POS hierarchy) ────────────────────────────────────
# depth 0 (coarsest)
FUNC_WORD    = 100
CONTENT_WORD = 200

# depth 1
ARTICLE      = 110
NOUN         = 210
VERB         = 220

# depth 2
DEF_ART      = 111   # "the"
INDEF_ART    = 112   # "a"

# Leaf-level token IDs (depth 2 for nouns/verbs — same POS slot)
CAT   = 2011
DOG   = 2012
BIRD  = 2013
FISH  = 2021
MOUSE = 2022
HORSE = 2031   # UNSEEN during training
COW   = 2032   # UNSEEN during training
RUNS  = 2211
SEES  = 2212


def inst(l0, l1, l2, r0, r1, r2):
    """3-depth-per-side content instance (6 visible attributes)."""
    return {
        0: {l0: 1.0},   # left  depth 0  (coarsest POS category)
        1: {l1: 1.0},   # left  depth 1
        2: {l2: 1.0},   # left  depth 2  (finest POS / leaf token)
        3: {r0: 1.0},   # right depth 0
        4: {r1: 1.0},   # right depth 1
        5: {r2: 1.0},   # right depth 2
    }


# ── path shorthand helpers ────────────────────────────────────────────────────
def article_def():    return (FUNC_WORD, ARTICLE, DEF_ART)
def article_indef():  return (FUNC_WORD, ARTICLE, INDEF_ART)
def noun_anim(leaf):  return (CONTENT_WORD, NOUN, leaf)
def noun_inanim(leaf):return (CONTENT_WORD, NOUN, leaf)
def verb(leaf):       return (CONTENT_WORD, VERB, leaf)


# ── training bigrams ──────────────────────────────────────────────────────────
# Repeat the most frequent NP-internal bigrams to create a tight cluster
# (mirrors the 10× repetition of "the cat" in test_hdp_recognition_words.py).
DET_NOUN_BIGRAMS = [
    # "the cat" x10 — dominant definite-article NP pair
    *[inst(*article_def(), *noun_anim(CAT))] * 10,
    inst(*article_def(),   *noun_anim(DOG)),
    inst(*article_def(),   *noun_anim(BIRD)),
    inst(*article_def(),   *noun_inanim(FISH)),
    inst(*article_def(),   *noun_inanim(MOUSE)),
    # "a cat" x5 — dominant indefinite-article NP pair (gives indef cluster
    # enough mass to outscore the Noun+Verb cluster for unseen nouns)
    *[inst(*article_indef(), *noun_anim(CAT))] * 5,
    inst(*article_indef(), *noun_anim(DOG)),
    inst(*article_indef(), *noun_anim(BIRD)),
    inst(*article_indef(), *noun_inanim(FISH)),
    inst(*article_indef(), *noun_inanim(MOUSE)),
]

NOUN_VERB_BIGRAMS = [
    inst(*noun_anim(CAT), *verb(RUNS)),
    inst(*noun_anim(DOG), *verb(RUNS)),
]

TRAINING = DET_NOUN_BIGRAMS + NOUN_VERB_BIGRAMS


# ── helpers ───────────────────────────────────────────────────────────────────
def print_scores(label: str, hdp: HDP, instance: dict):
    """Print HDP path-aware recognition statistics for *instance*."""
    leaf, path  = hdp.categorize_path(instance)
    stats       = hdp.score_along_path(instance)
    basic_node  = hdp.basic_level(instance)
    marginal_lp = hdp.log_prob_instance(instance)
    rec_score   = hdp.recognition_score(instance)

    root_stat  = stats[0]
    leaf_stat  = stats[-1]
    basic_stat = next(s for s in stats if s["node"] is basic_node)

    print(f"\n{'='*64}")
    print(f"  Query: {label}")
    print(f"{'='*64}")
    print(f"  marginal log p(x)   : {marginal_lp:.6f}")
    print(f"  root  recognition   : {root_stat['recognition']:.6f}"
          f"  (count={root_stat['count']})")
    print(f"  basic recognition   : {basic_stat['recognition']:.6f}"
          f"  (depth={basic_stat['depth']}, count={basic_stat['count']})"
          f"  ← hdp.recognition_score()")
    print(f"  leaf  recognition   : {leaf_stat['recognition']:.6f}"
          f"  (depth={leaf_stat['depth']}, count={leaf_stat['count']})")

    print(f"\n  Full path ({len(stats)} nodes):")
    for s in stats:
        marker = " ← basic-level" if s["node"] is basic_node else ""
        print(
            f"    [depth={s['depth']}]  count={s['count']:4d}"
            f"  ll={s['log_likelihood']:.4f}"
            f"  lf={s['log_frequency']:.4f}"
            f"  rec={s['recognition']:.4f}"
            f"{marker}"
        )

    assert rec_score == basic_stat["recognition"], (
        "recognition_score() must equal the basic-level row recognition"
    )


# ── main test ─────────────────────────────────────────────────────────────────
def test_hdp_recognition_paths():
    # Pin seed so Gibbs partitioning is deterministic across runs.
    random.seed(0)

    hdp = HDP(alpha=1.0, beta=0.1, max_depth=5)
    hdp.fit(TRAINING, n_passes=5)

    print(f"\nHDP: {hdp}")
    print(f"root.count={hdp.root.count}, n_stored={len(hdp)}")

    # 1. Frequent NP-internal bigram: Det+Noun (SEEN leaves)
    print_scores(
        'Det+Noun  →  "the cat"  (FREQUENT, NP-internal, SEEN leaves)',
        hdp,
        inst(*article_def(), *noun_anim(CAT)),
    )

    # 2. Rare boundary bigram: Noun+Verb (SEEN leaves)
    print_scores(
        'Noun+Verb  →  "cat runs"  (RARE, boundary-crossing, SEEN leaves)',
        hdp,
        inst(*noun_anim(CAT), *verb(RUNS)),
    )

    # 3. Unseen leaf, matching Det+Noun POS pattern
    # depth-0 (FUNC_WORD) and depth-1 (ARTICLE) still match the NP cluster
    # even though HORSE (leaf) was never seen during training.
    print_scores(
        'Det+UnseenNoun  →  "the horse"  (UNSEEN leaf, correct NP POS pattern)',
        hdp,
        inst(*article_def(), *noun_anim(HORSE)),
    )

    # 4. Unseen noun + seen verb: boundary pattern with novel left token
    # depth-0 mismatch (CONTENT_WORD/CONTENT_WORD vs NP cluster's FUNC_WORD/CONTENT_WORD)
    print_scores(
        'UnseenNoun+Verb  →  "horse runs"  (UNSEEN N, SEEN V, wrong POS pattern)',
        hdp,
        inst(*noun_anim(HORSE), *verb(RUNS)),
    )

    # 5. Fully unseen NP bigram: both leaves new but POS pattern matches
    # Should still score better than any Noun+Verb pair.
    print_scores(
        'Det+UnseenNoun  →  "a horse"  (UNSEEN leaf, INDEF article, NP pattern)',
        hdp,
        inst(*article_indef(), *noun_anim(HORSE)),
    )

    # ── ordering assertions ───────────────────────────────────────────────────
    r_det_noun       = hdp.recognition_score(inst(*article_def(),   *noun_anim(CAT)))
    r_noun_verb      = hdp.recognition_score(inst(*noun_anim(CAT),  *verb(RUNS)))
    r_det_unseen     = hdp.recognition_score(inst(*article_def(),   *noun_anim(HORSE)))
    r_unseen_verb    = hdp.recognition_score(inst(*noun_anim(HORSE), *verb(RUNS)))
    r_indef_unseen   = hdp.recognition_score(inst(*article_indef(), *noun_anim(HORSE)))

    # 1. Frequent NP outscores the rare boundary bigram (frequency effect).
    assert r_det_noun > r_noun_verb, (
        f"Expected frequent Det+Noun ({r_det_noun:.4f}) > "
        f"rare Noun+Verb ({r_noun_verb:.4f})"
    )

    # 2. Correct POS structure beats wrong POS structure even with an unseen leaf.
    # depth-0 FUNC_WORD match (Det+UnseenNoun) outranks CONTENT_WORD+CONTENT_WORD
    # (UnseenNoun+Verb), because the cluster mass at depth 0/1 is much larger.
    assert r_det_unseen > r_unseen_verb, (
        f"Expected Det+UnseenNoun ({r_det_unseen:.4f}) > "
        f"UnseenNoun+Verb ({r_unseen_verb:.4f})"
    )

    # 3. Both unseen Det+Noun variants (definite and indefinite) each outscore
    # the frequent *seen* Noun+Verb bigram — the cluster size difference alone
    # is not enough to flip the POS-structure advantage.
    assert r_det_unseen > r_noun_verb, (
        f"Expected Det+UnseenNoun ({r_det_unseen:.4f}) > "
        f"seen Noun+Verb ({r_noun_verb:.4f})"
    )
    assert r_indef_unseen > r_noun_verb, (
        f"Expected indef Det+UnseenNoun ({r_indef_unseen:.4f}) > "
        f"seen Noun+Verb ({r_noun_verb:.4f})"
    )

    print("\n✓ All ordering assertions passed.")


if __name__ == "__main__":
    test_hdp_recognition_paths()
