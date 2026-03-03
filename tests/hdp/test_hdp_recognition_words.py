"""
HDP Test 1: Recognition score probe using raw word IDs – bigram framing.

Mirrors tests/cobweb/test_logprob_words.py, replacing the CobwebDiscreteTree
scoring with HDP's recognition score:

    recognition(x) = log p(x | c_basic) + log(count_basic / count_root)

where ``c_basic`` is the basic-level node — the node along the MAP
categorisation path that *maximises* this combined score.

Training bigrams:
  - 10  Det+Noun  bigrams  (NP-internal, frequent and cohesive)
  - 2   Noun+Verb bigrams  (NP→VP boundary crossing, rare)

Each bigram instance is a 2-attribute record:
  {0: {left_token_id: 1}, 1: {right_token_id: 1}}

Expected ordering (recognition score, highest → lowest):
  1. Det+Noun  "the cat"    (FREQUENT, SEEN)         ← highest
  2. Det+UnseenNoun "the horse" (UNSEEN right)       ← near 1 due to left-slot match
  3. Noun+Verb "cat runs"  (RARE, SEEN)
  4. UnseenNoun+Verb "horse runs" (UNSEEN left + boundary)
  5. Noun+Det  "cat a"     (UNTRAINED reversed bigram) ← lowest

HDP-specific quantities printed:
  • marginal  log p(x)          — log Σ_k p(k)p(x|k) over all leaves
  • root      recognition       — recognition score at depth 0
  • basic     recognition       — recognition score at the adaptive basic level
  • leaf      recognition       — recognition score at the MAP leaf
  • full path table with per-node statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdp import HDP


# ── vocabulary ────────────────────────────────────────────────────────────────
# Determiners
THE  = 1
A    = 2

# Nouns (seen during training)
CAT   = 3
DOG   = 4
BIRD  = 5
FISH  = 6
MOUSE = 7

# Nouns *unseen* during training
HORSE = 8
COW   = 9

# Verbs
RUNS = 20
SEES = 21


def inst(left, right):
    """Build a 2-attribute bigram instance."""
    return {0: {left: 1.0}, 1: {right: 1.0}}


# ── training data ─────────────────────────────────────────────────────────────
DET_NOUN_BIGRAMS = [
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, CAT),
    inst(THE, DOG),
    inst(THE, BIRD),
    inst(THE, FISH),
    inst(THE, MOUSE),
    inst(A,   CAT),
    inst(A,   DOG),
    inst(A,   BIRD),
    inst(A,   FISH),
    inst(A,   MOUSE),
]

NOUN_VERB_BIGRAMS = [
    inst(CAT, RUNS),
    inst(DOG, RUNS),
]

TRAINING = DET_NOUN_BIGRAMS + NOUN_VERB_BIGRAMS


# ── helpers ───────────────────────────────────────────────────────────────────
def print_scores(label: str, hdp: HDP, instance: dict):
    """Print HDP recognition statistics for *instance*."""
    leaf, path = hdp.categorize_path(instance)
    stats       = hdp.score_along_path(instance)
    basic_node  = hdp.basic_level(instance)
    marginal_lp = hdp.log_prob_instance(instance)
    rec_score   = hdp.recognition_score(instance)

    root_stat  = stats[0]
    leaf_stat  = stats[-1]
    basic_stat = next(s for s in stats if s["node"] is basic_node)

    print(f"\n{'='*60}")
    print(f"  Query: {label}")
    print(f"{'='*60}")
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
def test_hdp_recognition_words():
    hdp = HDP(alpha=1.0, beta=0.1, max_depth=5)
    hdp.fit(TRAINING, n_passes=5)

    print(f"\nHDP: {hdp}")
    print(f"root.count={hdp.root.count}, n_stored={len(hdp)}")

    # 1. Frequent NP-internal bigram (SEEN)
    print_scores(
        'Det+Noun  →  "the cat"  (FREQUENT, NP-internal, SEEN)',
        hdp, inst(THE, CAT),
    )

    # 2. Rare boundary bigram (SEEN)
    print_scores(
        'Noun+Verb  →  "cat runs"  (RARE, boundary-crossing, SEEN)',
        hdp, inst(CAT, RUNS),
    )

    # 3. Unseen right token, matching left-slot type (DET)
    print_scores(
        'Det+UnseenNoun  →  "the horse"  (UNSEEN right, DET left matches cluster)',
        hdp, inst(THE, HORSE),
    )

    # 4. Unseen left + boundary pattern
    print_scores(
        'UnseenNoun+Verb  →  "horse runs"  (UNSEEN left + boundary-crossing)',
        hdp, inst(HORSE, RUNS),
    )

    # 5. Reversed / untrained bigram
    print_scores(
        'Noun+Det  →  "cat a"  (UNTRAINED reversed bigram)',
        hdp, inst(CAT, A),
    )

    # ── ordering assertions ───────────────────────────────────────────────────
    # Det+Noun (frequent) should be recognised more strongly than boundary bigrams
    r_det_noun     = hdp.recognition_score(inst(THE, CAT))
    r_noun_verb    = hdp.recognition_score(inst(CAT, RUNS))
    r_det_unseen   = hdp.recognition_score(inst(THE, HORSE))
    r_unseen_verb  = hdp.recognition_score(inst(HORSE, RUNS))

    assert r_det_noun > r_noun_verb, (
        f"Expected frequent Det+Noun ({r_det_noun:.4f}) > "
        f"rare Noun+Verb ({r_noun_verb:.4f})"
    )
    assert r_det_noun > r_unseen_verb, (
        f"Expected Det+Noun ({r_det_noun:.4f}) > "
        f"unseen Noun+Verb ({r_unseen_verb:.4f})"
    )
    assert r_det_unseen > r_unseen_verb, (
        f"Expected Det+UnseenNoun ({r_det_unseen:.4f}) > "
        f"UnseenNoun+Verb ({r_unseen_verb:.4f}), "
        f"since the left-slot category still matches"
    )

    print("\n✓ All ordering assertions passed.")


if __name__ == "__main__":
    test_hdp_recognition_words()
