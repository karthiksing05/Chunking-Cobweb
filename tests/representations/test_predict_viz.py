"""
Test 3: Visualize predict / predict_pmi outputs for bigrams and evaluate
predictive power as a chunking heuristic.

We train the same 3-depth path-encoded content tree as test_logprob_paths.py,
but now the instances represent surface bigrams from a simple English grammar:

    S → NP VP,  NP → Det N,  VP → V NP

Training bigrams:
  - 10  Det+Noun  bigrams  (NP-internal, cohesive)
  - 2   Noun+Verb bigrams  (NP→VP boundary, non-cohesive)

POS hierarchy:
    ROOT
    ├── FUNC_WORD (100)
    │   └── ARTICLE (110)
    │       ├── DEF_ART (111)    ← "the"
    │       └── INDEF_ART (112)  ← "a"
    └── CONTENT_WORD (200)
        ├── NOUN (210)
        │   ├── ANIM_NOUN leaf IDs: CAT=2011, DOG=2012, BIRD=2013
        │   └── INANIM_NOUN leaf IDs: FISH=2021, MOUSE=2022
        └── VERB (220)
            └── leaf IDs: RUNS=2211

Then for each query we:
  1. Give only the LEFT bigram token (attrs 0,1,2) as the "observed" part.
  2. Ask the tree to complete / predict the RIGHT token (attrs 3,4,5).
  3. Show the predicted distribution via predict() and predict_pmi().
  4. Score the ACTUAL right token against the predicted distribution.

Key expected results:
  • Observing "the" (DEF_ART) strongly predicts a NOUN on the right (NP path).
  • Observing "cat" (NOUN) predicts either RUNS (boundary bigram) or another
    NOUN, but with lower confidence since N+V only has 2 training instances.
  • Observing "runs" (VERB) predicts poorly – verbs don't appear as left
    tokens in the training data, so no coherent NP or VP chunk is predicted.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree

# ── concept IDs (POS hierarchy, same as test_logprob_paths.py) ───────────────
FUNC_WORD    = 100
CONTENT_WORD = 200
ARTICLE      = 110
NOUN         = 210
VERB         = 220
DEF_ART      = 111   # "the"
INDEF_ART    = 112   # "a"

THE   = 1011
A     = 1012
CAT   = 2011;  DOG  = 2012;  BIRD = 2013
FISH  = 2021;  MOUSE= 2022
HORSE = 2031;  COW  = 2032   # unseen leaves
RUNS  = 2211

# attr indices
L0, L1, L2 = 0, 1, 2   # left side depth 0,1,2
R0, R1, R2 = 3, 4, 5   # right side depth 0,1,2

# ID → readable name
NAMES = {
    100: "FUNC_WORD", 200: "CONTENT_WORD",
    110: "ARTICLE",   210: "NOUN",   220: "VERB",
    111: "DEF_ART",   112: "INDEF_ART",
    1011: "the",  1012: "a",
    2011: "cat",  2012: "dog",  2013: "bird",
    2021: "fish", 2022: "mouse",
    2031: "horse",2032: "cow",
    2211: "runs",
}
ATTR_NAMES = {0:"L-D0", 1:"L-D1", 2:"L-D2", 3:"R-D0", 4:"R-D1", 5:"R-D2"}

def name(v):
    return NAMES.get(v, str(v))

def inst(l0,l1,l2,r0,r1,r2):
    return {L0:{l0:1.}, L1:{l1:1.}, L2:{l2:1.},
            R0:{r0:1.}, R1:{r1:1.}, R2:{r2:1.}}

def article_def():      return (FUNC_WORD, ARTICLE, DEF_ART)     # "the"
def article_indef():    return (FUNC_WORD, ARTICLE, INDEF_ART)    # "a"
def noun(leaf):         return (CONTENT_WORD, NOUN, leaf)
def verb(leaf):         return (CONTENT_WORD, VERB, leaf)


# ── training bigrams (same POS hierarchy as test_logprob_paths.py) ────────────
TRAINING = [
    # NP-internal bigrams: Det + Noun (10 instances)
    inst(*article_def(),   *noun(CAT)),
    inst(*article_def(),   *noun(DOG)),
    inst(*article_def(),   *noun(BIRD)),
    inst(*article_def(),   *noun(FISH)),
    inst(*article_def(),   *noun(MOUSE)),
    inst(*article_indef(), *noun(CAT)),
    inst(*article_indef(), *noun(DOG)),
    inst(*article_indef(), *noun(BIRD)),
    inst(*article_indef(), *noun(FISH)),
    inst(*article_indef(), *noun(MOUSE)),
    # Boundary-crossing bigrams: Noun + Verb (2 instances)
    inst(*noun(CAT), *verb(RUNS)),
    inst(*noun(DOG), *verb(RUNS)),
]


# ── pretty-print helpers ──────────────────────────────────────────────────────
def _table(attr_dist, title, top_n=5):
    """Print a sorted probability table for one or more attributes."""
    print(f"\n  ┌─ {title}")
    rows = []
    for attr, val_map in sorted(attr_dist.items()):
        sorted_vals = sorted(val_map.items(), key=lambda kv: -kv[1])[:top_n]
        for val, p in sorted_vals:
            rows.append((attr, val, p))
    if not rows:
        print("  │  (empty)")
    for attr, val, p in rows:
        bar = "█" * int(p * 30)
        print(f"  │  {ATTR_NAMES.get(attr,'?'):6s}  {name(val):14s}  {p:.4f}  {bar}")
    print("  └" + "─" * 50)


# ── scoring helpers ──────────────────────────────────────────────────────────
LEFT_ATTRS  = (L0, L1, L2)
RIGHT_ATTRS = (R0, R1, R2)


def _directed_logp(tree, obs_inst, target_inst, max_nodes=100):
    """log p(target | obs) under tree.predict mixture."""
    pred = tree.predict(obs_inst, max_nodes, False)
    lp = 0.0
    for attr, val_map in target_inst.items():
        dist = pred.get(attr, {})
        for val, cnt in val_map.items():
            lp += cnt * math.log(max(dist.get(val, 1e-9), 1e-9))
    return lp


def _directed_pmi_logp(tree, obs_inst, target_inst, target_attrs, max_nodes=100):
    """Sum of log pmi_weighted_p(target_val | predict_pmi(obs, attr))."""
    lp = 0.0
    for attr in target_attrs:
        if attr not in target_inst:
            continue
        pmi_pred = tree.predict_pmi(obs_inst, attr, max_nodes, False)
        dist = pmi_pred.get(attr, {})
        for val, cnt in target_inst.get(attr, {}).items():
            lp += cnt * math.log(max(dist.get(val, 1e-9), 1e-9))
    return lp


def symmetric_logp(tree, left_inst, right_inst, max_nodes=100):
    """log p(R|L) + log p(L|R)"""
    return (_directed_logp(tree, left_inst,  right_inst, max_nodes)
            + _directed_logp(tree, right_inst, left_inst,  max_nodes))


def symmetric_pmi_logp(tree, left_inst, right_inst, max_nodes=100):
    """PMI-weighted version: pmilogp(R|L) + pmilogp(L|R)"""
    return (_directed_pmi_logp(tree, left_inst,  right_inst, RIGHT_ATTRS, max_nodes)
            + _directed_pmi_logp(tree, right_inst, left_inst,  LEFT_ATTRS,  max_nodes))


def _score_row(label, fwd, bwd, sym, sym_pmi, width=26):
    """Print one row of the symmetric score table."""
    bar = "█" * max(0, int((sym + 60) * 0.3))
    print(f"  {label:<{width}}  {fwd:+6.2f}  {bwd:+6.2f}  {sym:+6.2f}  {sym_pmi:+7.2f}  {bar}")


def predict_completion(tree, left_inst, label,
                       actual_right=None, max_nodes=100):
    """
    Given the LEFT side only, predict/complete the RIGHT side.
    Also show predict_pmi for each right attribute.
    """
    print(f"\n{'='*62}")
    print(f"  Completion query: {label}")
    print(f"{'='*62}")

    # 1. Full predict (mixture-weighted marginal over all attrs)
    full_pred = tree.predict(left_inst, max_nodes, False)
    right_pred = {k: v for k, v in full_pred.items() if k in (R0, R1, R2)}
    _table(right_pred, "predict()  →  right-side marginals (mixture-weighted)")

    # 2. predict_pmi for each right attribute separately
    pmi_results = {}
    for attr in (R0, R1, R2):
        try:
            pmi_pred = tree.predict_pmi(left_inst, attr, max_nodes, False)
            pmi_results.update(pmi_pred)
        except Exception as e:
            print(f"  predict_pmi attr={attr} failed: {e}")
    if pmi_results:
        _table(pmi_results, "predict_pmi()  →  right-side PMI-weighted")

    # 3. Directed + symmetric scores for the provided right side
    if actual_right is not None:
        fwd     = _directed_logp(tree, left_inst,   actual_right, max_nodes)
        bwd     = _directed_logp(tree, actual_right, left_inst,   max_nodes)
        sym     = fwd + bwd
        sym_pmi = symmetric_pmi_logp(tree, left_inst, actual_right, max_nodes)
        print(f"\n  Scores vs actual right side:")
        print(f"    forward  log p(R|L) = {fwd:+.4f}")
        print(f"    backward log p(L|R) = {bwd:+.4f}")
        print(f"    symmetric           = {sym:+.4f}  (fwd + bwd)")
        print(f"    symmetric PMI       = {sym_pmi:+.4f}") 


def test_predict_viz():
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)
    for item in TRAINING:
        tree.ifit(item)

    print(f"\nTree: {sum(1 for _ in _iter_nodes(tree.root))} nodes, "
          f"root.count={tree.root.count}")

    # ── Case 1: Left = "the" (DEF_ART) → predict right token ────────────────
    # Expect strong prediction of CONTENT_WORD→NOUN on the right (NP bigram).
    left_the   = {L0: {FUNC_WORD: 1.0}, L1: {ARTICLE: 1.0}, L2: {DEF_ART: 1.0}}
    actual_det_noun  = {R0: {CONTENT_WORD: 1.0}, R1: {NOUN: 1.0}, R2: {CAT: 1.0}}
    actual_det_verb  = {R0: {CONTENT_WORD: 1.0}, R1: {VERB: 1.0}, R2: {RUNS: 1.0}}
    actual_det_unseen= {R0: {CONTENT_WORD: 1.0}, R1: {NOUN: 1.0}, R2: {HORSE: 1.0}}

    predict_completion(tree, left_the,
        'Left = "the" (DEF_ART)  →  predict right token  [should predict NOUN strongly]',
        actual_right=actual_det_noun)

    # ── Case 2: Symmetric score comparison ──────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Symmetric score table: left=\"the\" vs multiple right tokens")
    print(f"{'='*62}")

    left_cat  = {L0: {CONTENT_WORD: 1.0}, L1: {NOUN: 1.0}, L2: {CAT: 1.0}}
    left_runs = {L0: {CONTENT_WORD: 1.0}, L1: {VERB: 1.0}, L2: {RUNS: 1.0}}
    left_a    = {L0: {FUNC_WORD: 1.0}, L1: {ARTICLE: 1.0}, L2: {INDEF_ART: 1.0}}

    actual_noun_verb   = {R0: {CONTENT_WORD: 1.0}, R1: {VERB: 1.0}, R2: {RUNS: 1.0}}
    actual_unseen_noun = {R0: {CONTENT_WORD: 1.0}, R1: {NOUN: 1.0}, R2: {HORSE: 1.0}}
    actual_verb_verb   = {R0: {CONTENT_WORD: 1.0}, R1: {VERB: 1.0}, R2: {RUNS: 1.0}}

    candidates = [
        ('the + cat   (Det+Noun, NP-internal)',   left_the,  actual_det_noun),
        ('the + horse (Det+UnseenNoun, NP-same)', left_the,  actual_det_unseen),
        ('the + runs  (Det+Verb, wrong pattern)', left_the,  actual_det_verb),
        ('cat + runs  (Noun+Verb, boundary)',     left_cat,  actual_noun_verb),
        ('a + cat     (IndefDet+Noun, NP-same)',  left_a,    actual_det_noun),
        ('runs + cat  (Verb+Noun, reverse)',      left_runs, actual_det_noun),
    ]

    print(f"\n  {'Pair':<40}  {'fwd':>6}  {'bwd':>6}  {'sym':>6}  {'sym_pmi':>7}  bar(sym)")
    print("  " + "-" * 84)
    scores = {}
    for lbl, l_inst, r_inst in candidates:
        fwd     = _directed_logp(tree,     l_inst, r_inst)
        bwd     = _directed_logp(tree,     r_inst, l_inst)
        sym     = fwd + bwd
        sym_pmi = symmetric_pmi_logp(tree, l_inst, r_inst)
        scores[lbl] = sym
        _score_row(lbl, fwd, bwd, sym, sym_pmi, width=40)

    best  = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    print(f"\n  Best  symmetric score → {best}")
    print(f"  Worst symmetric score → {worst}")

    # delta: NP-internal bigram advantage over boundary bigram
    np_sym  = scores['the + cat   (Det+Noun, NP-internal)']
    bnd_sym = scores['cat + runs  (Noun+Verb, boundary)']
    print(f"\n  Δ symmetric (Det+Noun NP vs Noun+Verb boundary) = {np_sym - bnd_sym:+.4f} nats")

    # ── Case 3: Left = "runs" (VERB) → predict right  ──────────────────────
    # Verbs never appear as left-side tokens in training data, so predict
    # should be diffuse / low-confidence.
    predict_completion(tree,
        {L0: {CONTENT_WORD: 1.0}, L1: {VERB: 1.0}, L2: {RUNS: 1.0}},
        'Left = "runs" (VERB)  →  predict right token  [should be diffuse/low]')


def _iter_nodes(node):
    yield node
    for c in node.children:
        yield from _iter_nodes(c)


if __name__ == "__main__":
    test_predict_viz()
