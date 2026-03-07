"""
Test 3: Context-hierarchy probe – distributional primitives from a toy grammar.

The *context hierarchy* in webster learns distributional representations of words
by training on their sliding-window neighbourhood: what words surround each
token in a sentence.

A context instance for word ``w`` at position ``i`` (context_length=2) is:

    {
      0: {left_1:     1.0},   # immediate left  neighbour  (slot mode)
      1: {left_2:     0.5},   # 2-away  left  neighbour
      2: {right_1:    1.0},   # immediate right neighbour
      3: {right_2:    0.5},   # 2-away  right neighbour
     -2: {COMPLEXITY: 1  },   # complexity/hidden marker
      4: {word_id:    1  },   # content-ref (the word itself, at 2*ctx_len)
    }

Missing neighbours leave the slot **absent** (key omitted entirely).
Weights follow the binary schedule: 1 / 2^(j+1) for j = 0, 1, …

Grammar used to generate the training corpus
─────────────────────────────────────────────
    S  → NP VP
    NP → Det N  |  Det N PP
    VP → V  |  V NP
    PP → P NP

    Det  → the | a
    N    → cat | dog | bird | fish | mouse
    V    → runs | sees | chases
    P    → with | in

All sentences are expanded without BOS/EOS markers so that sentence-boundary
slots are simply absent; this keeps the context representation clean.

Expected distributional behaviour
──────────────────────────────────
 • Determiners ("the", "a") cluster together:
       left  = absent (sentence start or post-PP verb) | noun
       right = a NOUN
 • Nouns ("cat", "dog", …) cluster together:
       left  = a DETERMINER
       right = VERB | end-of-NP | P
 • Verbs ("runs", "sees", "chases") cluster together:
       left  = a NOUN
       right = DETERMINER | absent

Probes (same heuristics as test_logprob_paths.py):
  1. "the" in sentence-initial Det slot            → HIGH  (frequent, typical)
  2. "a"   in same position-0 Det slot             → HIGH  (almost identical context)
  3. "this" (UNSEEN Det) in Det slot               → near-HIGH  (same context pattern)
  4. "runs" (SEEN verb) in VP-head slot            → HIGH
  5. "flies" (UNSEEN verb) in VP-head slot         → near-HIGH
  6. Verb "runs" queried in Det slot               → LOW  (context mismatch)
  7. "cat" (SEEN noun) in NP-head slot             → HIGH
  8. "horse" (UNSEEN noun) in NP-head slot         → near-HIGH
  9. LEFT-only vs. RIGHT-only split scores
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree
from viz import HTMLCobwebDrawer

import random


# ── vocabulary ────────────────────────────────────────────────────────────────
# 0 is reserved for EMPTYNULL (absent / padding).

COMPLEXITY = 99      # hidden marker – matches parse_mh convention

# Determiners
THE  = 10
A    = 11
THIS = 12            # UNSEEN during training

# Nouns (training)
CAT   = 20
DOG   = 21
BIRD  = 22
FISH  = 23
MOUSE = 24

# Nouns (unseen)
HORSE = 25
COW   = 26

# Verbs (training)
RUNS   = 30
SEES   = 31
CHASES = 32

# Verbs (unseen)
FLIES = 33

# Prepositions
WITH = 40
IN   = 41


# ── human-readable names ──────────────────────────────────────────────────────
WORD_NAMES = {
    0:      "∅",
    COMPLEXITY: "COMPLEXITY",
    THE:    "the",
    A:      "a",
    THIS:   "this (unseen)",
    CAT:    "cat",
    DOG:    "dog",
    BIRD:   "bird",
    FISH:   "fish",
    MOUSE:  "mouse",
    HORSE:  "horse (unseen)",
    COW:    "cow (unseen)",
    RUNS:   "runs",
    SEES:   "sees",
    CHASES: "chases",
    FLIES:  "flies (unseen)",
    WITH:   "with",
    IN:     "in",
}


# ── grammar-based sentence generator ─────────────────────────────────────────
# Grammar expressed as a list of integer token sequences.
# We enumerate all possible surface forms deterministically so the test
# is reproducible even without a fixed random seed.

DETERMINERS = [THE, A]
NOUNS       = [CAT, DOG, BIRD, FISH, MOUSE]
VERBS       = [RUNS, SEES, CHASES]
PREPS       = [WITH, IN]


def _all_nps():
    """All Det+N noun phrases."""
    return [[d, n] for d in DETERMINERS for n in NOUNS]


def _all_pps():
    """All P + Det + N prepositional phrases."""
    return [[p] + np for p in PREPS for np in _all_nps()]


def _all_vps():
    """All simple and transitive verb phrases (no PP inside object NP)."""
    vps = []
    for v in VERBS:
        vps.append([v])                         # intransitive
        for np in _all_nps():
            vps.append([v] + np)                # transitive
    return vps


def _all_sentences():
    """S → NP VP  (simple) and S → NP_PP VP (NP with PP modifier)."""
    sentences = []
    for subj_np in _all_nps():
        for vp in _all_vps():
            sentences.append(subj_np + vp)
    return sentences


# Build the full corpus deterministically, then keep a manageable subset
# so the test runs quickly.  We pick every sentence that involves a SEEN
# noun and verb (no cross-contamination with the unseen tokens).
_ALL_SENTS = _all_sentences()
# Keep only sentences whose tokens are all in the training vocabulary
_TRAIN_VOCAB = set(DETERMINERS + NOUNS + VERBS)
TRAINING_SENTENCES = [s for s in _ALL_SENTS
                      if all(t in _TRAIN_VOCAB for t in s)]


# ── context instance utilities ────────────────────────────────────────────────
CONTEXT_LENGTH = 2   # slots on each side


def context_weight(j: int) -> float:
    """Binary distance decay: 1 / 2^(j+1)."""
    return 1.0 / (2 ** (j + 1))


def build_context_instance(sentence_ids: list, position: int,
                           context_length: int = CONTEXT_LENGTH) -> dict:
    """
    Build a context-hierarchy instance for the token at *position*.

    Layout (context_length = 2, total 4 context attrs + complexity + content-ref):
        attr 0: immediate left  neighbour  (weight 0.5)
        attr 1: 2-away   left  neighbour  (weight 0.25)
        attr 2: immediate right neighbour  (weight 0.5)
        attr 3: 2-away   right neighbour  (weight 0.25)
        attr -2: COMPLEXITY hidden marker
        attr 2*context_length: content-ref (the word itself, weight 1)

    Missing slots are omitted entirely (no EMPTYNULL key).
    """
    cref_attr = 2 * context_length
    inst: dict = {}

    # before slots
    for j in range(context_length):
        src = position - (j + 1)
        if 0 <= src < len(sentence_ids):
            wid = sentence_ids[src]
            inst[j] = {wid: context_weight(j), 0: 0}
        # absent → omit slot entirely

    # after slots
    for j in range(context_length):
        src = position + (j + 1)
        attr = context_length + j
        if 0 <= src < len(sentence_ids):
            wid = sentence_ids[src]
            inst[attr] = {wid: context_weight(j), 0: 0}
        # absent → omit slot entirely

    # hidden complexity marker
    inst[-2] = {COMPLEXITY: 1}

    # content-ref: the word's own ID (visible, mirrors parse_mh convention)
    inst[cref_attr] = {sentence_ids[position]: 1}

    return inst


def probe_instance(word_id: int,
                   left_1=None, left_2=None,
                   right_1=None, right_2=None,
                   context_length: int = CONTEXT_LENGTH) -> dict:
    """
    Build a context instance from explicit neighbour word IDs.
    Pass ``None`` for absent neighbours (sentence boundary or unknown).
    """
    cref_attr = 2 * context_length
    inst: dict = {}

    for j, wid in enumerate([left_1, left_2]):
        if wid is not None:
            inst[j] = {wid: context_weight(j), 0: 0}

    for j, wid in enumerate([right_1, right_2]):
        if wid is not None:
            attr = context_length + j
            inst[attr] = {wid: context_weight(j), 0: 0}

    inst[-2] = {COMPLEXITY: 1}
    inst[cref_attr] = {word_id: 1}
    return inst


# ── scoring helpers ───────────────────────────────────────────────────────────
def count_concepts(node) -> int:
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


def _left_only(inst: dict, context_length: int = CONTEXT_LENGTH) -> dict:
    """Retain only the left-context slots (indices 0..context_length-1) + hidden."""
    kept = {k: v for k, v in inst.items()
            if k == -2 or (isinstance(k, int) and 0 <= k < context_length)}
    return kept


def _right_only(inst: dict, context_length: int = CONTEXT_LENGTH) -> dict:
    """Retain only the right-context slots (indices context_length..2*ctx_len-1) + hidden."""
    kept = {k: v for k, v in inst.items()
            if k == -2 or (isinstance(k, int)
                           and context_length <= k < 2 * context_length)}
    return kept


def _content_only(inst: dict, context_length: int = CONTEXT_LENGTH) -> dict:
    """Retain only the content-ref attribute + hidden (no context slots)."""
    cref = 2 * context_length
    return {k: v for k, v in inst.items() if k in (-2, cref)}


def print_scores(label: str, tree: CobwebDiscreteTree,
                 instance: dict, context_length: int = CONTEXT_LENGTH):
    """Mirror of print_scores from test_logprob_paths.py."""
    path = path_to_leaf(tree, instance)

    left_inst    = _left_only(instance, context_length)
    right_inst   = _right_only(instance, context_length)
    content_inst = _content_only(instance, context_length)

    tree_lp          = tree.log_prob(instance,      100, False)
    tree_class_lp    = tree.log_prob_class_given_instance(instance, 100, False)
    tree_left_lp     = tree.log_prob(left_inst,     100, False)
    tree_right_lp    = tree.log_prob(right_inst,    100, False)
    tree_content_lp  = tree.log_prob(content_inst,  100, False)
    root_lp          = path[0].log_prob_instance(instance)
    leaf_lp          = path[-1].log_prob_instance(instance)

    basic_node      = path[-1].get_basic(1000, 100)
    basic_lp        = basic_node.log_prob_instance(instance)
    basic_class_lp  = basic_node.log_prob_class_given_instance(instance)
    basic_depth     = basic_node.depth()

    # best node via get_best (peak log_prob_instance along root→leaf path)
    best_node      = path[-1].get_best(instance)
    best_lp        = best_node.log_prob_instance(instance)
    best_class_lp  = best_node.log_prob_class_given_instance(instance)
    best_depth     = best_node.depth()

    print(f"\n{'='*65}")
    print(f"  Query: {label}")
    print(f"{'='*65}")
    print(f"           tree log-prob : {tree_lp:.6f}")
    print(f"     tree class log-prob : {tree_class_lp:.6f}")
    print(f"      tree LEFT log-prob : {tree_left_lp:.6f}")
    print(f"     tree RIGHT log-prob : {tree_right_lp:.6f}")
    print(f"   tree CONTENT log-prob : {tree_content_lp:.6f}")
    print(f"           root log-prob : {root_lp:.6f}  (count={path[0].count})")
    print(f"           leaf log-prob : {leaf_lp:.6f}  (count={path[-1].count})")
    print(f"          basic log-prob : {basic_lp:.6f}"
          f"  (depth={basic_depth}, count={basic_node.count})")
    print(f"    basic-class log-prob : {basic_class_lp:.6f}")
    print(f"           best log-prob : {best_lp:.6f}"
          f"  (depth={best_depth}, count={best_node.count})")
    print(f"     best-class log-prob : {best_class_lp:.6f}")

    print(f"\n  Full path ({len(path)} nodes):")
    for i, n in enumerate(path):
        lp     = n.log_prob_instance(instance)
        markers = []
        if n is basic_node:
            markers.append("basic-level")
        if n is best_node:
            markers.append("best")
        marker = (" ← " + ", ".join(markers)) if markers else ""
        print(f"    [{i}]  depth={i}  count={n.count:5.0f}  lp={lp:.6f}{marker}")


# ── main test ─────────────────────────────────────────────────────────────────
def test_context_hierarchy():
    """
    Train the context hierarchy on all sentences generated from the toy grammar
    and probe it with several queries that span seen/unseen and matched/
    mismatched distributional contexts.
    """
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)

    # Collect all training context instances + shuffle for natural ordering
    all_instances = []
    for sentence in TRAINING_SENTENCES:
        for pos in range(len(sentence)):
            ctx_inst = build_context_instance(sentence, pos)
            all_instances.append(ctx_inst)

    random.shuffle(all_instances)
    for inst in all_instances:
        tree.ifit(inst)

    n_sents = len(TRAINING_SENTENCES)
    n_insts = len(all_instances)
    print(f"\nCorpus: {n_sents} training sentences → {n_insts} context instances")
    print(f"Tree  : {count_concepts(tree.root)} concepts, "
          f"root.count={tree.root.count}")

    # ------------------------------------------------------------------
    # Probe 1 – "the" as sentence-initial determiner before a noun
    #   Typical context: nothing to the left, CAT to the right.
    # ------------------------------------------------------------------
    print_scores(
        '"the" in Det slot  (left=∅  right=cat)  — SEEN, frequent',
        tree,
        probe_instance(THE, right_1=CAT),
    )

    # ------------------------------------------------------------------
    # Probe 2 – "a" in the same Det slot before a different SEEN noun
    #   Should score similarly to "the" because left=∅, right=NOUN
    # ------------------------------------------------------------------
    print_scores(
        '"a" in Det slot  (left=∅  right=dog)  — SEEN, frequent',
        tree,
        probe_instance(A, right_1=DOG),
    )

    # ------------------------------------------------------------------
    # Probe 3 – "this" (UNSEEN determiner) in the same Det slot
    #   Context pattern identical to "the"/"a" but word_id never trained.
    #   Expect score close to probes 1-2 because context matches.
    # ------------------------------------------------------------------
    print_scores(
        '"this" (UNSEEN Det) in Det slot  (left=∅  right=cat)  — UNSEEN word, SEEN context',
        tree,
        probe_instance(THIS, right_1=CAT),
    )

    # ------------------------------------------------------------------
    # Probe 4 – Verb "runs" (SEEN) queried in the WRONG (Det) slot
    #   Context is identical to probes 1-3 but the word is a verb.
    #   The content-ref attribute should pull the score down relative
    #   to a genuine determiner in the same slot.
    # ------------------------------------------------------------------
    print_scores(
        '"runs" (SEEN Verb) in Det slot  (left=∅  right=cat)  — context-MISMATCH',
        tree,
        probe_instance(RUNS, right_1=CAT),
    )

    # ------------------------------------------------------------------
    # Probe 5 – "cat" (SEEN noun) in its typical NP-head slot
    #   Left = THE (a determiner), right = RUNS (a verb)
    # ------------------------------------------------------------------
    print_scores(
        '"cat" in N slot  (left=the  right=runs)  — SEEN, frequent',
        tree,
        probe_instance(CAT, left_1=THE, right_1=RUNS),
    )

    # ------------------------------------------------------------------
    # Probe 6 – "horse" (UNSEEN noun) in the same N slot
    #   Same context pattern as "cat" but horse was never in the corpus.
    #   Should score close to probe 5 due to context match.
    # ------------------------------------------------------------------
    print_scores(
        '"horse" (UNSEEN N) in N slot  (left=the  right=runs)  — UNSEEN word, SEEN context',
        tree,
        probe_instance(HORSE, left_1=THE, right_1=RUNS),
    )

    # ------------------------------------------------------------------
    # Probe 7 – "runs" (SEEN verb) in VP-head slot
    #   Left = NOUN, right absent (sentence-final intransitive VP)
    # ------------------------------------------------------------------
    print_scores(
        '"runs" in V slot  (left=cat  right=∅)  — SEEN, intransitive VP-final',
        tree,
        probe_instance(RUNS, left_1=CAT),
    )

    # ------------------------------------------------------------------
    # Probe 8 – "flies" (UNSEEN verb) in the same VP-head slot
    #   Same context pattern as "runs" but flies was never trained.
    # ------------------------------------------------------------------
    print_scores(
        '"flies" (UNSEEN V) in V slot  (left=cat  right=∅)  — UNSEEN word, SEEN context',
        tree,
        probe_instance(FLIES, left_1=CAT),
    )

    # ------------------------------------------------------------------
    # Probe 9 – "runs" queried in N slot (verb in noun position)
    #   Left = THE, right = RUNS (noun context, verb content)
    #   Should score LOW – context mismatch in content-ref.
    # ------------------------------------------------------------------
    print_scores(
        '"runs" (Verb) in N slot  (left=the  right=runs)  — content MISMATCH',
        tree,
        probe_instance(RUNS, left_1=THE, right_1=RUNS),
    )

    # ------------------------------------------------------------------
    # Probe 10 – "the" with 2-slot context: left=cat (noun after noun),
    #   very unusual position for a determiner.
    # ------------------------------------------------------------------
    print_scores(
        '"the" in weird position  (left=cat  right=dog)  — distributional MISMATCH',
        tree,
        probe_instance(THE, left_1=CAT, right_1=DOG),
    )

    # ------------------------------------------------------------------
    # HTML tree visualisation
    # ------------------------------------------------------------------
    _wname = lambda wid: WORD_NAMES.get(wid, str(wid))

    attr_names = [
        f"Left-{j+1}"  for j in range(CONTEXT_LENGTH)
    ] + [
        f"Right-{j+1}" for j in range(CONTEXT_LENGTH)
    ]

    drawer = HTMLCobwebDrawer(
        attributes=attr_names,
        id_to_value=[],
        value_to_id={},
        attr_value_fn={i: _wname for i in range(2 * CONTEXT_LENGTH)},
    )

    output_path = os.path.join(
        os.path.dirname(__file__), "output", "context_hierarchy_tree"
    )
    try:
        html_file, png_file = drawer.draw_tree(tree.root, output_path, max_depth=3)
        print(f"\nTree visualisation saved to: {html_file}")
    except Exception as exc:
        import json
        html_file = output_path + ".html"
        d3_json = json.dumps(drawer._node_to_dict(tree.root))
        html_str = drawer._build_html(d3_json)
        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"\nTree visualisation (HTML only) saved to: {html_file}")
        print(f"  (PNG skipped: {exc})")


if __name__ == "__main__":
    test_context_hierarchy()
