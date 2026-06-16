"""
Boilerplate code for creating toy grammars with which to test our chunking algorithm.
"""

import random

# Define a simple context-free grammar using recursive structures (NO TERMINALS YET)
POS_GRAMMAR1 = {
    "S": [["NP", "VP"]],  # Sentence = Noun Phrase + Verb Phrase

    "NP": [
        ["Det", "AdjP", "N"],  # Noun Phrase = Determiner + Adjective Phrase + Noun
        ["Det", "N"]
    ],

    "AdjP": [
        ["Adj", "AdjP"],  # Adjective Phrase can recurse: Adj + AdjP
        ["Adj"],
        []  # Empty string to allow termination of recursion
    ],

    "VP": [
        ["V", "NP"],  # Verb Phrase = Verb + Noun Phrase
        ["V", "NP", "PP"],  # Verb Phrase with prepositional phrase
        ["V"]  # Simple verb
    ],

    "PP": [
        ["P", "NP"]  # Prepositional Phrase = Preposition + Noun Phrase
    ]
}

POS_CORPUS1 = ["Det", "N", "Adj", "V", "P"]

# MED grammar (TEST_GRAMMAR1). Every nonterminal expansion has
# exactly two right-hand-side elements, so derivation trees are
# strictly binary -- no post-hoc binarisation is required. PP
# attaches via the recursive S -> S PP production; AdjP is
# right-recursive and bottoms out at Adj+N.
TEST_GRAMMAR1 = {
    "S": [
        ["NP", "VP"],
        ["S", "PP"],
    ],

    "NP": [
        ["Det", "N"],
        ["Det", "AdjP"],
    ],

    "AdjP": [
        ["Adj", "AdjP"],
        ["Adj", "N"],
    ],

    "VP": [
        ["V", "NP"],
    ],

    "PP": [
        ["P", "NP"],
    ],

    "Det": [["the"], ["a"]],
    "N": [["cat"], ["dog"], ["man"], ["woman"], ["park"], ["telescope"]],
    "Adj": [["big"], ["small"], ["red"], ["quick"], ["lazy"]],
    "V": [["saw"], ["liked"], ["chased"], ["found"], ["admired"]],
    "P": [["with"], ["in"], ["on"], ["under"]]
}

TEST_CORPUS1 = (
    sum(TEST_GRAMMAR1["Det"], []) +
    sum(TEST_GRAMMAR1["N"], []) +
    sum(TEST_GRAMMAR1["Adj"], []) +
    sum(TEST_GRAMMAR1["V"], []) +
    sum(TEST_GRAMMAR1["P"], [])
)


# ── Terminal-sweep helper ─────────────────────────────────────────────
# Extended terminal pools for the lexicon-size variation experiment.
# Each pool overlaps with TEST_GRAMMAR1's terminals at the head so the
# "med" variant matches the canonical grammar exactly.
_TERMINAL_POOLS = {
    "Det": ["the", "a", "this", "that", "every", "some"],
    "N":   ["cat", "dog", "man", "woman", "park", "telescope",
            "robot", "apple", "book", "girl", "boy", "bird",
            "tree", "house", "child", "river"],
    "Adj": ["big", "small", "red", "quick", "lazy",
            "ancient", "blue", "tall", "wise", "friendly"],
    "V":   ["saw", "liked", "chased", "found", "admired",
            "carried", "read", "knew", "lost", "wanted"],
    "P":   ["with", "in", "on", "under", "near", "behind"],
}


def make_grammar_variant(lex_sizes):
    """Build a TEST_GRAMMAR1-shaped grammar with a varying lexicon.

    ``lex_sizes`` is a dict like ``{"Det": 2, "N": 6, "Adj": 5, "V": 5, "P": 4}``
    specifying how many terminal options each POS class has. Returns
    ``(grammar, corpus)`` ready to feed to the TRELLIS pipeline.

    The non-terminal productions (S, NP, VP, PP, AdjP) are kept
    identical to ``TEST_GRAMMAR1`` so structural complexity stays
    constant; only the lexicon size varies. This isolates the effect
    of lexical diversity on parser learning.
    """
    grammar = {
        "S":   list(TEST_GRAMMAR1["S"]),
        "NP":  list(TEST_GRAMMAR1["NP"]),
        "AdjP": list(TEST_GRAMMAR1["AdjP"]),
        "VP":  list(TEST_GRAMMAR1["VP"]),
        "PP":  list(TEST_GRAMMAR1["PP"]),
    }
    corpus = []
    for cls, n in lex_sizes.items():
        words = _TERMINAL_POOLS[cls][:n]
        if len(words) < n:
            raise ValueError(
                f"Pool for {cls} has {len(words)} words; requested {n}")
        grammar[cls] = [[w] for w in words]
        corpus.extend(words)
    return grammar, corpus


# Default variant catalogue for the terminal-sweep experiment.
# - ``low``  : 11 terminals — every POS class has just 2-3 words.
# - ``med``  : 22 terminals — matches TEST_GRAMMAR1 exactly.
# - ``high`` : 39 terminals — pool is doubled for maximum lexical
#              diversity (still grammatically equivalent).
LEXICON_VARIANTS = {
    "low":  {"Det": 2, "N": 3, "Adj": 2, "V": 2, "P": 2},
    "med":  {"Det": 2, "N": 6, "Adj": 5, "V": 5, "P": 4},
    "high": {"Det": 4, "N": 12, "Adj": 9, "V": 8, "P": 6},
}


# SMALL grammar (TEST_GRAMMAR2). Every nonterminal expansion has
# exactly two right-hand-side elements (strictly binary). The unary
# intransitive ``VP -> V`` production has been removed so every
# verb is treated as transitive.
TEST_GRAMMAR2 = {
    "S":   [["NP", "VP"]],
    "NP":  [["Det", "N"]],
    "VP":  [["V", "NP"]],
    "Det": [["the"], ["a"]],
    "N":   [["dog"], ["cat"], ["man"], ["woman"]],
    "V":   [["runs"], ["sees"], ["likes"], ["chases"]],
}

# Define a very simple grammar (no recursion, fewer rules)
ADDED_GRAMMAR2 = {
    "S": [["NP", "VP"], ["S", "Conj", "S"]], # Sentence can also coordinate two sentences

    "NP": [["Det", "N"]], # Noun Phrase = Determiner + Noun

    "VP": [["V", "NP"], ["V"]], # Verb Phrase = Verb (+ optional NP)

    "Det": [["the"], ["a"], ["an"]],
    "N": [["dog"], ["cat"], ["man"], ["woman"], ["pencil"], ["typewriter"], ["earring"], ["money"], ["light"], ["lock"]],
    "V": [["runs"], ["sees"], ["likes"], ["chases"], ["shows"], ["eats"], ["makes"], ["helps"], ["watches"]],
    "Conj": [["and"], ["or"]]
}

TEST_CORPUS2 = (
    sum(TEST_GRAMMAR2["Det"], []) +
    sum(TEST_GRAMMAR2["N"], []) +
    sum(TEST_GRAMMAR2["V"], [])
)

ADDED_CORPUS2 = sum(
    [["an"], # articles
     ["pencil"], ["typewriter"], ["earring"], ["money"], ["light"], ["lock"], # nouns
     ["shows"], ["eats"], ["makes"], ["helps"], ["watches"], # verbs
     ["and"], ["or"]], # conjunctions
    []
)

# Grammar with relative clauses and stacked adjectival phrases.
# NP is biased toward terminal expansions (3× weight on non-recursive
# productions) so RelClause recursion behaves like AdjP recursion in
# TEST_GRAMMAR1: a tail-heavy distribution that almost always
# terminates. Without this bias the uniform 50/50 choice between
# "no RelClause" and "with RelClause" leads to runaway expansion via
# RelClause → RelPro VP → V NP → ... → RelClause again, producing
# 20+ word sentences during _eval_omission that crippled grammar_large
# runtime in the May 28 chain.
# LARGE grammar (TEST_GRAMMAR3). Every nonterminal expansion has
# exactly two right-hand-side elements (strictly binary). NPs gain
# RelClause modification recursively via NP -> NP RelClause; PPs
# adjoin at the S level via S -> S PP (same convention as MED).
# The NP expansions are weighted to keep the terminal expansions
# dominant -- without the bias, RelClause + AdjP recursion together
# blow up the tail of sentence length.
TEST_GRAMMAR3 = {
    "S": [
        ["NP", "VP"],
        ["S", "PP"],
    ],

    "NP": [
        # Terminal expansions -- duplicated 6x so they dominate.
        ["Det", "N"], ["Det", "N"], ["Det", "N"],
        ["Det", "N"], ["Det", "N"], ["Det", "N"],
        ["Det", "AdjP"], ["Det", "AdjP"], ["Det", "AdjP"],
        ["Det", "AdjP"], ["Det", "AdjP"], ["Det", "AdjP"],
        # Recursive expansion -- single weight; ~7% per NP.
        ["NP", "RelClause"],
    ],

    # VP productions: strictly binary. Transitive (V NP) plus
    # intransitive-with-PP (V PP). The ternary V NP PP shape has
    # been removed; PPs now attach at the S level via S -> S PP.
    "VP": [
        ["V", "NP"],
        ["V", "PP"],
    ],

    "AdjP": [
        ["Adj", "AdjP"],
        ["Adj", "N"],
    ],

    "RelClause": [["RelPro", "VP"]],

    "PP": [["P", "NP"]],

    "Det": [["the"], ["a"], ["this"], ["that"]],
    "N": [["book"], ["boy"], ["girl"], ["teacher"], ["robot"], ["apple"]],
    "Adj": [["tall"], ["curious"], ["blue"], ["ancient"], ["friendly"]],
    "RelPro": [["who"], ["that"], ["which"]],
    "V": [["saw"], ["liked"], ["chased"], ["carried"], ["read"], ["admired"]],
    "P": [["with"], ["without"], ["near"]]
}

TEST_CORPUS3 = (
    sum(TEST_GRAMMAR3["Det"], []) +
    sum(TEST_GRAMMAR3["N"], []) +
    sum(TEST_GRAMMAR3["Adj"], []) +
    sum(TEST_GRAMMAR3["RelPro"], []) +
    sum(TEST_GRAMMAR3["V"], []) +
    sum(TEST_GRAMMAR3["P"], [])
)


# ── Canonical structural-complexity progression ────────────────────────
# SMALL  : minimal CFG — flat S, NP=Det N, VP=V (NP)
#          (TEST_GRAMMAR2). 4 nouns, 4 verbs, 2 dets, no AdjP/PP.
# MED    : +AdjP recursion, +PP, +ditransitive VP→V NP PP
#          (TEST_GRAMMAR1). The hand-annotated hollow corpus targets
#          this grammar, so it's the apples-to-apples baseline.
# LARGE  : +RelClause modifier on NP, +RelPro POS class
#          (TEST_GRAMMAR3). Same lexicon size as MED but with embedded
#          sentence-like structures inside NPs.
#
# The old numeric names (TEST_GRAMMAR1/2/3) remain as aliases so all
# existing scripts and tests continue to work without modification.
TEST_GRAMMAR_SMALL = TEST_GRAMMAR2
TEST_CORPUS_SMALL  = TEST_CORPUS2
TEST_GRAMMAR_MED   = TEST_GRAMMAR1
TEST_CORPUS_MED    = TEST_CORPUS1
TEST_GRAMMAR_LARGE = TEST_GRAMMAR3
TEST_CORPUS_LARGE  = TEST_CORPUS3


# Define a simple context-free grammar using recursive structures
TEST_EPOCH_GRAMMAR_EPOCH_1 = {
    "S": [["NP"]],  # Sentence = Noun Phrase + Verb Phrase

    "NP": [
        ["Det", "AdjP", "N"],  # Noun Phrase = Determiner + Adjective Phrase + Noun
        ["Det", "N"]
    ],

    "AdjP": [
        ["Adj", "AdjP"],  # Adjective Phrase can recurse: Adj + AdjP
        ["Adj"],
        []  # Empty string to allow termination of recursion
    ],

    "Det": [["the"], ["a"]],
    "N": [["cat"], ["dog"], ["man"], ["woman"], ["park"], ["telescope"]],
    "Adj": [["big"], ["small"], ["red"], ["quick"], ["lazy"]]
}

# Define a simple context-free grammar using recursive structures
TEST_EPOCH_GRAMMAR_EPOCH_2 = {
    "S": [["NP", "VP"]],  # Sentence = Noun Phrase + Verb Phrase

    "NP": [
        ["Det", "AdjP", "N"],  # Noun Phrase = Determiner + Adjective Phrase + Noun
        ["Det", "N"]
    ],

    "AdjP": [
        ["Adj", "AdjP"],  # Adjective Phrase can recurse: Adj + AdjP
        ["Adj"],
        []  # Empty string to allow termination of recursion
    ],

    "VP": [
        ["V", "NP"],  # Verb Phrase = Verb + Noun Phrase
        ["V", "NP", "PP"],  # Verb Phrase with prepositional phrase
        ["V"]  # Simple verb
    ],

    "PP": [
        ["P", "NP"]  # Prepositional Phrase = Preposition + Noun Phrase
    ],

    "Det": [["the"], ["a"]],
    "N": [["cat"], ["dog"], ["man"], ["woman"], ["park"], ["telescope"]],
    "Adj": [["big"], ["small"], ["red"], ["quick"], ["lazy"]],
    "V": [["saw"], ["liked"], ["chased"], ["found"], ["admired"]],
    "P": [["with"], ["in"], ["on"], ["under"]]
}

TEST_EPOCH_CORPUS = (
    sum(TEST_GRAMMAR1["Det"], []) +
    sum(TEST_GRAMMAR1["N"], []) +
    sum(TEST_GRAMMAR1["Adj"], []) +
    sum(TEST_GRAMMAR1["V"], []) +
    sum(TEST_GRAMMAR1["P"], [])
)


# ── Huge grammar with rich syntax and broad vocabulary ─────────────────
TEST_GRAMMAR_HUGE = {
    # Top-level sentence types
    "S": [
        ["NP", "VP"],
        ["S", "Conj", "S"],          # coordination
        ["AdvP", "NP", "VP"],         # fronted adverb
    ],

    # Noun phrases
    "NP": [
        ["Det", "N"],
        ["Det", "AdjP", "N"],
        ["Det", "N", "PP"],
        ["Det", "AdjP", "N", "PP"],
        ["Det", "N", "RelClause"],
        ["Det", "AdjP", "N", "RelClause"],
        ["ProperN"],                  # bare proper noun
        ["Pron"],                     # pronoun
    ],

    # Adjective phrases (recursive stacking)
    "AdjP": [
        ["Adj"],
        ["Adj", "AdjP"],
        ["AdvDeg", "Adj"],            # degree adverb + adjective ("very big")
    ],

    # Adverb phrase (sentence-level)
    "AdvP": [
        ["Adv"],
        ["AdvDeg", "Adv"],            # "very quickly"
    ],

    # Verb phrases
    "VP": [
        ["Vi"],                       # intransitive
        ["Vi", "AdvP"],               # intransitive + adverb
        ["Vt", "NP"],                 # transitive
        ["Vt", "NP", "AdvP"],         # transitive + adverb
        ["Vt", "NP", "PP"],           # transitive + PP
        ["Vt", "NP", "PP", "PP"],     # double PP attachment
        ["Vdt", "NP", "NP"],          # ditransitive ("gave the dog a bone")
        ["Vc", "AdjP"],              # copula ("is tall")
        ["Vc", "NP"],                # copula + NP ("is a teacher")
        ["Vt", "Comp"],              # sentential complement ("thinks that ...")
    ],

    # Prepositional phrases
    "PP": [
        ["P", "NP"],
    ],

    # Relative clauses
    "RelClause": [
        ["RelPro", "VP"],             # "who runs"
        ["RelPro", "NP", "Vt"],       # "that the dog chased"  (object-gap)
    ],

    # Complement clause
    "Comp": [
        ["CompC", "NP", "VP"],        # "that the cat ran"
    ],

    # ── Terminals ────────────────────────────────────────────────────────

    "Det": [["the"], ["a"], ["an"], ["this"], ["that"], ["every"],
            ["some"], ["each"], ["no"], ["my"], ["her"], ["his"]],

    "N": [["cat"], ["dog"], ["man"], ["woman"], ["child"], ["teacher"],
          ["student"], ["robot"], ["doctor"], ["artist"], ["book"], ["apple"],
          ["table"], ["car"], ["house"], ["tree"], ["river"], ["mountain"],
          ["city"], ["garden"], ["bridge"], ["letter"], ["window"], ["door"],
          ["phone"], ["camera"], ["ticket"], ["lamp"], ["clock"], ["key"]],

    "ProperN": [["Alice"], ["Bob"], ["Charlie"], ["Diana"], ["Eve"],
                ["Frank"], ["Grace"], ["Henry"]],

    "Pron": [["he"], ["she"], ["it"], ["they"], ["someone"], ["everyone"]],

    "Adj": [["big"], ["small"], ["red"], ["blue"], ["green"], ["old"],
            ["young"], ["tall"], ["short"], ["bright"], ["dark"], ["quiet"],
            ["loud"], ["fast"], ["slow"], ["heavy"], ["light"], ["clean"],
            ["dirty"], ["sharp"], ["soft"], ["warm"], ["cold"], ["ancient"],
            ["curious"], ["friendly"], ["angry"], ["brave"], ["gentle"], ["wise"]],

    "AdvDeg": [["very"], ["quite"], ["extremely"], ["rather"], ["somewhat"]],

    "Adv": [["quickly"], ["slowly"], ["carefully"], ["eagerly"], ["quietly"],
            ["loudly"], ["gently"], ["suddenly"], ["often"], ["never"],
            ["always"], ["sometimes"], ["rarely"], ["happily"], ["sadly"]],

    "Vi": [["ran"], ["slept"], ["arrived"], ["laughed"], ["cried"],
           ["waited"], ["vanished"], ["appeared"], ["fell"], ["jumped"]],

    "Vt": [["saw"], ["liked"], ["chased"], ["found"], ["admired"],
           ["carried"], ["read"], ["built"], ["opened"], ["watched"],
           ["painted"], ["loved"], ["heard"], ["followed"], ["caught"]],

    "Vdt": [["gave"], ["showed"], ["sent"], ["offered"], ["handed"],
            ["told"]],

    "Vc": [["is"], ["was"], ["seems"], ["became"], ["remains"]],

    "P": [["with"], ["in"], ["on"], ["under"], ["near"], ["behind"],
          ["beside"], ["above"], ["below"], ["across"], ["through"],
          ["around"], ["between"], ["among"], ["without"]],

    "RelPro": [["who"], ["that"], ["which"]],

    "CompC": [["that"]],

    "Conj": [["and"], ["but"], ["or"], ["yet"]],
}

TEST_CORPUS_HUGE = (
    sum(TEST_GRAMMAR_HUGE["Det"], []) +
    sum(TEST_GRAMMAR_HUGE["N"], []) +
    sum(TEST_GRAMMAR_HUGE["ProperN"], []) +
    sum(TEST_GRAMMAR_HUGE["Pron"], []) +
    sum(TEST_GRAMMAR_HUGE["Adj"], []) +
    sum(TEST_GRAMMAR_HUGE["AdvDeg"], []) +
    sum(TEST_GRAMMAR_HUGE["Adv"], []) +
    sum(TEST_GRAMMAR_HUGE["Vi"], []) +
    sum(TEST_GRAMMAR_HUGE["Vt"], []) +
    sum(TEST_GRAMMAR_HUGE["Vdt"], []) +
    sum(TEST_GRAMMAR_HUGE["Vc"], []) +
    sum(TEST_GRAMMAR_HUGE["P"], []) +
    sum(TEST_GRAMMAR_HUGE["RelPro"], []) +
    sum(TEST_GRAMMAR_HUGE["CompC"], []) +
    sum(TEST_GRAMMAR_HUGE["Conj"], [])
)


def generate_with_merges(symbol, grammar, flatten_at_parent=()):
    """Generate a sentence AND the hollow-style merge sequence,
    matching the hand-annotated convention in
    ``data/test_hollow_grammar_1/``.

    The hand-annotated hollow corpus follows three rules that this
    function reproduces exactly:

      1. **Inside an NP, PP, or AdjP**: right-binarize. So
         ``a small woman`` becomes ``(a (small woman))`` — the
         intermediate ``(small woman)`` is a grammatical noun-group.
         (Compare to left-binarization which would yield the bogus
         intermediate ``(a small)``.)

      2. **VP, RelClause, …**: **FLATTEN** at the parent level.
         Hollow has no ``(V NP)`` bracket for a transitive VP —
         instead, V and NP are merged INDIVIDUALLY into the running
         S-chunk. The set ``flatten_at_parent`` controls which
         non-terminals are flattened.

      3. **S** (the top level): **left-incremental** merge over the
         flattened sequence of children. So ``S = NP V NP PP`` is
         merged ``((NP V) NP) PP`` — picking up tokens left-to-right.
         This matches how a human reader processes a sentence
         incrementally.

    Together these rules produce bracket sets that match the
    hand-annotated hollow corpus for the same sentence (verified on
    e.g. ``a park admired the cat`` → both produce the same
    ``{(0,1), (0,2), (3,4), (0,4)}``). Parsers trained on these
    synthetic trees achieve hollow-comparable F1 because the
    intermediate constituents are grammatically meaningful, not
    binarization-artefacts.

    ``flatten_at_parent`` should be tuple of non-terminal names that
    should NOT form their own constituent bracket — their children
    bubble up to the parent's level. Defaults to ``("VP", "RelClause")``,
    correct for TEST_GRAMMAR1 and TEST_GRAMMAR3.

    Returns ``(sentence_str, merges_list)``.
    """
    text, merges, units, _n = _generate_with_pos(
        symbol, grammar, 0, 0, set(flatten_at_parent), is_top=True)
    return text, merges


def _generate_with_pos(symbol, grammar, start_idx, _adjp_depth,
                       flatten_set, is_top=False):
    """Recursive helper. Returns ``(text, merges, units, n_leaves)``:

      • ``text``    — joined surface form of this expansion.
      • ``merges``  — list of {"left", "right"} merge dicts for THIS
                       subtree (in valid apply-order).
      • ``units``   — list of ``(center, n_leaves)`` tuples representing
                       this symbol's "structural units" at the parent
                       level. If ``symbol`` is in ``flatten_set``, this
                       is the flat list of its CHILDREN's units
                       (pass-through). Otherwise this is a single-element
                       list with this symbol's own center.
      • ``n_leaves``— number of leaf tokens.
    """
    if symbol not in grammar:
        # Terminal: occupies one leaf position; no merges.
        return symbol, [], [(float(start_idx), 1)], 1

    # ── Production selection — mirrors generate() ────────────────
    if symbol == "AdjP":
        productions = grammar[symbol]
        recursive   = [p for p in productions if "AdjP" in p]
        terminating = [p for p in productions if "AdjP" not in p]
        non_empty_term = [p for p in terminating if p]
        if non_empty_term:
            terminating = non_empty_term
        p_continue = (1.0 / 3.0) ** _adjp_depth
        if recursive and (not terminating or random.random() < p_continue):
            production = random.choice(recursive)
            next_depth = _adjp_depth + 1
        elif terminating:
            production = random.choice(terminating)
            next_depth = _adjp_depth
        else:
            production = random.choice(productions)
            next_depth = _adjp_depth
    else:
        production = random.choice(grammar[symbol])
        next_depth = _adjp_depth

    # ── Recursively expand children, accumulating their units ───
    merges = []
    all_units = []
    cursor = start_idx
    child_texts = []
    for sym in production:
        adjp_d = next_depth if sym == "AdjP" else 0
        t, m, units, n = _generate_with_pos(
            sym, grammar, cursor, adjp_d, flatten_set)
        if n == 0:
            continue
        merges.extend(m)
        cursor += n
        child_texts.append(t)
        all_units.extend(units)

    n_leaves = cursor - start_idx
    if n_leaves == 0:
        return "", merges, [], 0
    text = " ".join(child_texts)

    # If THIS symbol is in flatten_set, don't form its own bracket —
    # just pass the units up to the parent.
    if symbol in flatten_set and not is_top:
        return text, merges, all_units, n_leaves

    # Single unit — pass through unchanged.
    if len(all_units) == 1:
        return text, merges, all_units, n_leaves

    # ── Binarize ──
    # - S (top level): LEFT-INCREMENTAL (NP + V → (NP V) + NP → ...).
    # - Everything else: RIGHT-BINARY ((AdjP N) → (Det (AdjP N))).
    if is_top or symbol == "S":
        cur = all_units[0][0]
        for nxt_c, _ in all_units[1:]:
            merges.append({"left": cur, "right": nxt_c})
            cur = (cur + nxt_c) / 2.0
    else:
        cur = all_units[-1][0]
        for prev_c, _ in reversed(all_units[:-1]):
            merges.append({"left": prev_c, "right": cur})
            cur = (prev_c + cur) / 2.0

    return text, merges, [(cur, n_leaves)], n_leaves


def generate(symbol, grammar, _adjp_depth: int = 0):
    """Recursively generate a sentence from the grammar starting with a symbol.

    AdjP recursion uses a multiplicatively-decaying continuation
    probability: ``P(continue) = (1/3)**depth`` where ``depth`` is
    the number of AdjP recursions already taken. So:
      depth 0 → p=1     (always go to a 2nd adjective)
      depth 1 → p=1/3
      depth 2 → p=1/9
      depth 3 → p=1/27
      ...

    Geometric decay rather than a hard cap — 5+ adjective AdjPs are
    possible but rare (~3.7% chance of 5+ adj when AdjP is invoked,
    ~1.2% for 6+).

    The empty AdjP terminating branch (``[]``) is also skipped when
    non-empty options exist, so every AdjP invocation produces at
    least one adjective.
    """
    if symbol not in grammar:
        return symbol  # Terminal symbol

    # Depth-aware override for AdjP: split productions into recursive
    # (those that re-invoke AdjP) and terminating, then pick a recursive
    # production with the depth-decayed probability.
    if symbol == "AdjP":
        productions = grammar[symbol]
        recursive   = [p for p in productions if "AdjP" in p]
        terminating = [p for p in productions if "AdjP" not in p]
        # Prefer non-empty terminating productions so AdjP always
        # yields at least one adjective when it's invoked.
        non_empty_term = [p for p in terminating if p]
        if non_empty_term:
            terminating = non_empty_term

        p_continue = (1.0 / 3.0) ** _adjp_depth
        if recursive and (not terminating or random.random() < p_continue):
            production = random.choice(recursive)
            next_depth = _adjp_depth + 1
        elif terminating:
            production = random.choice(terminating)
            next_depth = _adjp_depth
        else:
            production = random.choice(productions)
            next_depth = _adjp_depth
        result = []
        for sym in production:
            if sym == "AdjP":
                result.append(generate(sym, grammar, next_depth))
            else:
                result.append(generate(sym, grammar))
        return " ".join(filter(None, result))

    production = random.choice(grammar[symbol])  # Choose one production rule
    result = []

    for sym in production:
        result.append(generate(sym, grammar))  # Recursively expand each symbol

    final_sent = " ".join(filter(None, result))  # Join and remove empty strings

    return final_sent
