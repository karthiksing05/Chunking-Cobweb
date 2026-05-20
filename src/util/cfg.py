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

# Define a simple context-free grammar using recursive structures
TEST_GRAMMAR1 = {
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

TEST_CORPUS1 = (
    sum(TEST_GRAMMAR1["Det"], []) +
    sum(TEST_GRAMMAR1["N"], []) +
    sum(TEST_GRAMMAR1["Adj"], []) +
    sum(TEST_GRAMMAR1["V"], []) +
    sum(TEST_GRAMMAR1["P"], [])
)


# Define a very simple grammar (no recursion, fewer rules)
TEST_GRAMMAR2 = {
    "S": [["NP", "VP"]], # Sentence = Noun Phrase + Verb Phrase

    "NP": [["Det", "N"]], # Noun Phrase = Determiner + Noun

    "VP": [["V", "NP"], ["V"]], # Verb Phrase = Verb (+ optional NP)

    "Det": [["the"], ["a"]],
    "N": [["dog"], ["cat"], ["man"], ["woman"]],
    "V": [["runs"], ["sees"], ["likes"], ["chases"]]
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

# Grammar with relative clauses and stacked adjectival phrases
TEST_GRAMMAR3 = {
    "S": [["NP", "VP"]],

    "NP": [
        ["Det", "N"],
        ["Det", "AdjP", "N"],
        ["Det", "N", "RelClause"],
        ["Det", "AdjP", "N", "RelClause"]
    ],

    "VP": [["V", "NP"], ["V"], ["V", "PP"]],

    "AdjP": [["Adj"], ["Adj", "AdjP"]],

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


# ── Large grammar with rich syntax and broad vocabulary ─────────────────
TEST_GRAMMAR_LARGE = {
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

TEST_CORPUS_LARGE = (
    sum(TEST_GRAMMAR_LARGE["Det"], []) +
    sum(TEST_GRAMMAR_LARGE["N"], []) +
    sum(TEST_GRAMMAR_LARGE["ProperN"], []) +
    sum(TEST_GRAMMAR_LARGE["Pron"], []) +
    sum(TEST_GRAMMAR_LARGE["Adj"], []) +
    sum(TEST_GRAMMAR_LARGE["AdvDeg"], []) +
    sum(TEST_GRAMMAR_LARGE["Adv"], []) +
    sum(TEST_GRAMMAR_LARGE["Vi"], []) +
    sum(TEST_GRAMMAR_LARGE["Vt"], []) +
    sum(TEST_GRAMMAR_LARGE["Vdt"], []) +
    sum(TEST_GRAMMAR_LARGE["Vc"], []) +
    sum(TEST_GRAMMAR_LARGE["P"], []) +
    sum(TEST_GRAMMAR_LARGE["RelPro"], []) +
    sum(TEST_GRAMMAR_LARGE["CompC"], []) +
    sum(TEST_GRAMMAR_LARGE["Conj"], [])
)


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
