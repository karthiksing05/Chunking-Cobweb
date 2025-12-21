"""
Boilerplate code for creating toy grammars with which to test our chunking algorithm.
"""

import random

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

def generate(symbol, grammar):
    """Recursively generate a sentence from the grammar starting with a symbol."""
    if symbol not in grammar:
        return symbol  # Terminal symbol

    production = random.choice(grammar[symbol])  # Choose one production rule
    result = []

    for sym in production:
        result.append(generate(sym, grammar))  # Recursively expand each symbol

    final_sent = " ".join(filter(None, result))  # Join and remove empty strings

    return final_sent
