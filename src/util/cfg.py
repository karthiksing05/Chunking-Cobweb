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


TEST_CORPUS2 = (
    sum(TEST_GRAMMAR2["Det"], []) +
    sum(TEST_GRAMMAR2["N"], []) +
    sum(TEST_GRAMMAR2["V"], [])
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
