"""
Grammar Scaling Test - A test to evaluate how parsing scales at different granularities of grammars.

CFGs can be measured in terms of two separate factors: the number of potential relations and the number
of items in the vocabulary. Additionally, while there does exist a "size" property for CFGs which sums
both rules and symbols involved in rules, it'll be important to denote the differences in experimental
time complexity between having a lot of symbols and having a lot of rules. So, we'll implement two 
separate tests, which either fix the length of the vocabulary or fix the number of rules, and track
changes in two separate graphs.

For this implementation, we'll need to aptly summarize rules and 
*   To poll subsets of the vocabulary, we can randomly sample from the vocabulary
    *   Create some fixed grammar similar to the grammars and then sample a portion of the vocabulary
    *   Using 
*   To poll subsets of the grammar rules, we'll have to include different parts of speech incrementally
    *   To do this I think we're just going to have GPT construct different grammars of varying sizes and
        complexities but the same vocabulary length!

On each of these, the test should be some form of document analysis (do a timed test with 10 documents)
(parse and add documents and see how the line graphs change!)
"""

from util.cfg import generate
from parse import LanguageChunkingParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from copy import deepcopy

def obtain_corpus(grammar):
    terminal_keys = ["N", "V", "Det", "Adj", "Adv", "P", "RelPro"]
    return sum([sum(grammar.get(key, []), []) for key in terminal_keys], [])

base_vocab_grammar = {
    "S": [["NP", "VP"]],  # Sentence = Noun Phrase + Verb Phrase

    "NP": [["Det", "N"]],  # Noun Phrase = Determiner + Noun

    "VP": [["V", "NP"], ["V"]],  # Verb Phrase = Verb (+ optional NP)

    "Det": [["the"], ["a"]],
    
    # 50 nouns
    "N": [[noun] for noun in [
        "dog", "cat", "man", "woman", "car", "tree", "house", "bird", "child", "student",
        "teacher", "apple", "city", "river", "mountain", "computer", "phone", "book", "song", "flower",
        "friend", "family", "artist", "doctor", "lawyer", "engineer", "farmer", "player", "scientist", "driver",
        "nurse", "chef", "actor", "painter", "musician", "dancer", "writer", "poet", "athlete", "soldier",
        "neighbor", "parent", "student", "policeman", "pilot", "baker", "singer", "gardener", "fisherman", "photographer"
    ]],

    # 50 verbs
    "V": [[verb] for verb in [
        "runs", "sees", "likes", "chases", "eats", "drinks", "reads", "writes", "jumps", "walks",
        "talks", "sings", "dances", "listens", "looks", "builds", "paints", "drives", "teaches", "learns",
        "cooks", "helps", "finds", "makes", "fixes", "plays", "draws", "throws", "catches", "kicks",
        "opens", "closes", "starts", "stops", "moves", "watches", "thinks", "feels", "touches", "smells",
        "laughs", "cries", "jumps", "falls", "stands", "sits", "works", "sleeps", "reads", "writes"
    ]]
}

vocab_subset_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Generate grammars with reduced vocabularies
vocab_grammars = []

for size in vocab_subset_sizes:
    g = deepcopy(base_vocab_grammar)
    g["N"] = base_vocab_grammar["N"][:size]
    g["V"] = base_vocab_grammar["V"][:size]
    vocab_grammars.append(g)

# Five grammars of increasing complexity.
# Each grammar compounds all rules from prior grammars.
# Terminal vocabularies remain fixed at 5 words each.

complex_grammars = []

# -----------------------
# Grammar 1: Basic Sentence
# -----------------------
grammar1 = {
    "S": [["NP", "VP"]],
    "NP": [["Det", "N"]],
    "VP": [["V", "NP"], ["V"]],
    "Det": [["the"], ["a"], ["this"], ["that"], ["every"]],
    "N": [[n] for n in ["dog", "cat", "man", "woman", "child"]],
    "V": [[v] for v in ["runs", "sees", "likes", "chases", "finds"]],
}
complex_grammars.append(grammar1)

# -----------------------
# Grammar 2: Adds Adjective Phrase to NP
# -----------------------
grammar2 = dict(grammar1)  # Start from previous grammar
grammar2 = {k: [list(v) for v in val] for k, val in grammar2.items()}  # Deep copy

grammar2.update({
    "NP": [["Det", "AdjP", "N"], ["Det", "N"]],
    "AdjP": [["Adj"], ["Adv", "Adj"]],
    "Adj": [[a] for a in ["big", "small", "happy", "angry", "fast"]],
    "Adv": [[a] for a in ["very", "quite", "really", "so", "extremely"]],
})
complex_grammars.append(grammar2)

# -----------------------
# Grammar 3: Adds Prepositional Phrase (PP)
# -----------------------
grammar3 = dict(grammar2)
grammar3 = {k: [list(v) for v in val] for k, val in grammar3.items()}  # Deep copy

grammar3.update({
    "NP": [["Det", "AdjP", "N", "PP"], ["Det", "AdjP", "N"], ["Det", "N", "PP"], ["Det", "N"]],
    "VP": [["V", "NP"], ["V", "PP"], ["V"]],
    "PP": [["P", "NP"]],
    "P": [[p] for p in ["on", "in", "under", "with", "by"]],
})
complex_grammars.append(grammar3)

# -----------------------
# Grammar 4: Adds Adverb Phrase (AdvP) to VP
# -----------------------
grammar4 = dict(grammar3)
grammar4 = {k: [list(v) for v in val] for k, val in grammar4.items()}  # Deep copy

grammar4.update({
    "VP": [
        ["V", "NP", "AdvP"], ["V", "PP", "AdvP"], ["V", "NP"],
        ["V", "PP"], ["V", "AdvP"], ["V"]
    ],
    "AdvP": [["Adv"], ["Adv", "Adv"]],
})
complex_grammars.append(grammar4)

# -----------------------
# Grammar 5: Adds Relative Clause (RelClause)
# -----------------------
grammar5 = dict(grammar4)
grammar5 = {k: [list(v) for v in val] for k, val in grammar5.items()}  # Deep copy

grammar5.update({
    "NP": [
        ["Det", "AdjP", "N", "PP", "RelClause"], ["Det", "AdjP", "N", "RelClause"],
        ["Det", "N", "PP", "RelClause"], ["Det", "N", "RelClause"],
        ["Det", "AdjP", "N", "PP"], ["Det", "AdjP", "N"],
        ["Det", "N", "PP"], ["Det", "N"]
    ],
    "RelClause": [["RelPro", "VP"]],
    "RelPro": [[r] for r in ["who", "that", "which", "whom", "whose"]],
})
complex_grammars.append(grammar5)

corp_size = 50
vocab_documents = [
    [generate("S", grammar) for _ in range(corp_size)] for grammar in vocab_grammars
]

complex_documents = [
    [generate("S", grammar) for _ in range(corp_size)] for grammar in complex_grammars
]

vocab_datapoints = []

for i, doc_set in enumerate(vocab_documents):

    print(f"Evaluating Grammar with Vocab of size {vocab_subset_sizes[i]}")

    parser = LanguageChunkingParser(obtain_corpus(vocab_grammars[i]))

    start = time.time()
    
    for doc in doc_set:
        tree = parser.parse_input([doc])[0]
        parser.add_parse_tree(tree)

    end = time.time()

    vocab_datapoints.append((vocab_subset_sizes[i], end - start))

complex_datapoints = []

for i, doc_set in enumerate(complex_documents):

    print(f"Evaluating Grammar with Complexity of intensity {i}")

    parser = LanguageChunkingParser(obtain_corpus(complex_grammars[i]))

    start = time.time()

    for doc in doc_set:
        tree = parser.parse_input([doc])[0]
        parser.add_parse_tree(tree)

    end = time.time()

    complex_datapoints.append((i, end - start))

x, y = np.array(vocab_datapoints).T

def complexity_func(x, a, b):
    return a * x**b

popt, pcov = curve_fit(complexity_func, x, y)

a, b = popt
print(f"Fitted function: y = {a} * x^{b:.3f}")
print(f"→ Approximate time complexity: O(n^{b:.3f})")

x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 200)
y_fit = complexity_func(x_fit, *popt)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label=f'Fit: O(n^{b:.2f})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Vocab Size (n)')
plt.ylabel('Time (s)')
plt.title('Time Complexity of adjusting Vocabulary Size + Fit')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.savefig("demo/time_complexity_tests/grammar_vocab_scaling_test.png", dpi=300, bbox_inches='tight')

# plt.show()

x, y = np.array(complex_datapoints).T

def complexity_func(x, a, b):
    return a * x**b

popt, pcov = curve_fit(complexity_func, x, y)

a, b = popt
print(f"Fitted function: y = {a} * x^{b:.3f}")
print(f"→ Approximate time complexity: O(n^{b:.3f})")

x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 200)
y_fit = complexity_func(x_fit, *popt)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label=f'Fit: O(n^{b:.2f})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Complexity Factor (i)')
plt.ylabel('Time (s)')
plt.title('Time Complexity of adjusting grammar complexity + Fit')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.savefig("demo/time_complexity_tests/grammar_complexity_scaling_test.png", dpi=300, bbox_inches='tight')

# plt.show()