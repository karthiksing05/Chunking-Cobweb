"""
Vocab Expansion Test - A test to confirm whether expanding the grammar that our framework is exposed to
yields sustainable behavior: does adding new words with similar usage result in similar parses?

To conduct this experiment, we run the following steps:
*   Create a set of preliminary parses to guide future parses
*   Create a train set based on the initial grammar
*   Parse some amount of generated sentences
*   Create a test set based on the added vocabulary
    *   Create some sentences manually (based on parallels of fed parses)
    *   Generate some sentences with the grammar rules
    *   Make sure there are sentences in the end with entirely new vocabulary
*   Evaluate changes in old grammar and new grammar parsing - the prospective observation is that we see
    similarities between styles of content for parsing - nouns are parsed similarly before and after, and 
    *   Evaluate all partially new sentences first without adding, then add them, then evaluate completely
        new settings

We can evaluate this based on the grammar2 vocabularies constructed in the data directory.
*   A raw train doesn't end up working but can we generate a curated set of sentences with only one differing
    term and use those to our advantage?
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2, ADDED_CORPUS2, ADDED_GRAMMAR2
from parse import LanguageChunkingParser

parser = LanguageChunkingParser.load_state("data/grammar2_fullparse/ltm")

num_new_sents = 10
new_document = []

for _ in range(num_new_sents):
    sentence = generate("S", ADDED_GRAMMAR2)
    new_document.append(sentence)

## updating vocabulary with new content
for new_vocab in ADDED_CORPUS2:
    res = parser.add_to_vocab(new_vocab)

trees = parser.parse_input(new_document, end_behavior="converge", debug=False)

for i, tree in enumerate(trees):
    print(f"Generating TREE for Sentence {i}: " + new_document[i])
    tree.visualize(f"demo/vocab_expand_test/visualizations/parse_tree{i}")
    tree.to_json(f"demo/vocab_expand_test/json/parse_tree{i}.json")

print("Finished constructing new-vocab trees.")