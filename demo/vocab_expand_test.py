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
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2, ADDED_CORPUS2, ADDED_GRAMMAR2
from parse import LanguageChunkingParser

num_sentences = 50
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=True)

# TODO: OPTION TO COME BACK TO LATER - adding pregenerated parses, this is contingent on the LTM memory saving feature

for i, sentence in enumerate(document):
    parse_tree = parser.parse_input([sentence], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

## updating vocabulary with new content

for new_vocab in ADDED_CORPUS2:
    parser.add_to_vocab(new_vocab)

num_new_sents = 10
new_document = []

for _ in range(num_new_sents):
    sentence = generate("S", ADDED_GRAMMAR2)
    new_document.append(sentence)


