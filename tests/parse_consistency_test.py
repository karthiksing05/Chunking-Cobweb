"""
Parse Consistency Test - essentially, the idea is that we want to make sure that our
parse trees yield consistency when being parsed in between 

For this test, we'll run it according to the following loop:
*   First and foremost, we're testing two different things:
    *   Does a parse work within a subparse
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser

# Creating and printing toy sentences
num_sentences = 50
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=True)

initial_sentence = "the man chases the cat"
sub_sentence = "the man chases"

for i, doc in enumerate(document[:int(len(document) / 2)]):
    parse_tree = parser.parse_input([doc], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

initial_tree = parser.parse_input([initial_sentence], end_behavior="converge", debug=False)[0]
parser.add_parse_tree(initial_tree, debug=False)
initial_tree.visualize("tests/parse_consistency_test/initial_tree")

for i, doc in enumerate(document[int(len(document) / 2):]):
    parse_tree = parser.parse_input([doc], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

final_tree = parser.parse_input([initial_sentence], end_behavior="converge", debug=False)[0]
parser.add_parse_tree(final_tree, debug=False)
final_tree.visualize("tests/parse_consistency_test/final_tree")

subparse_tree = parser.parse_input([sub_sentence], end_behavior="converge", debug=False)[0]
parser.add_parse_tree(subparse_tree, debug=False)
subparse_tree.visualize("tests/parse_consistency_test/subparse_tree")