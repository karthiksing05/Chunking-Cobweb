"""
Parse Consistency Test - essentially, the idea is that we want to make sure that our
parse trees yield consistency when being parsed in between other tests

First and foremost, we're testing two different things:
*   Does a parse work to contribute a subparse? I.e. does a subparse mimic the
    style of an actual parse?
*   Do parses and subparses maintain consistency over large scales? I.e. does the
    presence of new data overwrite biases towards prior parses

Quite fortunately, we see rewarding results in favor of our chunking framework for
the above two questions. We see not only the presence of robustness against catastrophic 
forgetting, but we also see the consistency between different parses.
*   However, this seems to fail on larger grammars with not enough properly generated
    concepts / vocabulary
*   My current leading hypothesis is that if an error is ever discovered, it is magnified
    into a crazy proportion.
"""

from util.cfg import generate, TEST_GRAMMAR2
from parse import LanguageChunkingParser
import time

# Creating and printing toy sentences
num_sentences = 200
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser.load_state("unittests/gen_learn_test/final_ltm_data")

initial_sentence = "a woman sees the woman"
sub_sentence = "a woman sees"
new_sentence = "a woman sees the dog"
initial_tree = parser.parse_input([initial_sentence], end_behavior="converge", debug=False)[0]
parser.add_parse_tree(initial_tree, debug=False)
initial_tree.visualize("demo/parse_consistency_test/initial_parse")

start = time.time()

for i, sentence in enumerate(document):
    parse_tree = parser.parse_input([sentence], end_behavior="converge", debug=False)[0]
    parser.add_parse_tree(parse_tree, debug=False)

end = time.time()

print(f"Time taken to add {num_sentences} to LTM: {end - start} seconds.")

final_tree = parser.parse_input([initial_sentence], end_behavior="converge", debug=False)[0]
# parser.add_parse_tree(final_tree, debug=False)
final_tree.visualize("demo/parse_consistency_test/final_parse")

subparse_tree = parser.parse_input([sub_sentence], end_behavior="converge", debug=False)[0]
# parser.add_parse_tree(subparse_tree, debug=False)
subparse_tree.visualize("demo/parse_consistency_test/sub_parse")

new_parse_tree = parser.parse_input([new_sentence], end_behavior="converge", debug=False)[0]
# parser.add_parse_tree(new_parse_tree, debug=False)
new_parse_tree.visualize("demo/parse_consistency_test/new_parse")
