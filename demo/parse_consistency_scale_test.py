"""
Parse Consistency Test AT SCALE - We have a new goal!

Here's the plan:
*   Build various parse trees

"""

from util.cfg import generate, TEST_GRAMMAR1
from parse import LanguageChunkingParser
import time

# Creating and printing toy sentences
num_sentences = 1000
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser.load_state("data/grammar1_fullparse/ltm")

initial_sentence = "a woman saw the woman"
sub_sentence = "a woman saw"
new_sentence = "a woman saw the dog"
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
