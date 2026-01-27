"""
General Learning Test - to confirm the logic of learning is completely functional!

THIS TEST HAS PRIMITIVES!!! We're going to arbitrarily build the primitive hierarchy first
and then have "epochs" of shit just to confirm that
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse import LanguageChunkingParser
import shutil
import os

if os.path.isdir("unittests/primitive_only_test"):
    shutil.rmtree("unittests/primitive_only_test")

# Creating and printing toy sentences
CONTEXT_LENGTH = 3

num_sentences = 60
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS1, context_length=CONTEXT_LENGTH, merge_split=True)

# Iterate through training documents and parse them one at a time, saving every 10th parse tree to file
for i, doc in enumerate(document):
    threshold = 0
    print("Threshold:", threshold)
    parse_trees = parser.parse_input([doc], end_behavior=threshold, debug=False)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

parser.visualize_ltm("unittests/primitive_only_test/final_ltm", max_depth=4)

SAVE_DIR = "unittests/primitive_only_test/final_ltm_data"
parser.save_state(SAVE_DIR)
print(f"Saved LTM to \"{SAVE_DIR}\"!")