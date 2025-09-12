"""
LTM Analysis!

Goal is to analyze the long term memory and corresponding subtrees from the intermediary levels.
*   We can start with basic-level analysis and see what we can usurp from basic-level subtrees and
    all inherently related subtrees!
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser

# Creating and printing toy sentences
num_sentences = 200
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=False)

for i, doc in enumerate(document):
    parse_trees = parser.parse_input([doc], end_behavior="converge", debug=False)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

# calling the basic-level tree draw method!
# parser.visualize_ltm("tests/ltm_analysis/ltm")
parser.cobweb_drawer.save_basic_level_subtrees(parser.ltm_hierarchy.root, "tests/ltm_analysis/basic2_subtrees", level=2)
parser.cobweb_drawer.save_basic_level_subtrees(parser.ltm_hierarchy.root, "tests/ltm_analysis/basic3_subtrees", level=3)