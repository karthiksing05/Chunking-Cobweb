"""
LTM Analysis!

Goal is to analyze the long term memory and corresponding subtrees from the intermediary levels.
*   We can start with basic-level analysis and see what we can usurp from basic-level subtrees and
    all inherently related subtrees!
*   The goal with this test is to notice AND and OR nodes - AND nodes are nodes where the content
    is uniquely defined and the context varies, and OR nodes are nodes where the context is uniquely
    and specifically defined and the content varies.
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser

# Creating and printing toy sentences
load_ltm = ""
num_sentences = 51
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
if load_ltm != "":
    parser = LanguageChunkingParser.load_state(load_ltm)
else:
    parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=False)

for i, doc in enumerate(document):
    parse_trees = parser.parse_input([doc], end_behavior="converge", debug=False)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

    if i == 50:
        parser.visualize_ltm("demo/ltm_analysis/ltm")
        parser.save_state("demo/ltm_analysis/ltm_save")

# calling the basic-level tree draw method!
parser.cobweb_drawer.save_basic_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/basic_level_subtrees", debug=True)
parser.cobweb_drawer.save_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/level_2_subtrees", level=2)
parser.cobweb_drawer.save_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/level_3_subtrees", level=3)
