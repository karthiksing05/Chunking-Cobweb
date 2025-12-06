"""
LTM Analysis!

Goal is to analyze the long term memory and corresponding subtrees from the intermediary levels.
*   We can start with basic-level analysis and see what we can usurp from basic-level subtrees and
    all inherently related subtrees!
*   The goal with this test is to notice AND and OR nodes - AND nodes are nodes where the content
    is uniquely defined and the context varies, and OR nodes are nodes where the context is uniquely
    and specifically defined and the content varies.

We're going to start by saving basic-level nodes (which are super relevant) and then also maybe save
any nodes and subtrees by AND-likeness and OR-likeness.
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser
import os
import shutil

# Creating and printing toy sentences
load_ltm = "data/grammar2_partialparse_ct3/ltm"
num_sentences = 40
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
if load_ltm != "":
    parser = LanguageChunkingParser.load_state(load_ltm)
else:
    parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=True)

parser.visualize_ltm("demo/ltm_analysis/ltm")

for i, doc in enumerate(document):
    parse_trees = parser.parse_input([doc], end_behavior="converge", debug=False)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

parser.save_state("demo/ltm_analysis/ltm_save")
parser.visualize_ltm("demo/ltm_analysis/ltm")

while not os.path.exists("demo/ltm_analysis/ltm.png"):
    pass

# calling the basic-level tree draw method!
parser.cobweb_drawer.save_basic_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/basic_level_subtrees", debug=True)
parser.cobweb_drawer.save_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/level_2_subtrees", level=2)
parser.cobweb_drawer.save_level_subtrees(parser.ltm_hierarchy.root, "demo/ltm_analysis/level_3_subtrees", level=3)

# root = parser.ltm_hierarchy.root
# folder = "and_or_node_subtrees"

# if os.path.exists(folder):
#     try:
#         shutil.rmtree(folder)
#     except OSError as e:
#         print(f"Error deleting folder '{folder}': {e}")

# visited = [root]

# and_nodes = {}
# or_nodes = {}
# num_and_trees = 0
# num_or_trees = 0

# while len(visited) > 0:
#     depth, curr = visited.pop()
#     num_nodes += 1

# for key, bl_node in and_nodes.items():
#     filename = ""
#     parser.cobweb_drawer.draw_tree(bl_node, folder + ("/" if folder[-1] != "/" else "") + filename, max_depth=3)

# print(f"Saved {num_and_trees} AND-trees and {num_or_trees} OR-trees.")