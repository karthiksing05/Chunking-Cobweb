"""
Primitive Learning Test (Multi-Hierarchy) - confirms the logic of learning
is completely functional using the two-hierarchy (content + context) framework
defined in parse_mh.py / MULTIHIERARCHY.md.

ONLY TESTS PRIMITIVES - currently used for sandboxing

This is the best settings right now but still failing in disambiguation so distributional context coming in clutch!!
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, TEST_GRAMMAR2, TEST_CORPUS2, POS_GRAMMAR1, POS_CORPUS1
from parse_mh import WEBSTER
import shutil
import os
import random

OUT_DIR = "unittests/primitives_only_test_mh"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    
# Creating and printing toy sentences
CONTEXT_LENGTH = 3
CONTENT_LENGTH = 5

num_sentences = 40
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

THRESHOLD = 1e9

# Setting up the multi-hierarchy parser (WEBSTER)
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    content_length=CONTENT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-3,
    context_alpha=1e-4,
    content_bl_alpha=1,
    context_bl_alpha=1,
    bow=False,
    chunk_context=False,
    empty_weighting=True,
    weighting="binary",
    categorization_mode='dfs', # can be dfs, bfs, or bfs_pmi
    depth_max_content=1000,
    depth_max_context=1000,
    branch_max_content=1000,
    branch_max_context=1000,
)

# Iterate through training documents and parse them one at a time
for i, doc in enumerate(document):

    # parse_sentence with learning=True adds to both hierarchies automatically
    parse_tree = webster.parse_sentence(
        doc,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=True,
        debug=True,
    )

    if i < 5:
        webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=4)

    elif i < 21 and i % 5 == 0:
        webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=3)

SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
webster.save_state(SAVE_DIR)
print(f"Saved Final LTM to \"{SAVE_DIR}\"!")
webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)

# --- Basic-level nodes ---
basic_nodes = webster.get_basic_level_nodes()
print("\n=== Basic-Level Nodes ===")
print("Content hierarchy:")
for h, node, freq in sorted(basic_nodes["content"], key=lambda x: x[2], reverse=True):
    print(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}")
print("Context hierarchy:")
for h, node, freq in sorted(basic_nodes["context"], key=lambda x: x[2], reverse=True):
    print(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}")

# --- Redistribution ---
print("\n=== Redistributing ===")
# print("Redistributing content hierarchy...")
# webster.ltm.content_hierarchy.redistribute(10000)
print("Redistributing context hierarchy...")
webster.ltm.context_hierarchy.redistribute(5000)
print("Redistribution complete!")

REDIST_SAVE_DIR = f"{OUT_DIR}/final_ltm_data_redistributed"
webster.save_state(REDIST_SAVE_DIR)
print(f"Saved Redistributed LTM to \"{REDIST_SAVE_DIR}\"!")
webster.visualize_ltm(f"{OUT_DIR}/final_ltm_redistributed", max_depth=3)

# --- Basic-level nodes ---
basic_nodes = webster.get_basic_level_nodes()
print("\n=== REDISTRIBUTED Basic-Level Nodes ===")
print("Content hierarchy:")
for h, node, freq in sorted(basic_nodes["content"], key=lambda x: x[2], reverse=True):
    print(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}")
print("Context hierarchy:")
for h, node, freq in sorted(basic_nodes["context"], key=lambda x: x[2], reverse=True):
    print(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}")
