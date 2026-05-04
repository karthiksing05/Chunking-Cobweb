"""
Brief basic-level test which I programmed with Chris's framework to show the basic level changing with time!!
"""


from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, TEST_GRAMMAR_LARGE, TEST_CORPUS_LARGE
from parse_mh import WEBSTER
import shutil
import os
import random
import sys
from contextlib import redirect_stdout, redirect_stderr

CURR_GRAMMAR = TEST_GRAMMAR_LARGE
CURR_CORP = TEST_CORPUS_LARGE

OUT_DIR = "tests/basic-level/basic-level-test"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# Creating and printing toy sentences
CONTEXT_LENGTH = 3

num_sentences = 40
document = []

for _ in range(num_sentences):
    sentence = generate("S", CURR_GRAMMAR)
    document.append(sentence)

THRESHOLD = 1e9

# Ensure output directory exists for the basic-level nodes log
OUT_TEXT = OUT_DIR + "/output.txt"
os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)

# Clear previous basic-level output file (we will append basic-level node info only)
open(OUT_TEXT, "w").close()

# Run two tests sequentially and print all normal output to stdout
for run_idx, ctx_alpha in enumerate([1e-4, 1], start=1):
    print("\n\n===== RUN {}: context_alpha={} =====\n".format(run_idx, ctx_alpha))
    with open(OUT_TEXT, "a") as out:
        out.write("\n\n===== RUN {}: context_alpha={} ".format(run_idx, ctx_alpha) + "=====" * 40 + "\n")
    # Setting up the multi-hierarchy parser (WEBSTER)
    webster = WEBSTER(
        CURR_CORP,
        context_length=CONTEXT_LENGTH,
        threshold=THRESHOLD,
        content_alpha=1e-3,
        context_alpha=ctx_alpha,
        # content_bl_alpha=1,
        # context_bl_alpha=10,
        bow=False,
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
        parse_tree = webster.parse_sentence(
            doc,
            threshold=THRESHOLD,
            new_vocab=True,
            learning=True,
            debug=False,
        )

    SAVE_DIR = f"{OUT_DIR}/final_ltm_data_alpha-{ctx_alpha}"
    webster.save_state(SAVE_DIR)
    print(f"Saved Final LTM to \"{SAVE_DIR}\"!")
    webster.visualize_ltm(f"{OUT_DIR}/final_ltm_alpha-{ctx_alpha}", max_depth=3)

    # --- Basic-level nodes ---
    basic_level_alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    for ctx_bl_alpha in basic_level_alphas:
        webster.get_long_term_memory().context_bl_alpha = ctx_bl_alpha
        basic_nodes = webster.get_basic_level_nodes()

        # Print to console as before
        print(f"\n=== Basic-Level Nodes for a={ctx_bl_alpha} ===")
        print("Context hierarchy (top 10 most frequent basic level nodes):")
        for h, node, freq in sorted(basic_nodes["context"], key=lambda x: x[2], reverse=True)[:10]:
            print(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}")

        # Also append the basic-level nodes info to the output file ONLY
        with open(OUT_TEXT, "a") as out:
            out.write(f"\n=== Basic-Level Nodes for a={ctx_bl_alpha} ===\n")
            out.write("Context hierarchy (top 10 most frequent basic level nodes):\n")
            for h, node, freq in sorted(basic_nodes["context"], key=lambda x: x[2], reverse=True)[:10]:
                out.write(f"  - {h}  count={node.count}  depth={node.depth()}  leaf_freq={freq}\n")
