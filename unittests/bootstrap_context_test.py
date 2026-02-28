"""
Bootstrap Context Hierarchy Test (Multi-Hierarchy) - confirms that
bootstrap_context_hierarchy clusters words by distributional similarity
(POS) before any parsing takes place.

Uses an ObservationBuffer to incrementally accumulate context for each
word type and flush (ifit) to the context hierarchy every N observations.
"""

from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse_mh import WEBSTER
import shutil
import os


# ========================================================================
# Test script
# ========================================================================

OUT_DIR = "unittests/bootstrap_context_test"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# Creating and printing toy sentences
CONTEXT_LENGTH = 3
CONTENT_LENGTH = 10
FLUSH_EVERY = 10   # observations per word before flushing

num_sentences = 100
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

THRESHOLD = 0

# Setting up the multi-hierarchy parser (WEBSTER)
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    content_length=CONTENT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-4,
    context_alpha=1e-6,
    bow=True,
    empty_weighting=True,
    weighting="binary",
    categorization_mode='bfs_pmi'  # can be dfs, bfs, or bfs_pmi
)

# Bootstrap via ObservationBuffer
print(f"Running ObservationBuffer bootstrap (flush_every={FLUSH_EVERY}) ...")
buf = webster.create_observation_buffer(flush_every=FLUSH_EVERY, debug=True)
for sent in document:
    buf.observe_sentence(sent)
buf.flush_all()  # flush any remaining partial buffers
print(f"Total flushes: {buf._total_flushes}\n")

# -- Print word clusters at a configurable depth -------------------------
PRINT_DEPTH = 2  # 1 = root's children, 2 = grandchildren, etc.
BASIC_N_SAMPLES = 1000
BASIC_MAX_NODES = 1000


def _get_leaf_words(node, id_to_value):
    """Collect all word names under a node by reading attr -1 (content-ref) from leaves."""
    words = []
    if not node.children:
        refs = (node.av_count or {}).get(-1, {})
        for vid in refs:
            if 0 <= vid < len(id_to_value):
                words.append(id_to_value[vid])
    else:
        for child in node.children:
            words.extend(_get_leaf_words(child, id_to_value))
    return words


def _collect_at_depth(node, target_depth, current_depth=0):
    """Return all nodes at exactly *target_depth* levels below *node*."""
    if current_depth == target_depth:
        return [node]
    nodes = []
    for child in (node.children or []):
        nodes.extend(_collect_at_depth(child, target_depth, current_depth + 1))
    return nodes


def _collect_all_leaves(node):
    """Return all leaf nodes under *node*."""
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(_collect_all_leaves(child))
    return leaves


root = webster.ltm.context_hierarchy.root
id_to_val = webster.ltm.id_to_value

# --- Depth-based clusters ---
nodes_at_depth = _collect_at_depth(root, PRINT_DEPTH)
print(f"=== Word clusters at depth {PRINT_DEPTH} ({len(nodes_at_depth)} nodes) ===")
for i, nd in enumerate(nodes_at_depth):
    words = _get_leaf_words(nd, id_to_val)
    print(f"  Cluster {i}: ({len(words)} words) {sorted(words)}")

# --- Basic-level clusters ---
# print(f"\n=== Basic-level clusters (n_samples={BASIC_N_SAMPLES}, max_nodes={BASIC_MAX_NODES}) ===")
# leaves = _collect_all_leaves(root)
# basic_level_nodes = {}  # concept_hash -> node (deduplicated)
# for leaf in leaves:
#     bl_node = leaf.get_basic(BASIC_N_SAMPLES, BASIC_MAX_NODES)
#     bl_hash = bl_node.concept_hash()
#     if bl_hash not in basic_level_nodes:
#         basic_level_nodes[bl_hash] = bl_node
#
# print(f"  {len(leaves)} leaves -> {len(basic_level_nodes)} basic-level nodes")
# for i, (bl_hash, bl_node) in enumerate(basic_level_nodes.items()):
#     words = _get_leaf_words(bl_node, id_to_val)
#     print(f"  Basic Cluster {i}: ({len(words)} words) {sorted(words)}")

# Visualize the bootstrapped context hierarchy
webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)
print(f"\nDone! Visualisation saved to {OUT_DIR}/final_ltm")
