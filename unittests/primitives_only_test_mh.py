"""
Primitive Learning Test (Multi-Hierarchy) - confirms the logic of learning
is completely functional using the two-hierarchy (content + context) framework
defined in parse_mh.py / MULTIHIERARCHY.md.

ONLY TESTS PRIMITIVES - currently used for sandboxing
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import WEBSTER
import shutil
import os
import random

OUT_DIR = "unittests/primitives_only_test_mh"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    
# Creating and printing toy sentences
CONTEXT_LENGTH = 3
CONTENT_LENGTH = 10

num_sentences = 300
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
    content_alpha=5e-4,
    context_alpha=5e-4,
    bow=False,
    categorization_mode='bfs_pmi' # can be dfs, bfs, or bfs_pmi
)

train_size = 0.96

train_documents = document[:int(len(document) * train_size)]
test_documents = document[int(len(document) * train_size):]

# Iterate through training documents and parse them one at a time
for i, doc in enumerate(train_documents):

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
