"""
General Learning Test (Multi-Hierarchy) - confirms the logic of learning
is completely functional using the two-hierarchy (content + context) framework
defined in parse_mh.py / MULTIHIERARCHY.md.
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import WEBSTER
import shutil
import os
import random

OUT_DIR = "unittests/gen_learn_test_mh"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# Creating and printing toy sentences
CONTEXT_LENGTH = 3
CONTENT_LENGTH = 10

num_sentences = 100
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

THRESHOLD = -5.0

# Setting up the multi-hierarchy parser (WEBSTER)
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    content_length=CONTENT_LENGTH,
    threshold=THRESHOLD,
    alpha=1e-4
)

train_size = 0.90

PRIMITIVES_FIRST = 0

train_documents = document[:int(len(document) * train_size)]
test_documents = document[int(len(document) * train_size):]

# Iterate through training documents and parse them one at a time
for i, doc in enumerate(train_documents):
    threshold = (0 if i < PRIMITIVES_FIRST else THRESHOLD)  # should never trigger atp
    print("Threshold:", threshold)

    # parse_sentence with learning=True adds to both hierarchies automatically
    parse_tree = webster.parse_sentence(
        doc,
        threshold=threshold,
        new_vocab=True,
        learning=True,
        debug=True,
    )

    if i < 5:
        webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=4)

    if i % 5 == 0:
        parse_tree.visualize(f"{OUT_DIR}/train_trees/train_parse_tree{i}")

        if i < 21:
            webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=4)

webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=4)

# Visualize the test parse trees
for i, test in enumerate(test_documents):
    parse_tree = webster.parse_sentence(
        test,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=False,
    )
    parse_tree.visualize(f"{OUT_DIR}/test_trees/test_parse_tree{i}")
    print(f"Created parse tree {i} for sentence, \"{test}\"")

# Creating fake sentences, through completely random choice, to see if they parse!
fake_sentences = [
    " ".join([random.choice(TEST_CORPUS1) for _ in range(random.randint(3, 8))])
    for _ in range(10)
]
fake_sentences.append("the dog the dog")

for i, fake_sentence in enumerate(fake_sentences):
    parse_tree = webster.parse_sentence(
        fake_sentence,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=False,
    )
    parse_tree.visualize(f"{OUT_DIR}/fake_trees/fake_parse_tree{i}")
    print(f"Created fake parse tree for fake sentence, \"{fake_sentence}\"")

# Generating complete sentences from scratch
print("\n--- FROM-SCRATCH GENERATION ---")
gen_results_path = f"{OUT_DIR}/generated_sentences.txt"
os.makedirs(os.path.dirname(gen_results_path), exist_ok=True)
with open(gen_results_path, "w") as gen_f:
    for i in range(10):
        sentence, parse = webster.generate_sentence(debug=True)
        print(f"Generated sentence [{i}]: \"{sentence}\"")
        gen_f.write(f"[{i}] {sentence}\n")
        if parse and hasattr(parse, 'visualize'):
            parse.visualize(f"{OUT_DIR}/generated_trees/generated_parse_tree{i}")
print(f"Saved generated sentences to \"{gen_results_path}\"")

# Masked completion (random single-token mask)
print("\n--- MASKED COMPLETION (single token) ---")
for i in range(min(5, len(test_documents))):
    tokens = test_documents[i].split()
    if len(tokens) > 2:
        mask_idx = random.randint(1, len(tokens) - 1)
        tokens[mask_idx] = '[mask]'
    masked = ' '.join(tokens)
    print(f"  Masked input: \"{masked}\"")
    completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
    print(f"  Completed:    \"{completed}\"\n")

# Masked prediction: keep first half, replace second half with a single [mask]
print("\n--- MASKED PREDICTION (expand second half) ---")
mask_pred_path = f"{OUT_DIR}/masked_prediction_results.txt"
os.makedirs(os.path.dirname(mask_pred_path), exist_ok=True)
with open(mask_pred_path, "w") as mask_f:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 2:
            continue
        split_point = len(tokens) // 2
        prefix = tokens[:split_point]
        masked = ' '.join(prefix + ['[mask]'])
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"")
        mask_f.write(f"[{i}] original: {test_documents[i]}\n")
        mask_f.write(f"    masked:   {masked}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
        print(f"  Completed: \"{completed}\"\n")
        mask_f.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, 'visualize'):
            parse.visualize(f"{OUT_DIR}/masked_pred_trees/masked_pred_tree{i}")
print(f"Saved masked prediction results to \"{mask_pred_path}\"")

SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
webster.save_state(SAVE_DIR)
print(f"Saved Final LTM to \"{SAVE_DIR}\"!")
