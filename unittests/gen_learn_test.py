"""
General Learning Test - to confirm the logic of learning is completely functional!
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse import LanguageChunkingParser
import shutil
import os
import random

if os.path.exists("unittests/gen_learn_test"):
    shutil.rmtree("unittests/gen_learn_test")

# Creating and printing toy sentences
CONTEXT_LENGTH = 3

num_sentences = 100
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS1, context_length=CONTEXT_LENGTH, merge_split=True)

train_size = 0.90

PRIMITIVES_FIRST = 0
THRESHOLD = -5.5

train_documents = document[:int(len(document) * train_size)]
test_documents = document[int(len(document) * train_size):]

# Iterate through training documents and parse them one at a time, saving every 10th parse tree to file
for i, doc in enumerate(train_documents):
    threshold = (0 if i < PRIMITIVES_FIRST else THRESHOLD) # should never trigger atp
    print("Threshold:", threshold)
    parse_trees = parser.parse_input([doc], end_behavior=threshold, debug=True)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

    if i < 5:
        parser.visualize_ltm(f"unittests/gen_learn_test/ltms/cobweb_ltm{i}", max_depth=4)

    if i % 5 == 0:
        parse_tree.visualize(f"unittests/gen_learn_test/train_trees/train_parse_tree{i}")

        if i < 21:
            parser.visualize_ltm(f"unittests/gen_learn_test/ltms/cobweb_ltm{i}", max_depth=4)

parser.visualize_ltm("unittests/gen_learn_test/final_ltm", max_depth=4)

# visualize the test parse trees
for i, test in enumerate(test_documents):
    parse_tree = parser.parse_input([test], end_behavior=THRESHOLD, debug=False)[0]
    parse_tree.visualize(f"unittests/gen_learn_test/test_trees/test_parse_tree{i}")
    print(f"Created parse tree {i} for sentence, \"{test}\"")

# creating fake sentences, through completely random choice, to see if they parse!
fake_sentences = [" ".join([random.choice(TEST_CORPUS1) for _ in range(random.randint(3, 8))]) for _ in range(10)]
fake_sentences.append("the dog the dog")

for i, fake_sentence in enumerate(fake_sentences):
    parse_tree = parser.parse_input([fake_sentence], end_behavior=THRESHOLD, debug=False)[0]
    parse_tree.visualize(f"unittests/gen_learn_test/fake_trees/fake_parse_tree{i}")
    print(f"Created fake parse tree for fake sentence, \"{fake_sentence}\"")

# generating complete sentences! no prompt yet
for i in range(10):
    sentence, parse = parser.generate_sentence(debug=True)
    print(f"Generated sentence: {sentence}")
    # parse.visualize(f"unittests/gen_learn_test/generated_trees/fake_parse_tree{i}")

SAVE_DIR = "unittests/gen_learn_test/final_ltm_data"
parser.save_state(SAVE_DIR)
print(f"Saved Final LTM to \"{SAVE_DIR}\"!")