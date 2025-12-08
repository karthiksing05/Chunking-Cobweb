"""
General Learning Test - to confirm the logic of learning is completely functional!
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser

# Creating and printing toy sentences
CONTEXT_LENGTH = 3

num_sentences = 100
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS2, context_length=CONTEXT_LENGTH, merge_split=True)

train_size = 0.9

train_documents = document[:int(len(document) * train_size)]
test_documents = document[int(len(document) * train_size):]

# Iterate through training documents and parse them one at a time, saving every 10th parse tree to file
for i, doc in enumerate(train_documents):
    parse_trees = parser.parse_input([doc], end_behavior=-5, debug=True)
    parse_tree = parse_trees[0]

    parser.add_parse_tree(parse_tree, debug=False)

    if i < 5:
        parser.visualize_ltm(f"unittests/gen_learn_test/ltms/cobweb_ltm{i}")

    if i % 5 == 0:
        parse_tree.visualize(f"unittests/gen_learn_test/train_trees/train_parse_tree{i}")

        if i < 21:
            parser.visualize_ltm(f"unittests/gen_learn_test/ltms/cobweb_ltm{i}")

# parser.visualize_ltm("unittests/gen_learn_test/final_ltm")

# perhaps visualize the last ten parse trees based on new inputs
for i, test in enumerate(test_documents):
    parse_tree = parser.parse_input([test], end_behavior="converge", debug=False)[0]
    parse_tree.visualize(f"unittests/gen_learn_test/test_trees/test_parse_tree{i}")
    print(f"Created parse tree {i} for sentence, \"{test}\"")

SAVE_DIR = "unittests/gen_learn_test/final_ltm_data"
parser.save_state(SAVE_DIR)
print(f"Saved LTM to \"{SAVE_DIR}\"!")