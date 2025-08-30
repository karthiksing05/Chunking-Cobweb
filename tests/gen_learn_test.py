"""
General Learning Test - to confirm the logic of learning is completely functional
"""

from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse import LanguageChunkingParser

# Creating and printing toy sentences
num_sentences = 100
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS1)

train_size = 0.9

train_documents = document[:int(len(document) * train_size)]
test_documents = document[int(len(document) * train_size):]

# Iterate through training documents and parse them one at a time, saving every 10th parse tree to file
for i, doc in enumerate(train_documents):
    parse_trees = parser.parse_input([doc], debug=False)
    parse_tree = parse_trees[0]

    if i % 5 == 0:
        parse_tree.visualize(f"tests/gen_learn_test/train_trees/parse_tree{i}")

        if i < 21:
            parser.visualize_ltm(f"tests/gen_learn_test/ltms/cobweb_ltm{i}")
    
    parser.add_parse_tree(parse_tree)

# perhaps visualize the last ten parse trees based on new inputs
for test in test_documents:
    parse_tree = parser.parse_input([test])
    parse_tree.visualize(f"tests/gen_learn_test/test_trees/test_parse_tree{i}")
