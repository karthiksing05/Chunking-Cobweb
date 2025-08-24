from pprint import pprint
from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from chunking import LanguageChunkingParser
from parse import ParseTree

# Creating and printing toy sentences
num_sentences = 3
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

print("Generated Sentences:")
pprint(document)

# Setting up the parser
parser = LanguageChunkingParser(TEST_CORPUS1)

trees = parser.parse_input(document, debug=False) # TODO debug flag needs to be implemented

for i, tree in enumerate(trees):
    print(f"TREE for Sentence {i}: " + document[i])
    tree.visualize(f"tests/visualizations/parse_tree{i}")
    tree.to_json(f"tests/json/parse_tree{i}.json")

# Loading parse_tree0.json to verify
tree = ParseTree.from_json("tests/json/parse_tree0.json", ltm_hierarchy=parser.get_long_term_memory(), filepath=True)
tree.visualize("tests/visualizations/parse_tree0_from_json")