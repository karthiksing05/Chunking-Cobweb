from pprint import pprint
from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse import LanguageChunkingParser, ParseTree

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

trees = parser.parse_input(document, debug=True)

for i, tree in enumerate(trees):
    print(f"TREE for Sentence {i}: " + document[i])
    tree.visualize(f"tests/parse_tree_test/visualizations/parse_tree{i}")
    tree.to_json(f"tests/parse_tree_test/json/parse_tree{i}.json")

# Loading parse_tree0.json to verify
tree = ParseTree.from_json("tests/parse_tree_test/json/parse_tree0.json", ltm_hierarchy=parser.get_long_term_memory(), filepath=True)
tree.visualize("tests/parse_tree_test/visualizations/parse_tree0_from_json")