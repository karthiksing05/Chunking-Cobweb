from pprint import pprint
from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse import LanguageChunkingParser, FiniteParseTree

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

trees = parser.parse_input(document, end_behavior="converge", debug=False)
# using end behavior of -8 is good for now but TBD because the measure isn't as valid
# we can also set debug to equal true!

for i, tree in enumerate(trees):
    print(f"Generating TREE for Sentence {i}: " + document[i])
    tree.visualize(f"unittests/parse_tree_test/visualizations/parse_tree{i}")
    tree.to_json(f"unittests/parse_tree_test/json/parse_tree{i}.json")

# Loading parse_tree0.json to verify
tree = FiniteParseTree.from_json("unittests/parse_tree_test/json/parse_tree0.json", ltm_hierarchy=parser.get_long_term_memory(), filepath=True)
tree.visualize("unittests/parse_tree_test/visualizations/parse_tree0_from_json")

print("Confirming that all chunk instances are appropriately scraped:")
for chunk_inst in tree.get_chunk_instances():
    print("- ", chunk_inst)