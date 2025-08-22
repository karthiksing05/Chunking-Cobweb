from pprint import pprint
from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from src.chunking import LanguageChunkingParser

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