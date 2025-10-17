"""
Sandbox!

Currently working on testing json!
"""
print()

from parse import LanguageChunkingParser

parser = LanguageChunkingParser.load_state("unittests/ltm_analysis/halfway_test")

root = parser.get_long_term_memory().root

queue = [root]

id_set = set()

while len(queue) > 0:

    curr = queue.pop()

    print(curr.concept_hash())
    id_set.add(curr.concept_hash())
    for child in curr.children:
        queue.append(child)

print("len", len(id_set))