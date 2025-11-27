"""
Sandbox!

Currently working on testing what happens when we just add the first granularity of candidate chunks
to Cobweb and see how the hierarchy emerges.
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser
from cobweb.cobweb_discrete import CobwebTree
import os

if os.path.exists("sandbox/sandbox_ltm.png"):
    os.remove("sandbox/sandbox_ltm.png")

CONTEXT_LENGTH = 2

# function to scrape instances from sentence (not using the parse tree stuff for this)
# that'll eventually be rewritten!
def get_chunk_candidates(sentence: str, value_to_id: dict, context_length: int = CONTEXT_LENGTH):
    """
    Produce merge-candidate instances for all adjacent word pairs in `sentence`.

    Instances follow the numeric-key format used by `PrimitiveParseNode`/`CompositeParseNode`:
      - 0: content-left dict
      - 1: content-right dict
      - 2..2+context_length-1: per-index context-before dicts (0 = immediate left)
      - 2+context_length..2+2*context_length-1: per-index context-after dicts (0 = immediate right)

    Missing context slots are represented as `{0: 0}` to keep compatibility with existing code.
    """

    words = [value_to_id[w] for w in sentence.split(" ")]
    insts = []

    for i in range(len(words) - 1):
        content_left = words[i]
        content_right = words[i + 1]

        inst = {
            0: {content_left: 1, 0: 0},
            1: {content_right: 1, 0: 0}
        }

        # build per-index context-before (0 = immediate left of `content_left`)
        for k in range(context_length):
            idx_before = i - 1 - k
            if idx_before >= 0:
                d = {words[idx_before]: 1.0, 0: 0}
            else:
                d = {0: 0}
            inst[2 + k] = d

        # build per-index context-after (0 = immediate right of `content_right`)
        for k in range(context_length):
            idx_after = i + 2 + k
            if idx_after < len(words):
                d = {words[idx_after]: 1.0, 0: 0}
            else:
                d = {0: 0}
            inst[2 + context_length + k] = d

        insts.append(inst)

    return insts


num_sentences = 40
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

parser = LanguageChunkingParser(TEST_CORPUS2, context_length=CONTEXT_LENGTH)

tree = CobwebTree()

for sentence in document:
    instances = get_chunk_candidates(sentence, parser.value_to_id)

    for inst in instances:
        tree.ifit(inst, 0, True)

while not os.path.exists("sandbox/sandbox_ltm.png"):
    parser.cobweb_drawer.draw_tree(tree.root, "sandbox/sandbox_ltm")