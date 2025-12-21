"""
Sandbox!

Currently working on testing what happens when we just add the first granularity of candidate chunks
to Cobweb and see how the hierarchy emerges.

So far, a safe score that we're seeing is -5 for this setting right here. Everything is
equally valuable earlier (obviously)!

What we can do is store costs prior to our chunking to acknowledge our chunks
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser, FiniteParseTree, custom_categorize
from cobweb.cobweb_discrete import CobwebTree
import os
from pprint import pprint

if os.path.exists("sandbox/sandbox_ltm.png"):
    os.remove("sandbox/sandbox_ltm.png")

CONTEXT_LENGTH = 3

# function to scrape instances from sentence (not using the parse tree stuff for this)
# that'll eventually be rewritten!
def get_composite_chunk_candidates(sentence: str, value_to_id: dict, context_length: int = CONTEXT_LENGTH):
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

def get_primitive_chunk_candidates(sentence: str, value_to_id: dict, context_length: int = CONTEXT_LENGTH):
    """
    Produce merge-candidate instances for words in `sentence`.

    Instances follow the numeric-key format used by `PrimitiveParseNode`/`CompositeParseNode`:
      - 0: NONE
      - 1: NONE
      - 2..2+context_length-1: per-index context-before dicts (0 = immediate left)
      - 2+context_length..2+2*context_length-1: per-index context-after dicts (0 = immediate right)
      - 2+2*context_length-1: singular content 

    Missing context slots are represented as `{0: 0}` to keep compatibility with existing code.
    """

    words = [value_to_id[w] for w in sentence.split(" ")]
    insts = []

    for i in range(len(words)):

        inst = {
            0: {0: 0},
            1: {0: 0}
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
            idx_after = i + 1 + k
            if idx_after < len(words):
                d = {words[idx_after]: 1.0, 0: 0}
            else:
                d = {0: 0}
            inst[2 + context_length + k] = d

        inst[2 + 2 * context_length] = {words[i]: 1.0, 0: 0}

        insts.append(inst)

    return insts

num_sentences = 20
document = []

num_primitive_docs = 0.5

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

primitive_doc = None

parser = LanguageChunkingParser(TEST_CORPUS2, context_length=CONTEXT_LENGTH)

# tree = CobwebTree(10, False, 0, True, False)
tree = CobwebTree(0.1, False, 0, True, False)

for sentence in document:

    instances = get_composite_chunk_candidates(sentence, parser.value_to_id)

    for inst in instances:
        tree.ifit(inst, 0, True)

parser.cobweb_drawer.save_basic_level_subtrees(tree.root, "sandbox")

print("All sentences:")
pprint(document)

# while not os.path.exists("sandbox/sandbox_ltm.png"):
#     parser.cobweb_drawer.draw_tree(tree.root, "sandbox/sandbox_ltm")

test_sentence = input("enter input sentence: ")
candidates = get_composite_chunk_candidates(test_sentence, parser.value_to_id)

print("Test Sentence:", test_sentence)

costs = []
counts = []
root_costs = []
best_log_prob_idxs = []
best_avg_log_probs = []
log_prob_avgs = []

for i, candidate in enumerate(candidates):
    print(f"Candidate {i}:")
    node, categorize_ids, node_categorize_path = custom_categorize(candidate, tree)
    print("Stats:")
    score_stats = FiniteParseTree._score_function(node_categorize_path, candidate)
    pprint(score_stats)
    costs.append(score_stats["cost"])
    counts.append(score_stats["normed_count"])
    root_costs.append(score_stats["root_cost"])
    best_log_prob_idxs.append(score_stats["best_log_prob_idx"])
    best_avg_log_probs.append(score_stats["best_avg_log_prob"])

print(costs)
print(root_costs)
print(counts)