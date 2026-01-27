"""
Sandbox!

Currently working on testing what happens when we just add the first granularity of candidate chunks
to Cobweb and see how the hierarchy emerges.

So far, a safe score that we're seeing is -5 for this setting right here. Everything is
equally valuable earlier (obviously)!

What we can do is store costs prior to our chunking to acknowledge our chunks

Instance representations - pad EMPTYNULL with 1s for composite singular-content attribute and for
primitive content-left and content-right attributes
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, POS_GRAMMAR1, POS_CORPUS1
from parse import LanguageChunkingParser, FiniteParseTree, custom_categorize
from cobweb.cobweb_discrete import CobwebTree
import os
from pprint import pprint

if os.path.exists("sandbox/sandbox_ltm.png"):
    os.remove("sandbox/sandbox_ltm.png")

CONTEXT_LENGTH = 3

# function to scrape instances from sentence (not using the parse tree stuff for this)
# that'll eventually be rewritten!
def get_composite_chunk_candidates(sentence: str, value_to_id: dict, context_length: int = CONTEXT_LENGTH, bow=False):
    """
    Produce merge-candidate instances for all adjacent word pairs in `sentence`.

    Instances follow the numeric-key format used by `PrimitiveParseNode`/`CompositeParseNode`:
      - 0: content-left dict
      - 1: content-right dict
      - 2..2+context_length-1: per-index context-before dicts (0 = immediate left)
      - 2+context_length..2+2*context_length-1: per-index context-after dicts (0 = immediate right)

    Missing context slots are represented as `{0: 0}` to keep compatibility with existing code.

    'bow' represents a parameter of how the context should be associated - positionally OR with 
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

        if bow:

            # build per-index context-before (0 = immediate left of `content_left`)
            d = {0: 0}
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d[words[idx_before]] = 1.0 / (k + 1)
                inst[2 + k] = {0: 0}

            inst[2] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            d = {0: 0}
            for k in range(context_length):
                idx_after = i + 2 + k
                if idx_after < len(words):
                    d[words[idx_after]] = 1.0 / (k + 1)
                inst[2 + context_length + k] = {0: 0}

            inst[2 + context_length] = d

            inst[2 + 2 * context_length] = {0: 1}

            insts.append(inst)
            
        else:

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

            inst[2 + 2 * context_length] = {0: 1}

            insts.append(inst)

    return insts

def get_primitive_chunk_candidates(sentence: str, value_to_id: dict, context_length: int = CONTEXT_LENGTH, bow=False):
    """
    Produce merge-candidate instances for words in `sentence`.

    Instances follow the numeric-key format used by `PrimitiveParseNode`/`CompositeParseNode`:
      - 0: NONE
      - 1: NONE
      - 2..2+context_length-1: per-index context-before dicts (0 = immediate left)
      - 2+context_length..2+2*context_length-1: per-index context-after dicts (0 = immediate right)
      - 2+2*context_length-1: singular content 

    Missing context slots are represented as `{0: 0}` to keep compatibility with existing code.

    'bow' is a keyword that signifies the type of 
    """

    words = [value_to_id[w] for w in sentence.split(" ")]
    insts = []

    for i in range(len(words)):

        inst = {
            0: {0: 1},
            1: {0: 1}
        }

        if bow:

            # build per-index context-before (0 = immediate left of `content_left`)
            d = {0: 0}
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d[words[idx_before]] = 1.0 / (k + 1)
                inst[2 + k] = {0: 0}

            inst[2] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            d = {0: 0}
            for k in range(context_length):
                idx_after = i + 2 + k
                if idx_after < len(words):
                    d[words[idx_after]] = 1.0 / (k + 1)
                inst[2 + context_length + k] = {0: 0}

            inst[2 + context_length] = d

            inst[2 + 2 * context_length] = {words[i]: 1.0, 0: 0}

            insts.append(inst)
            
        else:

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


num_sentences = 2000
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

primitives_factor = 0.66

primitive_doc = document[:int(len(document) * primitives_factor)]
composite_doc = document[int(len(document) * primitives_factor):]

parser = LanguageChunkingParser(TEST_CORPUS1, context_length=CONTEXT_LENGTH)

# tree = CobwebTree(10, False, 0, True, False)
tree = CobwebTree(0.0005, False, 0, True, False)

for sentence in primitive_doc:

    instances = get_primitive_chunk_candidates(sentence, parser.value_to_id, bow=True)

    for inst in instances:
        tree.ifit(inst, 0, True)

for sentence in composite_doc:

    instances = get_composite_chunk_candidates(sentence, parser.value_to_id, bow=True)

    for inst in instances:
        tree.ifit(inst, 0, True)

parser.cobweb_drawer.save_basic_level_subtrees(tree.root, "sandbox", debug=True)

# print("All sentences:")
# pprint(document)

MAX_DEPTH = 4

while not os.path.exists(f"sandbox/sandbox_ltm_{MAX_DEPTH}.png"):
    # parser.cobweb_drawer.draw_tree(tree.root, "sandbox/sandbox_ltm")
    parser.cobweb_drawer.draw_tree(tree.root, f"sandbox/sandbox_ltm_{MAX_DEPTH}", max_depth=MAX_DEPTH)

# test_sentence = input("enter input sentence: ")
# candidates = get_composite_chunk_candidates(test_sentence, parser.value_to_id)

# print("Test Sentence:", test_sentence)

# costs = []
# counts = []
# root_costs = []
# best_log_prob_idxs = []
# best_avg_log_probs = []
# log_prob_avgs = []

# for i, candidate in enumerate(candidates):
#     print(f"Candidate {i}:")
#     node, categorize_ids, node_categorize_path = custom_categorize(candidate, tree)
#     print("Stats:")
#     score_stats = FiniteParseTree._score_function(node_categorize_path, candidate)
#     pprint(score_stats)
#     costs.append(score_stats["cost"])
#     counts.append(score_stats["normed_count"])
#     root_costs.append(score_stats["root_cost"])
#     best_log_prob_idxs.append(score_stats["best_log_prob_idx"])
#     best_avg_log_probs.append(score_stats["best_avg_log_prob"])

# print(costs)
# print(root_costs)
# print(counts)