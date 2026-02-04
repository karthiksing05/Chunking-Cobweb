"""
Sandbox!

Currently working on better understanding representations - need normalized attributes across all dimensions
so that we can calculate scores that make sense
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, POS_GRAMMAR1, POS_CORPUS1, TEST_EPOCH_GRAMMAR_EPOCH_1, TEST_EPOCH_GRAMMAR_EPOCH_2, TEST_EPOCH_CORPUS
from parse import LanguageChunkingParser
from cobweb.cobweb_discrete import CobwebDiscreteTree
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
    Bag-Of-Word contexts
    """

    words = [value_to_id[w] for w in sentence.split(" ")]
    insts = []

    for i in range(len(words) - 1):
        content_left = words[i]
        content_right = words[i + 1]

        inst = {
            0: {content_left: 0.5, 0: 0},
            1: {content_right: 0.5, 0: 0}
        }

        if bow:

            # build per-index context-before (0 = immediate left of `content_left`)
            d = {0: 0}
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d[words[idx_before]] = 1.0 / (2 ** (k + 1))
                else:
                    d[0] += 1.0 / (2 ** (k + 1))

            inst[2] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            d = {0: 0}
            for k in range(context_length):
                idx_after = i + 2 + k
                if idx_after < len(words):
                    d[words[idx_after]] = 1.0 / (2 ** (k + 1))
                else:
                    d[0] += 1.0 / (2 ** (k + 1))

            inst[2 + context_length] = d
            
        else:

            # build per-index context-before (0 = immediate left of `content_left`)
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d = {words[idx_before]: 1.0 / (2 ** (k + 1)), 0: 0}
                    # d = {words[idx_before]: 1.0 / ((k + 2)), 0: 0}
                    # d = {words[idx_before]: 1.0 / (context_length), 0: 0}
                else:
                    # d = {0: 1.0 / (2 ** (k + 1))}
                    # d = {0: 1.0 / (k + 2)}
                    # d = {0: 1.0 / (context_length)}
                    d = {0: 0}
                inst[2 + k] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            for k in range(context_length):
                idx_after = i + 1 + k
                if idx_after < len(words):
                    d = {words[idx_after]: 1.0 / (2 ** (k + 1)), 0: 0}
                    # d = {words[idx_after]: 1.0 / ((k + 2)), 0: 0}
                    # d = {words[idx_after]: 1.0 / (context_length), 0: 0}
                else:
                    # d = {0: 1.0 / (2 ** (k + 1))}
                    # d = {0: 1.0 / (k + 2)}
                    # d = {0: 1.0 / (context_length)}
                    d = {0: 0}
                inst[2 + context_length + k] = d

        inst[2 + 2 * context_length] = {0: 0}

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

    'bow' represents a parameter of how the context should be associated - positionally OR with
    Bag-Of-Word contexts
    """

    words = [value_to_id[w] for w in sentence.split(" ")]
    insts = []

    for i in range(len(words)):

        inst = {
            0: {0: 0},
            1: {0: 0}
        }

        if bow:

            # build per-index context-before (0 = immediate left of `content_left`)
            d = {0: 0}
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d[words[idx_before]] = 1.0 / (2 ** (k + 1))
                else:
                    d[0] += 1.0 / (2 ** (k + 1))

            inst[2] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            d = {0: 0}
            for k in range(context_length):
                idx_after = i + 2 + k
                if idx_after < len(words):
                    d[words[idx_after]] = 1.0 / (2 ** (k + 1))
                else:
                    d[0] += 1.0 / (2 ** (k + 1))

            inst[2 + context_length] = d
            
        else:

            # build per-index context-before (0 = immediate left of `content_left`)
            for k in range(context_length):
                idx_before = i - 1 - k
                if idx_before >= 0:
                    d = {words[idx_before]: 1.0 / (2 ** (k + 1)), 0: 0}
                    # d = {words[idx_before]: 1.0 / ((k + 2)), 0: 0}
                    # d = {words[idx_before]: 1.0 / (context_length), 0: 0}
                else:
                    # d = {0: 1.0 / (2 ** (k + 1))}
                    # d = {0: 1.0 / (k + 2)}
                    # d = {0: 1.0 / (context_length)}
                    d = {0: 0}
                inst[2 + k] = d

            # build per-index context-after (0 = immediate right of `content_right`)
            for k in range(context_length):
                idx_after = i + 1 + k
                if idx_after < len(words):
                    # d = {words[idx_after]: 1.0 / (2 ** (k + 1)), 0: 0}
                    d = {words[idx_after]: 1.0 / ((k + 2)), 0: 0}
                    # d = {words[idx_after]: 1.0 / (context_length), 0: 0}
                else:
                    # d = {0: 1.0 / (2 ** (k + 1))}
                    # d = {0: 1.0 / (k + 2)}
                    # d = {0: 1.0 / (context_length)}
                    d = {0: 0}
                inst[2 + context_length + k] = d

        inst[2 + 2 * context_length] = {words[i]: 1, 0: 0}

        insts.append(inst)

    return insts


num_sentences = 300
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_EPOCH_GRAMMAR_EPOCH_1)
    document.append(sentence)

primitives_factor = 0.5

primitive_doc = document[:int(len(document) * primitives_factor)]
composite_doc = document[int(len(document) * primitives_factor):]

parser = LanguageChunkingParser(TEST_EPOCH_CORPUS, context_length=CONTEXT_LENGTH)

# tree = CobwebTree(10, False, 0, True, False)
tree = CobwebDiscreteTree(alpha=1e-4, weight_attr=True)

for sentence in primitive_doc:

    instances = get_primitive_chunk_candidates(sentence, parser.value_to_id, bow=False)

    for inst in instances:
        # check total sum of the instance
        total_value_sum = 0
        for k, d in inst.items():
            # print(f"{k}: {sum(d.values())}")
            total_value_sum += sum(d.values())

        print(total_value_sum)

    for inst in instances:
        tree.ifit(inst)

for sentence in composite_doc:

    instances = get_composite_chunk_candidates(sentence, parser.value_to_id, bow=False)

    for inst in instances:
        # check total sum of the instance
        total_value_sum = 0
        for k, d in inst.items():
            # print(f"{k}: {sum(d.values())}")
            total_value_sum += sum(d.values())

        print(total_value_sum)


    for inst in instances:
        tree.ifit(inst)

parser.cobweb_drawer.save_basic_level_subtrees(tree.root, "sandbox", debug=True)

# print("All sentences:")
# pprint(document)

MAX_DEPTH = 3

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