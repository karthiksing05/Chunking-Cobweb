"""
Sandbox!

Currently working on testing pycobweb!
"""
print()

from viz import TextCobwebDrawer
from cobweb.cobweb_discrete import CobwebTree, CobwebNode
import json

MERGE_SPLIT = True

tree = CobwebTree()

attrs = ["color-head", "color-body", "color-foot", "color-hand"]
values = list(range(30))

drawer = TextCobwebDrawer(attrs, values)

sim_insts = [
    {0: {0: 1}, 1: {1: 1}, 2: {0: 1}}, # red-head, yellow-body, red-foot
    {0: {1: 1}, 1: {2: 1}, 2: {2: 1}}, # yellow-head, blue-body, blue-foot
    {0: {0: 1}, 1: {2: 1}, 2: {0: 1}}, # red-head, blue-body, red-foot
    {0: {1: 1}, 1: {0: 1}, 2: {1: 1}}, # yellow-head, yellow-body, yellow-foot
    {0: {1: 1}, 1: {2: 1}, 2: {0: 1}},  # yellow-head, blue-body, red-foot
]

comp_sim_insts = [
    {0: {0: 1, 29: 0}, 1: {1: 1, 29: 0}, 2: {0: 1, 29: 0}}, # red-head, yellow-body, red-foot
    {0: {1: 1, 29: 0}, 1: {2: 1, 29: 0}, 2: {2: 1, 29: 0}}, # yellow-head, blue-body, blue-foot
    {0: {0: 1, 29: 0}, 1: {2: 1, 29: 0}, 2: {0: 1, 29: 0}}, # red-head, blue-body, red-foot
    {0: {1: 1, 29: 0}, 1: {0: 1, 29: 0}, 2: {1: 1, 29: 0}}, # yellow-head, yellow-body, yellow-foot
    {0: {1: 1, 29: 0}, 1: {2: 1, 29: 0}, 2: {0: 1, 29: 0}},  # yellow-head, blue-body, red-foot
]

# results in an error!!!
countercase_insts = [
    {0: {0: 0, 22: 1}, 1: {0: 0, 2: 1}, 2: {0: 0, }, 3: {0: 0, }},
    {0: {0: 0, 22: 1}, 1: {0: 0, 11: 1}, 2: {0: 0, }, 3: {0: 0, 2: 1.0}},
    {0: {0: 0, 22: 1}, 1: {0: 0, 12: 1}, 2: {0: 0, }, 3: {0: 0, 11: 1.0, 2: 0.5}}
]

logs = []

for i, inst in enumerate(sim_insts):
    node, time_ops, log = tree.ifit(inst, (0 if MERGE_SPLIT else 4), debug=True)
    drawer.visualize_tree(tree.root)
    log = [json.loads(x) for x in log]
    logs.append(log)
    print("Iteration", i, "complete!")

drawer.visualize_tree(tree.root)

print("Logs:")
for log in logs:
    print(log)

# tree.root.set_av_count({0: {1: 1}, 1: {2: 1}, 2: {1: 1}})

# drawer.visualize_tree(tree.root)

sample_inst = {0: {1: 1}, 1: {2: 1}, 2: {1: 1}}

leaf = tree.categorize(sample_inst)
# leaf.set_av_count({0: {1: 1}, 1: {2: 1}, 2: {1: 1}})
bl_node = leaf.get_basic_level() # TODO WHY THE FREAK IS THIS NONDETERMINISTIC

# help(CobwebNode)

print(bl_node.concept_hash(), leaf.concept_hash())