"""
Sandbox!

Currently working on testing pycobweb!
"""
print()

from viz import TextCobwebDrawer
from cobweb.cobweb_discrete import CobwebTree, CobwebNode
from tqdm import tqdm

tree = CobwebTree()

attrs = ["color-head", "color-body", "color-foot"]
values = ["red", "yellow", "blue"]

drawer = TextCobwebDrawer(attrs, values)

sim_insts = [
    {0: {0: 1}, 1: {1: 1}, 2: {0: 1}}, # red-head, yellow-body, red-foot
    {0: {1: 1}, 1: {2: 1}, 2: {2: 1}}, # yellow-head, blue-body, blue-foot
    {0: {0: 1}, 1: {2: 1}, 2: {0: 1}}, # red-head, blue-body, red-foot
    {0: {1: 1}, 1: {0: 1}, 2: {1: 1}}, # yellow-head, yellow-body, yellow-foot
    {0: {1: 1}, 1: {2: 1}, 2: {0: 1}},  # yellow-head, blue-body, red-foot
]

logs = []

for i, inst in enumerate(sim_insts):
    node, time_ops, log = tree.ifit(inst, 0, debug=True)
    drawer.visualize_tree(tree.root)
    logs.append(log)
    print("Iteration", i, "complete!")


print("Logs:")
for log in logs:
    print(log)

drawer.visualize_tree(tree.root)

# tree.root.set_av_count({0: {1: 1}, 1: {2: 1}, 2: {1: 1}})

# drawer.visualize_tree(tree.root)

sample_inst = {0: {1: 1}, 1: {2: 1}, 2: {1: 1}}

leaf = tree.categorize(sample_inst)
# leaf.set_av_count({0: {1: 1}, 1: {2: 1}, 2: {1: 1}})
bl_node = leaf.get_basic_level()

# help(CobwebNode)

print(bl_node.concept_hash(), leaf.concept_hash())