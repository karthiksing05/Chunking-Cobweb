"""
Sandbox!

Currently working on testing pycobweb!
"""
print()

from viz import TextCobwebDrawer
from util.pycobweb import CobwebTree

tree = CobwebTree()

sim_insts = [
    {0: {0: 1}, 1: {1: 1}, 2: {0: 1}}, # red-head, yellow-body, red-foot
    {0: {1: 1}, 1: {2: 1}, 2: {2: 1}}, # yellow-head, blue-body, blue-foot
    {0: {0: 1}, 1: {2: 1}, 2: {0: 1}}, # red-head, blue-body, red-foot
    {0: {1: 1}, 1: {0: 1}, 2: {1: 1}}, # yellow-head, yellow-body, yellow-foot
    {0: {1: 1}, 1: {2: 1}, 2: {0: 1}}  # yellow-head, blue-body, red-foot
]

for inst in sim_insts:
    tree.ifit(inst, 0)

attrs = ["color-head", "color-body", "color-foot"]
values = ["red", "yellow", "blue"]

drawer = TextCobwebDrawer(attrs, values)
drawer.visualize_tree(tree.root)

sample_inst = {0: {1: 1}, 1: {2: 1}, 2: {1: 1}}

leaf = tree.categorize(sample_inst)
bl_node = leaf.get_basic_level()

print(bl_node.concept_hash(), leaf.concept_hash())