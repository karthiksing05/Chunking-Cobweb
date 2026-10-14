"""TRELLIS: concepts and chunks in Cobweb hierarchies.

v2's grammar distillation (cuts read off the hierarchies by description
length, night and incremental learners) is archived at git tag
``archive/pcfg-v2``. What remains here is shared by v3: Cobweb trees (with
insertion and removal), element descriptions, the domains and their data,
and evaluation."""
from .cobweb import CobwebNode, CobwebTree
from .data import Example, Tree, load_corpus, tree_from_merges, v1_split
from .memory import Memory
