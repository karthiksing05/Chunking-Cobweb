"""
Confirm `tree.get_basic_frontier()` recovers the canonical basic-level nodes
on the seven hierarchies defined in corter_gluck_hierarchies.py.

For each dataset, we:
  1. Build a FrozenCobwebDiscreteTree from the hierarchy.
  2. Call tree.get_basic_frontier() — the antichain DFS over the
     alpha-agnostic per-attribute squared-deviation score.
  3. Check that the frontier matches the dataset's labeled 'Basic' level
     (and report the score of every node so the limit shape is visible).
  4. Verify the one-BL-per-path property holds.
"""

import os
import sys
import importlib.util
from collections import Counter

import numpy as np

# Make src/ importable so we can pull in the FrozenCobwebDiscreteTree helper.
_HERE     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Pull FrozenCobwebDiscreteTree directly from cobweb-private/src/cobweb/.
_FD_PATH = os.path.join(
    _HERE, "..", "..", "cobweb-private", "src", "cobweb", "frozen_discrete.py",
)
_fd_spec = importlib.util.spec_from_file_location("_frozen_discrete_local", _FD_PATH)
_fd_mod  = importlib.util.module_from_spec(_fd_spec)
_fd_spec.loader.exec_module(_fd_mod)
FrozenCobwebDiscreteTree = _fd_mod.FrozenCobwebDiscreteTree

# Pull the hierarchy definitions.
import corter_gluck_hierarchies as cgh

OUT_DIR = os.path.join(_HERE, "basic_frontier_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Convert a Corter & Gluck-style hierarchy into the nested dict format
# expected by FrozenCobwebDiscreteTree.
# ---------------------------------------------------------------------------

def encode_instance(row, feature_idx):
    """Row is a single X[i] vector; encode into INSTANCE_TYPE dict."""
    return {int(fidx): {int(row[fidx]): 1.0}
            for _, fidx in feature_idx.items()}


def build_frozen_hierarchy(node_members, levels, X, feature_idx):
    """
    Convert the dataset's flat node→members map into a nested hierarchy dict.

    Treats `levels` as ordered (top = first key, bottom = last key).  Inner
    nodes are recovered by greedy subset matching: each parent's children
    are the largest-subset nodes at the next level down whose members are
    contained in the parent's members.
    """
    level_names = list(levels.keys())                   # top → leaf order …
    # …but `levels` is currently written leaf→root, so reverse it.
    level_names = level_names[::-1]

    # Map node-name → member-tuple for set ops.
    members_of = {n: tuple(sorted(node_members[n])) for n in node_members}

    # Walk top to bottom, attaching children.
    def build_node(name):
        m = set(members_of[name])
        # Find children in any deeper level whose members are a strict subset of m.
        my_depth = next(i for i, lvl in enumerate(level_names)
                        if name in levels[lvl])
        for deeper in level_names[my_depth + 1:]:
            for cand in levels[deeper]:
                if cand == name:
                    continue
                cm = set(members_of[cand])
                if cm.issubset(m) and cm != m:
                    # cand is a descendant.  Check it's a *direct* child by
                    # ensuring no intermediate node sits between.
                    is_direct = True
                    for mid_lvl in level_names[my_depth + 1: deeper_idx(deeper)]:
                        for mid in levels[mid_lvl]:
                            if mid == name or mid == cand:
                                continue
                            mid_m = set(members_of[mid])
                            if cm.issubset(mid_m) and mid_m.issubset(m) and mid_m != m and mid_m != cm:
                                is_direct = False
                                break
                        if not is_direct:
                            break
                    if is_direct:
                        return [(deeper, cand)]
            # fall through (shouldn't happen)
        return []

    def deeper_idx(name):
        return level_names.index(name)

    # Simpler approach: directly group children by partition.
    def find_direct_children(parent_name, parent_members):
        parent_depth = next(i for i, lvl in enumerate(level_names)
                            if parent_name in levels[lvl])
        for next_lvl_i in range(parent_depth + 1, len(level_names)):
            candidates = []
            for cand in levels[level_names[next_lvl_i]]:
                cm = set(members_of[cand])
                if cm.issubset(parent_members) and cm != set():
                    candidates.append(cand)
            # Take candidates whose union covers parent_members and are
            # pairwise disjoint — those are this level's direct children.
            covered = set()
            disjoint = True
            for c in candidates:
                cm = set(members_of[c])
                if covered & cm:
                    disjoint = False
                    break
                covered |= cm
            if candidates and disjoint and covered == parent_members:
                return candidates
        return []

    def recurse(name):
        m = set(members_of[name])
        children_names = find_direct_children(name, m)
        if not children_names:
            # Leaf-level node — emit its instances as raw observations.
            return {"instances": [encode_instance(X[i], feature_idx)
                                  for i in members_of[name]]}
        return {"children": [recurse(c) for c in children_names]}

    # Top of the hierarchy:
    top_nodes = levels[level_names[0]]
    if len(top_nodes) == 1:
        root_name = top_nodes[0]
        return recurse(root_name)
    return {"children": [recurse(n) for n in top_nodes]}


# ---------------------------------------------------------------------------
# Run on each dataset
# ---------------------------------------------------------------------------

def name_of_node(node, dataset):
    """Best-effort: return the dataset's name for `node` (by member set)."""
    member_idxs = collect_descendant_leaf_indices(node, dataset)
    target = tuple(sorted(member_idxs))
    for name, members in dataset["node_members"].items():
        if tuple(sorted(members)) == target:
            return name
    return f"<unnamed depth={node.depth()} count={int(node.count)}>"


# The frozen tree's instance set lives in the leaves' av_count.  Reconstruct
# member-instance indices by matching attribute distributions to X.  Some
# datasets contain *identical* instances (e.g. classical_guitar and
# folk_guitar in Music), so we greedily consume X-indices as we match.
def collect_descendant_leaf_indices(node, dataset):
    X = dataset["X"]
    feature_idx = dataset["feature_idx"]
    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)

    def matches(leaf, i):
        for _, fidx in feature_idx.items():
            a = int(fidx)
            v = int(X[i, fidx])
            if a not in leaf.av_count:
                return False
            if v not in leaf.av_count.get(a, {}):
                return False
        return True

    used = set()
    members = []
    for leaf in leaves:
        # Each leaf may hold count > 1 (merged identical instances).
        for _ in range(int(leaf.count)):
            for i in range(len(X)):
                if i in used:
                    continue
                if matches(leaf, i):
                    members.append(i)
                    used.add(i)
                    break
    return members


def verify_dataset(ds_name, dataset, out_file):
    out_file.write(f"\n{'='*78}\n")
    out_file.write(f"  Dataset: {ds_name}\n")
    out_file.write(f"{'='*78}\n")
    print(f"\n{'='*78}\n  Dataset: {ds_name}\n{'='*78}")

    X           = dataset["X"]
    feature_idx = dataset["feature_idx"]
    node_members= dataset["node_members"]
    levels      = dataset["levels"]
    expected_basic = set(levels.get("Basic", []))

    hierarchy = build_frozen_hierarchy(node_members, levels, X, feature_idx)
    tree = FrozenCobwebDiscreteTree(hierarchy, alpha=1.0, weight_attr=True)

    # Print per-node scores by level
    out_file.write("\nPer-node basic_level_score (α-agnostic):\n")
    print("\nPer-node basic_level_score (α-agnostic):")
    # Collect all nodes via BFS over tree.root, count leaves for L.
    queue, all_nodes = [tree.root], []
    while queue:
        n = queue.pop(0)
        all_nodes.append(n)
        for c in n.children:
            queue.append(c)
    n_leaves = sum(1 for n in all_nodes if not n.children)
    # Map node-object → dataset name (by member-set)
    obj_to_name = {}
    for n in all_nodes:
        obj_to_name[id(n)] = name_of_node(n, dataset)
    for lvl_name, names_at_lvl in levels.items():
        out_file.write(f"  [{lvl_name}]\n")
        print(f"  [{lvl_name}]")
        for n in all_nodes:
            label = obj_to_name[id(n)]
            if label in names_at_lvl:
                s = n.basic_level_score(n_leaves)
                line = f"    {label:<24} depth={n.depth()} count={int(n.count):>3} score={s:+.5f}"
                out_file.write(line + "\n")
                print(line)

    # Run the frontier (FrozenCobwebDiscreteTree wraps a CobwebDiscreteTree).
    frontier = tree._tree.get_basic_frontier()
    frontier_names = sorted(obj_to_name[id(n)] for n in frontier)
    out_file.write(f"\nFrontier ({len(frontier)} nodes):\n")
    print(f"\nFrontier ({len(frontier)} nodes):")
    for n in frontier:
        label = obj_to_name[id(n)]
        s = n.basic_level_score(n_leaves)
        line = f"  {label:<24} depth={n.depth()} count={int(n.count):>3} score={s:+.5f}"
        out_file.write(line + "\n")
        print(line)

    # Expected basic level match
    found = set(frontier_names)
    if expected_basic:
        match = found == expected_basic
        line = (f"\nExpected basic level: {sorted(expected_basic)}"
                f"\n  Frontier match: {'YES ✓' if match else 'NO ✗'}")
        out_file.write(line + "\n")
        print(line)
        missing = expected_basic - found
        extra = found - expected_basic
        if missing:
            out_file.write(f"  Missing from frontier: {sorted(missing)}\n")
            print(f"  Missing from frontier: {sorted(missing)}")
        if extra:
            out_file.write(f"  Extra in frontier: {sorted(extra)}\n")
            print(f"  Extra in frontier: {sorted(extra)}")

    # Verify one-BL-per-path
    frontier_ids = {id(n) for n in frontier}
    leaves = [n for n in all_nodes if not n.children]
    counts = Counter()
    for leaf in leaves:
        node = leaf
        c = 0
        while node is not None:
            if id(node) in frontier_ids:
                c += 1
            node = node.parent
        counts[c] += 1
    out_file.write(f"\nOne-BL-per-path check ({len(leaves)} leaves):\n")
    print(f"\nOne-BL-per-path check ({len(leaves)} leaves):")
    for k in sorted(counts.keys()):
        flag = "" if k == 1 else "  ← violation"
        line = f"  {k} frontier nodes on path : {counts[k]}{flag}"
        out_file.write(line + "\n")
        print(line)


def main():
    out_path = os.path.join(OUT_DIR, "corter_gluck_basic_frontier.txt")
    with open(out_path, "w") as out_file:
        out_file.write(
            "Alpha-agnostic basic_level_score and get_basic_frontier on the\n"
            "Corter & Gluck hierarchies (Murphy & Smith + Begriffshierarchien\n"
            "+ Fruit + Music + Furniture).\n"
        )
        for ds_name, dataset in cgh.datasets.items():
            verify_dataset(ds_name, dataset, out_file)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
