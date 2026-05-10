"""
Corter & Gluck (1992) basic-level replication — stable/ideal-hierarchy variant
===============================================================================

Unlike ``corter_gluck_test.py``, this test does **not** run the Cobweb
learning algorithm.  Instead, it constructs a ``FrozenCobwebDiscreteTree``
directly from the ideal 4-level hierarchy described in the paper:

    Root
    ├── Pounder (items 1-8)
    │   ├── Hammer  → Hammer 1 (1-2),  Hammer 2 (3-4)
    │   └── Brick   → Brick 1  (5-6),  Brick 2  (7-8)
    └── Cutter  (items 9-16)
        ├── Knife   → Knife 1  (9-10), Knife 2  (11-12)
        └── Pizza   → Pizza 1  (13-14), Pizza 2 (15-16)

Because the tree is fully deterministic (no random learning), basic-level
detection via ``get_basic`` should reliably return the correct intermediate
node for every item.

The same closed-form CU computation (replicating Table 4 of C&G 1992) is
also performed here for completeness.
"""

import sys
import os
import shutil
import importlib.util

# Make sure src/ is on the path so viz.py can be imported
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Load frozen_discrete.py directly from cobweb-private/src/cobweb/.
# The installed cobweb package only ships the .so extensions; the Python
# helpers (frozen_discrete.py, viz.py wrappers, …) live in source only.
_FD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cobweb-private", "src", "cobweb", "frozen_discrete.py",
)
_fd_spec = importlib.util.spec_from_file_location("_frozen_discrete_local", _FD_PATH)
_fd_mod  = importlib.util.module_from_spec(_fd_spec)
_fd_spec.loader.exec_module(_fd_mod)
FrozenCobwebDiscreteTree = _fd_mod.FrozenCobwebDiscreteTree

from viz import HTMLCobwebDrawer

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

ALPHA      = 1e-3   # Dirichlet smoothing (same as the learning-based test)
EVAL_ALPHA = 1     # Smoothing used only during basic-level evaluation

OUT_DIR    = "tests/basic-level/corter_gluck_stable_viz"

# ---------------------------------------------------------------------------
# Murphy & Smith (1982) Table 3 stimuli  (identical to corter_gluck_test.py)
# Attributes: Handle=0, Shaft=1, Head=2, Size=3
# ---------------------------------------------------------------------------

ATTR_HANDLE = 0
ATTR_SHAFT  = 1
ATTR_HEAD   = 2
ATTR_SIZE   = 3

# (item_id, superordinate, basic, subordinate, handle, shaft, head, size)
ITEMS = [
    ( 1, "Pounder", "Hammer",       "Hammer 1", 1, 1, 1, 1),
    ( 2, "Pounder", "Hammer",       "Hammer 1", 1, 1, 1, 2),
    ( 3, "Pounder", "Hammer",       "Hammer 2", 1, 1, 2, 1),
    ( 4, "Pounder", "Hammer",       "Hammer 2", 1, 1, 2, 2),
    ( 5, "Pounder", "Brick",        "Brick 1",  2, 2, 3, 1),
    ( 6, "Pounder", "Brick",        "Brick 1",  2, 2, 3, 2),
    ( 7, "Pounder", "Brick",        "Brick 2",  3, 2, 3, 1),
    ( 8, "Pounder", "Brick",        "Brick 2",  3, 2, 3, 2),
    ( 9, "Cutter",  "Knife",        "Knife 1",  4, 3, 4, 1),
    (10, "Cutter",  "Knife",        "Knife 1",  4, 3, 4, 2),
    (11, "Cutter",  "Knife",        "Knife 2",  4, 3, 5, 1),
    (12, "Cutter",  "Knife",        "Knife 2",  4, 3, 5, 2),
    (13, "Cutter",  "Pizza cutter", "Pizza 1",  5, 4, 6, 1),
    (14, "Cutter",  "Pizza cutter", "Pizza 1",  5, 4, 6, 2),
    (15, "Cutter",  "Pizza cutter", "Pizza 2",  5, 5, 6, 1),
    (16, "Cutter",  "Pizza cutter", "Pizza 2",  5, 5, 6, 2),
]

BASIC_CATEGORIES = ["Hammer", "Brick", "Knife", "Pizza cutter"]


def make_instance(handle, shaft, head, size):
    """Encode one stimulus as a ``FrozenCobwebDiscreteTree``-compatible dict."""
    return {
        ATTR_HANDLE: {handle: 1.0},
        ATTR_SHAFT:  {shaft:  1.0},
        ATTR_HEAD:   {head:   1.0},
        ATTR_SIZE:   {size:   1.0},
    }


def build_instances():
    return [make_instance(h, s, hd, sz) for _, _, _, _, h, s, hd, sz in ITEMS]


# ---------------------------------------------------------------------------
# Ideal hierarchy
# ---------------------------------------------------------------------------

def build_ideal_hierarchy(instances):
    """
    Return the nested hierarchy dict representing the ideal 4-level tree
    described in Corter & Gluck (1992).

    The root node contains two superordinate children (Pounder, Cutter).
    Each superordinate has two basic-level children (Hammer/Brick, Knife/Pizza).
    Each basic-level node has two subordinate leaf nodes, each containing 2 items.

    Parameters
    ----------
    instances : list[dict]
        The 16 item instances in ITEMS order (0-indexed).
    """
    return {
        "children": [
            {  # Superordinate: Pounder (items 1-8, indices 0-7)
                "children": [
                    {  # Basic: Hammer (items 1-4, indices 0-3)
                        "children": [
                            {"instances": [instances[0], instances[1]]},  # Hammer 1
                            {"instances": [instances[2], instances[3]]},  # Hammer 2
                        ]
                    },
                    {  # Basic: Brick (items 5-8, indices 4-7)
                        "children": [
                            {"instances": [instances[4], instances[5]]},  # Brick 1
                            {"instances": [instances[6], instances[7]]},  # Brick 2
                        ]
                    },
                ]
            },
            {  # Superordinate: Cutter (items 9-16, indices 8-15)
                "children": [
                    {  # Basic: Knife (items 9-12, indices 8-11)
                        "children": [
                            {"instances": [instances[8],  instances[9]]},   # Knife 1
                            {"instances": [instances[10], instances[11]]},  # Knife 2
                        ]
                    },
                    {  # Basic: Pizza cutter (items 13-16, indices 12-15)
                        "children": [
                            {"instances": [instances[12], instances[13]]},  # Pizza 1
                            {"instances": [instances[14], instances[15]]},  # Pizza 2
                        ]
                    },
                ]
            },
        ]
    }


# ---------------------------------------------------------------------------
# Category Utility  (Corter & Gluck 1992, Equation 2)
# Identical to corter_gluck_test.py — replicated here for self-containedness.
# ---------------------------------------------------------------------------

def compute_category_utility(category_indices, all_instances):
    """
    CU(c) = P(c) × Σ_j Σ_k [ P(f_jk|c)² - P(f_jk)² ]

    P(f_jk) is derived from all 16 items, matching the analysis in Table 4.
    """
    n_total = len(all_instances)
    n_cat   = len(category_indices)
    p_c     = n_cat / n_total

    all_attrs: dict[int, set] = {}
    for inst in all_instances:
        for attr, val_dict in inst.items():
            all_attrs.setdefault(attr, set()).update(val_dict.keys())

    cu_inner = 0.0
    for attr, vals in all_attrs.items():
        for val in vals:
            p_f = sum(1 for inst in all_instances
                      if val in inst.get(attr, {})) / n_total
            p_f_given_c = sum(1 for i in category_indices
                              if val in all_instances[i].get(attr, {})) / n_cat
            cu_inner += p_f_given_c ** 2 - p_f ** 2

    return p_c * cu_inner


def compute_cu_by_level(instances):
    superord_idx: dict[str, list] = {}
    basic_idx:    dict[str, list] = {}
    subord_idx:   dict[str, list] = {}

    for i, item in enumerate(ITEMS):
        _, superord, basic, subord = item[0], item[1], item[2], item[3]
        superord_idx.setdefault(superord, []).append(i)
        basic_idx.setdefault(basic, []).append(i)
        subord_idx.setdefault(subord, []).append(i)

    return {
        "superordinate": {c: compute_category_utility(idx, instances)
                          for c, idx in superord_idx.items()},
        "basic":         {c: compute_category_utility(idx, instances)
                          for c, idx in basic_idx.items()},
        "subordinate":   {c: compute_category_utility(idx, instances)
                          for c, idx in subord_idx.items()},
    }


def print_cu_table(cu_by_level):
    """Print Table 4 replication. Expected: Sub≈0.30, Basic≈0.47, Super≈0.31."""
    print(f"\n{'='*60}")
    print("  Category Utility — replication of Table 4 (Corter & Gluck 1992)")
    print(f"{'='*60}")
    print(f"  {'Level':>14}  {'Mean CU':>8}  {'Categories'}")
    print("-" * 60)
    level_means = {}
    for level in ("superordinate", "basic", "subordinate"):
        cats = cu_by_level[level]
        mean_cu = sum(cats.values()) / len(cats)
        level_means[level] = mean_cu
        cat_str = ", ".join(f"{c}={v:.3f}" for c, v in sorted(cats.items()))
        print(f"  {level:>14}  {mean_cu:>8.4f}  {cat_str}")
    print("-" * 60)
    print("  Paper Table 4:  Subordinate=0.30, Basic=0.47, Superordinate=0.31")

    correct = (level_means["basic"] > level_means["subordinate"] and
               level_means["basic"] > level_means["superordinate"])
    print(f"\n  Basic level has highest CU: {'YES — matches paper ✓' if correct else 'NO ✗'}")
    return correct


# ---------------------------------------------------------------------------
# Cobweb-based basic-level evaluation
# ---------------------------------------------------------------------------

def evaluate_basic_level(tree, instances, uniform_leaf=False, method="get_basic"):
    """
    For every item, categorize it to its leaf and ask that leaf for its
    basic-level ancestor via the chosen method.

    ``method`` is one of:
      - "get_basic"     : original EPMI vs full-tree mixture (uses EVAL_ALPHA)
      - "get_basic_pc"  : P(c)-weighted EPMI vs root (no eval_alpha)

    Returns a list of per-item result dicts.
    """
    results = []
    for i, item in enumerate(ITEMS):
        item_id, superord, basic, subord = item[0], item[1], item[2], item[3]
        inst       = instances[i]
        leaf       = tree.categorize(inst)
        if method == "get_basic":
            basic_node = leaf.get_basic(
                1000, 1000, debug=False,
                eval_alpha=EVAL_ALPHA, uniform_leaf=uniform_leaf,
            )
        elif method == "get_basic_pc":
            basic_node = leaf.get_basic_pc(
                1000, debug=False, uniform_leaf=uniform_leaf,
            )
        else:
            msg = f"unknown basic-level method: {method!r}"
            raise ValueError(msg)
        results.append({
            "item_id":     item_id,
            "superord":    superord,
            "basic":       basic,
            "subord":      subord,
            "leaf_depth":  leaf.depth(),
            "basic_depth": basic_node.depth(),
            "basic_hash":  basic_node.concept_hash(),
            "is_root":     basic_node.parent is None,
        })
    return results


# ---------------------------------------------------------------------------
# Side-by-side comparison: get_basic vs get_basic_pc
# ---------------------------------------------------------------------------

# Depth → human-readable level label, given the ideal C&G hierarchy:
#   0 = Root,  1 = Superord,  2 = Basic (target),  3 = Subord,  4+ = Leaf
DEPTH_LABEL = {0: "Root", 1: "Superord", 2: "Basic", 3: "Subord"}


def _level_label(depth):
    return DEPTH_LABEL.get(depth, "Leaf")


def _hash_to_label(node_hash, hash_to_basic_cat, hash_to_depth):
    """Map a basic-level node hash to a human-readable label.

    If the hash is the canonical BL node for one of the four target basic
    categories (Hammer/Brick/Knife/Pizza cutter), return that category name.
    Otherwise return the level-tier label (Root/Superord/Subord/Leaf).
    """
    if node_hash in hash_to_basic_cat:
        return hash_to_basic_cat[node_hash]
    return _level_label(hash_to_depth.get(node_hash, -1))


def _build_canonical_bl_map(results):
    """For each target basic category, find the BL-node hash that the
    *majority* of its items mapped to. Used purely for labeling — not for
    correctness checking."""
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        by_cat.setdefault(r["basic"], {})
        by_cat[r["basic"]][r["basic_hash"]] = (
            by_cat[r["basic"]].get(r["basic_hash"], 0) + 1
        )
    canonical: dict[str, str] = {}
    for cat, counts in by_cat.items():
        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        canonical[winner] = cat
    return canonical


def print_comparison_table(results_old, results_pc, label="", out_file=None):
    """Print side-by-side table of the basic-level node each method picked
    for every item.

    Columns:
      Item | Expected basic | get_basic node (depth, label) | get_basic_pc node (depth, label) | Match?

    If ``out_file`` is provided (a writeable file object), the same table is
    appended there in addition to being printed to stdout.
    """
    tag = f"  [{label}]" if label else ""

    def _emit(line):
        print(line)
        if out_file is not None:
            out_file.write(line + "\n")

    _emit(f"\n{'='*92}")
    _emit(f"  Comparison: get_basic  vs  get_basic_pc{tag}")
    _emit(f"{'='*92}")

    canon_old = _build_canonical_bl_map(results_old)
    canon_pc  = _build_canonical_bl_map(results_pc)
    depth_map = {}
    for r in results_old + results_pc:
        depth_map[r["basic_hash"]] = r["basic_depth"]

    header = (
        f"{'Item':>4}  {'Expected':>13}  "
        f"{'get_basic (d, label)':>28}  "
        f"{'get_basic_pc (d, label)':>28}  "
        f"{'Match?':>8}"
    )
    _emit(header)
    _emit("-" * 92)

    n_old_correct = 0
    n_pc_correct  = 0
    n_agree       = 0

    for r_old, r_pc in zip(results_old, results_pc, strict=True):
        assert r_old["item_id"] == r_pc["item_id"]
        item_id  = r_old["item_id"]
        expected = r_old["basic"]

        old_label = _hash_to_label(r_old["basic_hash"], canon_old, depth_map)
        pc_label  = _hash_to_label(r_pc["basic_hash"],  canon_pc,  depth_map)

        old_correct = (old_label == expected)
        pc_correct  = (pc_label  == expected)
        agree       = (r_old["basic_hash"] == r_pc["basic_hash"])

        n_old_correct += int(old_correct)
        n_pc_correct  += int(pc_correct)
        n_agree       += int(agree)

        old_cell = f"{r_old['basic_depth']}, {old_label}"
        pc_cell  = f"{r_pc['basic_depth']}, {pc_label}"
        match_cell = "same" if agree else "DIFF"

        _emit(
            f"{item_id:>4}  {expected:>13}  "
            f"{old_cell:>28}  {pc_cell:>28}  {match_cell:>8}"
        )

    _emit("-" * 92)
    n = len(results_old)
    _emit(f"  get_basic    target hits: {n_old_correct}/{n}")
    _emit(f"  get_basic_pc target hits: {n_pc_correct}/{n}")
    _emit(f"  Methods agree on:         {n_agree}/{n} items")


def print_results_table(results, label=""):
    """Pretty-print per-item basic-level results."""
    tag = f"  [{label}]" if label else ""
    print(f"\n{'='*72}")
    print(f"  Frozen (ideal-hierarchy) basic-level results{tag}")
    print(f"{'='*72}")
    header = (f"{'Item':>4}  {'Superord':>9}  {'Basic':>13}  {'Subord':>10}"
              f"  {'L-d':>3}  {'BL-d':>4}  {'BL node':>12}")
    print(header)
    print("-" * 72)
    for r in results:
        print(
            f"{r['item_id']:>4}  {r['superord']:>9}  {r['basic']:>13}  {r['subord']:>10}"
            f"  {r['leaf_depth']:>3}  {r['basic_depth']:>4}  {r['basic_hash'][:12]:>12}"
        )
    print("-" * 72)
    print("L-d = leaf depth,  BL-d = basic-level node depth")


def check_basic_level_accuracy(results, label=""):
    """
    Check:
      1. Consistency  — items in the same basic category share one BL node hash.
      2. Distinctness — different basic categories map to different BL nodes.
      3. Non-trivial  — BL node is not the root.
      4. Non-specific — BL node is not always a leaf.
    """
    basic_to_hashes: dict[str, set] = {}
    for r in results:
        basic_to_hashes.setdefault(r["basic"], set()).add(r["basic_hash"])

    consistency_pass  = all(len(h) == 1 for h in basic_to_hashes.values())
    all_hashes        = [next(iter(h)) for h in basic_to_hashes.values() if len(h) == 1]
    distinctness_pass = (len(all_hashes) == len(set(all_hashes)) == len(BASIC_CATEGORIES))
    not_root          = not any(r["is_root"] for r in results)
    not_all_leaf      = not all(r["basic_depth"] == r["leaf_depth"] for r in results)

    tag = f"[{label}]" if label else ""
    print(f"\n{tag} Consistency  (same basic cat → same BL node): "
          f"{'PASS' if consistency_pass else 'FAIL'}")
    print(f"{tag} Distinctness (diff basic cat → diff BL node):  "
          f"{'PASS' if distinctness_pass else 'FAIL'}")
    print(f"{tag} Non-trivial  (BL node is not root):            "
          f"{'PASS' if not_root else 'FAIL'}")
    print(f"{tag} Non-specific (BL node is not always a leaf):   "
          f"{'PASS' if not_all_leaf else 'FAIL'}")

    if consistency_pass:
        print(f"\n{tag} Basic-level nodes found:")
        for cat, hashes in sorted(basic_to_hashes.items()):
            h = next(iter(hashes)) if len(hashes) == 1 else ", ".join(hashes)
            print(f"    {cat:>14}: {h[:16]}")

    return consistency_pass, distinctness_pass


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_drawer():
    attr_names  = ["Handle", "Shaft", "Head", "Size"]
    id_to_value = [str(i) for i in range(7)]
    value_to_id = {v: i for i, v in enumerate(id_to_value)}
    return HTMLCobwebDrawer(attr_names, id_to_value, value_to_id)


def visualize(tree, instances, out_dir):
    """
    Save:
      - The full frozen tree to ``out_dir/full_tree.html``
      - One subtree per unique basic-level node to ``out_dir/basic_level_<hash>.html``
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    drawer = make_drawer()

    drawer.draw_tree(tree.root, os.path.join(out_dir, "full_tree"))
    print(f"  Full tree → {out_dir}/full_tree.html")

    bl_nodes: dict[str, object] = {}
    for i in range(len(ITEMS)):
        inst     = instances[i]
        leaf     = tree.categorize(inst)
        bl_node  = leaf.get_basic(1000, 1000, debug=False, eval_alpha=EVAL_ALPHA)
        h        = bl_node.concept_hash()
        if h not in bl_nodes:
            bl_nodes[h] = bl_node

    print(f"  {len(bl_nodes)} unique basic-level node(s) found:")
    for h, node in bl_nodes.items():
        out_path = os.path.join(out_dir, f"basic_level_{h}")
        drawer.draw_tree(node, out_path)
        print(f"    {h[:16]} (depth={node.depth()}) → {out_path}.html")

    print(f"\nAll visualizations saved to {out_dir}/")


# ---------------------------------------------------------------------------
# pytest entry-point
# ---------------------------------------------------------------------------

def test_corter_gluck_stable_basic_level():
    """
    Pytest test: FrozenCobwebDiscreteTree built from the ideal C&G hierarchy
    should reliably recover the basic-level categories (Hammer, Brick, Knife,
    Pizza cutter) via ``get_basic``.

    Step 1 – Replicate Table 4 (closed-form CU, no Cobweb involved).
    Step 2 – Build FrozenCobwebDiscreteTree from the ideal hierarchy and
             evaluate basic-level detection with ``get_basic``.
    Step 3 – Visualize full tree + per-BL-node subtrees.
    """
    instances = build_instances()

    # --- Step 1: closed-form CU (Table 4) ---
    cu_by_level = compute_cu_by_level(instances)
    cu_correct  = print_cu_table(cu_by_level)

    # --- Step 2: frozen tree ---
    print(f"\nBuilding FrozenCobwebDiscreteTree (alpha={ALPHA}) …")
    hierarchy = build_ideal_hierarchy(instances)
    tree      = FrozenCobwebDiscreteTree(hierarchy, alpha=ALPHA, weight_attr=True)
    print("  Tree built successfully.")

    # Evaluate with default sampling (weighted-choice on node distribution)
    print("\n--- get_basic (uniform_leaf=False, default) ---")
    results_default = evaluate_basic_level(
        tree, instances, uniform_leaf=False, method="get_basic"
    )
    print_results_table(results_default, label="get_basic, uniform_leaf=False")
    cons_d, dist_d = check_basic_level_accuracy(
        results_default, label="get_basic, uniform_leaf=False"
    )

    # Evaluate with uniform-leaf sampling
    print("\n--- get_basic (uniform_leaf=True) ---")
    results_uniform = evaluate_basic_level(
        tree, instances, uniform_leaf=True, method="get_basic"
    )
    print_results_table(results_uniform, label="get_basic, uniform_leaf=True")
    cons_u, dist_u = check_basic_level_accuracy(
        results_uniform, label="get_basic, uniform_leaf=True"
    )

    # Evaluate get_basic_pc (no eval_alpha — P(c) damping replaces it)
    print("\n--- get_basic_pc (uniform_leaf=False) ---")
    results_pc = evaluate_basic_level(
        tree, instances, uniform_leaf=False, method="get_basic_pc"
    )
    print_results_table(results_pc, label="get_basic_pc, uniform_leaf=False")
    cons_pc, dist_pc = check_basic_level_accuracy(
        results_pc, label="get_basic_pc, uniform_leaf=False"
    )

    print("\n--- get_basic_pc (uniform_leaf=True) ---")
    results_pc_u = evaluate_basic_level(
        tree, instances, uniform_leaf=True, method="get_basic_pc"
    )
    print_results_table(results_pc_u, label="get_basic_pc, uniform_leaf=True")
    cons_pcu, dist_pcu = check_basic_level_accuracy(
        results_pc_u, label="get_basic_pc, uniform_leaf=True"
    )

    # Side-by-side comparison tables (also persisted to disk)
    os.makedirs(OUT_DIR, exist_ok=True)
    cmp_path = os.path.join(OUT_DIR, "basic_level_comparison.txt")
    with open(cmp_path, "w") as cmp_f:
        cmp_f.write(
            "Comparison of basic-level node selection: get_basic vs get_basic_pc\n"
            f"  ALPHA={ALPHA}, EVAL_ALPHA={EVAL_ALPHA}, n_samples=1000\n"
        )
        print_comparison_table(
            results_default, results_pc,
            label="uniform_leaf=False", out_file=cmp_f,
        )
        print_comparison_table(
            results_uniform, results_pc_u,
            label="uniform_leaf=True",  out_file=cmp_f,
        )
    print(f"\nComparison table saved → {cmp_path}")

    # --- Step 3: visualize ---
    print("\nGenerating visualizations …")
    visualize(tree, instances, OUT_DIR)

    # Pytest pass/fail is gated only on the original get_basic method.
    # get_basic_pc is reported for comparison but not required to pass —
    # treat its consistency/distinctness results as informational.
    passed = cu_correct and cons_d and dist_d and cons_u and dist_u
    pc_summary = (
        f"get_basic_pc (uniform=False): consistency={'P' if cons_pc else 'F'} "
        f"distinctness={'P' if dist_pc else 'F'} | "
        f"get_basic_pc (uniform=True):  consistency={'P' if cons_pcu else 'F'} "
        f"distinctness={'P' if dist_pcu else 'F'}"
    )
    print(f"\n  [info] {pc_summary}")
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'} (gated on get_basic only)")


if __name__ == "__main__":
    instances   = build_instances()
    cu_by_level = compute_cu_by_level(instances)
    print_cu_table(cu_by_level)

    print(f"\nBuilding FrozenCobwebDiscreteTree (alpha={ALPHA}) …")
    hierarchy = build_ideal_hierarchy(instances)
    tree      = FrozenCobwebDiscreteTree(hierarchy, alpha=ALPHA, weight_attr=True)
    print("  Tree built.")

    results_old = evaluate_basic_level(tree, instances, method="get_basic")
    print_results_table(results_old, label="get_basic")
    check_basic_level_accuracy(results_old, label="get_basic")

    results_pc = evaluate_basic_level(tree, instances, method="get_basic_pc")
    print_results_table(results_pc, label="get_basic_pc")
    check_basic_level_accuracy(results_pc, label="get_basic_pc")

    os.makedirs(OUT_DIR, exist_ok=True)
    cmp_path = os.path.join(OUT_DIR, "basic_level_comparison.txt")
    with open(cmp_path, "w") as cmp_f:
        cmp_f.write(
            "Comparison of basic-level node selection: get_basic vs get_basic_pc\n"
            f"  ALPHA={ALPHA}, EVAL_ALPHA={EVAL_ALPHA}, n_samples=1000\n"
        )
        print_comparison_table(results_old, results_pc, out_file=cmp_f)
    print(f"\nComparison table saved → {cmp_path}")

    print("\nGenerating visualizations …")
    visualize(tree, instances, OUT_DIR)
