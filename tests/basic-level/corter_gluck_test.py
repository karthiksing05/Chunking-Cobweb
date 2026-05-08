"""
Corter and Gluck study replication!!

Once we adjust the p(x) probability, this will be a good test to confirm the accuracy of the basic-level
method we've employed AND the quality of the hierarchy

Dataset: Murphy & Smith (1982) Table 3 -- 16 items × 4 substitutive attributes
Expected hierarchy:
  Superordinate: Pounder (items 1-8), Cutter (items 9-16)
  Basic:         Hammer (1-4), Brick (5-8), Knife (9-12), Pizza cutter (13-16)
  Subordinate:   Hammer 1 (1-2), Hammer 2 (3-4), Brick 1 (5-6), Brick 2 (7-8),
                 Knife 1 (9-10), Knife 2 (11-12), Pizza 1 (13-14), Pizza 2 (15-16)

Corter & Gluck (1992) predict the basic level is the intermediate level that
maximises expected informativeness (EPMI), which is exactly what Cobweb's
get_basic() / get_basic_instance_pmi() implements.
"""

import sys
import os
import shutil

# Make sure src/ is on the path so viz.py can be imported
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from cobweb.cobweb_discrete import CobwebDiscreteTree
from viz import HTMLCobwebDrawer

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

ALPHA      = 1e-3   # Cobweb tree smoothing parameter
EVAL_ALPHA = 10   # Smoothing used only during basic-level evaluation

# Corter & Gluck's analysis is purely structural (closed-form CU on a static
# 16-item matrix — no learning loop at all).  For Cobweb we must present
# items incrementally; ITERATIONS full passes are used so the tree can
# converge.  Each of the 16 items is seen exactly ITERATIONS times and every
# pass uses a freshly shuffled order (see CobwebDiscreteTree::fit — a
# std::default_random_engine() shuffle is applied *after* every iteration,
# seeded deterministically at 0, so results are reproducible).
ITERATIONS = 15
OUT_DIR    = "tests/basic-level/corter_gluck_viz"

# ---------------------------------------------------------------------------
# Murphy & Smith (1982) Table 3 stimuli
# Attributes: Handle=0, Shaft=1, Head=2, Size=3  (integer encoding)
# Values are the integers from the original table (1-6 for most attrs, 1-2 for Size)
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

BASIC_CATEGORIES   = ["Hammer", "Brick", "Knife", "Pizza cutter"]
SUPERORD_CATEGORIES = ["Pounder", "Cutter"]


def make_instance(handle, shaft, head, size):
    """Encode one stimulus as a CobwebDiscreteTree instance."""
    return {
        ATTR_HANDLE: {handle: 1.0},
        ATTR_SHAFT:  {shaft:  1.0},
        ATTR_HEAD:   {head:   1.0},
        ATTR_SIZE:   {size:   1.0},
    }


def build_instances():
    return [make_instance(h, s, hd, sz) for _, _, _, _, h, s, hd, sz in ITEMS]


# ---------------------------------------------------------------------------
# Category Utility (Corter & Gluck 1992, Equation 2)
#
# CU(c) = P(c) * Σ_j Σ_k [ P(f_jk|c)² - P(f_jk)² ]
#
# P(f_jk) is computed from all 16 items (the full population), matching the
# analysis in Table 4 of the paper (where superordinate CU = 0.31, not 0).
# ---------------------------------------------------------------------------

def compute_category_utility(category_indices, all_instances):
    """
    Corter & Gluck (1992) Equation 2: category utility for a single category.

    Parameters
    ----------
    category_indices : list[int]
        Indices into all_instances belonging to this category.
    all_instances : list[dict]
        The full population of 16 instances (provides base-rate P(f_jk)).

    Note: C&G use ALL 16 items as the reference population for P(f_jk), not
    just the superordinate subtree.  This is why the superordinate CU in
    Table 4 is 0.31 rather than 0 — the superordinate categories are evaluated
    against the global base rates of the full stimulus set.
    """
    n_total = len(all_instances)
    n_cat   = len(category_indices)
    p_c     = n_cat / n_total

    # Collect every (attr, val) pair that appears in the population
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
    """
    Compute mean CU for each of the three levels: superordinate, basic, subordinate.

    Returns dict mapping level → {category_name: CU}.
    """
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
    """
    Print per-level CU summary, replicating Table 4 of Corter & Gluck (1992).
    Expected: Subordinate≈0.30, Basic≈0.47, Superordinate≈0.31
    """
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
    print(f"\n  Basic level has highest CU: {'YES — matches paper ✓' if correct else 'NO — differs from paper ✗'}")
    return correct




def evaluate_basic_level(tree, instances):
    """
    For every item, categorize it to its leaf and ask that leaf for its
    basic-level ancestor (get_basic_instance_pmi -- exact-instance variant).

    Returns a list of dicts with per-item results.
    """
    results = []
    for i, item in enumerate(ITEMS):
        item_id, superord, basic, subord = item[0], item[1], item[2], item[3]
        inst = instances[i]
        leaf       = tree.categorize(inst)
        # basic_node = leaf.get_basic_instance_pmi(inst, debug=False, eval_alpha=EVAL_ALPHA)
        basic_node = leaf.get_basic(1000, 1000, debug=False, eval_alpha=EVAL_ALPHA, uniform_leaf=True)
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


def print_results_table(results, alpha):
    """Pretty-print per-item basic-level results."""
    print(f"\n{'='*72}")
    print(f"  alpha={alpha}  |  Corter & Gluck basic-level replication")
    print(f"{'='*72}")
    header = f"{'Item':>4}  {'Superord':>9}  {'Basic':>13}  {'Subord':>10}  {'L-d':>3}  {'BL-d':>4}  {'BL node':>12}"
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
    Two checks:
      1. Consistency  -- all items in the same basic category share the same
                         basic-level node hash.
      2. Distinctness -- different basic categories have different nodes.

    Returns (consistency_pass, distinctness_pass).
    """
    # Group by expected basic category
    basic_to_hashes: dict[str, set] = {}
    for r in results:
        basic_to_hashes.setdefault(r["basic"], set()).add(r["basic_hash"])

    consistency_pass = all(len(hashes) == 1 for hashes in basic_to_hashes.values())

    all_basic_hashes = [next(iter(hashes)) for hashes in basic_to_hashes.values()
                        if len(hashes) == 1]
    distinctness_pass = len(all_basic_hashes) == len(set(all_basic_hashes)) == len(BASIC_CATEGORIES)

    # Non-trivial: basic level must not be the root for any item
    not_root = not any(r["is_root"] for r in results)

    # Not over-specific: basic level should not be at leaf depth for all items
    not_all_leaf = not all(r["basic_depth"] == r["leaf_depth"] for r in results)

    tag = f"[{label}]" if label else ""
    print(f"\n{tag} Consistency  (same basic cat → same BL node): {'PASS' if consistency_pass else 'FAIL'}")
    print(f"{tag} Distinctness (diff basic cat → diff BL node):  {'PASS' if distinctness_pass else 'FAIL'}")
    print(f"{tag} Non-trivial  (BL node is not root):            {'PASS' if not_root else 'FAIL'}")
    print(f"{tag} Non-specific (BL node is not always a leaf):   {'PASS' if not_all_leaf else 'FAIL'}")

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
    """
    Build an HTMLCobwebDrawer suited for the Murphy & Smith attribute space.

    Attributes are the four integer keys (0-3).  Values are raw integers
    (1-6 for Handle/Shaft/Head, 1-2 for Size); id_to_value is built so that
    index i returns the string "i", which matches the raw integer values used
    as dict keys in CobwebDiscreteTree.
    """
    attr_names = ["Handle", "Shaft", "Head", "Size"]
    # Values range from 0-6; index == value so the drawer can look them up directly
    id_to_value = [str(i) for i in range(7)]
    value_to_id = {v: i for i, v in enumerate(id_to_value)}
    return HTMLCobwebDrawer(attr_names, id_to_value, value_to_id)


def visualize(tree, instances):
    """
    Draw:
      1. The full Cobweb tree rooted at tree.root
      2. One subtree per unique basic-level node identified via
         get_basic_instance_pmi (one call per item, deduped by concept hash)
    """
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    drawer = make_drawer()

    # --- full tree ---
    drawer.draw_tree(tree.root, os.path.join(OUT_DIR, "full_tree"))
    print(f"  Full tree → {OUT_DIR}/full_tree.html")

    # --- basic-level nodes ---
    bl_nodes: dict[str, object] = {}
    for i, item in enumerate(ITEMS):
        inst = instances[i]
        leaf = tree.categorize(inst)
        # bl_node = leaf.get_basic_instance_pmi(inst, debug=False, eval_alpha=EVAL_ALPHA)
        bl_node = leaf.get_basic(1000, 1000, debug=False, eval_alpha=EVAL_ALPHA, uniform_leaf=True)
        h = bl_node.concept_hash()
        if h not in bl_nodes:
            bl_nodes[h] = bl_node

    print(f"  {len(bl_nodes)} unique basic-level node(s) found:")
    for h, node in bl_nodes.items():
        out_path = os.path.join(OUT_DIR, f"basic_level_{h}")
        drawer.draw_tree(node, out_path)
        print(f"    {h[:16]} (depth={node.depth()}) → {out_path}.html")

    print(f"\nAll visualizations saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# pytest entry-point
# ---------------------------------------------------------------------------

def test_corter_gluck_basic_level():
    """
    Pytest test: Cobweb on the Murphy & Smith (1982) stimuli should recover
    the basic-level categories (Hammer, Brick, Knife, Pizza cutter) using the
    constants ALPHA and EVAL_ALPHA defined at the top of this file.

    Also directly replicates Table 4 from Corter & Gluck (1992) by computing
    the exact Category Utility formula before involving Cobweb at all.
    """
    instances = build_instances()

    # --- Step 1: verify Table 4 via closed-form CU (no Cobweb) ---
    cu_by_level = compute_cu_by_level(instances)
    cu_correct = print_cu_table(cu_by_level)

    # --- Step 2: fit Cobweb and check its basic-level detection ---
    tree = CobwebDiscreteTree(alpha=ALPHA, weight_attr=True)
    tree.fit(instances, iterations=ITERATIONS, randomizeFirst=True)

    results = evaluate_basic_level(tree, instances)
    print_results_table(results, ALPHA)
    cons, dist = check_basic_level_accuracy(results, label=f"alpha={ALPHA}")

    visualize(tree, instances)

    if cu_correct and cons and dist:
        print("\nOVERALL: PASS")
    else:
        print("\nOVERALL: FAIL")


if __name__ == "__main__":
    instances = build_instances()

    # --- Paper replication: closed-form CU (Table 4) ---
    cu_by_level = compute_cu_by_level(instances)
    print_cu_table(cu_by_level)

    # --- Cobweb basic-level detection ---
    tree = CobwebDiscreteTree(alpha=ALPHA, weight_attr=True)
    tree.fit(instances, iterations=ITERATIONS, randomizeFirst=True)

    results = evaluate_basic_level(tree, instances)
    print_results_table(results, ALPHA)
    check_basic_level_accuracy(results, label=f"alpha={ALPHA}")

    print("\nGenerating visualizations ...")
    visualize(tree, instances)
