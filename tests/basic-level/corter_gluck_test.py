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

from cobweb.cobweb_discrete import CobwebDiscreteTree

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
# Core evaluation
# ---------------------------------------------------------------------------

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
        basic_node = leaf.get_basic_instance_pmi(inst, debug=False)
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
# Main sweep over alpha values
# ---------------------------------------------------------------------------

def run_sweep(alpha_values=None, iterations=10):
    if alpha_values is None:
        alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    instances = build_instances()
    summary = {}

    for alpha in alpha_values:
        tree = CobwebDiscreteTree(alpha=alpha, weight_attr=True)
        tree.fit(instances, iterations=iterations, randomizeFirst=True)

        results = evaluate_basic_level(tree, instances)
        print_results_table(results, alpha)
        cons, dist = check_basic_level_accuracy(results, label=f"alpha={alpha}")
        summary[alpha] = {"consistency": cons, "distinctness": dist}

    print("\n\n" + "="*72)
    print("SUMMARY across alpha values")
    print("="*72)
    print(f"{'alpha':>8}  {'Consistency':>12}  {'Distinctness':>13}")
    print("-" * 40)
    for alpha, res in summary.items():
        c = "PASS" if res["consistency"] else "FAIL"
        d = "PASS" if res["distinctness"] else "FAIL"
        print(f"{alpha:>8.0e}  {c:>12}  {d:>13}")

    return summary


# ---------------------------------------------------------------------------
# pytest entry-point
# ---------------------------------------------------------------------------

def test_corter_gluck_basic_level():
    """
    Pytest test: Cobweb on the Murphy & Smith (1982) stimuli should recover
    the basic-level categories (Hammer, Brick, Knife, Pizza cutter) at a
    range of alpha values, matching the Corter & Gluck (1992) prediction.
    """
    instances = build_instances()

    passed_any = False
    for alpha in [1e-4, 1e-3, 1e-2]:
        tree = CobwebDiscreteTree(alpha=alpha, weight_attr=True)
        tree.fit(instances, iterations=15, randomizeFirst=True)
        results = evaluate_basic_level(tree, instances)
        cons, dist = check_basic_level_accuracy(results, label=f"alpha={alpha}")
        if cons and dist:
            passed_any = True
            break

    assert passed_any, (
        "Cobweb failed to recover basic-level categories from Murphy & Smith stimuli "
        "at any tested alpha value. Check tree structure / alpha sweep."
    )


if __name__ == "__main__":
    run_sweep(alpha_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], iterations=10)
