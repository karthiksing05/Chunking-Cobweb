"""
tests/alpha/test_alpha_ablation.py

Self-contained alpha ablation study using **synthetic** Cobweb trees.

Each fixture builds a tree whose ground-truth category structure is known
by construction, so we can reason about *where* the basic level should be
and watch how the smoothing parameter (eval_alpha) moves it.

Hierarchies
───────────
  1. balanced    — 2 super-categories × 2 sub-categories, even instance counts.
                   Expected basic level: the 2 super-category nodes.
  2. deep_chain  — A → B → C → D → leaf, each level adds distinguishing attrs.
                   Tests that basic level avoids the root *and* the leaf.
  3. asymmetric  — one deep branch (3 levels) vs one shallow branch (1 level).
                   Tests that basic level adapts to local tree depth.
  4. overlapping — two categories sharing some attributes but differing in others.
                   Tests basic level when categories are not crisply separated.

Tests
─────
  test_visualize_hierarchies   — text dump + graphviz-style node diagrams (PNG)
  test_score_curves_per_alpha  — score-curve grids for each hierarchy × alpha
  test_alpha_sweep_agreement   — line chart: alpha vs basic-level depth for each hierarchy
  test_alpha_sweep_grid        — combined grid across all hierarchies
"""

import csv
import os
import sys
import time
import math
import random
import pytest

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree, CobwebDiscreteNode
from viz import HTMLCobwebDrawer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 10.0, 100.0]

EPMI_N     = 200       # samples for ground-truth EPMI
MAX_NODES  = 100


# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary constants — plain ints for attribute IDs and value IDs
# ═══════════════════════════════════════════════════════════════════════════

# -- Attributes --
COLOUR  = 0
SHAPE   = 1
SIZE    = 2
TEXTURE = 3
LEGS    = 4
SOUND   = 5

# -- Values --
RED = 1; BLUE = 2; GREEN = 3; YELLOW = 4; ORANGE = 5; PURPLE = 6
CIRCLE = 10; SQUARE = 11; TRIANGLE = 12; DIAMOND = 13; STAR = 14; HEXAGON = 15
SMALL = 20; MEDIUM = 21; LARGE = 22
ROUGH = 30; SMOOTH = 31; FUZZY = 32
TWO = 40; FOUR = 41; SIX = 42; EIGHT = 43
BARK = 50; MEOW = 51; CHIRP = 52; HISS = 53


def _inst(**kw):
    """Build an instance dict from keyword args like colour=RED."""
    name_to_attr = {
        "colour": COLOUR, "shape": SHAPE, "size": SIZE,
        "texture": TEXTURE, "legs": LEGS, "sound": SOUND,
    }
    return {name_to_attr[k]: {v: 1.0} for k, v in kw.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchy builders — each returns (tree, description, expected_basic_depth)
# ═══════════════════════════════════════════════════════════════════════════

def _build_balanced(alpha=1e-3):
    """
    Balanced: 2 super-categories × 2 sub-categories × 10 instances each.

    Super-A: red things   (sub-A1: red+circle,  sub-A2: red+square)
    Super-B: blue things  (sub-B1: blue+triangle, sub-B2: blue+diamond)

    Each sub-category varies the SIZE attribute randomly.
    Basic level should be at the super-category (depth 1): red vs blue.
    """
    tree = CobwebDiscreteTree(alpha, False)
    rng = random.Random(42)
    sizes = [SMALL, MEDIUM, LARGE]

    for _ in range(10):
        tree.ifit(_inst(colour=RED,  shape=CIRCLE,   size=rng.choice(sizes)))
    for _ in range(10):
        tree.ifit(_inst(colour=RED,  shape=SQUARE,   size=rng.choice(sizes)))
    for _ in range(10):
        tree.ifit(_inst(colour=BLUE, shape=TRIANGLE, size=rng.choice(sizes)))
    for _ in range(10):
        tree.ifit(_inst(colour=BLUE, shape=DIAMOND,  size=rng.choice(sizes)))

    return tree, "Balanced (2×2, colour is super-cat)", 1


def _build_deep_chain(alpha=1e-3):
    """
    Deep chain: 4-level deep hierarchy, each level adds a distinguishing attribute.

    Level 0 (root): everything
    Level 1: colour (red vs blue) — 2 branches
    Level 2: +shape (circle vs square, within red) — deeper branch
    Level 3: +texture (rough vs smooth, within red+circle)
    Level 4: +size (small vs large, within red+circle+rough) — leaves

    We only train the deep branch (red → circle → rough → small/large)
    plus a shallow blue branch for contrast.

    Basic level should *not* be the leaf or root — likely depth 1 or 2.
    """
    tree = CobwebDiscreteTree(alpha, False)
    rng = random.Random(42)

    # Deep branch: red, circle, rough, small/large
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=ROUGH, size=SMALL))
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=ROUGH, size=LARGE))
    # Sibling at depth 3: red, circle, smooth
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=SMOOTH, size=rng.choice([SMALL, MEDIUM, LARGE])))
    # Sibling at depth 2: red, square
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=SQUARE, texture=rng.choice([ROUGH, SMOOTH, FUZZY]),
                        size=rng.choice([SMALL, MEDIUM, LARGE])))
    # Shallow contrast: blue, anything
    for _ in range(8):
        tree.ifit(_inst(colour=BLUE, shape=rng.choice([TRIANGLE, DIAMOND, STAR]),
                        texture=rng.choice([ROUGH, SMOOTH, FUZZY]),
                        size=rng.choice([SMALL, MEDIUM, LARGE])))

    return tree, "Deep chain (red→circle→rough→size)", 2


def _build_asymmetric(alpha=1e-3):
    """
    Asymmetric: one deep branch (3 levels), one shallow branch (1 level).

    Deep: colour=RED → shape=CIRCLE → texture=ROUGH → leaves (8 each)
    Shallow: colour=BLUE → leaves (20 instances, mixed shapes)

    Tests whether basic level adapts — deep-branch leaves should have an
    intermediate basic level; shallow-branch leaves should pick something
    close to the leaf.
    """
    tree = CobwebDiscreteTree(alpha, False)
    rng = random.Random(42)

    # Deep branch
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=ROUGH, size=SMALL, sound=BARK))
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=ROUGH, size=LARGE, sound=MEOW))
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=CIRCLE, texture=SMOOTH, size=MEDIUM, sound=CHIRP))
    for _ in range(8):
        tree.ifit(_inst(colour=RED, shape=SQUARE, texture=FUZZY, size=LARGE, sound=HISS))

    # Shallow branch — lots of variation
    for _ in range(20):
        tree.ifit(_inst(colour=BLUE,
                        shape=rng.choice([TRIANGLE, DIAMOND, STAR, HEXAGON]),
                        texture=rng.choice([ROUGH, SMOOTH, FUZZY]),
                        size=rng.choice([SMALL, MEDIUM, LARGE]),
                        sound=rng.choice([BARK, MEOW, CHIRP, HISS])))

    return tree, "Asymmetric (deep-red vs shallow-blue)", 1


def _build_overlapping(alpha=1e-3):
    """
    Overlapping categories: two groups that share COLOUR but differ in SHAPE and TEXTURE.

    Group A: red or blue, always circle, always rough     (20 instances)
    Group B: red or blue, always square, always smooth    (20 instances)

    COLOUR is uninformative (shared); SHAPE+TEXTURE distinguish the groups.
    Basic level should cluster by shape+texture, not by colour.
    """
    tree = CobwebDiscreteTree(alpha, False)
    rng = random.Random(42)

    for _ in range(10):
        tree.ifit(_inst(colour=rng.choice([RED, BLUE]), shape=CIRCLE, texture=ROUGH,
                        size=rng.choice([SMALL, MEDIUM, LARGE])))
    for _ in range(10):
        tree.ifit(_inst(colour=rng.choice([RED, BLUE]), shape=CIRCLE, texture=ROUGH,
                        size=rng.choice([SMALL, MEDIUM, LARGE])))
    for _ in range(10):
        tree.ifit(_inst(colour=rng.choice([RED, BLUE]), shape=SQUARE, texture=SMOOTH,
                        size=rng.choice([SMALL, MEDIUM, LARGE])))
    for _ in range(10):
        tree.ifit(_inst(colour=rng.choice([RED, BLUE]), shape=SQUARE, texture=SMOOTH,
                        size=rng.choice([SMALL, MEDIUM, LARGE])))

    return tree, "Overlapping (shared colour, differ shape+texture)", 1


# ── All hierarchies as a list for parametrisation ────────────────────────────

HIERARCHY_BUILDERS = [
    ("balanced",    _build_balanced),
    ("deep_chain",  _build_deep_chain),
    ("asymmetric",  _build_asymmetric),
    ("overlapping", _build_overlapping),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

VAL_NAMES = {
    RED: "red", BLUE: "blue", GREEN: "grn", YELLOW: "yel", ORANGE: "org", PURPLE: "pur",
    CIRCLE: "circ", SQUARE: "sq", TRIANGLE: "tri", DIAMOND: "dia", STAR: "star", HEXAGON: "hex",
    SMALL: "S", MEDIUM: "M", LARGE: "L",
    ROUGH: "rough", SMOOTH: "smooth", FUZZY: "fuzzy",
    TWO: "2", FOUR: "4", SIX: "6", EIGHT: "8",
    BARK: "bark", MEOW: "meow", CHIRP: "chirp", HISS: "hiss",
}
ATTR_NAMES = {COLOUR: "colour", SHAPE: "shape", SIZE: "size",
              TEXTURE: "texture", LEGS: "legs", SOUND: "sound"}


def _node_label(node):
    """Compact label: show the dominant (mode) value per attribute."""
    parts = []
    for attr in sorted(node.av_count):
        vals = node.av_count[attr]
        if not vals:
            continue
        mode_v = max(vals, key=vals.get)
        a_name = ATTR_NAMES.get(attr, str(attr))
        v_name = VAL_NAMES.get(mode_v, str(mode_v))
        parts.append(f"{a_name}={v_name}")
    return ", ".join(parts) if parts else "(empty)"


def _collect_leaves(tree):
    if tree.root is None:
        return []
    leaves, stack = [], [tree.root]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)
    return leaves


def _walk_path_to_root(leaf):
    path, curr = [], leaf
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path


def _znorm(arr):
    import numpy as np
    a = np.array(arr, dtype=float)
    std = a.std()
    return (a - a.mean()) / (std if std > 1e-12 else 1.0)


def _csv_write(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_drawer():
    """Build an HTMLCobwebDrawer for the test vocabulary."""
    # attributes list indexed by attribute id (0-5)
    attributes = ["colour", "shape", "size", "texture", "legs", "sound"]

    # id_to_value list — needs to cover the largest value id (HISS = 53)
    _max_val = max(VAL_NAMES.keys())
    id_to_value = [str(i) for i in range(_max_val + 1)]
    for vid, vname in VAL_NAMES.items():
        id_to_value[vid] = vname
    value_to_id = {v: k for k, v in enumerate(id_to_value)}

    return HTMLCobwebDrawer(attributes, id_to_value, value_to_id)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Visualise all four hierarchies as text + tree-diagram PNGs
# ═══════════════════════════════════════════════════════════════════════════

def test_visualize_hierarchies(capsys):
    """
    For each mock hierarchy:
      - Print a text tree showing every node with its label, depth, count.
      - Save an interactive HTML tree (+ PNG screenshot) via HTMLCobwebDrawer.

    Saved to: tests/alpha/output/tree_{name}.html, tree_{name}.png
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    drawer = _make_drawer()

    for hier_name, builder in HIERARCHY_BUILDERS:
        tree, desc, _ = builder()

        # ── Text dump ────────────────────────────────────────────────────
        with capsys.disabled():
            print(f"\n{'═' * 80}")
            print(f"  Hierarchy: {hier_name}  —  {desc}")
            print(f"{'═' * 80}")

        def _print_node(node, indent=0):
            prefix = "  " + "│   " * indent + "├── " if indent > 0 else "  "
            label = _node_label(node)
            with capsys.disabled():
                print(f"{prefix}[d={node.depth()} n={node.count:.0f}] {label}")
            for child in node.children:
                _print_node(child, indent + 1)

        _print_node(tree.root)

        # ── HTML + PNG via HTMLCobwebDrawer ───────────────────────────────
        out_base = os.path.join(OUTPUT_DIR, f"tree_{hier_name}")
        try:
            drawer.draw_tree(tree.root, out_base)
        except Exception:
            # Playwright not available — fall back to HTML-only
            import json as _json
            d3_json = _json.dumps(drawer._node_to_dict(tree.root))
            html_str = drawer._build_html(d3_json)
            with open(out_base + ".html", "w", encoding="utf-8") as f:
                f.write(html_str)

    with capsys.disabled():
        print(f"\n[html] Tree visualisations saved to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Score curves per alpha — for each hierarchy, show how alpha
#         changes the shape of inst-PMI and EPMI score curves
# ═══════════════════════════════════════════════════════════════════════════

N_CURVE_LEAVES = 3  # leaves to sample per panel

def test_score_curves_per_alpha(capsys):
    """
    For each hierarchy × a subset of alphas, plot the inst-PMI score curve
    (and sampled EPMI) along leaf-to-root paths.

    Layout per hierarchy: rows = alphas, cols = leaves.
    Each panel: x = depth, y = score, two lines (EPMI, inst-PMI).
    Vertical dashed lines mark each method's argmax.

    Saved to: tests/alpha/output/score_curves_{name}.png
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DISPLAY_ALPHAS = [1e-4, 1e-2, 1e-1, 1.0, 10.0]

    for hier_name, builder in HIERARCHY_BUILDERS:
        tree, desc, _ = builder()
        leaves = _collect_leaves(tree)
        # Prefer deeper leaves
        sample = sorted(leaves, key=lambda l: -l.depth())[:N_CURVE_LEAVES]
        if not sample:
            continue

        nrow = len(DISPLAY_ALPHAS)
        ncol = len(sample)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4.5 * ncol, 3.5 * nrow), squeeze=False)

        for row, alpha in enumerate(DISPLAY_ALPHAS):
            for col, leaf in enumerate(sample):
                ax = axes[row][col]
                path = list(reversed(_walk_path_to_root(leaf)))  # root-first
                depths = [n.depth() for n in path]

                # Mode instance
                inst = ({a: {max(v, key=v.get): 1.0}
                         for a, v in leaf.av_count.items()} if leaf.av_count else {})
                root = path[0]
                lp_root = root.log_prob_instance(inst, alpha) if inst else 0.0

                # Inst-PMI scores along path
                inst_scores = [n.log_prob_instance(inst, alpha) - lp_root
                               for n in path]

                # Sampled EPMI scores (ground truth, uses the *same* alpha)
                epmi_scores = [n.expected_pmi(EPMI_N, MAX_NODES, alpha)
                               for n in path]

                # Plot raw scores (not z-normed, so we see absolute shape)
                ax.plot(depths, epmi_scores, color="steelblue", lw=1.5,
                        marker="o", markersize=3, label="EPMI (sampled)")
                ax.plot(depths, inst_scores, color="tomato", lw=1.5,
                        marker="s", markersize=3, linestyle="--", label="inst-PMI")

                epmi_best = depths[int(np.argmax(epmi_scores))]
                inst_best = depths[int(np.argmax(inst_scores))]
                ax.axvline(epmi_best, color="steelblue", alpha=0.4, lw=1, ls=":")
                ax.axvline(inst_best, color="tomato",    alpha=0.4, lw=1, ls=":")

                ax.set_xlabel("depth")
                if col == 0:
                    ax.set_ylabel(f"α={alpha:.0e}\nscore")
                ax.set_title(f"leaf d={leaf.depth()}, n={leaf.count:.0f}",
                             fontsize=8)
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, loc="upper left")

        fig.suptitle(
            f"Score curves: {hier_name}\n{desc}\n"
            f"(rows = eval_alpha, cols = sampled leaves, "
            f"EPMI n={EPMI_N})",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"score_curves_{hier_name}.png"),
                    dpi=130)
        plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Score curve grids saved to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Alpha sweep — how does the chosen basic-level depth change
#         with alpha for each method and each hierarchy?
# ═══════════════════════════════════════════════════════════════════════════

def test_alpha_sweep_agreement(capsys):
    """
    For each hierarchy, sweep eval_alpha across ALPHAS.
    At each alpha compute:
      - sampled basic level (get_basic with that alpha)
      - inst_pmi basic level (get_basic_instance_pmi with that alpha)
    Report: mean basic-level depth, hash-agreement with sampled, depth-agreement.

    Saved CSV:  tests/alpha/output/alpha_sweep.csv
    Saved plot: tests/alpha/output/alpha_sweep_{name}.png  (per hierarchy)
          — x = alpha (log), y = mean depth chosen, one line per method.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    METHODS = {
        "sampled":  lambda leaf, a: leaf.get_basic(EPMI_N, MAX_NODES, False, a),
        "inst_pmi": lambda leaf, a: leaf.get_basic_instance_pmi(
            {attr: {max(v, key=v.get): 1.0} for attr, v in leaf.av_count.items()}
            if leaf.av_count else {}, False, a),
    }

    METHOD_COLORS = {
        "sampled":  "black",
        "inst_pmi": "orange",
    }

    csv_rows = []

    for hier_name, builder in HIERARCHY_BUILDERS:
        tree, desc, expected_depth = builder()
        leaves = _collect_leaves(tree)
        if not leaves:
            continue

        # Collect per-alpha, per-method results
        # method -> list of (alpha, mean_depth, agree_pct)
        sweep_data = {m: [] for m in METHODS}

        with capsys.disabled():
            print(f"\n{'═' * 90}")
            print(f"  Alpha sweep: {hier_name}  —  {desc}")
            print(f"  {'alpha':>10}  ", end="")
            for m in METHODS:
                print(f"  {m:>12}", end="")
            print("   (mean basic-level depth)")
            print(f"  {'-'*10}  " + "  ".join(["-" * 12] * len(METHODS)))

        for alpha in ALPHAS:
            row_str = f"  {alpha:>10.1e}  "
            method_results = {}
            for m_name, m_fn in METHODS.items():
                results = [m_fn(leaf, alpha) for leaf in leaves]
                depths = [r.depth() for r in results]
                hashes = [r.concept_hash() for r in results]
                mean_d = sum(depths) / len(depths)
                method_results[m_name] = (results, depths, hashes, mean_d)
                row_str += f"  {mean_d:>12.2f}"

            with capsys.disabled():
                print(row_str)

            # Compute agreement with sampled
            s_hashes = method_results["sampled"][2]
            s_depths = method_results["sampled"][1]
            for m_name in METHODS:
                _, depths, hashes, mean_d = method_results[m_name]
                h_agree = sum(h == s for h, s in zip(hashes, s_hashes))
                d_agree = sum(d == s for d, s in zip(depths, s_depths))
                n = len(leaves)
                csv_rows.append({
                    "hierarchy": hier_name, "alpha": alpha, "method": m_name,
                    "n_leaves": n, "mean_depth": round(mean_d, 4),
                    "hash_agree": h_agree,
                    "hash_agree_pct": round(100 * h_agree / n, 2),
                    "depth_agree": d_agree,
                    "depth_agree_pct": round(100 * d_agree / n, 2),
                })
                sweep_data[m_name].append((alpha, mean_d))

        with capsys.disabled():
            print(f"{'═' * 90}")

        # ── Per-hierarchy plot: mean depth vs alpha ──────────────────────
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m_name, pts in sweep_data.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", markersize=5, lw=1.8,
                    color=METHOD_COLORS[m_name], label=m_name)

        # Mark the expected basic-level depth
        ax.axhline(expected_depth, color="gray", ls="--", lw=1, alpha=0.6,
                   label=f"expected depth={expected_depth}")

        ax.set_xscale("log")
        ax.set_xlabel("eval_alpha (log scale)")
        ax.set_ylabel("Mean basic-level depth")
        ax.set_title(f"{hier_name}: {desc}\nMean basic-level depth vs eval_alpha",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"alpha_sweep_{hier_name}.png"),
                    dpi=130)
        plt.close(fig)

    _csv_write(os.path.join(OUTPUT_DIR, "alpha_sweep.csv"),
               ["hierarchy", "alpha", "method", "n_leaves", "mean_depth",
                "hash_agree", "hash_agree_pct", "depth_agree", "depth_agree_pct"],
               csv_rows)

    with capsys.disabled():
        print(f"\n[csv]   alpha_sweep.csv saved to {OUTPUT_DIR}/")
        print(f"[plots] alpha_sweep_{{name}}.png saved to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Combined grid — all hierarchies on one figure
# ═══════════════════════════════════════════════════════════════════════════

def test_alpha_sweep_grid(capsys):
    """
    Combined figure: 2×2 grid (one panel per hierarchy).
    Each panel: x = alpha (log), y = mean basic-level depth, one line per method.
    Horizontal dashed line = expected basic-level depth.

    Saved to: tests/alpha/output/alpha_sweep_grid.png
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    METHODS = {
        "sampled":  lambda leaf, a: leaf.get_basic(EPMI_N, MAX_NODES, False, a),
        "inst_pmi": lambda leaf, a: leaf.get_basic_instance_pmi(
            {attr: {max(v, key=v.get): 1.0} for attr, v in leaf.av_count.items()}
            if leaf.av_count else {}, False, a),
    }

    METHOD_COLORS = {
        "sampled":  "black",
        "inst_pmi": "orange",
    }

    nrow, ncol = 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow), squeeze=False)

    for idx, (hier_name, builder) in enumerate(HIERARCHY_BUILDERS):
        row, col = divmod(idx, ncol)
        ax = axes[row][col]

        tree, desc, expected_depth = builder()
        leaves = _collect_leaves(tree)
        if not leaves:
            continue

        max_leaf_depth = max(l.depth() for l in leaves)

        for m_name, m_fn in METHODS.items():
            mean_depths = []
            for alpha in ALPHAS:
                results = [m_fn(leaf, alpha) for leaf in leaves]
                mean_d = sum(r.depth() for r in results) / len(results)
                mean_depths.append(mean_d)
            ax.plot(ALPHAS, mean_depths, marker="o", markersize=4, lw=1.6,
                    color=METHOD_COLORS[m_name], label=m_name)

        ax.axhline(expected_depth, color="gray", ls="--", lw=1, alpha=0.6)
        ax.axhline(0, color="lightgray", ls=":", lw=0.5)
        ax.axhline(max_leaf_depth, color="lightgray", ls=":", lw=0.5)

        ax.set_xscale("log")
        ax.set_xlabel("eval_alpha")
        ax.set_ylabel("Mean basic-level depth")
        ax.set_title(f"{hier_name}\n{desc}", fontsize=9)
        ax.set_ylim(-0.3, max_leaf_depth + 0.5)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

        # Annotate root/leaf lines
        ax.text(ALPHAS[0], 0.15, "root", fontsize=7, color="gray")
        ax.text(ALPHAS[0], max_leaf_depth - 0.3, "leaf", fontsize=7, color="gray")

    fig.suptitle(
        "Alpha ablation across synthetic hierarchies\n"
        "(mean basic-level depth vs eval_alpha, dashed = expected)",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "alpha_sweep_grid.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Saved alpha_sweep_grid.png to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Score heatmap — depth × alpha heatmap for inst-PMI score
# ═══════════════════════════════════════════════════════════════════════════

def test_score_heatmap(capsys):
    """
    For each hierarchy, pick one representative leaf and create a heatmap:
      x-axis = alpha (log), y-axis = ancestor depth, colour = inst-PMI score.

    This directly visualises how the score surface shifts as alpha changes:
    at low alpha the leaf row dominates; at higher alpha intermediate rows
    become competitive.

    Saved to: tests/alpha/output/score_heatmap_{name}.png
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    HEATMAP_ALPHAS = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                      1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0]

    for hier_name, builder in HIERARCHY_BUILDERS:
        tree, desc, _ = builder()
        leaves = _collect_leaves(tree)
        if not leaves:
            continue
        # Pick deepest leaf
        leaf = max(leaves, key=lambda l: l.depth())
        path = list(reversed(_walk_path_to_root(leaf)))  # root-first
        depths = [n.depth() for n in path]

        inst = ({a: {max(v, key=v.get): 1.0}
                 for a, v in leaf.av_count.items()} if leaf.av_count else {})
        if not inst:
            continue

        # Build score matrix: rows = depths (root→leaf), cols = alphas
        scores = np.zeros((len(path), len(HEATMAP_ALPHAS)))
        for j, alpha in enumerate(HEATMAP_ALPHAS):
            lp_root = path[0].log_prob_instance(inst, alpha)
            for i, node in enumerate(path):
                scores[i, j] = node.log_prob_instance(inst, alpha) - lp_root

        # Mark the argmax at each alpha
        argmax_rows = np.argmax(scores, axis=0)

        fig, ax = plt.subplots(figsize=(max(8, len(HEATMAP_ALPHAS) * 0.8), 5))
        im = ax.imshow(scores, aspect="auto", cmap="RdYlBu_r",
                       origin="lower")
        # Overlay argmax
        ax.plot(range(len(HEATMAP_ALPHAS)), argmax_rows,
                "k*-", markersize=10, lw=1.5, label="argmax (basic level)")

        ax.set_xticks(range(len(HEATMAP_ALPHAS)))
        ax.set_xticklabels([f"{a:.0e}" for a in HEATMAP_ALPHAS],
                           fontsize=7, rotation=45)
        ax.set_yticks(range(len(path)))
        ax.set_yticklabels([f"d={d}" for d in depths], fontsize=8)
        ax.set_xlabel("eval_alpha")
        ax.set_ylabel("Ancestor (depth)")
        ax.set_title(f"{hier_name}: inst-PMI score heatmap\n"
                     f"{desc}  |  leaf depth={leaf.depth()}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=ax, label="inst-PMI score")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"score_heatmap_{hier_name}.png"),
                    dpi=130)
        plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Score heatmaps saved to {OUTPUT_DIR}/")
