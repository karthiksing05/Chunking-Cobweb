"""
Corter & Gluck hierarchies — α slider using the C++ Cobweb
``expected_pmi(use_root=True)`` implementation.

Mirrors ``corter_gluck_hierarchies.py`` but instead of computing the
empirical PMI in pure Python, it:

  1. Builds a ``FrozenCobwebDiscreteTree`` for each dataset from its
     ``node_members``/``levels`` definition.
  2. Maps each named node (e.g. ``MS_Hammer``, ``MS_Pounder``) to the
     corresponding tree node by member-set match.
  3. For each level, computes the mean of
     ``node.expected_pmi(..., use_root=True)`` at the slider's current α.

The C++ ``expected_pmi(use_root=True)`` was rewritten to compute the
closed-form empirical average

    (1/N_c) Σ_{i ∈ members(c)} [log P_c(x_i) − log P_root(x_i)]
    = Σ_a Σ_v (n_v^{c,a}/N_c) · [log P_c(v|a) − log P_root(v|a)]

with smoothing P(v|a) = (n_v + α)/(N + K_a · α) — the exact formula in
``corter_gluck_hierarchies.py``. So this slider should reproduce the
Python plot (modulo float noise from av_count enumeration order).
"""

import os
import sys
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Imports — dataset definitions from the Python slider file, plus the
# FrozenCobwebDiscreteTree wrapper from cobweb-private/src/cobweb/.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corter_gluck_hierarchies as cgh  # datasets, level_x, all_level_names

_FD_PATH = os.path.join(
    _HERE, "..", "..", "cobweb-private", "src", "cobweb", "frozen_discrete.py",
)
_fd_spec = importlib.util.spec_from_file_location("_frozen_discrete_local", _FD_PATH)
_fd_mod  = importlib.util.module_from_spec(_fd_spec)
_fd_spec.loader.exec_module(_fd_mod)
FrozenCobwebDiscreteTree = _fd_mod.FrozenCobwebDiscreteTree


# ---------------------------------------------------------------------------
# Convert each dataset's flat (node_members, levels) description into the
# nested {"children":[...], "instances":[...]} format consumed by the
# frozen tree constructor.
# ---------------------------------------------------------------------------

def encode_instance(row, feature_idx):
    """Row X[i] vector → INSTANCE_TYPE dict {attr_id: {val_id: 1.0}}."""
    return {int(fidx): {int(row[fidx]): 1.0}
            for _, fidx in feature_idx.items()}


def build_frozen_hierarchy(node_members, levels, X, feature_idx):
    """Greedy reconstruct the nested hierarchy from the flat dataset spec."""
    # `levels` is leaf→root in cgh, so reverse to root→leaf for top-down build.
    level_names = list(levels.keys())[::-1]
    members_of  = {n: tuple(sorted(node_members[n])) for n in node_members}

    def find_direct_children(parent_name, parent_members):
        parent_depth = next(i for i, lvl in enumerate(level_names)
                            if parent_name in levels[lvl])
        for next_lvl_i in range(parent_depth + 1, len(level_names)):
            candidates = []
            for cand in levels[level_names[next_lvl_i]]:
                cm = set(members_of[cand])
                if cm.issubset(parent_members) and cm:
                    candidates.append(cand)
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
            return {"instances": [encode_instance(X[i], feature_idx)
                                  for i in members_of[name]]}
        return {"children": [recurse(c) for c in children_names]}

    top_nodes = levels[level_names[0]]
    if len(top_nodes) == 1:
        return recurse(top_nodes[0])
    return {"children": [recurse(n) for n in top_nodes]}


# ---------------------------------------------------------------------------
# Map each named dataset node (e.g. ``MS_Hammer``) to its corresponding tree
# node by walking the dataset hierarchy and the tree in lockstep. Avoids the
# identical-X-leaf ambiguity that arises if you try to recover names from
# av_count alone (floor_lamp / desk_lamp etc.).
# ---------------------------------------------------------------------------

def map_names_to_nodes(tree_root, dataset):
    node_members = dataset["node_members"]
    levels       = dataset["levels"]
    level_names  = list(levels.keys())[::-1]  # root → leaves
    members_of   = {n: tuple(sorted(node_members[n])) for n in node_members}

    def find_direct_children(parent_name, parent_members_set):
        parent_depth = next(i for i, lvl in enumerate(level_names)
                            if parent_name in levels[lvl])
        for next_lvl_i in range(parent_depth + 1, len(level_names)):
            candidates = []
            for cand in levels[level_names[next_lvl_i]]:
                cm = set(members_of[cand])
                if cm.issubset(parent_members_set) and cm:
                    candidates.append(cand)
            covered = set()
            disjoint = True
            for c in candidates:
                cm = set(members_of[c])
                if covered & cm:
                    disjoint = False
                    break
                covered |= cm
            if candidates and disjoint and covered == parent_members_set:
                return candidates
        return []

    out = {}

    def walk(name, tree_node):
        out[name] = tree_node
        children_names = find_direct_children(name, set(members_of[name]))
        if not children_names:
            return
        if len(tree_node.children) != len(children_names):
            return  # topology drift — bail rather than mis-pair
        for child_name, tree_child in zip(children_names, tree_node.children):
            walk(child_name, tree_child)

    top_nodes = levels[level_names[0]]
    if len(top_nodes) == 1:
        walk(top_nodes[0], tree_root)
    else:
        if len(tree_root.children) == len(top_nodes):
            for tn, tc in zip(top_nodes, tree_root.children):
                walk(tn, tc)
    return out


# ---------------------------------------------------------------------------
# Build all trees once.
# ---------------------------------------------------------------------------

print("Building frozen trees for all hierarchies …")
trees     = {}
name_maps = {}
for ds_name, dataset in cgh.datasets.items():
    hierarchy = build_frozen_hierarchy(
        dataset["node_members"], dataset["levels"],
        dataset["X"], dataset["feature_idx"],
    )
    tree = FrozenCobwebDiscreteTree(hierarchy, alpha=1.0, weight_attr=False)
    trees[ds_name]     = tree
    name_maps[ds_name] = map_names_to_nodes(tree.root, dataset)
    print(f"  {ds_name}: {len(name_maps[ds_name])} named nodes mapped")


# ---------------------------------------------------------------------------
# Per-α level scores via C++ expected_pmi(use_root=True). n_samples=0 means
# use every leaf under the node (closed-form, matches Python exactly).
# n_samples>0 samples that many leaves uniformly with replacement and the
# PMI is the empirical average over the sampled subset.
# ---------------------------------------------------------------------------

def compute_level_scores(ds_name, dataset, alpha, n_samples):
    name_to_node = name_maps[ds_name]
    out = {}
    for level_name, names in dataset["levels"].items():
        scores = []
        for nm in names:
            if nm not in name_to_node:
                continue
            node = name_to_node[nm]
            scores.append(node.expected_pmi(
                int(n_samples), 0,
                eval_alpha=alpha,
                uniform_leaf=False,
                use_root=True,
            ))
        if scores:
            out[level_name] = float(np.mean(scores))
    return out


# ---------------------------------------------------------------------------
# Plot + sliders (log₁₀ α and n_samples)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 7.5))
gs  = gridspec.GridSpec(3, 1, height_ratios=[15, 1, 1], hspace=0.45)
ax           = fig.add_subplot(gs[0])
ax_alpha     = fig.add_subplot(gs[1])
ax_nsamples  = fig.add_subplot(gs[2])

alpha_slider = Slider(
    ax=ax_alpha,
    label="log₁₀(α)",
    valmin=-3, valmax=5,
    valinit=np.log10(1),
    valstep=0.05,
)

n_slider = Slider(
    ax=ax_nsamples,
    label="n_samples (0 = all leaves)",
    valmin=0, valmax=64,
    valinit=0,
    valstep=1,
)


def draw(alpha, n_samples):
    ax.clear()
    for ds_name, dataset in cgh.datasets.items():
        level_scores = compute_level_scores(ds_name, dataset, alpha, n_samples)
        color = dataset["color"]
        xs = [cgh.level_x(ln) for ln in level_scores.keys()]
        ys = list(level_scores.values())
        ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                markersize=8, label=ds_name, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color=color, ha="center")

    ax.set_xticks(range(len(cgh.all_level_names)))
    ax.set_xticklabels(cgh.all_level_names, fontsize=11)
    ax.set_ylabel(r"Mean $E_{x|c}\,[\, PMI(x;c)\,]$  (C++ expected_pmi, use_root=True)",
                  fontsize=11)
    sub = "all leaves" if n_samples <= 0 else f"{int(n_samples)} sampled leaves"
    ax.set_title(
        f"Mean PMI by level — FrozenCobwebDiscreteTree + expected_pmi(use_root=True)  "
        f"(α = {alpha:.4f}, {sub})",
        fontsize=12,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.3, len(cgh.all_level_names) - 0.7)
    fig.canvas.draw_idle()


def on_change(_val):
    draw(10 ** alpha_slider.val, n_slider.val)


alpha_slider.on_changed(on_change)
n_slider.on_changed(on_change)
draw(10 ** alpha_slider.valinit, n_slider.valinit)
plt.show()
