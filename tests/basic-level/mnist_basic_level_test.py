"""
MNIST basic-level subtree test
==============================

Build a continuous Cobweb tree on a subset of MNIST, then for every test
instance find its basic-level subtree via ``CobwebContinuousNode.get_basic_pc``
(P(c)-weighted KL against root) and visualise them.

Outputs (in tests/basic-level/mnist_basic_level_output/):

  - basic_level_subtrees.png : one row per unique basic-level node, showing
    its prototype (mean image), class histogram, and a few member digits.
  - cobweb_tree_labels.png   : same tree-with-class-bars drawing as
    tests/moc/mnist_example.py, top 4 depths.
  - per_subtree_membership.csv : for each basic-level node, member count,
    depth, and dominant digit class.

This script is meant to be run directly (not via pytest) so it only takes
a few minutes — it uses 5k training samples and 1k test samples by default.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from cobweb.cobweb_continuous import CobwebContinuousTree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "mnist_basic_level_output")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

N_TRAIN     = 5_000
N_TEST      = 1_000
PRIOR_VAR   = 0.05854983152
COVAR_FROM  = 2
TREE_DEPTH_FOR_LABEL_FIG = 3

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

transform = transforms.ToTensor()
trainset  = torchvision.datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
testset   = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)


def to_numpy(dataset, n):
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    imgs, lbls = next(iter(loader))
    return imgs.view(n, -1).numpy(), lbls.numpy()


X,      y      = to_numpy(trainset, N_TRAIN)
X_test, y_test = to_numpy(testset,  N_TEST)
X      = X.astype(np.float32, copy=False)
X_test = X_test.astype(np.float32, copy=False)

# ---------------------------------------------------------------------------
# Build the continuous Cobweb tree
# ---------------------------------------------------------------------------

print(f"Building CobwebContinuousTree on {N_TRAIN} MNIST samples …")
tree = CobwebContinuousTree(
    size=X.shape[1],
    num_labels=0,
    covar_from=COVAR_FROM,
    prior_var=PRIOR_VAR,
)
_empty_label = np.zeros(0, dtype=np.float32)
for i, x in enumerate(X):
    tree.ifit(x, _empty_label)
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{N_TRAIN} inserted")
print("  Tree built.")


# ---------------------------------------------------------------------------
# Greedy descent to get a leaf for an instance
# ---------------------------------------------------------------------------

def greedy_descend(root, x):
    """Pick the highest-log-prob child at each level until reaching a leaf."""
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob(x, _empty_label))
    return node


# ---------------------------------------------------------------------------
# Map every test instance to its basic-level subtree
# ---------------------------------------------------------------------------

print("Mapping test instances to basic-level subtrees via get_basic_pc …")
bl_members = {}  # id(node) -> dict{node, instance_indices, labels, depth}
bl_test_assignment = []

for i in range(len(X_test)):
    leaf  = greedy_descend(tree.root, X_test[i])
    bl    = leaf.get_basic_pc(debug=False)
    nid   = id(bl)
    if nid not in bl_members:
        bl_members[nid] = {
            "node":       bl,
            "depth":      bl.depth(),
            "indices":    [],
            "labels":     [],
        }
    bl_members[nid]["indices"].append(i)
    bl_members[nid]["labels"].append(int(y_test[i]))
    bl_test_assignment.append(nid)

print(f"  {len(bl_members)} unique basic-level subtree(s) recovered.")
print(f"  size distribution: "
      f"min={min(len(m['indices']) for m in bl_members.values())}, "
      f"max={max(len(m['indices']) for m in bl_members.values())}, "
      f"mean={np.mean([len(m['indices']) for m in bl_members.values()]):.1f}")

# ---------------------------------------------------------------------------
# Per-subtree visualisation: prototype image + class histogram + sample digits
# ---------------------------------------------------------------------------

CMAP = plt.get_cmap("tab10")
SAMPLE_COLS = 8   # # member digits to show per row

def plot_basic_level_subtrees(bl_members, X_test, y_test, out_path):
    """
    One row per unique BL subtree, sorted by membership descending.
    Columns:
      [prototype mean image]  [class histogram bar]  [sample member digits ...]
    """
    sorted_bls = sorted(
        bl_members.values(),
        key=lambda m: len(m["indices"]),
        reverse=True,
    )
    n_rows = len(sorted_bls)
    n_cols = 2 + SAMPLE_COLS   # prototype + histogram + sample digits

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.2, n_rows * 1.4),
        squeeze=False,
    )
    fig.suptitle(
        f"Basic-level subtrees recovered via get_basic_pc  "
        f"(N_test={len(X_test)}, n_subtrees={n_rows})",
        fontsize=11,
    )

    for row, m in enumerate(sorted_bls):
        node    = m["node"]
        idxs    = m["indices"]
        labels  = np.array(m["labels"])
        n_mem   = len(idxs)
        depth   = m["depth"]
        # prototype = mean image accumulated by the node itself
        proto = np.array(node.mean).reshape(28, 28)

        # column 0: prototype image
        ax0 = axes[row, 0]
        ax0.imshow(proto, cmap="gray")
        ax0.set_xticks([]); ax0.set_yticks([])
        dom = int(np.bincount(labels, minlength=10).argmax())
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom}",
            fontsize=6, rotation=0, labelpad=22, va="center",
        )
        if row == 0:
            ax0.set_title("prototype", fontsize=7)

        # column 1: class histogram (10 digits, normalized)
        ax1 = axes[row, 1]
        cls_counts = np.bincount(labels, minlength=10)
        cls_props  = cls_counts / max(cls_counts.sum(), 1)
        ax1.bar(np.arange(10), cls_props, color=[CMAP(c) for c in range(10)])
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(range(10), fontsize=5)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis="y", labelsize=5)
        if row == 0:
            ax1.set_title("class histogram", fontsize=7)

        # remaining columns: a handful of member digits
        sample_idxs = idxs[:SAMPLE_COLS] if n_mem >= SAMPLE_COLS else idxs
        for k in range(SAMPLE_COLS):
            ax = axes[row, 2 + k]
            if k < len(sample_idxs):
                img = X_test[sample_idxs[k]].reshape(28, 28)
                ax.imshow(img, cmap="gray")
                ax.set_title(str(int(y_test[sample_idxs[k]])), fontsize=6, pad=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and k == 0:
                pass  # leave column-0 title alone

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Subtree visualisation saved → {out_path}")


plot_basic_level_subtrees(
    bl_members, X_test, y_test,
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees.png"),
)


# ---------------------------------------------------------------------------
# CSV summary of basic-level membership
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUT_DIR, "per_subtree_membership.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subtree_idx", "depth", "node_count", "test_members",
                "dominant_digit", "class_distribution"])
    for i, m in enumerate(sorted(bl_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["labels"])
        cls_counts = np.bincount(labels, minlength=10)
        dom = int(cls_counts.argmax())
        dist = "/".join(str(int(c)) for c in cls_counts)
        w.writerow([
            i,
            m["depth"],
            int(m["node"].count),
            len(m["indices"]),
            dom,
            dist,
        ])
print(f"  CSV summary saved → {csv_path}")


# ---------------------------------------------------------------------------
# Tree-with-class-bars (top TREE_DEPTH_FOR_LABEL_FIG depths)
# Mirrors the layout from tests/moc/mnist_example.py for visual continuity.
# ---------------------------------------------------------------------------

def compute_node_label_counts(root, X_instances, y_labels, max_depth=3):
    n_classes = int(y_labels.max()) + 1
    counts   = {}
    node_obj = {}

    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid]   = np.zeros(n_classes, dtype=np.int32)
            node_obj[nid] = node
        return nid

    for x, label in zip(X_instances, y_labels):
        node = root
        for d in range(max_depth + 1):
            _ensure(node)
            counts[id(node)][int(label)] += 1
            if not node.children or d == max_depth:
                break
            node = max(node.children, key=lambda c: c.log_prob(x, _empty_label))

    return counts, node_obj


def plot_cobweb_tree_labels(root, label_counts_map, node_obj_map,
                            max_depth=3, out_path=None,
                            highlight_ids=None):
    """Same drawing logic as mnist_example.py, with optional BL highlighting."""
    tab10        = plt.get_cmap("tab10")
    digit_colors = [tab10(i) for i in range(10)]
    highlight_ids = highlight_ids or set()

    def leaf_span(node, depth, max_d):
        if depth >= max_d or not node.children:
            return 1
        return sum(leaf_span(c, depth + 1, max_d) for c in node.children)

    pos = {}

    def assign_pos(node, depth, x_left):
        span = leaf_span(node, depth, max_depth)
        x_centre = x_left + span / 2.0
        pos[id(node)] = (x_centre, depth)
        if depth < max_depth and node.children:
            cursor = x_left
            for child in node.children:
                child_span = leaf_span(child, depth + 1, max_depth)
                assign_pos(child, depth + 1, cursor)
                cursor += child_span
        return span

    assign_pos(root, 0, 0.0)
    total_width = leaf_span(root, 0, max_depth)

    bar_w   = 0.7
    bar_h   = 0.35
    y_gap   = 1.0
    fig_w   = max(14, total_width * 0.9)
    fig_h   = (max_depth + 1) * 2.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Cobweb-Continuous Tree — Label Distributions (depths 0–{max_depth})\n"
        f"red border = basic-level node selected by get_basic_pc",
        fontsize=11,
    )

    def draw_edges(node, depth):
        if depth >= max_depth or not node.children:
            return
        px, py = pos[id(node)]
        for child in node.children:
            cx, cy = pos[id(child)]
            ax.plot([px, cx],
                    [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            draw_edges(child, depth + 1)

    draw_edges(root, 0)

    def draw_node(node, depth):
        nid = id(node)
        if nid not in label_counts_map:
            return
        counts = label_counts_map[nid].astype(float)
        total  = counts.sum()
        if total == 0:
            return
        props = counts / total

        x_c, _ = pos[nid]
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2

        cursor = x_left
        for digit in range(10):
            seg_w = props[digit] * bar_w
            if seg_w > 0:
                rect = plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                     color=digit_colors[digit], lw=0)
                ax.add_patch(rect)
                cursor += seg_w

        is_bl = nid in highlight_ids
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h,
            fill=False,
            edgecolor=("red" if is_bl else "black"),
            lw=(1.5 if is_bl else 0.5),
        ))

        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=5)

        if depth < max_depth and node.children:
            for child in node.children:
                draw_node(child, depth + 1)

    draw_node(root, 0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=digit_colors[d], label=str(d))
        for d in range(10)
    ]
    ax.legend(handles=legend_handles, title="digit", loc="lower right",
              ncol=5, fontsize=7, title_fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


print("Computing node label distributions for tree figure …")
label_counts_map, node_obj_map = compute_node_label_counts(
    tree.root, X, y, max_depth=TREE_DEPTH_FOR_LABEL_FIG,
)

# IDs of nodes selected as basic-level (those that appear in bl_members)
highlight_ids = set(bl_members.keys())

plot_cobweb_tree_labels(
    tree.root, label_counts_map, node_obj_map,
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"),
    highlight_ids=highlight_ids,
)
print(f"  Tree figure saved → {os.path.join(OUT_DIR, 'cobweb_tree_labels.png')}")

print("\nDone.")
