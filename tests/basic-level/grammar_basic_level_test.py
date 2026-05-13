"""
Grammar basic-level test — α-agnostic frontier vs original get_basic.
=====================================================================

Builds a Cobweb-Discrete context tree on a TEST_GRAMMAR3 corpus, then
identifies basic-level nodes via two methods and compares:

  1. NEW  : ``tree.get_basic_frontier()`` — single DFS over the closed-form,
            α-agnostic, sampling-free score (CobwebDiscreteNode.basic_level_score).
  2. OLD  : ``leaf.get_basic(n_samples, max_nodes, eval_alpha=…)`` — Monte
            Carlo expected PMI against the full-tree mixture marginal.

For every test token we record:
  - which frontier node covers it (NEW)
  - which basic-level node the leaf walks up to under the old method
  - per-token agreement

Plus a side-by-side visual: the tree with red borders on frontier nodes,
and a second figure with green borders on the (deduplicated) BL nodes
returned by the old method.

Outputs (in ``tests/basic-level/grammar_basic_level_output/``):
  - basic_level_subtrees_frontier.png : POS hist + center words + context
    table, one row per frontier subtree.
  - basic_level_subtrees_get_basic.png : same view, one row per BL node
    selected by the old method.
  - cobweb_tree_labels_frontier.png  : tree, red borders on frontier nodes.
  - cobweb_tree_labels_get_basic.png : tree, green borders on old-method
    BL nodes.
  - method_comparison.txt : counts of unique BL nodes per method, per-leaf
    agreement, and a short summary.
  - per_subtree_membership_frontier.csv : depth, count, dominant POS, POS
    distribution.

Run directly: ``python tests/basic-level/grammar_basic_level_test.py``.
"""

import os
import sys
import csv
import random
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make src/ importable
_HERE     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from util.cfg import TEST_GRAMMAR3, generate
from cobweb.cobweb_discrete import CobwebDiscreteTree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = os.path.join(_HERE, "grammar_basic_level_output")
os.makedirs(OUT_DIR, exist_ok=True)

N_SENTENCES = 1000
WINDOW      = 3
ALPHA       = 1e-3
EVAL_ALPHA  = 10.0     # smoothing used inside the *old* get_basic
N_SAMPLES   = 200       # Monte Carlo samples for the old get_basic
MAX_NODES   = 100       # priority-queue budget for old get_basic's marginal
SEED        = 42
TREE_DEPTH_FOR_LABEL_FIG = 3
TOP_WORDS_PER_OFFSET     = 3

random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if sent:
        sentences.append(sent)

vocab   = sorted({w for s in sentences for w in s})
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V       = len(vocab)
print(f"  Vocab size: {V}  |  Total tokens: {sum(len(s) for s in sentences)}")

# POS mapping
_TERMINAL_POS = ["Det", "N", "Adj", "RelPro", "V", "P"]
word2pos = {}
for _pos_tag in _TERMINAL_POS:
    if _pos_tag in TEST_GRAMMAR3:
        for _prod in TEST_GRAMMAR3[_pos_tag]:
            if len(_prod) == 1 and _prod[0] not in word2pos:
                word2pos[_prod[0]] = _pos_tag
for _w in vocab:
    if _w not in word2pos:
        word2pos[_w] = "Unk"
pos_tags = sorted(set(word2pos.values()))
pos2id   = {p: i for i, p in enumerate(pos_tags)}
id2pos   = {i: p for p, i in pos2id.items()}
N_POS    = len(pos_tags)
print(f"  POS tags ({N_POS}): {pos_tags}")

CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def offset_for_attr(attr_id):
    return CONTEXT_OFFSETS[attr_id]


def make_context_instance(sentence, pos):
    instance = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sentence):
            instance[pos2attr[offset]] = {word2id[sentence[ctx]]: 1.0}
    return instance


instances_raw = []
center_words  = []
labels_all    = []
for sent in sentences:
    for pos, word in enumerate(sent):
        instances_raw.append(make_context_instance(sent, pos))
        center_words.append(word2id[word])
        labels_all.append(pos2id[word2pos[word]])
center_words = np.array(center_words, dtype=np.int32)
labels_all   = np.array(labels_all,   dtype=np.int32)

rng       = np.random.default_rng(SEED)
idx       = rng.permutation(len(instances_raw))
split     = int(0.8 * len(instances_raw))
train_idx, test_idx = idx[:split], idx[split:]
instances_train = [instances_raw[i] for i in train_idx]
instances_test  = [instances_raw[i] for i in test_idx]
center_test     = center_words[test_idx]
y               = labels_all[train_idx]
y_test          = labels_all[test_idx]
print(f"  Train: {len(instances_train)}  |  Test: {len(instances_test)}")

# ---------------------------------------------------------------------------
# Build the tree
# ---------------------------------------------------------------------------

print(f"Building Cobweb Discrete tree (alpha={ALPHA}) …")
tree = CobwebDiscreteTree(alpha=ALPHA, weight_attr=True)
for i, inst in enumerate(instances_train):
    tree.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(instances_train)} inserted")
print("  Tree built.")


def greedy_descend(root, instance):
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(instance))
    return node


# ---------------------------------------------------------------------------
# NEW: alpha-agnostic frontier via tree.get_basic_frontier()
# ---------------------------------------------------------------------------

print("\nComputing α-agnostic frontier via tree.get_basic_frontier() …")
frontier_nodes  = tree.get_basic_frontier()
frontier_id_set = {id(n) for n in frontier_nodes}
print(f"  Frontier size: {len(frontier_nodes)} nodes")


def assign_to_frontier(leaf):
    """Walk leaf → root; return the first ancestor (or self) in the
    frontier set.  By the antichain property exactly one ancestor qualifies."""
    node = leaf
    while node is not None:
        if id(node) in frontier_id_set:
            return node
        node = node.parent
    return None


# ---------------------------------------------------------------------------
# OLD: Monte Carlo get_basic per leaf
# ---------------------------------------------------------------------------

print(f"Running OLD get_basic (n_samples={N_SAMPLES}, eval_alpha={EVAL_ALPHA}) per leaf …")
# Cache per-leaf to avoid redoing the Monte Carlo walk for repeated leaves.
_old_cache = {}

def get_basic_old(leaf):
    key = id(leaf)
    if key in _old_cache:
        return _old_cache[key]
    bl = leaf.get_basic(N_SAMPLES, MAX_NODES, debug=False, eval_alpha=EVAL_ALPHA)
    _old_cache[key] = bl
    return bl


# ---------------------------------------------------------------------------
# Map test tokens via both methods
# ---------------------------------------------------------------------------

print("\nMapping test tokens via both methods …")

new_members = {}   # id(frontier_node) -> {node, depth, indices, center_words, pos_labels}
old_members = {}
agreements  = 0    # # test tokens where new and old map to the same node

for i, inst in enumerate(instances_test):
    leaf = greedy_descend(tree.root, inst)

    bl_new = assign_to_frontier(leaf)
    bl_old = get_basic_old(leaf)
    same   = (bl_new is not None) and (id(bl_new) == id(bl_old))
    agreements += int(same)

    for bl, store in [(bl_new, new_members), (bl_old, old_members)]:
        if bl is None:
            continue
        nid = id(bl)
        if nid not in store:
            store[nid] = {
                "node":         bl,
                "depth":        bl.depth(),
                "indices":      [],
                "center_words": [],
                "pos_labels":   [],
            }
        store[nid]["indices"].append(i)
        store[nid]["center_words"].append(int(center_test[i]))
        store[nid]["pos_labels"].append(int(y_test[i]))

print(f"  NEW: {len(new_members)} unique BL nodes covering {len(instances_test)} tokens")
print(f"  OLD: {len(old_members)} unique BL nodes covering {len(instances_test)} tokens")
print(f"  Tokens where NEW and OLD agree exactly: "
      f"{agreements}/{len(instances_test)} ({100*agreements/len(instances_test):.1f}%)")

# One-BL-per-path check for NEW
counts = Counter()
def all_leaves(root):
    leaves, stack = [], [root]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)
    return leaves
for leaf in all_leaves(tree.root):
    node = leaf
    c = 0
    while node is not None:
        if id(node) in frontier_id_set:
            c += 1
        node = node.parent
    counts[c] += 1
n_leaves_tree = sum(counts.values())
print("\nOne-BL-per-path check (NEW, over all tree leaves):")
for k in sorted(counts.keys()):
    flag = "" if k == 1 else "  ← violation"
    print(f"  {k} frontier nodes on path: {counts[k]} ({100*counts[k]/n_leaves_tree:.1f}%){flag}")


# ---------------------------------------------------------------------------
# Per-subtree visualisation (shared helper)
# ---------------------------------------------------------------------------

CMAP = plt.get_cmap("tab20") if N_POS > 10 else plt.get_cmap("tab10")
pos_colors = [CMAP(i / max(N_POS - 1, 1)) for i in range(N_POS)]
TOP_CENTER_WORDS = 6


def _top_context_words(node, k=TOP_WORDS_PER_OFFSET):
    out = {}
    for attr_id, val_map in node.av_count.items():
        if attr_id < 0 or attr_id not in {pos2attr[o] for o in CONTEXT_OFFSETS}:
            continue
        offset = offset_for_attr(attr_id)
        items  = sorted(val_map.items(), key=lambda kv: -kv[1])[:k]
        total  = sum(val_map.values()) or 1
        out[offset] = [(id2word.get(int(v), f"<{v}>"), c / total)
                       for v, c in items]
    return out


def plot_subtrees(members, title, out_path):
    sorted_bls = sorted(members.values(),
                        key=lambda m: len(m["indices"]), reverse=True)
    n_rows = len(sorted_bls)
    if n_rows == 0:
        return
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, max(2.0, n_rows * 1.6)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.4, 2.5]},
    )
    fig.suptitle(title, fontsize=11)

    for row, m in enumerate(sorted_bls):
        node    = m["node"]
        labels  = np.array(m["pos_labels"])
        centers = np.array(m["center_words"])
        n_mem   = len(m["indices"])
        depth   = m["depth"]
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom_pos    = id2pos[int(cls_counts.argmax())]

        ax0 = axes[row, 0]
        cls_props = cls_counts / max(cls_counts.sum(), 1)
        ax0.bar(np.arange(N_POS), cls_props,
                color=pos_colors, edgecolor="white", linewidth=0.4)
        ax0.set_xticks(range(N_POS))
        ax0.set_xticklabels([id2pos[i] for i in range(N_POS)],
                            rotation=45, ha="right", fontsize=6)
        ax0.set_ylim(0, 1.0)
        ax0.tick_params(axis="y", labelsize=5)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom_pos}",
            fontsize=6, rotation=0, labelpad=28, va="center",
        )
        if row == 0:
            ax0.set_title("POS histogram", fontsize=8)

        ax1 = axes[row, 1]
        cw_counts = {}
        for c in centers:
            cw_counts[int(c)] = cw_counts.get(int(c), 0) + 1
        top_cw = sorted(cw_counts.items(), key=lambda kv: -kv[1])[:TOP_CENTER_WORDS]
        if top_cw:
            words   = [id2word[w] for w, _ in top_cw]
            counts_ = [c for _, c in top_cw]
            colors_ = [pos_colors[pos2id[word2pos[id2word[w]]]] for w, _ in top_cw]
            ax1.barh(np.arange(len(words))[::-1], counts_,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(words))[::-1])
            ax1.set_yticklabels(words, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0:
            ax1.set_title("top center words", fontsize=8)

        ax2 = axes[row, 2]
        ax2.axis("off")
        ctx_top = _top_context_words(node, k=TOP_WORDS_PER_OFFSET)
        offsets = sorted(ctx_top.keys())
        if offsets:
            n_off = len(offsets)
            x_step = 1.0 / max(n_off, 1)
            for ci, off in enumerate(offsets):
                cx = (ci + 0.5) * x_step
                ax2.text(cx, 0.95, f"{off:+d}", ha="center", va="top",
                         fontsize=7, fontweight="bold",
                         transform=ax2.transAxes)
                for li, (w, frac) in enumerate(ctx_top[off]):
                    cy = 0.85 - li * 0.20
                    ax2.text(cx, cy, f"{w} ({frac:.2f})",
                             ha="center", va="top", fontsize=6,
                             color=pos_colors[pos2id.get(word2pos.get(w, "Unk"), 0)],
                             transform=ax2.transAxes)
        if row == 0:
            ax2.set_title("top context word per offset", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Subtree visualisation saved → {out_path}")


plot_subtrees(
    new_members,
    title=(f"NEW frontier — tree.get_basic_frontier()  "
           f"(N_test={len(instances_test)}, n_subtrees={len(new_members)})"),
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees_frontier.png"),
)
plot_subtrees(
    old_members,
    title=(f"OLD get_basic (Monte Carlo, eval_alpha={EVAL_ALPHA})  "
           f"(N_test={len(instances_test)}, n_subtrees={len(old_members)})"),
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees_get_basic.png"),
)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUT_DIR, "per_subtree_membership_frontier.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subtree_idx", "depth", "node_count", "test_members",
                "dominant_pos", "pos_distribution"])
    for i, m in enumerate(sorted(new_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["pos_labels"])
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom = id2pos[int(cls_counts.argmax())]
        dist = "/".join(f"{id2pos[k]}:{int(v)}" for k, v in enumerate(cls_counts))
        w.writerow([i, m["depth"], int(m["node"].count),
                    len(m["indices"]), dom, dist])
print(f"  CSV summary saved → {csv_path}")


# ---------------------------------------------------------------------------
# Tree-with-POS-bars
# ---------------------------------------------------------------------------

def compute_node_label_counts_disc(root, instances_tr, y_tr, max_depth=3):
    counts, node_obj = {}, {}
    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid] = np.zeros(N_POS, dtype=np.int32)
            node_obj[nid] = node
    for inst, label in zip(instances_tr, y_tr):
        node = root
        for depth in range(max_depth + 1):
            _ensure(node)
            counts[id(node)][int(label)] += 1
            if not node.children or depth == max_depth:
                break
            node = max(node.children, key=lambda c: c.log_prob_instance(inst))
    return counts, node_obj


print("Computing node label distributions (greedy descent) …")
label_counts_map, node_obj_map = compute_node_label_counts_disc(
    tree.root, instances_train, y, max_depth=TREE_DEPTH_FOR_LABEL_FIG,
)


def plot_tree_pos_labels(root, label_counts_map, out_path,
                         max_depth=3, highlight_ids=None,
                         highlight_color="red", title=""):
    highlight_ids = highlight_ids or set()
    def leaf_span(node, depth, md):
        if depth >= md or not node.children: return 1
        return sum(leaf_span(c, depth + 1, md) for c in node.children)
    pos = {}
    def assign_pos(node, depth, x_left):
        span = leaf_span(node, depth, max_depth)
        x_centre = x_left + span / 2.0
        pos[id(node)] = (x_centre, depth)
        if depth < max_depth and node.children:
            cursor = x_left
            for child in node.children:
                cs = leaf_span(child, depth + 1, max_depth)
                assign_pos(child, depth + 1, cursor)
                cursor += cs
    assign_pos(root, 0, 0.0)
    total_width = leaf_span(root, 0, max_depth)

    bar_w, bar_h, y_gap = 0.7, 0.35, 1.0
    fig, ax = plt.subplots(
        figsize=(max(14, total_width * 0.9), (max_depth + 1) * 2.2)
    )
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.7, max_depth * y_gap + 0.7)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title(title, fontsize=11)

    def draw_edges(node, depth):
        if depth >= max_depth or not node.children: return
        px, py = pos[id(node)]
        for child in node.children:
            cx, cy = pos[id(child)]
            ax.plot([px, cx], [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            draw_edges(child, depth + 1)
    draw_edges(root, 0)

    def draw_node(node, depth):
        nid = id(node)
        if nid not in label_counts_map: return
        cnts = label_counts_map[nid].astype(float)
        total = cnts.sum()
        if total == 0: return
        props = cnts / total
        x_c, _ = pos[nid]
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2
        cursor = x_left
        for cls_id in range(N_POS):
            seg_w = props[cls_id] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                           color=pos_colors[cls_id], lw=0))
                cursor += seg_w
        is_bl = nid in highlight_ids
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h, fill=False,
            edgecolor=(highlight_color if is_bl else "black"),
            lw=(3.0 if is_bl else 0.4),
            zorder=(5 if is_bl else 2)))
        ax.text(x_c, depth * y_gap + bar_h / 2 + 0.05,
                f"n={int(total)}", ha="center", va="top", fontsize=5)
        if depth < max_depth and node.children:
            for child in node.children:
                draw_node(child, depth + 1)
    draw_node(root, 0)

    legend_handles = [plt.Rectangle((0,0),1,1, color=pos_colors[i], label=id2pos[i])
                      for i in range(N_POS)]
    ax.legend(handles=legend_handles, title="POS", loc="lower right",
              ncol=max(1, N_POS // 4), fontsize=6, title_fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


plot_tree_pos_labels(
    tree.root, label_counts_map,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels_frontier.png"),
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    highlight_ids=set(new_members.keys()),
    highlight_color="red",
    title=("Cobweb Discrete Context Tree — POS Distributions  "
           "(red = α-agnostic frontier nodes)"),
)
plot_tree_pos_labels(
    tree.root, label_counts_map,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels_get_basic.png"),
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    highlight_ids=set(old_members.keys()),
    highlight_color="green",
    title=(f"Cobweb Discrete Context Tree — POS Distributions  "
           f"(green = old get_basic, eval_alpha={EVAL_ALPHA})"),
)
print(f"  Tree figures saved.")


# ---------------------------------------------------------------------------
# Mean basic-level score by depth
# Mirrors corter_gluck_hierarchies.py's plot, but x-axis is depth (int)
# instead of named categorical levels.
# ---------------------------------------------------------------------------

print("Computing basic_level_score for every node in the tree …")
all_nodes_full = []
stack = [tree.root]
while stack:
    n = stack.pop()
    all_nodes_full.append(n)
    stack.extend(n.children)
n_leaves_for_score = sum(1 for n in all_nodes_full if not n.children)
print(f"  {len(all_nodes_full)} nodes, {n_leaves_for_score} leaves")

depth_to_scores = {}
for n in all_nodes_full:
    d = n.depth()
    depth_to_scores.setdefault(d, []).append(n.basic_level_score(n_leaves_for_score))

# Mark which depths contain frontier nodes
depth_to_n_frontier = {}
for n in frontier_nodes:
    d = n.depth()
    depth_to_n_frontier[d] = depth_to_n_frontier.get(d, 0) + 1

depths_sorted = sorted(depth_to_scores.keys())
means = [np.mean(depth_to_scores[d]) for d in depths_sorted]
mins  = [np.min(depth_to_scores[d])  for d in depths_sorted]
maxs  = [np.max(depth_to_scores[d])  for d in depths_sorted]

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(depths_sorted, mins, maxs, alpha=0.15, color="#1f77b4",
                label="min–max range")
ax.plot(depths_sorted, means, marker="o", linewidth=2, color="#1f77b4",
        label="mean basic_level_score", zorder=3)

# Annotate each point with mean value
for d, m in zip(depths_sorted, means):
    ax.annotate(f"{m:.3f}", (d, m),
                textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha="center", color="#1f77b4")

# Mark depths where frontier nodes live
for d, n in depth_to_n_frontier.items():
    ax.axvline(d, color="red", alpha=0.25, linestyle="--", linewidth=1.2,
               zorder=1)
    ax.text(d, ax.get_ylim()[1], f"frontier × {n}",
            color="red", fontsize=7, ha="center", va="bottom",
            rotation=0)

ax.set_xlabel("Tree depth (root = 0)", fontsize=11)
ax.set_ylabel("Mean basic_level_score (α-agnostic)", fontsize=11)
ax.set_title("Mean α-agnostic basic-level score by depth  "
             "(red dashed = depth contains a frontier node)", fontsize=11)
ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
ax.set_xticks(depths_sorted)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
depth_plot_path = os.path.join(OUT_DIR, "score_by_depth.png")
plt.savefig(depth_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Score-by-depth plot saved → {depth_plot_path}")


# ---------------------------------------------------------------------------
# Method comparison summary
# ---------------------------------------------------------------------------

cmp_path = os.path.join(OUT_DIR, "method_comparison.txt")
with open(cmp_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write(" Frontier (NEW) vs get_basic (OLD) comparison\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Settings:\n")
    f.write(f"  N_SENTENCES = {N_SENTENCES}\n")
    f.write(f"  ALPHA       = {ALPHA}\n")
    f.write(f"  EVAL_ALPHA  = {EVAL_ALPHA}   (only used by old get_basic)\n")
    f.write(f"  N_SAMPLES   = {N_SAMPLES}    (only used by old get_basic)\n")
    f.write(f"  Train tokens: {len(instances_train)}\n")
    f.write(f"  Test  tokens: {len(instances_test)}\n\n")

    f.write(f"NEW: {len(new_members)} unique frontier nodes\n")
    for i, m in enumerate(sorted(new_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["pos_labels"])
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom = id2pos[int(cls_counts.argmax())]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  count={int(m['node'].count):>5}  "
                f"members={len(m['indices']):>4}  dom={dom}\n")

    f.write(f"\nOLD: {len(old_members)} unique BL nodes\n")
    for i, m in enumerate(sorted(old_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["pos_labels"])
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom = id2pos[int(cls_counts.argmax())]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  count={int(m['node'].count):>5}  "
                f"members={len(m['indices']):>4}  dom={dom}\n")

    f.write(f"\nPer-token agreement: {agreements}/{len(instances_test)} "
            f"({100*agreements/len(instances_test):.1f}%)\n")
    f.write(f"\nOne-BL-per-path (NEW frontier, all tree leaves):\n")
    for k in sorted(counts.keys()):
        f.write(f"  {k} frontier nodes on path: {counts[k]} "
                f"({100*counts[k]/n_leaves_tree:.1f}%)\n")
print(f"  Comparison saved → {cmp_path}")

print("\nDone.")
