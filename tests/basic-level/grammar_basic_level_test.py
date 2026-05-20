"""
Grammar basic-level test — leaf.get_basic(use_root=True, eval_alpha=10).
========================================================================

Builds a Cobweb-Discrete context tree on a TEST_GRAMMAR3 corpus, then for
each test token walks its greedy-descent leaf up to root using the new
``CobwebDiscreteNode::get_basic(use_root=True)`` — empirical PMI against
the root marginal (closed form when ``n_samples=0``; same formula as
``tests/basic-level/corter_gluck_hierarchies_cobweb.py`` and the Python
``corter_gluck_hierarchies.py`` reference).

For every test token we record which BL node it lands at, then collect:
  - the unique BL nodes,
  - per-BL POS distributions, top center words, and top context words,
  - mean empirical-PMI by depth across the whole tree.

Outputs (in tests/basic-level/grammar_basic_level_output/):
  - basic_level_subtrees.png       : POS hist + center words + context
                                     table, one row per BL node.
  - cobweb_tree_labels.png         : tree with red borders on BL nodes.
  - per_subtree_membership.csv     : depth, count, dominant POS, POS
                                     distribution.
  - method_summary.txt             : per-BL summary text.
  - score_by_depth.png             : mean expected_pmi(use_root=True,
                                     eval_alpha=EVAL_ALPHA) by depth,
                                     vertical marks where BL nodes live.
"""

import os
import sys
import csv
import random

import numpy as np
import matplotlib
# Leave the default backend in place so the interactive α-slider can
# pop up (cf. ``corter_gluck_hierarchies.py``). To run the test
# headless (CI / no display) set ``MPLBACKEND=Agg`` in the environment
# before invocation — ``plt.savefig`` keeps working on any backend.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

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
EVAL_ALPHA  = 10.0     # high-α smoothing inside the new get_basic
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
# Basic level per leaf via the new get_basic(use_root=True). Closed form:
# n_samples=0 evaluates expected_pmi over every leaf under each candidate
# ancestor — exact empirical PMI against the root marginal.
# ---------------------------------------------------------------------------

print(f"\nRunning get_basic(use_root=True, eval_alpha={EVAL_ALPHA}) per leaf …")
_cache = {}
def get_basic_node(leaf):
    key = id(leaf)
    if key in _cache:
        return _cache[key]
    bl = leaf.get_basic(0, 0, debug=False, eval_alpha=EVAL_ALPHA, use_root=True)
    _cache[key] = bl
    return bl


# ---------------------------------------------------------------------------
# Map test tokens to BL nodes
# ---------------------------------------------------------------------------

print("Mapping test tokens to basic-level nodes …")

bl_members = {}   # id(bl_node) -> {node, depth, indices, center_words, pos_labels}

for i, inst in enumerate(instances_test):
    leaf = greedy_descend(tree.root, inst)
    bl   = get_basic_node(leaf)
    if bl is None:
        continue
    nid = id(bl)
    if nid not in bl_members:
        bl_members[nid] = {
            "node":         bl,
            "depth":        bl.depth(),
            "indices":      [],
            "center_words": [],
            "pos_labels":   [],
        }
    bl_members[nid]["indices"].append(i)
    bl_members[nid]["center_words"].append(int(center_test[i]))
    bl_members[nid]["pos_labels"].append(int(y_test[i]))

print(f"  {len(bl_members)} unique BL nodes covering {len(instances_test)} tokens")


# ---------------------------------------------------------------------------
# Per-subtree visualisation
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
    bl_members,
    title=(f"Basic-level subtrees — get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(N_test={len(instances_test)}, n_subtrees={len(bl_members)})"),
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees.png"),
)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUT_DIR, "per_subtree_membership.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subtree_idx", "depth", "node_count", "test_members",
                "dominant_pos", "pos_distribution"])
    for i, m in enumerate(sorted(bl_members.values(),
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
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"),
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    highlight_ids=set(bl_members.keys()),
    highlight_color="red",
    title=(f"Cobweb Discrete Context Tree — POS Distributions  "
           f"(red = basic-level nodes, eval_alpha={EVAL_ALPHA})"),
)
print(f"  Tree figure saved.")


# ---------------------------------------------------------------------------
# Mean expected_pmi by depth — the new α-dependent score evaluated at
# EVAL_ALPHA via the closed-form path (n_samples=0).
# ---------------------------------------------------------------------------

print(f"Computing expected_pmi(use_root=True, eval_alpha={EVAL_ALPHA}) for every node …")
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
    score = n.expected_pmi(0, 0, eval_alpha=EVAL_ALPHA,
                           uniform_leaf=False, use_root=True)
    depth_to_scores.setdefault(d, []).append(score)

depth_to_n_bl = {}
for m in bl_members.values():
    d = m["depth"]
    depth_to_n_bl[d] = depth_to_n_bl.get(d, 0) + 1

depths_sorted = sorted(depth_to_scores.keys())
means = [np.mean(depth_to_scores[d]) for d in depths_sorted]
mins  = [np.min(depth_to_scores[d])  for d in depths_sorted]
maxs  = [np.max(depth_to_scores[d])  for d in depths_sorted]

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(depths_sorted, mins, maxs, alpha=0.15, color="#1f77b4",
                label="min–max range")
ax.plot(depths_sorted, means, marker="o", linewidth=2, color="#1f77b4",
        label="mean expected_pmi", zorder=3)

for d, m in zip(depths_sorted, means):
    ax.annotate(f"{m:.3f}", (d, m),
                textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha="center", color="#1f77b4")

for d, n in depth_to_n_bl.items():
    ax.axvline(d, color="red", alpha=0.25, linestyle="--", linewidth=1.2,
               zorder=1)
    ax.text(d, ax.get_ylim()[1], f"BL × {n}",
            color="red", fontsize=7, ha="center", va="bottom",
            rotation=0)

ax.set_xlabel("Tree depth (root = 0)", fontsize=11)
ax.set_ylabel(f"Mean expected_pmi (use_root=True, eval_alpha={EVAL_ALPHA})",
              fontsize=11)
ax.set_title("Mean empirical PMI against root by depth  "
             "(red dashed = depth contains a BL node)", fontsize=11)
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
# Summary
# ---------------------------------------------------------------------------

summary_path = os.path.join(OUT_DIR, "method_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write(" Basic-level summary — get_basic(use_root=True)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Settings:\n")
    f.write(f"  N_SENTENCES = {N_SENTENCES}\n")
    f.write(f"  ALPHA       = {ALPHA}\n")
    f.write(f"  EVAL_ALPHA  = {EVAL_ALPHA}\n")
    f.write(f"  Train tokens: {len(instances_train)}\n")
    f.write(f"  Test  tokens: {len(instances_test)}\n\n")

    f.write(f"{len(bl_members)} unique basic-level nodes:\n")
    for i, m in enumerate(sorted(bl_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["pos_labels"])
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom = id2pos[int(cls_counts.argmax())]
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  count={int(m['node'].count):>5}  "
                f"members={len(m['indices']):>4}  dom={dom}\n")
print(f"  Summary saved → {summary_path}")


# ---------------------------------------------------------------------------
# Interactive α-slider for score_by_depth (mirrors corter_gluck_hierarchies.py)
# ---------------------------------------------------------------------------
# The static ``score_by_depth.png`` is computed at EVAL_ALPHA. This slider
# lets you sweep ``log_10(eval_alpha)`` and watch the depth curve update
# live — the same exploratory affordance as
# ``tests/basic-level/corter_gluck_hierarchies.py``. The BL vertical
# markers stay anchored at the initial EVAL_ALPHA (changing the slider
# would also shift basic-level membership; we keep the BL set frozen
# so the slider only shows how the *depth-score curve itself* moves
# with α).

print("\nOpening interactive α-slider for score_by_depth …")

fig_sl = plt.figure(figsize=(10, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.32)
ax_plot   = fig_sl.add_subplot(gs[0])
ax_slider = fig_sl.add_subplot(gs[1])

slider = Slider(
    ax=ax_slider,
    label="log₁₀(eval_alpha)",
    valmin=-3, valmax=5,
    valinit=float(np.log10(EVAL_ALPHA)),
    valstep=0.05,
)


def _redraw(eval_a):
    ax_plot.clear()
    # Recompute expected_pmi per node at the new α.
    d2s: dict = {}
    for n in all_nodes_full:
        d = n.depth()
        score = n.expected_pmi(0, 0, eval_alpha=eval_a,
                               uniform_leaf=False, use_root=True)
        d2s.setdefault(d, []).append(score)
    depths_sorted = sorted(d2s.keys())
    means = [float(np.mean(d2s[d])) for d in depths_sorted]
    mins  = [float(np.min(d2s[d]))  for d in depths_sorted]
    maxs  = [float(np.max(d2s[d]))  for d in depths_sorted]

    ax_plot.fill_between(depths_sorted, mins, maxs, alpha=0.15,
                          color="#1f77b4", label="min–max range")
    ax_plot.plot(depths_sorted, means, marker="o", linewidth=2,
                  color="#1f77b4", label="mean expected_pmi", zorder=3)
    for d, m in zip(depths_sorted, means):
        ax_plot.annotate(f"{m:.3f}", (d, m),
                          textcoords="offset points", xytext=(0, 8),
                          fontsize=8, ha="center", color="#1f77b4")
    # BL markers from the original EVAL_ALPHA run (frozen).
    for d, n in depth_to_n_bl.items():
        ax_plot.axvline(d, color="red", alpha=0.25, linestyle="--",
                         linewidth=1.2, zorder=1)
        ax_plot.text(d, ax_plot.get_ylim()[1], f"BL × {n}",
                      color="red", fontsize=7, ha="center", va="bottom")

    ax_plot.set_xlabel("Tree depth (root = 0)", fontsize=11)
    ax_plot.set_ylabel(f"Mean expected_pmi  (eval_alpha = {eval_a:.4g})",
                        fontsize=11)
    ax_plot.set_title(
        f"Interactive α sweep — score_by_depth   "
        f"(BL markers frozen at initial α = {EVAL_ALPHA})",
        fontsize=11)
    ax_plot.axhline(0, color="black", linewidth=0.8, linestyle=":",
                     alpha=0.4)
    ax_plot.set_xticks(depths_sorted)
    ax_plot.grid(axis="y", alpha=0.25)
    ax_plot.legend(loc="best", fontsize=9)
    fig_sl.canvas.draw_idle()


slider.on_changed(lambda _val: _redraw(10 ** slider.val))
_redraw(10 ** slider.valinit)
plt.show()

print("\nDone.")
