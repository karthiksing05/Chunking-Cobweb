"""
Grammar basic-level subtree test
================================

Mirror of ``tests/moc/grammar_example.py`` focused on basic-level analysis.

We build a Cobweb-Discrete *context tree* over context-window instances
generated from a CFG corpus (TEST_GRAMMAR3), then for every test token
locate its basic-level subtree via
``CobwebDiscreteNode.get_basic_pc`` (P(c)-weighted EPMI against root)
and visualise them.

Outputs (in ``tests/basic-level/grammar_basic_level_output/``):

  - ``basic_level_subtrees.png`` : one row per unique basic-level node.
    Per row: POS histogram, top center words that landed there, and the
    top context word at each offset position.
  - ``cobweb_tree_labels.png``   : same POS-bar tree drawing as
    ``tests/moc/grammar_example.py``, top 4 depths.  Basic-level nodes
    selected by ``get_basic_pc`` are highlighted with a red border.
  - ``per_subtree_membership.csv`` : depth, member count, dominant POS,
    POS distribution.

Run directly: ``python tests/basic-level/grammar_basic_level_test.py``.
"""

import os
import sys
import csv
import random
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
WINDOW      = 3       # context half-window  (offsets ±1, ±2, ±3, excluding self)
ALPHA       = 1e-3
SEED        = 42
TREE_DEPTH_FOR_LABEL_FIG = 3
TOP_WORDS_PER_OFFSET     = 3  # top context words to display per offset

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Generate CFG corpus
# ---------------------------------------------------------------------------

print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if sent:
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
id2word    = {i: w for w, i in word2id.items()}
V          = len(vocab)
print(f"  Vocab size: {V}  |  Total tokens: {len(all_tokens)}")

# ---------------------------------------------------------------------------
# Word → POS mapping (from CFG terminal rules)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Build context-window instances
# ---------------------------------------------------------------------------

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
center_words  = []   # word id of the center word being represented
labels_all    = []   # POS id of the center word

for sent in sentences:
    for pos, word in enumerate(sent):
        instances_raw.append(make_context_instance(sent, pos))
        center_words.append(word2id[word])
        labels_all.append(pos2id[word2pos[word]])

center_words = np.array(center_words, dtype=np.int32)
labels_all   = np.array(labels_all,   dtype=np.int32)
print(f"  Total instances (tokens): {len(instances_raw)}")

# 80 / 20 train-test split (test set is what we descend through the tree)
rng       = np.random.default_rng(SEED)
idx       = rng.permutation(len(instances_raw))
split     = int(0.8 * len(instances_raw))
train_idx, test_idx = idx[:split], idx[split:]

instances_train = [instances_raw[i] for i in train_idx]
instances_test  = [instances_raw[i] for i in test_idx]
center_train    = center_words[train_idx]
center_test     = center_words[test_idx]
y               = labels_all[train_idx]
y_test          = labels_all[test_idx]
print(f"  Train: {len(instances_train)}  |  Test: {len(instances_test)}")

# ---------------------------------------------------------------------------
# Build the Cobweb Discrete tree
# ---------------------------------------------------------------------------

print(f"Building Cobweb Discrete tree (alpha={ALPHA}) …")
tree = CobwebDiscreteTree(alpha=ALPHA, weight_attr=True)
for i, inst in enumerate(instances_train):
    tree.ifit(inst)
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(instances_train)} inserted")
print("  Tree built.")


# ---------------------------------------------------------------------------
# Greedy descent → leaf for each test instance
# ---------------------------------------------------------------------------

def greedy_descend(root, instance):
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(instance))
    return node


# ---------------------------------------------------------------------------
# Map every test token to its basic-level subtree via get_basic_pc
# ---------------------------------------------------------------------------

print("Mapping test tokens to basic-level subtrees via get_basic_pc …")
bl_members = {}   # id(node) -> dict
for i, inst in enumerate(instances_test):
    leaf = greedy_descend(tree.root, inst)
    bl   = leaf.get_basic_pc(200, debug=False)
    nid  = id(bl)
    if nid not in bl_members:
        bl_members[nid] = {
            "node":          bl,
            "depth":         bl.depth(),
            "indices":       [],
            "center_words":  [],
            "pos_labels":    [],
        }
    bl_members[nid]["indices"].append(i)
    bl_members[nid]["center_words"].append(int(center_test[i]))
    bl_members[nid]["pos_labels"].append(int(y_test[i]))

print(f"  {len(bl_members)} unique basic-level subtree(s) recovered.")
sizes = [len(m["indices"]) for m in bl_members.values()]
print(f"  size distribution: min={min(sizes)}, max={max(sizes)}, "
      f"mean={np.mean(sizes):.1f}")


# ---------------------------------------------------------------------------
# Per-subtree visualisation
# ---------------------------------------------------------------------------

CMAP = plt.get_cmap("tab20") if N_POS > 10 else plt.get_cmap("tab10")
pos_colors = [CMAP(i / max(N_POS - 1, 1)) for i in range(N_POS)]

TOP_CENTER_WORDS = 6   # # most-frequent center words to display per row


def _top_context_words(node, k=TOP_WORDS_PER_OFFSET):
    """For each context offset, return the top-k words by attribute count."""
    out = {}  # offset -> [(word_str, count_frac)]
    for attr_id, val_map in node.av_count.items():
        if attr_id < 0 or attr_id not in {pos2attr[o] for o in CONTEXT_OFFSETS}:
            continue
        offset = offset_for_attr(attr_id)
        items  = sorted(val_map.items(), key=lambda kv: -kv[1])[:k]
        total  = sum(val_map.values()) or 1
        out[offset] = [(id2word.get(int(v), f"<{v}>"), c / total)
                       for v, c in items]
    return out


def plot_basic_level_subtrees(bl_members, out_path):
    """One row per BL subtree.

    Columns:
      [POS histogram] [top center words] [context offsets table]
    """
    sorted_bls = sorted(bl_members.values(),
                        key=lambda m: len(m["indices"]), reverse=True)
    n_rows = len(sorted_bls)
    n_cols = 3   # POS hist | center words | context table

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, max(2.0, n_rows * 1.6)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.4, 2.5]},
    )
    fig.suptitle(
        f"Basic-level subtrees on context tree  —  Grammar (TEST_GRAMMAR3)\n"
        f"selected via get_basic_pc  "
        f"(N_test={sum(len(m['indices']) for m in sorted_bls)}, "
        f"n_subtrees={n_rows})",
        fontsize=11,
    )

    for row, m in enumerate(sorted_bls):
        node      = m["node"]
        labels    = np.array(m["pos_labels"])
        centers   = np.array(m["center_words"])
        n_mem     = len(m["indices"])
        depth     = m["depth"]
        cls_counts = np.bincount(labels, minlength=N_POS)
        dom_pos    = id2pos[int(cls_counts.argmax())]

        # column 0: POS histogram (normalized)
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

        # column 1: top-K center words landing here
        ax1 = axes[row, 1]
        cw_counts = {}
        for c in centers:
            cw_counts[int(c)] = cw_counts.get(int(c), 0) + 1
        top_cw = sorted(cw_counts.items(), key=lambda kv: -kv[1])[:TOP_CENTER_WORDS]
        if top_cw:
            words   = [id2word[w] for w, _ in top_cw]
            counts  = [c for _, c in top_cw]
            colors_ = [pos_colors[pos2id[word2pos[id2word[w]]]] for w, _ in top_cw]
            ax1.barh(np.arange(len(words))[::-1], counts,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(words))[::-1])
            ax1.set_yticklabels(words, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0:
            ax1.set_title("top center words", fontsize=8)

        # column 2: context table — top-k word per offset
        ax2 = axes[row, 2]
        ax2.axis("off")
        ctx_top = _top_context_words(node, k=TOP_WORDS_PER_OFFSET)
        # render as a small text table, one column per offset
        offsets = sorted(ctx_top.keys())
        if offsets:
            n_off = len(offsets)
            x_step = 1.0 / max(n_off, 1)
            for ci, off in enumerate(offsets):
                cx = (ci + 0.5) * x_step
                ax2.text(cx, 0.95, f"{off:+d}",
                         ha="center", va="top",
                         fontsize=7, fontweight="bold",
                         transform=ax2.transAxes)
                for li, (w, frac) in enumerate(ctx_top[off]):
                    cy = 0.85 - li * 0.20
                    ax2.text(cx, cy,
                             f"{w} ({frac:.2f})",
                             ha="center", va="top",
                             fontsize=6,
                             color=pos_colors[pos2id.get(
                                 word2pos.get(w, "Unk"), 0)],
                             transform=ax2.transAxes)
        if row == 0:
            ax2.set_title("top context word per offset", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Subtree visualisation saved → {out_path}")


plot_basic_level_subtrees(
    bl_members,
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees.png"),
)


# ---------------------------------------------------------------------------
# CSV summary
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
# Tree-with-POS-bars figure (mirrors grammar_example.py), red borders on BL nodes
# ---------------------------------------------------------------------------

def compute_node_label_counts_disc(root, instances_tr, y_tr, max_depth=3):
    n_classes = N_POS
    counts    = {}
    node_obj  = {}

    def _ensure(node):
        nid = id(node)
        if nid not in counts:
            counts[nid]   = np.zeros(n_classes, dtype=np.int32)
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
                         max_depth=3, highlight_ids=None):
    highlight_ids = highlight_ids or set()

    def leaf_span(node, depth, md):
        if depth >= md or not node.children:
            return 1
        return sum(leaf_span(c, depth + 1, md) for c in node.children)

    pos = {}

    def assign_pos(node, depth, x_left):
        span     = leaf_span(node, depth, max_depth)
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
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Cobweb Discrete Context Tree — POS Distributions (depths 0–{max_depth})\n"
        f"red border = basic-level node selected by get_basic_pc",
        fontsize=11,
    )

    def draw_edges(node, depth):
        if depth >= max_depth or not node.children:
            return
        px, py = pos[id(node)]
        for child in node.children:
            cx, cy = pos[id(child)]
            ax.plot([px, cx], [py * y_gap + bar_h / 2, cy * y_gap - bar_h / 2],
                    color="gray", lw=0.8, zorder=0)
            draw_edges(child, depth + 1)

    draw_edges(root, 0)

    def draw_node(node, depth):
        nid = id(node)
        if nid not in label_counts_map:
            return
        cnts  = label_counts_map[nid].astype(float)
        total = cnts.sum()
        if total == 0:
            return
        props  = cnts / total
        x_c, _ = pos[nid]
        x_left = x_c - bar_w / 2
        y_top  = depth * y_gap - bar_h / 2
        cursor = x_left
        for cls_id in range(N_POS):
            seg_w = props[cls_id] * bar_w
            if seg_w > 0:
                rect = plt.Rectangle((cursor, y_top), seg_w, bar_h,
                                     color=pos_colors[cls_id], lw=0)
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
        plt.Rectangle((0, 0), 1, 1, color=pos_colors[i],
                       label=id2pos[i])
        for i in range(N_POS)
    ]
    ax.legend(handles=legend_handles, title="POS", loc="lower right",
              ncol=max(1, N_POS // 4), fontsize=6, title_fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


highlight_ids = set(bl_members.keys())
plot_tree_pos_labels(
    tree.root, label_counts_map,
    out_path=os.path.join(OUT_DIR, "cobweb_tree_labels.png"),
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    highlight_ids=highlight_ids,
)
print(f"  Tree figure saved → {os.path.join(OUT_DIR, 'cobweb_tree_labels.png')}")

print("\nDone.")
