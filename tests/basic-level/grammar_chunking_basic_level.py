"""
Grammar Chunking basic-level example — TopK-Disc-Cnt1 representation.
=====================================================================

Mirrors ``tests/moc/grammar_chunking_example.py`` for the corpus,
context tree, and TopK-Disc-Cnt1 pair encoding, then layers on the new
basic-level pipeline used in ``tests/basic-level/grammar_basic_level_test.py``:

  1. Build a Cobweb-Discrete *context* tree on TEST_GRAMMAR3 tokens.
  2. For each adjacent bigram (w_i, w_{i+1}), pull top-K context-tree
     node ids at depth ``TOPK_DEPTH`` per side → TopK-Disc-Cnt1 instance
     ``{0:{top-K-L:1.0}, 1:{top-K-R:1.0}}``.
  3. Train a Cobweb-Discrete *content* tree on those instances.
  4. For every test pair, greedy-descend to a content leaf and call
     ``leaf.get_basic(use_root=True, eval_alpha=EVAL_ALPHA)`` — the new
     empirical-PMI-against-root basic level.

Outputs (in tests/moc/grammar_chunking_basic_level_output/):
  - basic_level_subtrees.png       : per-BL row showing
                                     (L_POS, R_POS) joint heat-map +
                                     top center bigrams + top per-side
                                     context-attr signatures.
  - content_tree_labels.png        : content tree with stacked L/R POS
                                     bars at every node; red borders
                                     mark BL nodes.
  - per_subtree_membership.csv     : depth, count, dominant pair POS,
                                     joint POS distribution.
  - method_summary.txt             : per-BL summary text.
  - score_by_depth.png             : mean expected_pmi(use_root=True,
                                     eval_alpha=EVAL_ALPHA) by depth in
                                     the content tree.
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

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from util.cfg import TEST_GRAMMAR3, generate
from cobweb.cobweb_discrete import CobwebDiscreteTree

OUT_DIR = os.path.join(_HERE, "grammar_chunking_basic_level_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants (same shape as grammar_chunking_example.py) ────────────────────
N_SENTENCES   = 1000
WINDOW        = 3
TOP_K         = 5
TOPK_DEPTH    = 4
ALPHA_CONTEXT = 1e-3
ALPHA_CONTENT = 1e-3
EVAL_ALPHA    = 10.0     # for get_basic / expected_pmi (use_root=True)
SEED          = 42
TREE_DEPTH_FOR_LABEL_FIG = 3
TOP_CENTER_BIGRAMS = 6
TOP_CTX_NODES      = 5
random.seed(SEED); np.random.seed(SEED)

# ── Corpus ───────────────────────────────────────────────────────────────────
print(f"Generating {N_SENTENCES} sentences from TEST_GRAMMAR3 …")
sentences = []
for _ in range(N_SENTENCES):
    sent = [w for w in generate("S", TEST_GRAMMAR3).split() if w]
    if len(sent) >= 2:
        sentences.append(sent)

all_tokens = [w for sent in sentences for w in sent]
vocab      = sorted(set(all_tokens))
word2id    = {w: i for i, w in enumerate(vocab)}
id2word    = {i: w for w, i in word2id.items()}
V          = len(vocab)
print(f"  Sentences: {len(sentences)} | Vocab: {V} | Tokens: {len(all_tokens)}")

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


def pair_label(left_w, right_w):
    return pos2id[word2pos[left_w]] * N_POS + pos2id[word2pos[right_w]]


def split_pair_label(lbl):
    return lbl // N_POS, lbl % N_POS


# ── Context-window encoding ──────────────────────────────────────────────────
CONTEXT_OFFSETS = [p for p in range(-WINDOW, WINDOW + 1) if p != 0]
pos2attr        = {p: i for i, p in enumerate(CONTEXT_OFFSETS)}


def make_context_instance(sent, pos):
    inst = {}
    for offset in CONTEXT_OFFSETS:
        ctx = pos + offset
        if 0 <= ctx < len(sent):
            inst[pos2attr[offset]] = {word2id[sent[ctx]]: 1.0}
    return inst


# Sentence split — no leakage between train/test pairs
rng         = np.random.default_rng(SEED)
sent_perm   = rng.permutation(len(sentences))
split_s     = int(0.8 * len(sentences))
train_sents = [sentences[i] for i in sent_perm[:split_s]]
test_sents  = [sentences[i] for i in sent_perm[split_s:]]
print(f"  Sentence split: train={len(train_sents)} test={len(test_sents)}")

context_train_tokens = [make_context_instance(s, p)
                        for s in train_sents for p in range(len(s))]
print(f"  Context tree training tokens: {len(context_train_tokens)}")


def build_pair_set(sents):
    L, R, labels, words_L, words_R = [], [], [], [], []
    for s in sents:
        for i in range(len(s) - 1):
            L.append(make_context_instance(s, i))
            R.append(make_context_instance(s, i + 1))
            labels.append(pair_label(s[i], s[i + 1]))
            words_L.append(word2id[s[i]])
            words_R.append(word2id[s[i + 1]])
    return (L, R,
            np.array(labels,  dtype=np.int32),
            np.array(words_L, dtype=np.int32),
            np.array(words_R, dtype=np.int32))


train_L, train_R, y_train, train_wL, train_wR = build_pair_set(train_sents)
test_L,  test_R,  y_test,  test_wL,  test_wR  = build_pair_set(test_sents)
print(f"  Pairs:  train={len(train_L)}  test={len(test_L)}")


# ── Context tree ─────────────────────────────────────────────────────────────
print(f"\nBuilding Context tree (CobwebDiscreteTree, alpha={ALPHA_CONTEXT}) …")
context_tree = CobwebDiscreteTree(alpha=ALPHA_CONTEXT, weight_attr=True)
for i, inst in enumerate(context_train_tokens):
    context_tree.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(context_train_tokens)} inserted")
print("  Context tree built.")


def collect_by_depth_nodes(root):
    by_depth, queue = {}, [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        by_depth.setdefault(d, []).append(node)
        for c in node.children:
            queue.append((c, d + 1))
    return by_depth


by_depth     = collect_by_depth_nodes(context_tree.root)
depth_counts = {d: len(v) for d, v in by_depth.items()}
print(f"  Nodes per depth: {dict(sorted(depth_counts.items()))}")
topk_depth      = min(TOPK_DEPTH, max(depth_counts))
topk_pool_nodes = by_depth[topk_depth]
if topk_depth != TOPK_DEPTH:
    print(f"  WARN: requested TOPK_DEPTH={TOPK_DEPTH} clamped to {topk_depth}")
print(f"  TopK pool: depth {topk_depth} ({len(topk_pool_nodes)} nodes)  k={TOP_K}")


# ── TopK-Disc-Cnt1 encoding ──────────────────────────────────────────────────
def encode_logpost_disc(instances, nodes):
    out = np.empty((len(instances), len(nodes)), dtype=np.float64)
    for j, node in enumerate(nodes):
        for i, inst in enumerate(instances):
            out[i, j] = node.log_prob_instance(inst)
    return out


def topk_indices(Z_raw, k):
    k = min(k, Z_raw.shape[1])
    return np.argpartition(Z_raw, -k, axis=1)[:, -k:]


print("Encoding TopK pool log-probs …")
_pool_train_L = encode_logpost_disc(train_L, topk_pool_nodes)
_pool_train_R = encode_logpost_disc(train_R, topk_pool_nodes)
_pool_test_L  = encode_logpost_disc(test_L,  topk_pool_nodes)
_pool_test_R  = encode_logpost_disc(test_R,  topk_pool_nodes)

topk_idx_train_L = topk_indices(_pool_train_L, TOP_K)
topk_idx_train_R = topk_indices(_pool_train_R, TOP_K)
topk_idx_test_L  = topk_indices(_pool_test_L,  TOP_K)
topk_idx_test_R  = topk_indices(_pool_test_R,  TOP_K)


def topkdisc_cnt1_inst(idx_L, idx_R):
    return {0: {int(j): 1.0 for j in idx_L},
            1: {int(j): 1.0 for j in idx_R}}


pair_train = [topkdisc_cnt1_inst(iL, iR)
              for iL, iR in zip(topk_idx_train_L, topk_idx_train_R)]
pair_test  = [topkdisc_cnt1_inst(iL, iR)
              for iL, iR in zip(topk_idx_test_L,  topk_idx_test_R)]
print(f"  TopK-Disc-Cnt1 instances built: train={len(pair_train)}  test={len(pair_test)}")


# ── Content tree ─────────────────────────────────────────────────────────────
print(f"\nBuilding Content tree (CobwebDiscreteTree, alpha={ALPHA_CONTENT}) …")
content_tree = CobwebDiscreteTree(alpha=ALPHA_CONTENT, weight_attr=True)
for i, inst in enumerate(pair_train):
    content_tree.ifit(inst)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(pair_train)} inserted")
print("  Content tree built.")


def greedy_descend(root, instance):
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(instance))
    return node


# ── Basic level via leaf.get_basic(use_root=True) ────────────────────────────
print(f"\nRunning get_basic(use_root=True, eval_alpha={EVAL_ALPHA}) per leaf …")
_cache = {}
def get_basic_node(leaf):
    key = id(leaf)
    if key in _cache:
        return _cache[key]
    bl = leaf.get_basic(0, 0, debug=False, eval_alpha=EVAL_ALPHA, use_root=True)
    _cache[key] = bl
    return bl


print("Mapping test pairs to basic-level nodes …")
bl_members = {}
for i, inst in enumerate(pair_test):
    leaf = greedy_descend(content_tree.root, inst)
    bl   = get_basic_node(leaf)
    if bl is None:
        continue
    nid = id(bl)
    if nid not in bl_members:
        bl_members[nid] = {
            "node":     bl,
            "depth":    bl.depth(),
            "indices":  [],
            "wL":       [],
            "wR":       [],
            "labels":   [],
        }
    bl_members[nid]["indices"].append(i)
    bl_members[nid]["wL"].append(int(test_wL[i]))
    bl_members[nid]["wR"].append(int(test_wR[i]))
    bl_members[nid]["labels"].append(int(y_test[i]))

print(f"  {len(bl_members)} unique BL nodes covering {len(pair_test)} test pairs")


# ── Per-subtree visualisation ────────────────────────────────────────────────
CMAP_POS    = plt.get_cmap("tab10") if N_POS <= 10 else plt.get_cmap("tab20")
pos_colors  = [CMAP_POS(i / max(N_POS - 1, 1)) for i in range(N_POS)]


def top_context_attrs(node, k=TOP_CTX_NODES):
    """Top-k pool-node ids stored at this content node, per side (attr 0 = L, 1 = R)."""
    out = {0: [], 1: []}
    for attr in (0, 1):
        if attr not in node.av_count:
            continue
        items = sorted(node.av_count[attr].items(), key=lambda kv: -kv[1])[:k]
        total = sum(node.av_count[attr].values()) or 1
        out[attr] = [(int(pool_id), c / total) for pool_id, c in items]
    return out


def plot_subtrees(members, title, out_path):
    sorted_bls = sorted(members.values(),
                        key=lambda m: len(m["indices"]), reverse=True)
    n_rows = len(sorted_bls)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, max(2.4, n_rows * 1.9)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.4, 1.6, 2.2]},
    )
    fig.suptitle(title, fontsize=11)

    for row, m in enumerate(sorted_bls):
        node    = m["node"]
        labels  = np.array(m["labels"])
        wL_arr  = np.array(m["wL"])
        wR_arr  = np.array(m["wR"])
        n_mem   = len(m["indices"])
        depth   = m["depth"]

        # Joint (L_POS, R_POS) heatmap
        joint = np.zeros((N_POS, N_POS), dtype=np.int32)
        for lbl in labels:
            lp, rp = split_pair_label(int(lbl))
            joint[lp, rp] += 1
        flat_top = joint.flatten().argmax()
        dom_l, dom_r = flat_top // N_POS, flat_top % N_POS
        dom_pair = f"{id2pos[dom_l]}-{id2pos[dom_r]}"

        ax0 = axes[row, 0]
        im = ax0.imshow(joint / max(joint.sum(), 1), cmap="Blues",
                        vmin=0, vmax=1, aspect="equal")
        ax0.set_xticks(range(N_POS))
        ax0.set_yticks(range(N_POS))
        ax0.set_xticklabels([id2pos[i] for i in range(N_POS)],
                            rotation=45, ha="right", fontsize=6)
        ax0.set_yticklabels([id2pos[i] for i in range(N_POS)], fontsize=6)
        ax0.set_xlabel("R POS", fontsize=6)
        ax0.set_ylabel(
            f"d={depth}\nn={n_mem}\ncnt={int(node.count)}\ndom={dom_pair}\n\nL POS",
            fontsize=6, rotation=0, labelpad=42, va="center",
        )
        if row == 0:
            ax0.set_title("(L,R) POS joint", fontsize=8)

        # Top center bigrams
        ax1 = axes[row, 1]
        bigram_counts = Counter(zip(wL_arr.tolist(), wR_arr.tolist()))
        top_big = bigram_counts.most_common(TOP_CENTER_BIGRAMS)
        if top_big:
            labels_str = [f"{id2word[wl]} {id2word[wr]}" for (wl, wr), _ in top_big]
            counts_    = [c for _, c in top_big]
            colors_    = [pos_colors[pos2id[word2pos[id2word[wl]]]]
                          for (wl, _), _ in top_big]
            ax1.barh(np.arange(len(labels_str))[::-1], counts_,
                     color=colors_, edgecolor="white", linewidth=0.4)
            ax1.set_yticks(np.arange(len(labels_str))[::-1])
            ax1.set_yticklabels(labels_str, fontsize=6)
            ax1.tick_params(axis="x", labelsize=5)
        if row == 0:
            ax1.set_title("top center bigrams", fontsize=8)

        # Top context-tree node ids stored at this content node, per side
        ax2 = axes[row, 2]
        ax2.axis("off")
        ctx_top = top_context_attrs(node, k=TOP_CTX_NODES)
        for ci, side in enumerate(("L", "R")):
            cx = (ci + 0.5) / 2.0
            ax2.text(cx, 0.95, side, ha="center", va="top",
                     fontsize=7, fontweight="bold", transform=ax2.transAxes)
            for li, (pool_id, frac) in enumerate(ctx_top[ci]):
                cy = 0.85 - li * 0.16
                ax2.text(cx, cy, f"ctx#{pool_id} ({frac:.2f})",
                         ha="center", va="top", fontsize=6,
                         color="black", transform=ax2.transAxes)
        if row == 0:
            ax2.set_title(f"top context pool-node ids (k={TOP_CTX_NODES})",
                          fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Subtree visualisation saved → {out_path}")


plot_subtrees(
    bl_members,
    title=(f"Basic-level subtrees — TopK-Disc-Cnt1 content tree, "
           f"get_basic(use_root=True, eval_alpha={EVAL_ALPHA})  "
           f"(N_test={len(pair_test)}, n_subtrees={len(bl_members)})"),
    out_path=os.path.join(OUT_DIR, "basic_level_subtrees.png"),
)


# ── CSV ──────────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "per_subtree_membership.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subtree_idx", "depth", "node_count", "test_members",
                "dominant_pair", "top_pair_dist"])
    for i, m in enumerate(sorted(bl_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["labels"])
        joint  = np.bincount(labels, minlength=N_POS * N_POS)
        top5 = np.argsort(-joint)[:5]
        dl, dr = split_pair_label(int(top5[0]))
        dist_str = "/".join(
            f"{id2pos[split_pair_label(int(p))[0]]}-"
            f"{id2pos[split_pair_label(int(p))[1]]}:{int(joint[p])}"
            for p in top5 if joint[p] > 0
        )
        w.writerow([i, m["depth"], int(m["node"].count),
                    len(m["indices"]),
                    f"{id2pos[dl]}-{id2pos[dr]}", dist_str])
print(f"  CSV summary saved → {csv_path}")


# ── Content-tree with stacked L/R POS bars + BL highlights ───────────────────
def make_layout(root, max_depth):
    all_nodes   = [root]
    children_of = {0: []}
    depth_of    = {0: 0}
    queue       = [0]
    while queue:
        idx  = queue.pop(0)
        node = all_nodes[idx]
        if depth_of[idx] < max_depth:
            for c in node.children:
                cidx = len(all_nodes)
                all_nodes.append(c)
                children_of[idx].append(cidx)
                children_of[cidx] = []
                depth_of[cidx]    = depth_of[idx] + 1
                queue.append(cidx)
    return all_nodes, children_of, depth_of


def descend_layout_disc(all_nodes, children_of, inst, max_depth):
    visited, cur = [0], 0
    for _ in range(max_depth):
        ch = children_of[cur]
        if not ch:
            break
        best = max(ch, key=lambda i: all_nodes[i].log_prob_instance(inst))
        visited.append(best)
        cur = best
    return visited


def compute_pair_label_counts_idx(all_nodes, children_of,
                                  train_data, y_tr, max_depth):
    cL, cR = {}, {}
    for x, lbl in zip(train_data, y_tr):
        l_pos, r_pos = split_pair_label(int(lbl))
        for idx in descend_layout_disc(all_nodes, children_of, x, max_depth):
            if idx not in cL:
                cL[idx] = np.zeros(N_POS, dtype=np.int32)
                cR[idx] = np.zeros(N_POS, dtype=np.int32)
            cL[idx][l_pos] += 1
            cR[idx][r_pos] += 1
    return cL, cR


# Map id(bl_node) -> layout idx (needed for tree-plot highlights).
print("Computing content-tree layout + per-node label counts …")
layout_nodes, layout_children, layout_depth = make_layout(
    content_tree.root, max_depth=TREE_DEPTH_FOR_LABEL_FIG,
)
cL_map, cR_map = compute_pair_label_counts_idx(
    layout_nodes, layout_children, pair_train, y_train,
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
)

# BL nodes that happen to live within the rendered depth band
bl_node_hashes = {m["node"].concept_hash() for m in bl_members.values()}
highlight_idx = set()
for idx, n in enumerate(layout_nodes):
    if n.concept_hash() in bl_node_hashes:
        highlight_idx.add(idx)


def plot_pair_tree_idx(children_of, depth_of, counts_L, counts_R,
                        title, out_path, max_depth, highlight_idx=None):
    highlight_idx = highlight_idx or set()
    def leaf_span(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return 1
        return sum(leaf_span(c, depth + 1) for c in children_of[idx])

    pos = {}
    def assign_pos(idx, depth, x_left):
        span     = leaf_span(idx, depth)
        x_centre = x_left + span / 2.0
        pos[idx] = (x_centre, depth)
        if depth < max_depth and children_of[idx]:
            cur = x_left
            for c in children_of[idx]:
                cs = leaf_span(c, depth + 1)
                assign_pos(c, depth + 1, cur)
                cur += cs

    assign_pos(0, 0, 0.0)
    total_w = leaf_span(0, 0)

    bar_w, bar_h, gap, y_unit = 0.7, 0.18, 0.05, 1.0
    fig, ax = plt.subplots(figsize=(max(14, total_w * 0.9),
                                    (max_depth + 1) * 2.4))
    ax.set_xlim(0, total_w)
    ax.set_ylim(-0.7, max_depth * y_unit + 0.7)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title(title, fontsize=11)

    def draw_edges(idx, depth):
        if depth >= max_depth or not children_of[idx]:
            return
        px, py = pos[idx]
        for c in children_of[idx]:
            cx, cy = pos[c]
            y_par = py * y_unit + bar_h + gap / 2
            y_chi = cy * y_unit - bar_h - gap / 2
            ax.plot([px, cx], [y_par, y_chi], color="gray", lw=0.8, zorder=0)
            draw_edges(c, depth + 1)
    draw_edges(0, 0)

    def _draw_bar(x_left, y_top, props, label_text, is_bl):
        cur = x_left
        for p_idx in range(N_POS):
            seg_w = props[p_idx] * bar_w
            if seg_w > 0:
                ax.add_patch(plt.Rectangle((cur, y_top), seg_w, bar_h,
                                            color=pos_colors[p_idx], lw=0))
                cur += seg_w
        ax.add_patch(plt.Rectangle(
            (x_left, y_top), bar_w, bar_h, fill=False,
            edgecolor=("red" if is_bl else "black"),
            lw=(3.0 if is_bl else 0.5),
            zorder=(5 if is_bl else 2),
        ))
        ax.text(x_left - 0.05, y_top + bar_h / 2, label_text,
                ha="right", va="center", fontsize=5)

    def draw_node(idx, depth):
        if idx not in counts_L:
            return
        cntL = counts_L[idx].astype(float)
        cntR = counts_R[idx].astype(float)
        tot  = cntL.sum()
        if tot == 0:
            return
        propsL = cntL / tot
        propsR = cntR / tot
        is_bl  = idx in highlight_idx
        x_c, _ = pos[idx]
        x_left = x_c - bar_w / 2
        y_top_L = depth * y_unit - bar_h - gap / 2
        y_top_R = depth * y_unit + gap / 2
        _draw_bar(x_left, y_top_L, propsL, "L", is_bl)
        _draw_bar(x_left, y_top_R, propsR, "R", is_bl)
        ax.text(x_c, depth * y_unit - bar_h - gap / 2 - 0.04,
                f"n={int(tot)}", ha="center", va="bottom", fontsize=5)
        if depth < max_depth and children_of[idx]:
            for c in children_of[idx]:
                draw_node(c, depth + 1)
    draw_node(0, 0)

    legend_h = [plt.Rectangle((0, 0), 1, 1, color=pos_colors[i], label=id2pos[i])
                for i in range(N_POS)]
    ax.legend(handles=legend_h,
              title="POS (top bar=L, bottom=R; red border=BL)",
              loc="lower right", ncol=max(1, N_POS // 4),
              fontsize=6, title_fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


tree_fig_path = os.path.join(OUT_DIR, "content_tree_labels.png")
plot_pair_tree_idx(
    layout_children, layout_depth, cL_map, cR_map,
    title=(f"Content tree (TopK-Disc-Cnt1) — Pair POS distributions  "
           f"(red border = basic-level node, eval_alpha={EVAL_ALPHA})"),
    out_path=tree_fig_path,
    max_depth=TREE_DEPTH_FOR_LABEL_FIG,
    highlight_idx=highlight_idx,
)
print(f"  Content tree figure saved → {tree_fig_path}")


# ── Mean expected_pmi by depth (content tree) ────────────────────────────────
print(f"Computing expected_pmi(use_root=True, eval_alpha={EVAL_ALPHA}) for every content node …")
all_nodes_full = []
stack = [content_tree.root]
while stack:
    n = stack.pop()
    all_nodes_full.append(n)
    stack.extend(n.children)
print(f"  {len(all_nodes_full)} nodes total")

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
            color="red", fontsize=7, ha="center", va="bottom")
ax.set_xlabel("Tree depth (root = 0)", fontsize=11)
ax.set_ylabel(f"Mean expected_pmi (use_root=True, eval_alpha={EVAL_ALPHA})",
              fontsize=11)
ax.set_title("Content tree — mean empirical PMI against root by depth  "
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


# ── Summary ──────────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "method_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 64 + "\n")
    f.write(" Grammar chunking basic-level summary — TopK-Disc-Cnt1\n")
    f.write(" Method: leaf.get_basic(use_root=True, eval_alpha={:.1f})\n".format(EVAL_ALPHA))
    f.write("=" * 64 + "\n\n")
    f.write(f"Settings:\n")
    f.write(f"  N_SENTENCES = {N_SENTENCES}\n")
    f.write(f"  WINDOW      = {WINDOW}\n")
    f.write(f"  TOP_K       = {TOP_K}\n")
    f.write(f"  TOPK_DEPTH  = {topk_depth} (clamped from {TOPK_DEPTH})\n")
    f.write(f"  ALPHA_CONTEXT = {ALPHA_CONTEXT}\n")
    f.write(f"  ALPHA_CONTENT = {ALPHA_CONTENT}\n")
    f.write(f"  EVAL_ALPHA  = {EVAL_ALPHA}\n")
    f.write(f"  Train pairs: {len(pair_train)}\n")
    f.write(f"  Test  pairs: {len(pair_test)}\n")
    f.write(f"  Content tree nodes: {len(all_nodes_full)}\n\n")

    f.write(f"{len(bl_members)} unique basic-level nodes:\n")
    for i, m in enumerate(sorted(bl_members.values(),
                                  key=lambda m: len(m["indices"]),
                                  reverse=True)):
        labels = np.array(m["labels"])
        joint  = np.bincount(labels, minlength=N_POS * N_POS)
        top1   = int(joint.argmax())
        dl, dr = split_pair_label(top1)
        f.write(f"  [{i:>2}] depth={m['depth']:>2}  count={int(m['node'].count):>6}  "
                f"members={len(m['indices']):>4}  dom={id2pos[dl]}-{id2pos[dr]}\n")
print(f"  Summary saved → {summary_path}")

print(f"\nDone. All outputs in {OUT_DIR}")
