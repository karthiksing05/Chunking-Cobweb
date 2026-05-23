"""
Grammar Log-Probability Test (met5)
====================================

Sanity-checks that ``log_prob_instance`` on the trained WEBSTER cobweb
trees behaves the way the grammar-distillation hypothesis assumes —
i.e. it actually discriminates chunk classes.

The hypothesis (from Methodology 5.2 in MULTIHIERARCHY.md and the
basic-level findings in ``tests/representations/test_logprob_paths.py``):

  log_prob_instance(I) at a cobweb node N is the log-likelihood of
  observing instance I under the smoothed multinomial stored at N:

      log P(I | N) = Σ_(a,v in I) count(a,v) · log P(a=v | N)
      P(a=v | N)   = (av_count[N][a][v] + α) /
                     (Σ_v' av_count[N][a][v'] + |V|·α)

If the cobweb tree clusters chunks by grammatical category, log-prob
should:

  (P1)  **Class separability.** For a chunk of gold class C, the
        frontier NT whose dominant class is C should rank near the top
        by ``log_prob_instance(chunk_bag)``.

  (P2)  **Monotonicity along path.** Descending root → … → leaf along
        the chunk's categorize path, log-prob should *increase*
        monotonically (smoothly is fine; strict monotonicity is too
        much to ask but the overall trend must trend up).

  (P3)  **Within-class agreement.** Two chunks of the same gold class
        should have highly-correlated log-prob VECTORS over the
        frontier (i.e. they peak at the same NTs).

  (P4)  **Between-class divergence.** Two chunks of different gold
        classes should have *less*-correlated log-prob vectors than
        same-class pairs.

If any of these fail, the cobweb representations aren't class-
discriminative enough to support unsupervised grammar distillation.

Phases & outputs
----------------
0. Train WEBSTER (mirrors grammar_threshold_test / hollow_learn_test).
1. Collect test chunks + gold head-based chunk class.
2. **Heatmap** (rows=chunks sorted by gold class, cols=frontier NTs,
   cell=log-prob): block structure expected. Two heatmaps, one for
   ``fixed_d3``, one for ``basic_level``.
3. **Purity@K**: bar chart of top-K accuracy (the dominant gold class
   of the argmax NT matches the chunk's class).
4. **Path monotonicity**: for a sample of chunks, plot log-prob vs
   depth along the categorize path. Annotated with the basic-level
   ancestor and the climbing-ancestor (the gate used by build()).
5. **Within- vs between-class correlation**: histogram of cosine
   similarity between pairs of chunks' log-prob vectors, split by
   same-class vs different-class.

Artefacts (under ``grammar_logprob_test_output/``):
  logprob_heatmap_{fixed_d3,basic_level}.png
  purity_at_k.png
  path_monotonicity.png
  within_vs_between.png
  logprob_per_chunk_{fixed_d3,basic_level}.csv
  purity_at_k.csv
"""

import os
import sys
import csv
import glob
import json
import math
import random
import shutil
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import (WEBSTER, FiniteParseTree, PrimitiveParseNode,
                      CompositeParseNode)
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR           = os.path.join(_HERE, "grammar_logprob_test_output")
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
CONTEXT_LENGTH    = 3
THRESHOLD         = 30
PRIMITIVES_FIRST  = 200
EVAL_ALPHA        = 10.0
SEED              = 13
N_PATH_VIZ        = 8     # # of chunks to plot path-monotonicity for
N_PAIR_SAMPLE     = 2000  # # of chunk pairs to sample for within/between

random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

POS_LIST     = ["Det", "N", "Adj", "V", "P"]
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
WORD_TO_POS  = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR1[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
LABEL_COLOR = {
    "Det":   "#2ca02c", "N":     "#8c564b", "Adj":   "#1f77b4",
    "V":     "#17becf", "P":     "#7f7f7f", "NP":    "#ff7f0e",
    "AdjP":  "#9467bd", "PP":    "#bcbd22", "VP":    "#e377c2",
    "S":     "#d62728", "OTHER": "#cccccc",
}

# =============================================================================
# PHASE 0 — Train WEBSTER
# =============================================================================
print("=== PHASE 0: Train WEBSTER ===")
webster = WEBSTER(
    TEST_CORPUS1,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-4,
    context_alpha=1e-4,
    content_bl_alpha=10,
    context_bl_alpha=10,
    bow=False,
    empty_weighting=True,
    chunk_context=False,
    weighting="binary",
    categorization_mode="dfs",
    depth_max_content=1000,
    depth_max_context=1000,
    branch_max_content=1000,
    branch_max_context=1000,
    content_top_k=7,
    content_pool_depth=4,
)

print(f"  Phase 0a: {PRIMITIVES_FIRST} primitive-only sentences")
for i in range(PRIMITIVES_FIRST):
    s = generate("S", TEST_GRAMMAR1)
    webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                           learning=True, debug=False)
    if (i + 1) % 50 == 0: print(f"    [{i+1}/{PRIMITIVES_FIRST}]")

print(f"  Phase 0b: hollow corpus replay")
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try: data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
train_hollow = hollow_corpus[:_split]
test_hollow  = hollow_corpus[_split:]
print(f"  Loaded {len(hollow_corpus)} hollow trees · "
      f"train={len(train_hollow)}  test={len(test_hollow)}")
for i, hollow in enumerate(train_hollow):
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(hollow["sentence"], threshold=THRESHOLD)
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)

cnt_root = webster.ltm.content_hierarchy.root


# =============================================================================
# Shared helpers
# =============================================================================
def _walk(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for c in n.children:
            stack.append(c)

def _chunk_span(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            out.append(int(n.position_idx)); return
        for _, c in getattr(n, "children", []):
            w(c)
    w(node)
    if not out: return None, None
    return min(out), max(out)

def _chunk_yield(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is None or wid < 0 or wid >= len(webster.ltm.id_to_value):
                return
            pos = WORD_TO_POS.get(webster.ltm.id_to_value[wid])
            if pos: out.append(pos)
            return
        for _, c in sorted(getattr(n, "children", []),
                           key=lambda x: x[0] if x[0] is not None else 0):
            w(c)
    w(node)
    return out

def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode): return
    if not getattr(node, "is_global_root", False):
        yield node
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)

def classify_chunk(node, sentence_len):
    pos_seq = _chunk_yield(node)
    if not pos_seq: return None
    if len(pos_seq) == 1: return pos_seq[0]
    s, e = _chunk_span(node)
    if s == 0 and e == sentence_len - 1: return "S"
    if "V" in pos_seq:                                  return "VP"
    if pos_seq[0] == "P":                               return "PP"
    if all(p == "Adj" for p in pos_seq):                return "AdjP"
    if pos_seq[0] == "Adj" and "N" in pos_seq:          return "AdjP"
    if "N" in pos_seq or pos_seq[0] == "Det":           return "NP"
    return "OTHER"

def fixed_depth_frontier(root, target_depth, min_count=0):
    seen = {}
    def walk(node, d):
        if (d == target_depth) or (not node.children):
            cur = node
            while (min_count > 0 and cur.count < min_count
                   and getattr(cur, "parent", None) is not None):
                cur = cur.parent
            seen[str(cur.concept_hash())] = cur
            return
        for c in node.children:
            walk(c, d + 1)
    walk(root, 0)
    return list(seen.values())

def basic_level_frontier(root, eval_alpha=EVAL_ALPHA):
    seen = {}
    for node in _walk(root):
        if node.children: continue
        bl = node.get_basic(0, 0, debug=False,
                            eval_alpha=eval_alpha, use_root=True)
        seen[str(bl.concept_hash())] = bl
    return list(seen.values())

def _categorize_path(bag, root):
    """DFS root → leaf along the argmax-log_prob path. Returns the
    list of nodes including root and leaf."""
    node = root
    path = [node]
    while node.children:
        node = max(node.children, key=lambda c: c.log_prob_instance(bag))
        path.append(node)
    return path


# =============================================================================
# PHASE 1 — Collect test chunks (with gold classes for evaluation only)
# =============================================================================
# Use the train fold's gold merges to harvest a labeled chunk inventory
# (the gold labels are used ONLY to measure log-prob class purity, never
# to compute the log-probs themselves).
print("\n=== PHASE 1: Collect labeled test chunks ===")
chunks = []   # [{bag, gold_class, tokens, sentence}]
for hollow in test_hollow:
    sent = hollow["sentence"]; sent_len = len(sent.split())
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sent, threshold="converge")
    for m in hollow["merges"]:
        try: tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    for comp in _walk_composites(tree.global_root_node):
        ci = comp.get_content_instance()
        if not ci: continue
        cls = classify_chunk(comp, sent_len)
        if cls is None: continue
        s, e = _chunk_span(comp)
        toks = sent.split()[s:e+1]
        chunks.append({"bag": ci, "gold": cls, "tokens": " ".join(toks),
                       "sentence": sent})
print(f"  Collected {len(chunks)} labeled chunks "
      f"({dict(Counter(c['gold'] for c in chunks))})")


# =============================================================================
# PHASE 2 — Compute log-prob heatmap (chunks × frontier NTs)
# =============================================================================
print("\n=== PHASE 2: log_prob_instance(chunk × frontier_NT) ===")

FRONTIERS = {
    "fixed_d3":           fixed_depth_frontier(cnt_root, 3),
    "fixed_d3_pruned":    fixed_depth_frontier(cnt_root, 3, min_count=15),
    "basic_level":        basic_level_frontier(cnt_root),
}
print(f"  Frontiers:")
for k, v in FRONTIERS.items():
    print(f"    {k:<22} size={len(v):>3}  "
          f"counts (min/med/max) = "
          f"{min(int(n.count) for n in v)}/"
          f"{int(np.median([int(n.count) for n in v]))}/"
          f"{max(int(n.count) for n in v)}")

def label_nts(frontier, chunks):
    """For diagnostic purposes only: assign each NT its dominant gold
    class based on argmax log-prob assignments of train chunks. The
    labels are used to MEASURE log-prob purity but do not participate
    in the log-prob computation."""
    counts_per_nt = defaultdict(Counter)
    for c in chunks:
        bag = c["bag"]
        # Sort by log-prob only (cobweb nodes aren't comparable directly).
        scored = [(f.log_prob_instance(bag), idx)
                  for idx, f in enumerate(frontier)]
        scored.sort(key=lambda kv: -kv[0])
        winner = frontier[scored[0][1]]
        counts_per_nt[str(winner.concept_hash())][c["gold"]] += 1
    labels = {}
    seen_class = defaultdict(int)
    # Sort by frequency (most-attractive NT first) so the suffix
    # numbering is deterministic.
    nts_sorted = sorted(frontier, key=lambda n: -int(n.count))
    for n in nts_sorted:
        h = str(n.concept_hash())
        c = counts_per_nt.get(h, Counter()).most_common(1)
        dom = c[0][0] if c else "?"
        seen_class[dom] += 1
        labels[h] = f"NT_{dom}_{chr(ord('a') + seen_class[dom] - 1)}"
    return labels


def compute_heatmap(frontier, chunks):
    """Build the log-prob matrix: rows = chunks, cols = frontier NTs."""
    n_c = len(chunks); n_n = len(frontier)
    M = np.zeros((n_c, n_n))
    for i, c in enumerate(chunks):
        bag = c["bag"]
        for j, nt in enumerate(frontier):
            M[i, j] = nt.log_prob_instance(bag)
    return M


def plot_heatmap(M, chunks, frontier, frontier_labels, frontier_name,
                 out_path):
    """Rows sorted by gold class so block structure (if any) is
    visible. Annotate each column with its NT label + dominant
    gold class color."""
    n_c, n_n = M.shape
    # Sort rows by gold class for block structure.
    order_r = sorted(range(n_c),
                     key=lambda i: (chunks[i]["gold"], chunks[i]["tokens"]))
    Ms = M[order_r, :]
    chunks_s = [chunks[i] for i in order_r]

    # Z-normalize rows so colors are comparable across rows.
    row_mean = Ms.mean(axis=1, keepdims=True)
    row_std  = Ms.std(axis=1, keepdims=True) + 1e-12
    Mz = (Ms - row_mean) / row_std

    fig_w = max(8, 0.4 * n_n + 4)
    fig_h = max(6, 0.04 * n_c + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Mz, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("z-scored log_prob_instance (row-normalized)", fontsize=9)

    # Y-tick labels: gold class only at class boundaries.
    cur, last = None, -1
    for i, c in enumerate(chunks_s):
        if c["gold"] != cur:
            cur = c["gold"]
            ax.axhline(i - 0.5, color="black", linewidth=0.4, alpha=0.4)
            ax.text(-1.5, i + 1, c["gold"], ha="right", va="center",
                    fontsize=9, fontweight="bold",
                    color=LABEL_COLOR.get(c["gold"], "#444"))
    ax.set_yticks([])
    ax.set_ylabel(f"{n_c} test chunks (grouped by gold class)", fontsize=10)

    # X-tick labels: NT_i_class, colored by dominant class.
    nt_labels = [frontier_labels.get(str(n.concept_hash()), "?")
                 for n in frontier]
    ax.set_xticks(range(n_n))
    ax.set_xticklabels(nt_labels, rotation=45, ha="right", fontsize=8)
    for i, lab in enumerate(nt_labels):
        if "_" in lab:
            cls = lab.split("_")[1]
            ax.get_xticklabels()[i].set_color(LABEL_COLOR.get(cls, "#444"))

    ax.set_title(f"log-prob heatmap — {frontier_name} frontier  "
                 f"(rows sorted by gold class, "
                 f"row-z-scored)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


labels_per_frontier = {}
M_per_frontier      = {}
for f_name, frontier in FRONTIERS.items():
    if not frontier: continue
    labels = label_nts(frontier, chunks)
    labels_per_frontier[f_name] = labels
    M = compute_heatmap(frontier, chunks)
    M_per_frontier[f_name] = M
    plot_heatmap(M, chunks, frontier, labels, f_name,
                 os.path.join(OUT_DIR, f"logprob_heatmap_{f_name}.png"))
    print(f"  Heatmap (n_chunks={len(chunks)} × n_NT={len(frontier)})"
          f" → logprob_heatmap_{f_name}.png")

    # Persist the raw matrix.
    csv_path = os.path.join(OUT_DIR,
                            f"logprob_per_chunk_{f_name}.csv")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        nt_cols = [labels.get(str(n.concept_hash()), f"NT_{i}")
                   for i, n in enumerate(frontier)]
        w.writerow(["sentence", "tokens", "gold"] + nt_cols)
        for c, row in zip(chunks, M):
            w.writerow([c["sentence"], c["tokens"], c["gold"]]
                       + [f"{x:.4f}" for x in row])


# =============================================================================
# PHASE 3 — Purity @ K (does argmax NT have the chunk's class?)
# =============================================================================
print("\n=== PHASE 3: Purity@K ===")

def nt_dominant_class(label):
    """NT_<class>_<suffix> → <class>."""
    parts = label.split("_")
    return parts[1] if len(parts) >= 3 else "?"

def purity_at_k(M, chunks, frontier, labels, K_list=(1, 2, 3)):
    rows = []
    for K in K_list:
        n_hit = 0
        for i, c in enumerate(chunks):
            # Top-K NT indices for this chunk.
            top = np.argsort(-M[i])[:K]
            top_classes = [nt_dominant_class(
                              labels.get(str(frontier[j].concept_hash()), "?"))
                           for j in top]
            if c["gold"] in top_classes:
                n_hit += 1
        rows.append((K, n_hit, n_hit / max(len(chunks), 1)))
    return rows

purity_rows = []
print(f"  {'frontier':<22} {'P@1':>6} {'P@2':>6} {'P@3':>6}")
for f_name, M in M_per_frontier.items():
    rows = purity_at_k(M, chunks, FRONTIERS[f_name],
                       labels_per_frontier[f_name])
    purity_rows.append((f_name, rows))
    print(f"  {f_name:<22} "
          f"{100*rows[0][2]:>5.1f}%  "
          f"{100*rows[1][2]:>5.1f}%  "
          f"{100*rows[2][2]:>5.1f}%")

# Bar chart.
fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(purity_rows))
width = 0.27
for i, K in enumerate([1, 2, 3]):
    vals = [r[i][2] for _, r in purity_rows]
    ax.bar(xs + (i - 1) * width, vals, width,
           label=f"P@{K}",
           color=["#1f77b4", "#2ca02c", "#d62728"][i],
           edgecolor="black", linewidth=0.5)
    for k, v in enumerate(vals):
        ax.text(xs[k] + (i - 1) * width, v + 0.015,
                f"{100*v:.0f}", ha="center", fontsize=7)
ax.axhline(1/5, color="#888", linestyle=":", linewidth=0.7,
           label=f"chance (5 classes)")
ax.set_xticks(xs)
ax.set_xticklabels([n for n, _ in purity_rows], rotation=15, ha="right")
ax.set_ylim(0, 1.1); ax.set_ylabel("Purity")
ax.set_title("Log-prob purity@K — argmax frontier NT has the chunk's gold class")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "purity_at_k.png"),
            dpi=140, bbox_inches="tight")
plt.close()
print(f"  Bar chart → purity_at_k.png")

with open(os.path.join(OUT_DIR, "purity_at_k.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["frontier", "K", "hits", "rate"])
    for f_name, rows in purity_rows:
        for K, hits, rate in rows:
            w.writerow([f_name, K, hits, f"{rate:.4f}"])


# =============================================================================
# PHASE 4 — Path monotonicity (root → leaf)
# =============================================================================
# For each sampled chunk, walk root → leaf along the argmax-log_prob
# path in the content tree, recording log_prob_instance at every
# node. Per the hypothesis, log-prob should grow as we descend toward
# the leaf that BEST matches the chunk.
# =============================================================================
print("\n=== PHASE 4: Path monotonicity (root → leaf) ===")
fig, axes = plt.subplots(2, N_PATH_VIZ // 2,
                          figsize=(2.5 * (N_PATH_VIZ // 2), 8),
                          sharey=False)
axes = axes.ravel()

# Sample a variety of chunks across gold classes.
by_class = defaultdict(list)
for c in chunks: by_class[c["gold"]].append(c)
sample = []
classes_cycle = list(by_class.keys())
random.shuffle(classes_cycle)
for cls in classes_cycle:
    if by_class[cls]:
        sample.append(random.choice(by_class[cls]))
    if len(sample) >= N_PATH_VIZ:
        break
while len(sample) < N_PATH_VIZ:
    sample.append(random.choice(chunks))

monotonic_count = 0
for ax, c in zip(axes, sample[:N_PATH_VIZ]):
    bag = c["bag"]
    path = _categorize_path(bag, cnt_root)
    lps  = [n.log_prob_instance(bag) for n in path]
    ds   = [n.depth() for n in path]
    cls_color = LABEL_COLOR.get(c["gold"], "#444")
    ax.plot(ds, lps, marker="o", color=cls_color, linewidth=2)
    # Annotate basic-level node on the path.
    leaf = path[-1]
    try:
        bl = leaf.get_basic(0, 0, debug=False,
                            eval_alpha=EVAL_ALPHA, use_root=True)
        for n, d, lp in zip(path, ds, lps):
            if str(n.concept_hash()) == str(bl.concept_hash()):
                ax.plot(d, lp, marker="*", color="red", markersize=14,
                        zorder=5)
                ax.annotate("basic-level", (d, lp), xytext=(5, -15),
                            textcoords="offset points", fontsize=7,
                            color="red")
                break
    except Exception:
        pass

    # Check monotonicity (do log-probs trend upward?)
    is_mono = lps[-1] > lps[0]
    if is_mono: monotonic_count += 1
    ax.axhline(lps[0], color="#bbb", linestyle=":", linewidth=0.6)
    ax.set_xlabel("depth")
    ax.set_ylabel("log p")
    ax.set_title(f"{c['gold']}: \"{c['tokens'][:30]}\"  "
                 f"{'↑' if is_mono else '↓'}",
                 fontsize=9, color=cls_color)
    ax.grid(alpha=0.3)

plt.suptitle(
    f"Path monotonicity — log_prob_instance(chunk) vs depth  "
    f"(root → leaf along argmax path); "
    f"{monotonic_count}/{N_PATH_VIZ} chunks trend upward",
    fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, "path_monotonicity.png"),
            dpi=140, bbox_inches="tight")
plt.close()
print(f"  {monotonic_count}/{N_PATH_VIZ} sampled chunks show upward "
      f"log-prob trend root→leaf")
print(f"  Plot → path_monotonicity.png")


# =============================================================================
# PHASE 5 — Within-class vs between-class log-prob correlation
# =============================================================================
# For each pair of chunks (i, j), compute cosine similarity of their
# log-prob vectors over the frontier. Compare the distribution for
# same-gold-class pairs vs different-class pairs. If log-probs are
# class-discriminative, the same-class distribution should be shifted
# higher.
# =============================================================================
print("\n=== PHASE 5: Within-class vs between-class log-prob correlation ===")

def cosine(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
mean_within_per_frontier  = {}
mean_between_per_frontier = {}
for ax, (f_name, M) in zip(axes, M_per_frontier.items()):
    n = M.shape[0]
    pairs_idx = [(random.randrange(n), random.randrange(n))
                 for _ in range(N_PAIR_SAMPLE)]
    pairs_idx = [(i, j) for i, j in pairs_idx if i != j]
    within, between = [], []
    for i, j in pairs_idx:
        sim = cosine(M[i], M[j])
        if chunks[i]["gold"] == chunks[j]["gold"]:
            within.append(sim)
        else:
            between.append(sim)
    mean_within_per_frontier[f_name]  = np.mean(within) if within else float("nan")
    mean_between_per_frontier[f_name] = np.mean(between) if between else float("nan")

    bins = np.linspace(min(within + between), 1.0, 40)
    ax.hist(within,  bins=bins, alpha=0.6, color="#2ca02c",
            edgecolor="black", linewidth=0.3,
            label=f"within ({len(within)})  mean={np.mean(within):.3f}")
    ax.hist(between, bins=bins, alpha=0.6, color="#d62728",
            edgecolor="black", linewidth=0.3,
            label=f"between ({len(between)})  mean={np.mean(between):.3f}")
    ax.axvline(np.mean(within),  color="#2ca02c", linestyle="--", lw=1.5)
    ax.axvline(np.mean(between), color="#d62728", linestyle="--", lw=1.5)
    sep = np.mean(within) - np.mean(between)
    ax.set_title(f"{f_name}\nseparation = {sep:.3f}",
                 fontsize=11)
    ax.set_xlabel("cosine similarity of log-prob vectors")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("# pairs")
plt.suptitle(
    "Within-class vs between-class log-prob vector correlation  "
    "(higher within-mean = more class-discriminative log-probs)",
    fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, "within_vs_between.png"),
            dpi=140, bbox_inches="tight")
plt.close()

print(f"  {'frontier':<22} {'mean within':>12} {'mean between':>14} "
      f"{'separation':>12}")
for f_name in M_per_frontier:
    w = mean_within_per_frontier[f_name]
    b = mean_between_per_frontier[f_name]
    print(f"  {f_name:<22} {w:>12.3f} {b:>14.3f} {w - b:>12.3f}")
print(f"  Histogram → within_vs_between.png")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY — does log_prob_instance encode chunk class?")
print("=" * 70)
print("  Hypothesis: cobweb log-probs should be class-discriminative.")
print()
for f_name in M_per_frontier:
    p1 = next(r[2] for K, h, r in purity_rows
              for K2, h2, r2 in [r] if K == 1) \
         if False else None
    # Pull P@1 cleanly.
    for n, rows in purity_rows:
        if n == f_name:
            p1_val = rows[0][2]
            break
    sep = (mean_within_per_frontier[f_name]
           - mean_between_per_frontier[f_name])
    print(f"  [{f_name}]")
    print(f"      Purity@1         : {100*p1_val:.1f}%  (chance = 20%)")
    print(f"      Within-mean cos  : {mean_within_per_frontier[f_name]:.3f}")
    print(f"      Between-mean cos : {mean_between_per_frontier[f_name]:.3f}")
    print(f"      Separation       : {sep:.3f}  "
          f"({'YES' if sep > 0.05 else 'WEAK'} class-discriminative)")
print(f"\nArtefacts in {OUT_DIR}/:")
print("  logprob_heatmap_{fixed_d3,fixed_d3_pruned,basic_level}.png")
print("  purity_at_k.png  ·  purity_at_k.csv")
print("  path_monotonicity.png")
print("  within_vs_between.png")
print("  logprob_per_chunk_*.csv")
