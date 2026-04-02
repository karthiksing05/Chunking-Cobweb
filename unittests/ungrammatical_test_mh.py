"""
Ungrammatical-sentence testing script for a saved WEBSTER LTM.

This script loads a previously saved LTM and evaluates how WEBSTER handles
sentences that violate the grammar used during training. It tests various
types of ungrammaticality (word-order violations, category violations,
missing constituents, duplicated constituents, extra words, nonsense
sequences) and compares parsing behaviour against grammatical baselines.
Results are written to `unittests/ungrammatical_test_mh`.
"""

import os
import shutil
import random
import math

from parse_mh import WEBSTER, PrimitiveParseNode, CompositeParseNode
from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1

LOAD_LTM_DIR = "unittests/gen_learn_test_mh/final_ltm_data"
OUT_DIR = "unittests/ungrammatical_test_mh"

if not os.path.exists(LOAD_LTM_DIR):
    print(f"Saved LTM not found at {LOAD_LTM_DIR}. Aborting ungrammatical test run.")
    raise SystemExit(1)

# reset output directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

webster = WEBSTER.load_state(LOAD_LTM_DIR)

THRESHOLD = 30

random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary pools drawn from the grammar
# ---------------------------------------------------------------------------
DETS = ["the", "a"]
NOUNS = ["cat", "dog", "man", "woman", "park", "telescope"]
ADJS = ["big", "small", "red", "quick", "lazy"]
VERBS = ["saw", "liked", "chased", "found", "admired"]
PREPS = ["with", "in", "on", "under"]
ALL_WORDS = DETS + NOUNS + ADJS + VERBS + PREPS

# ---------------------------------------------------------------------------
# Helper: annotate parse tree nodes with content/context log probabilities
# ---------------------------------------------------------------------------
def _annotate_tree_with_scores(parse_tree, webster):
    """Walk all nodes in a parse tree and compute content/context log probs.

    Sets ``node._content_log_prob``, ``node._context_log_prob``,
    ``node._content_basic_count``, and ``node._context_basic_count`` on each
    node so the visualizer can display them.  Returns a list of per-node
    score dicts for aggregation.
    """
    ltm = webster.ltm
    cnt_bl_alpha = getattr(ltm, 'content_bl_alpha', None)
    ctx_bl_alpha = getattr(ltm, 'context_bl_alpha', None)
    scores = []

    def _basic_count(tree, instance, bl_alpha):
        """Categorize instance into tree, find basic-level node, return its count."""
        try:
            leaf = tree.categorize(instance)
            _alpha = bl_alpha if bl_alpha is not None else -1.0
            basic = leaf.get_basic(200, 100, eval_alpha=_alpha)
            c = basic.count
            return c if not math.isnan(c) else None
        except Exception:
            return None

    def walk(node):
        if isinstance(node, PrimitiveParseNode):
            ctx_inst = node.get_context_instance()
            ctx_lp = None
            ctx_bc = None
            if ctx_inst:
                try:
                    ctx_lp = float(ltm.context_hierarchy.log_prob(ctx_inst, 200, False))
                    if math.isnan(ctx_lp):
                        ctx_lp = None
                except Exception:
                    ctx_lp = None
                ctx_bc = _basic_count(ltm.context_hierarchy, ctx_inst, ctx_bl_alpha)
            node._content_log_prob = None
            node._context_log_prob = ctx_lp
            node._content_basic_count = None
            node._context_basic_count = ctx_bc
            scores.append({"content_lp": None, "context_lp": ctx_lp,
                           "content_bc": None, "context_bc": ctx_bc})

        elif isinstance(node, CompositeParseNode) and not node.is_global_root:
            cnt_inst = node.get_content_instance()
            ctx_inst = node.get_context_instance()
            cnt_lp = None
            ctx_lp = None
            cnt_bc = None
            ctx_bc = None
            if cnt_inst:
                try:
                    cnt_lp = float(ltm.content_hierarchy.log_prob(cnt_inst, 200, False))
                    if math.isnan(cnt_lp):
                        cnt_lp = None
                except Exception:
                    cnt_lp = None
                cnt_bc = _basic_count(ltm.content_hierarchy, cnt_inst, cnt_bl_alpha)
            if ctx_inst:
                try:
                    ctx_lp = float(ltm.context_hierarchy.log_prob(ctx_inst, 200, False))
                    if math.isnan(ctx_lp):
                        ctx_lp = None
                except Exception:
                    ctx_lp = None
                ctx_bc = _basic_count(ltm.context_hierarchy, ctx_inst, ctx_bl_alpha)
            node._content_log_prob = cnt_lp
            node._context_log_prob = ctx_lp
            node._content_basic_count = cnt_bc
            node._context_basic_count = ctx_bc
            scores.append({"content_lp": cnt_lp, "context_lp": ctx_lp,
                           "content_bc": cnt_bc, "context_bc": ctx_bc})

        for _, ch in getattr(node, "children", []):
            walk(ch)

    for _, ch in parse_tree.global_root_node.children:
        walk(ch)

    return scores


def _avg(values):
    """Average of non-None values, or None if empty."""
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _fmt(val):
    """Format a float or None for display."""
    return f"{val:.4f}" if val is not None else "N/A"


# ---------------------------------------------------------------------------
# Helper: parse a sentence, annotate scores, print result, visualize
# ---------------------------------------------------------------------------
def _try_parse(webster, sentence, label, idx, viz_dir):
    try:
        parse_tree = webster.parse_sentence(
            sentence,
            threshold=THRESHOLD,
            new_vocab=True,
            learning=False,
            debug=False,
        )
        # Annotate nodes with content/context log probs
        node_scores = _annotate_tree_with_scores(parse_tree, webster)

        print(f"  [{idx}] {label}: \"{sentence}\"")
        if parse_tree and hasattr(parse_tree, "visualize"):
            parse_tree.visualize(os.path.join(viz_dir, f"{label}_{idx}"))
        return parse_tree, node_scores
    except Exception as e:
        print(f"  [{idx}] FAILED {label}: \"{sentence}\" — {e}")
        return None, []


def _print_category_summary(category_name, all_scores):
    """Print average content/context log probs and basic-level counts for a category."""
    all_cnt = [s["content_lp"] for scores in all_scores for s in scores]
    all_ctx = [s["context_lp"] for scores in all_scores for s in scores]
    all_cnt_bc = [s["content_bc"] for scores in all_scores for s in scores]
    all_ctx_bc = [s["context_bc"] for scores in all_scores for s in scores]
    avg_cnt = _avg(all_cnt)
    avg_ctx = _avg(all_ctx)
    avg_cnt_bc = _avg(all_cnt_bc)
    avg_ctx_bc = _avg(all_ctx_bc)
    n_chunks = len([v for v in all_cnt if v is not None])
    n_prims = len([v for v in all_ctx if v is not None]) - n_chunks
    print(f"  >> {category_name} AVERAGES: "
          f"content_lp={_fmt(avg_cnt)} ({n_chunks} chunks), "
          f"context_lp={_fmt(avg_ctx)} ({n_prims} prims + {n_chunks} chunks), "
          f"content_bc={_fmt(avg_cnt_bc)}, context_bc={_fmt(avg_ctx_bc)}")
    return {"category": category_name, "avg_content_lp": avg_cnt,
            "avg_context_lp": avg_ctx, "avg_content_bc": avg_cnt_bc,
            "avg_context_bc": avg_ctx_bc, "n_chunks": n_chunks,
            "n_context": len([v for v in all_ctx if v is not None])}


# Track per-category summary data
category_summaries = []
category_raw_data = []  # parallel list of (name, all_scores)

# ===== GRAMMATICAL BASELINE =================================================
print("\n--- GRAMMATICAL BASELINE ---")
baseline_dir = os.path.join(OUT_DIR, "baseline_grammatical")
os.makedirs(baseline_dir, exist_ok=True)

baseline_sentences = [generate("S", TEST_GRAMMAR1) for _ in range(10)]
baseline_all_scores = []
for i, sent in enumerate(baseline_sentences):
    _, scores = _try_parse(webster, sent, "baseline", i, baseline_dir)
    baseline_all_scores.append(scores)
category_summaries.append(_print_category_summary("Grammatical Baseline", baseline_all_scores))
category_raw_data.append(("Grammatical Baseline", baseline_all_scores))


# ===== 1. WORD-ORDER VIOLATIONS =============================================
# Swap two adjacent words in an otherwise grammatical sentence.
print("\n--- WORD-ORDER VIOLATIONS (adjacent swap) ---")
swap_dir = os.path.join(OUT_DIR, "word_order_swap")
os.makedirs(swap_dir, exist_ok=True)

swap_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    if len(tokens) >= 2:
        idx = random.randint(0, len(tokens) - 2)
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
    swap_sentences.append(" ".join(tokens))

swap_all_scores = []
for i, sent in enumerate(swap_sentences):
    _, scores = _try_parse(webster, sent, "swap", i, swap_dir)
    swap_all_scores.append(scores)
category_summaries.append(_print_category_summary("Word-Order Swap", swap_all_scores))
category_raw_data.append(("Word-Order Swap", swap_all_scores))


# ===== 2. REVERSED SENTENCES ================================================
print("\n--- REVERSED SENTENCES ---")
rev_dir = os.path.join(OUT_DIR, "reversed")
os.makedirs(rev_dir, exist_ok=True)

reversed_sentences = [" ".join(generate("S", TEST_GRAMMAR1).split()[::-1]) for _ in range(10)]
rev_all_scores = []
for i, sent in enumerate(reversed_sentences):
    _, scores = _try_parse(webster, sent, "reversed", i, rev_dir)
    rev_all_scores.append(scores)
category_summaries.append(_print_category_summary("Reversed", rev_all_scores))
category_raw_data.append(("Reversed", rev_all_scores))


# ===== 3. CATEGORY VIOLATIONS ===============================================
# Replace a word with one from a different syntactic category.
print("\n--- CATEGORY VIOLATIONS ---")
cat_dir = os.path.join(OUT_DIR, "category_violation")
os.makedirs(cat_dir, exist_ok=True)

def _category_of(word):
    if word in DETS:
        return "Det"
    if word in NOUNS:
        return "N"
    if word in ADJS:
        return "Adj"
    if word in VERBS:
        return "V"
    if word in PREPS:
        return "P"
    return None

def _random_from_other_category(word):
    cat = _category_of(word)
    pools = {"Det": DETS, "N": NOUNS, "Adj": ADJS, "V": VERBS, "P": PREPS}
    other_cats = [k for k in pools if k != cat]
    chosen_cat = random.choice(other_cats)
    return random.choice(pools[chosen_cat])

cat_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    if tokens:
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = _random_from_other_category(tokens[idx])
    cat_sentences.append(" ".join(tokens))

cat_all_scores = []
for i, sent in enumerate(cat_sentences):
    _, scores = _try_parse(webster, sent, "cat_viol", i, cat_dir)
    cat_all_scores.append(scores)
category_summaries.append(_print_category_summary("Category Violation", cat_all_scores))
category_raw_data.append(("Category Violation", cat_all_scores))


# ===== 4. MISSING CONSTITUENTS ==============================================
# Drop a random word from a grammatical sentence.
print("\n--- MISSING CONSTITUENTS (word deletion) ---")
del_dir = os.path.join(OUT_DIR, "missing_constituent")
os.makedirs(del_dir, exist_ok=True)

del_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    if len(tokens) > 1:
        drop_idx = random.randint(0, len(tokens) - 1)
        tokens.pop(drop_idx)
    del_sentences.append(" ".join(tokens))

del_all_scores = []
for i, sent in enumerate(del_sentences):
    _, scores = _try_parse(webster, sent, "missing", i, del_dir)
    del_all_scores.append(scores)
category_summaries.append(_print_category_summary("Missing Constituent", del_all_scores))
category_raw_data.append(("Missing Constituent", del_all_scores))


# ===== 5. DUPLICATED CONSTITUENTS ============================================
# Repeat a random word in place.
print("\n--- DUPLICATED CONSTITUENTS ---")
dup_dir = os.path.join(OUT_DIR, "duplicated_constituent")
os.makedirs(dup_dir, exist_ok=True)

dup_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    if tokens:
        dup_idx = random.randint(0, len(tokens) - 1)
        tokens.insert(dup_idx, tokens[dup_idx])
    dup_sentences.append(" ".join(tokens))

dup_all_scores = []
for i, sent in enumerate(dup_sentences):
    _, scores = _try_parse(webster, sent, "duplicated", i, dup_dir)
    dup_all_scores.append(scores)
category_summaries.append(_print_category_summary("Duplicated Constituent", dup_all_scores))
category_raw_data.append(("Duplicated Constituent", dup_all_scores))


# ===== 6. EXTRA INSERTION ====================================================
# Insert a random word at a random position.
print("\n--- EXTRA WORD INSERTION ---")
ins_dir = os.path.join(OUT_DIR, "extra_insertion")
os.makedirs(ins_dir, exist_ok=True)

ins_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    ins_idx = random.randint(0, len(tokens))
    tokens.insert(ins_idx, random.choice(ALL_WORDS))
    ins_sentences.append(" ".join(tokens))

ins_all_scores = []
for i, sent in enumerate(ins_sentences):
    _, scores = _try_parse(webster, sent, "insertion", i, ins_dir)
    ins_all_scores.append(scores)
category_summaries.append(_print_category_summary("Extra Insertion", ins_all_scores))
category_raw_data.append(("Extra Insertion", ins_all_scores))


# ===== 7. PURE RANDOM WORD SEQUENCES ========================================
# Completely random sequences of in-vocabulary words (no structure).
print("\n--- PURE RANDOM WORD SEQUENCES ---")
rand_dir = os.path.join(OUT_DIR, "random_sequences")
os.makedirs(rand_dir, exist_ok=True)

rand_sentences = [
    " ".join(random.choices(ALL_WORDS, k=random.randint(3, 8)))
    for _ in range(10)
]

rand_all_scores = []
for i, sent in enumerate(rand_sentences):
    _, scores = _try_parse(webster, sent, "random", i, rand_dir)
    rand_all_scores.append(scores)
category_summaries.append(_print_category_summary("Random Sequences", rand_all_scores))
category_raw_data.append(("Random Sequences", rand_all_scores))


# ===== 8. AGREEMENT / STRUCTURAL VIOLATIONS =================================
# Hand-crafted sentences that are structurally wrong in specific ways.
print("\n--- HAND-CRAFTED STRUCTURAL VIOLATIONS ---")
struct_dir = os.path.join(OUT_DIR, "structural_violations")
os.makedirs(struct_dir, exist_ok=True)

structural_violations = [
    # Det Det N V  (double determiner)
    "the a dog saw",
    # V N Det  (verb-first, backwards NP)
    "chased cat the",
    # N V N V  (two VPs mashed together)
    "dog saw cat liked",
    # P P P N  (stacked prepositions)
    "with in on dog",
    # Det Adj  (incomplete NP — no noun)
    "the big",
    # V V V  (verb spam)
    "saw liked chased",
    # N N N N  (noun spam)
    "cat dog man woman",
    # Det V Det N  (verb where noun should be)
    "the saw the cat",
    # Adj Adj Adj  (adjectives only)
    "big small red",
    # Single word
    "telescope",
    # Empty-ish: determiner only
    "a",
    # Long run of determiners
    "the the the the the",
    # Prep at start with no NP before it
    "under the cat saw",
    # Double verb phrase
    "the cat saw liked the dog",
]

struct_all_scores = []
for i, sent in enumerate(structural_violations):
    _, scores = _try_parse(webster, sent, "structural", i, struct_dir)
    struct_all_scores.append(scores)
category_summaries.append(_print_category_summary("Structural Violations", struct_all_scores))
category_raw_data.append(("Structural Violations", struct_all_scores))


# ===== 9. OUT-OF-VOCABULARY WORDS ============================================
# Sentences with words not in the training vocabulary.
print("\n--- OUT-OF-VOCABULARY WORDS ---")
oov_dir = os.path.join(OUT_DIR, "out_of_vocabulary")
os.makedirs(oov_dir, exist_ok=True)

oov_words = ["elephant", "quickly", "above", "colorful", "destroyed"]
oov_sentences = []
for _ in range(10):
    tokens = generate("S", TEST_GRAMMAR1).split()
    if tokens:
        replace_idx = random.randint(0, len(tokens) - 1)
        tokens[replace_idx] = random.choice(oov_words)
    oov_sentences.append(" ".join(tokens))

oov_all_scores = []
for i, sent in enumerate(oov_sentences):
    _, scores = _try_parse(webster, sent, "oov", i, oov_dir)
    oov_all_scores.append(scores)
category_summaries.append(_print_category_summary("Out-of-Vocabulary", oov_all_scores))
category_raw_data.append(("Out-of-Vocabulary", oov_all_scores))


# ===== 10. MASKED COMPLETION OF UNGRAMMATICAL PREFIXES =======================
# Give WEBSTER an ungrammatical prefix + [mask] and see what it generates.
print("\n--- MASKED COMPLETION OF UNGRAMMATICAL PREFIXES ---")
mask_ungram_dir = os.path.join(OUT_DIR, "masked_ungrammatical")
os.makedirs(mask_ungram_dir, exist_ok=True)

ungrammatical_prefixes = [
    "saw the [mask]",
    "the the [mask]",
    "big big big [mask]",
    "dog cat [mask]",
    "under under [mask]",
    "a a a [mask]",
    "chased liked [mask]",
    "the [mask]",
    "saw saw [mask]",
    "in on under [mask]",
]

mask_results_path = os.path.join(OUT_DIR, "masked_ungrammatical_results.txt")
with open(mask_results_path, "w", encoding="utf-8") as mf:
    for i, masked in enumerate(ungrammatical_prefixes):
        print(f"  Masked input: \"{masked}\"")
        mf.write(f"[{i}] masked: {masked}\n")
        try:
            completed, parse = webster.generate_sentence(masked_sentence=masked, debug=False)
            print(f"  Completed:    \"{completed}\"\n")
            mf.write(f"    completed: {completed}\n\n")
            if parse and hasattr(parse, "visualize"):
                # Annotate the completed parse tree with scores too
                _annotate_tree_with_scores(parse, webster)
                parse.visualize(os.path.join(mask_ungram_dir, f"mask_ungram_{i}"))
        except Exception as e:
            print(f"  FAILED: {e}\n")
            mf.write(f"    FAILED: {e}\n\n")
print(f"Saved masked ungrammatical results to \"{mask_results_path}\"")


# ===== SUMMARY ===============================================================
print(f"\n{'='*100}")
print("CATEGORY SUMMARY — Average Log Probabilities & Basic-Level Counts")
print(f"{'='*100}")
print(f"{'Category':<30} {'Avg CntLP':>12} {'Avg CtxLP':>12} {'Avg CntBC':>12} {'Avg CtxBC':>12} {'#Chunks':>10} {'#Context':>10}")
print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
for s in category_summaries:
    print(f"{s['category']:<30} {_fmt(s['avg_content_lp']):>12} {_fmt(s['avg_context_lp']):>12} {_fmt(s['avg_content_bc']):>12} {_fmt(s['avg_context_bc']):>12} {s['n_chunks']:>10} {s['n_context']:>10}")
print(f"{'='*100}")

# Write summary to file
summary_path = os.path.join(OUT_DIR, "category_summary.txt")
with open(summary_path, "w", encoding="utf-8") as sf:
    sf.write("CATEGORY SUMMARY — Average Log Probabilities & Basic-Level Counts\n")
    sf.write(f"{'='*100}\n")
    sf.write(f"{'Category':<30} {'Avg CntLP':>12} {'Avg CtxLP':>12} {'Avg CntBC':>12} {'Avg CtxBC':>12} {'#Chunks':>10} {'#Context':>10}\n")
    sf.write(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}\n")
    for s in category_summaries:
        sf.write(f"{s['category']:<30} {_fmt(s['avg_content_lp']):>12} {_fmt(s['avg_context_lp']):>12} {_fmt(s['avg_content_bc']):>12} {_fmt(s['avg_context_bc']):>12} {s['n_chunks']:>10} {s['n_context']:>10}\n")
    sf.write(f"{'='*100}\n")
print(f"Summary saved to '{summary_path}'")

# Write summary to CSV
import csv
csv_path = os.path.join(OUT_DIR, "category_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["Category", "Avg CntLP", "Avg CtxLP", "Avg CntBC", "Avg CtxBC", "#Chunks", "#Context"])
    for s in category_summaries:
        writer.writerow([
            s["category"],
            _fmt(s["avg_content_lp"]),
            _fmt(s["avg_context_lp"]),
            _fmt(s["avg_content_bc"]),
            _fmt(s["avg_context_bc"]),
            s["n_chunks"],
            s["n_context"],
        ])
print(f"CSV saved to '{csv_path}'")

# ===== HISTOGRAMS =============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

hist_dir = os.path.join(OUT_DIR, "histograms")
os.makedirs(hist_dir, exist_ok=True)

def _extract(all_scores, key):
    """Flatten per-node values for *key*, dropping Nones."""
    return [s[key] for scores in all_scores for s in scores if s[key] is not None]

# Gather all values across categories to compute shared axis limits
all_content_lps, all_context_lps = [], []
all_content_bcs, all_context_bcs = [], []
for _, raw in category_raw_data:
    all_content_lps.extend(_extract(raw, "content_lp"))
    all_context_lps.extend(_extract(raw, "context_lp"))
    all_content_bcs.extend(_extract(raw, "content_bc"))
    all_context_bcs.extend(_extract(raw, "context_bc"))

# Shared bin edges so histograms are visually comparable
def _bins(values, n=20):
    if not values:
        return np.linspace(0, 1, n + 1)
    lo, hi = min(values), max(values)
    margin = (hi - lo) * 0.05 if hi > lo else 1.0
    return np.linspace(lo - margin, hi + margin, n + 1)

cnt_lp_bins = _bins(all_content_lps)
ctx_lp_bins = _bins(all_context_lps)
cnt_bc_bins = _bins(all_content_bcs)
ctx_bc_bins = _bins(all_context_bcs)

for cat_name, raw in category_raw_data:
    cnt_lp = _extract(raw, "content_lp")
    ctx_lp = _extract(raw, "context_lp")
    cnt_bc = _extract(raw, "content_bc")
    ctx_bc = _extract(raw, "context_bc")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(cat_name, fontsize=14, fontweight="bold")

    # Content Log-Prob
    ax = axes[0, 0]
    if cnt_lp:
        ax.hist(cnt_lp, bins=cnt_lp_bins, color="steelblue", edgecolor="white")
    ax.set_title("Content Log-Prob")
    ax.set_xlabel("log-prob")
    ax.set_ylabel("count")
    ax.set_xlim(cnt_lp_bins[0], cnt_lp_bins[-1])

    # Context Log-Prob
    ax = axes[0, 1]
    if ctx_lp:
        ax.hist(ctx_lp, bins=ctx_lp_bins, color="darkorange", edgecolor="white")
    ax.set_title("Context Log-Prob")
    ax.set_xlabel("log-prob")
    ax.set_ylabel("count")
    ax.set_xlim(ctx_lp_bins[0], ctx_lp_bins[-1])

    # Content Basic-Level Count
    ax = axes[1, 0]
    if cnt_bc:
        ax.hist(cnt_bc, bins=cnt_bc_bins, color="seagreen", edgecolor="white")
    ax.set_title("Content Basic-Level Count")
    ax.set_xlabel("count")
    ax.set_ylabel("frequency")
    ax.set_xlim(cnt_bc_bins[0], cnt_bc_bins[-1])

    # Context Basic-Level Count
    ax = axes[1, 1]
    if ctx_bc:
        ax.hist(ctx_bc, bins=ctx_bc_bins, color="indianred", edgecolor="white")
    ax.set_title("Context Basic-Level Count")
    ax.set_xlabel("count")
    ax.set_ylabel("frequency")
    ax.set_xlim(ctx_bc_bins[0], ctx_bc_bins[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_name = cat_name.lower().replace(" ", "_").replace("-", "_")
    fig_path = os.path.join(hist_dir, f"{safe_name}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved: {fig_path}")

print(f"All histograms saved to '{hist_dir}/'")

print(f"\nUngrammatical test outputs written to '{OUT_DIR}'")
