"""
Hollow Learning Test (Multi-Hierarchy) — trains WEBSTER from a corpus of
human-annotated "hollow" parse trees (merge recipes), then evaluates
learning + generation quality.

Hollow JSON format:
  { "sentence": "the dog chased the cat",
    "merges": [{"left": 0, "right": 1}, ...] }

Workflow:
  1. Load hollow JSONs from HOLLOW_CORPUS_DIR
  2. For each hollow tree, build primitives and replay merges via WEBSTER
  3. Learn from each completed tree
  4. Evaluate on held-out test sentences (auto-parsed)
  5. Run generation tests (from-scratch, masked, multi-mask, prefix)
"""

import sys, os

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, TEST_GRAMMAR2, TEST_CORPUS2
from parse_mh import WEBSTER, FiniteParseTree
import shutil
import json
import random
import glob

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = "unittests/hollow_learn_test_mh"
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"  # where hollow JSONs live
VIZ_INTERMEDIATES = True

CONTEXT_LENGTH = 3
THRESHOLD = 30
PRIMITIVES_FIRST = 200  # first N trees train with infinite threshold (primitives only)

corpus = TEST_CORPUS1
grammar = TEST_GRAMMAR1

# ── Setup ──────────────────────────────────────────────────────────────────
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# ── Load hollow corpus ────────────────────────────────────────────────────
hollow_paths = sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json")))
hollow_corpus: list[dict] = []
for p in hollow_paths:
    with open(p, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON: {p}")
            continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
    else:
        print(f"[WARN] Skipping non-hollow JSON (missing sentence/merges): {p}")

print(f"Loaded {len(hollow_corpus)} hollow parse trees from {HOLLOW_CORPUS_DIR}")

if not hollow_corpus:
    print("[ERROR] No hollow parse trees found. Create some with the hollow editor first.")
    sys.exit(1)

# ── Initialise WEBSTER ────────────────────────────────────────────────────
webster = WEBSTER(
    corpus,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-4,
    context_alpha=1e-3,
    content_bl_alpha=10,
    context_bl_alpha=1,
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
    content_pool_depth=4
    # context_attr_weights={6: 2.0},   # attr 6 = content-ref when context_length=3
    # content_attr_weights={0: 1.0, 1: 1.0},  # boost left & right child attrs
)

# ── Phase 1: primitives-only on random sentences ──────────────────────────
print(f"\n=== PHASE 1: PRIMITIVES ONLY ({PRIMITIVES_FIRST} random sentences) ===")
for i in range(PRIMITIVES_FIRST):
    sentence = generate("S", grammar)
    parse_tree = webster.parse_sentence(
        sentence,
        threshold=1e9,
        new_vocab=True,
        learning=True,
        debug=False,
    )

    if i % 5 == 0 and VIZ_INTERMEDIATES:
        parse_tree.visualize(f"{OUT_DIR}/train_trees/primitives_tree{i}")
        webster.visualize_ltm(f"{OUT_DIR}/ltms/primitives_ltm{i}", max_depth=3)

    print(f"  [{i+1}/{PRIMITIVES_FIRST}] Primitives: \"{sentence}\"")

# ── Phase 2: replay hollow trees with merges ──────────────────────────────
print(f"\n=== PHASE 2: HOLLOW CORPUS TRAINING (size = {len(hollow_corpus)}) ===")
for i, hollow in enumerate(hollow_corpus):
    sentence = hollow["sentence"]
    merges = hollow["merges"]

    # Build primitives for this sentence
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold=THRESHOLD)

    # Replay the human-annotated merge sequence
    for m in merges:
        try:
            tree.apply_candidate(m["left"], m["right"])
        except Exception as e:
            print(f"  [WARN] Merge ({m['left']}, {m['right']}) failed on sentence "
                  f"\"{sentence}\": {e}")

    # Learn from the completed tree
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)

    if i % 5 == 0 and VIZ_INTERMEDIATES:
        tree.visualize(f"{OUT_DIR}/train_trees/train_parse_tree{i}")
        webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=3)

    print(f"  [{i+1}/{len(hollow_corpus)}] Trained on: \"{sentence}\"  "
          f"({len(merges)} merges)")

# ── Save state ─────────────────────────────────────────────────────────────
SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
webster.save_state(SAVE_DIR)
print(f"\nSaved Final LTM to \"{SAVE_DIR}\"!")
webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)

# ============================================================================
# DIAGNOSTIC PHASE — does supervised training actually stick?
# ============================================================================
# The supervised regime (apply_candidate bypassing the threshold gate)
# guarantees each annotated chunk is added to the LTM. But re-parsing
# with the SAME threshold-gated greedy chunker may pick different
# merges than the human did, because the threshold gate is computed
# at recognition time using basic_level_count and tree_log_prob —
# both of which depend on aggregate training statistics, not on
# whether THIS specific chunk was annotated.
#
# This diagnostic section measures:
#   A. Re-parse fidelity per sentence (recall of gold spans).
#   B. Threshold sweep — at which threshold does recall plateau?
#   C. Chunk memorization — for each gold chunk, where does its
#      content_instance land in the content tree, and what's that
#      leaf's training-count?
#   D. Over/under-generalization breakdown.
#   E. Generation precision — for each of K sampled training chunks,
#      can we regenerate the exact tokens from the corresponding
#      content-tree leaf?
# ============================================================================
print("\n=== DIAGNOSTIC: supervised memorization + re-parse fidelity ===")

import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse_mh import PrimitiveParseNode, CompositeParseNode

DIAG_DIR = f"{OUT_DIR}/diagnostic"
os.makedirs(DIAG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _extract_spans(parse_tree):
    """{(start, end)} for every non-root composite. Inclusive."""
    spans: set = set()
    def walk(node):
        if isinstance(node, PrimitiveParseNode):
            pos = int(node.position_idx)
            return (pos, pos)
        starts, ends = [], []
        for _, c in sorted(getattr(node, "children", []),
                            key=lambda x: x[0] if x[0] is not None else 0):
            s, e = walk(c)
            if s is not None:
                starts.append(s); ends.append(e)
        if not starts:
            return (None, None)
        rng = (min(starts), max(ends))
        if not getattr(node, "is_global_root", False):
            spans.add(rng)
        return rng
    walk(parse_tree.global_root_node)
    return spans

def _gold_spans_from_merges(merges, sentence):
    """Replay the hollow merge sequence symbolically → set of inclusive spans."""
    tokens = sentence.split()
    n = len(tokens)
    centers = [float(i) for i in range(n)]
    spans   = [(i, i) for i in range(n)]
    out: set = set()
    for m in merges:
        l_c, r_c = m.get("left"), m.get("right")
        if l_c is None or r_c is None:
            return None
        try:
            li = centers.index(l_c); ri = centers.index(r_c)
        except ValueError:
            return None
        if abs(li - ri) != 1:
            return None
        a, b = (li, ri) if li < ri else (ri, li)
        new_span = (spans[a][0], spans[b][1])
        out.add(new_span)
        centers[a:b+1] = [(centers[a] + centers[b]) / 2.0]
        spans[a:b+1]   = [new_span]
    return out

def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode):
        return
    if not getattr(node, "is_global_root", False):
        yield node
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)

def _greedy_descend(root, instance):
    n = root
    while n.children:
        n = max(n.children, key=lambda c: c.log_prob_instance(instance))
    return n

# ──────────────────────────────────────────────────────────────────────
# A. Re-parse fidelity (at the trained THRESHOLD)
# ──────────────────────────────────────────────────────────────────────
print(f"\n  --- A. Re-parse fidelity (threshold={THRESHOLD}) ---")
fidelity_rows = []
agg_gold = agg_pred = agg_match = 0
for h in hollow_corpus:
    sentence = h["sentence"]
    gold = _gold_spans_from_merges(h["merges"], sentence)
    if gold is None or not gold:
        continue
    try:
        pt = webster.parse_sentence(sentence, threshold=THRESHOLD,
                                     new_vocab=False, learning=False, debug=False)
    except Exception:
        continue
    pred = _extract_spans(pt)
    match = pred & gold
    agg_gold += len(gold); agg_pred += len(pred); agg_match += len(match)
    fidelity_rows.append({
        "sentence": sentence, "n_words": len(sentence.split()),
        "n_gold": len(gold), "n_pred": len(pred),
        "n_match": len(match),
        "recall": len(match) / max(1, len(gold)),
        "precision": len(match) / max(1, len(pred)),
        "missed_spans": sorted(gold - pred),
        "extra_spans":  sorted(pred - gold),
    })

agg_p = agg_match / max(1, agg_pred)
agg_r = agg_match / max(1, agg_gold)
agg_f1 = 2 * agg_p * agg_r / max(1e-12, agg_p + agg_r)
print(f"    Aggregate over {len(fidelity_rows)} sentences:")
print(f"      precision = {100*agg_p:5.1f}%   recall = {100*agg_r:5.1f}%   F1 = {100*agg_f1:5.1f}%")
print(f"      total_gold = {agg_gold}   total_pred = {agg_pred}   match = {agg_match}")

# Recall histogram
recalls = [r["recall"] for r in fidelity_rows]
bins = np.linspace(0, 1, 11)
hist, _ = np.histogram(recalls, bins=bins)
print(f"    Per-sentence recall histogram:")
for i, c in enumerate(hist):
    bar = "#" * min(50, int(c))
    print(f"      [{bins[i]:.1f}, {bins[i+1]:.1f}): {c:>4} {bar}")

with open(f"{DIAG_DIR}/A_per_sentence.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["sentence", "n_words", "n_gold", "n_pred", "n_match",
                "precision", "recall", "missed_spans", "extra_spans"])
    for r in fidelity_rows:
        w.writerow([r["sentence"], r["n_words"], r["n_gold"], r["n_pred"],
                    r["n_match"], f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                    "|".join(f"{s},{e}" for s, e in r["missed_spans"]),
                    "|".join(f"{s},{e}" for s, e in r["extra_spans"])])

# ──────────────────────────────────────────────────────────────────────
# B. Threshold sweep
# ──────────────────────────────────────────────────────────────────────
print(f"\n  --- B. Threshold sweep ---")
sweep_thresholds = [-2, 0, 1, 3, 5, 10, 20, 30, 50, 100, 200]
sweep_rows = []
for t in sweep_thresholds:
    tg = tp = tm = 0
    for h in hollow_corpus:
        sentence = h["sentence"]
        gold = _gold_spans_from_merges(h["merges"], sentence)
        if gold is None or not gold:
            continue
        try:
            pt = webster.parse_sentence(sentence, threshold=t,
                                         new_vocab=False, learning=False, debug=False)
        except Exception:
            continue
        pred = _extract_spans(pt)
        tg += len(gold); tp += len(pred); tm += len(pred & gold)
    p = tm / max(1, tp); r = tm / max(1, tg)
    f1 = 2 * p * r / max(1e-12, p + r)
    sweep_rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1,
                       "total_pred": tp, "total_gold": tg, "match": tm})
    print(f"    threshold={t:>5}  P={100*p:5.1f}%  R={100*r:5.1f}%  F1={100*f1:5.1f}%  "
          f"(pred={tp}, match={tm}, gold={tg})")

with open(f"{DIAG_DIR}/B_threshold_sweep.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "precision", "recall", "f1",
                "total_pred", "total_gold", "match"])
    for row in sweep_rows:
        w.writerow([row["threshold"], f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}", f"{row['f1']:.4f}",
                    row["total_pred"], row["total_gold"], row["match"]])

# Plot threshold sweep
plt.figure(figsize=(8, 4))
xs = [r["threshold"] for r in sweep_rows]
plt.plot(xs, [r["precision"] for r in sweep_rows], "o-", label="Precision")
plt.plot(xs, [r["recall"]    for r in sweep_rows], "o-", label="Recall")
plt.plot(xs, [r["f1"]        for r in sweep_rows], "o-", label="F1")
plt.axvline(THRESHOLD, color="red", linestyle="--", alpha=0.4,
            label=f"trained threshold ({THRESHOLD})")
plt.xlabel("Re-parse threshold"); plt.ylabel("Score")
plt.title("Re-parse fidelity vs threshold (on hollow training set)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{DIAG_DIR}/B_threshold_sweep.png", dpi=140)
plt.close()

# ──────────────────────────────────────────────────────────────────────
# C. Chunk memorization — leaf-count distribution
# ──────────────────────────────────────────────────────────────────────
print(f"\n  --- C. Chunk memorization (content-tree leaf counts) ---")
chunk_leaf_rows = []
for h in hollow_corpus:
    sentence = h["sentence"]
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold=THRESHOLD)
    fail = 0
    for m in h["merges"]:
        try:
            tree.apply_candidate(m["left"], m["right"])
        except Exception:
            fail += 1
    if fail:
        continue
    for comp in _walk_composites(tree.global_root_node):
        ci = comp.get_content_instance()
        if not ci:
            continue
        leaf = _greedy_descend(webster.ltm.content_hierarchy.root, ci)
        cplx = getattr(comp, "complexity", None)
        chunk_leaf_rows.append({
            "complexity": cplx,
            "leaf_count": int(getattr(leaf, "count", 0)),
            "leaf_depth": int(leaf.depth()),
        })

if chunk_leaf_rows:
    leaf_counts = [r["leaf_count"] for r in chunk_leaf_rows]
    print(f"    Inspected {len(chunk_leaf_rows)} training chunks")
    print(f"    Leaf count: min={min(leaf_counts)}  "
          f"median={int(np.median(leaf_counts))}  "
          f"max={max(leaf_counts)}  mean={np.mean(leaf_counts):.1f}")
    buckets = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 100), (100, int(1e9))]
    print(f"    Distribution:")
    for lo, hi in buckets:
        c = sum(1 for x in leaf_counts if lo <= x < hi)
        pct = 100 * c / max(1, len(leaf_counts))
        bar = "#" * min(50, int(pct))
        print(f"      [{lo:>4}, {hi if hi < 1e9 else '∞':>4}): {c:>4} ({pct:5.1f}%) {bar}")

    with open(f"{DIAG_DIR}/C_chunk_memorization.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["complexity", "leaf_count", "leaf_depth"])
        for r in chunk_leaf_rows:
            w.writerow([r["complexity"], r["leaf_count"], r["leaf_depth"]])

    plt.figure(figsize=(8, 4))
    plt.hist(leaf_counts, bins=30, color="#1f77b4", edgecolor="white")
    plt.axvline(THRESHOLD, color="red", linestyle="--",
                label=f"trained threshold ({THRESHOLD})")
    plt.xlabel("Content-tree leaf count where chunk lands")
    plt.ylabel("# training chunks")
    plt.title("Chunk memorization: training chunks vs landing-leaf count")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{DIAG_DIR}/C_chunk_memorization.png", dpi=140)
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# D. Over/under-generalization
# ──────────────────────────────────────────────────────────────────────
print(f"\n  --- D. Over/under-generalization (threshold={THRESHOLD}) ---")
over = agg_pred - agg_match
under = agg_gold - agg_match
print(f"    Predicted spans  : {agg_pred}")
print(f"    Gold spans       : {agg_gold}")
print(f"    Matched          : {agg_match}")
over_pct  = 100 * over  / max(1, agg_pred)
under_pct = 100 * under / max(1, agg_gold)
print(f"    Overgeneralized  : {over:>4} "
      f"({over_pct:.1f}% of pred — chunks the parser invented)")
print(f"    Undergeneralized : {under:>4} "
      f"({under_pct:.1f}% of gold — chunks the supervisor gave that parser missed)")

# Per-length breakdown
print(f"    By chunk length:")
len_buckets: dict = {}
for r in fidelity_rows:
    sentence_tokens = r["sentence"].split()
    for s, e in r["missed_spans"]:
        length = e - s + 1
        len_buckets.setdefault(length, {"missed": 0, "extra": 0})
        len_buckets[length]["missed"] += 1
    for s, e in r["extra_spans"]:
        length = e - s + 1
        len_buckets.setdefault(length, {"missed": 0, "extra": 0})
        len_buckets[length]["extra"] += 1
print(f"      {'span_len':>10s} {'missed (under)':>16s} {'extra (over)':>15s}")
for length in sorted(len_buckets.keys()):
    b = len_buckets[length]
    print(f"      {length:>10d} {b['missed']:>16d} {b['extra']:>15d}")

with open(f"{DIAG_DIR}/D_generalization.txt", "w") as f:
    f.write(f"Re-parse threshold:        {THRESHOLD}\n")
    f.write(f"Total predicted spans:     {agg_pred}\n")
    f.write(f"Total gold spans:          {agg_gold}\n")
    f.write(f"Matched (TP):              {agg_match}\n")
    f.write(f"Overgeneralized  (FP):     {over} ({over_pct:.1f}% of pred)\n")
    f.write(f"Undergeneralized (FN):     {under} ({under_pct:.1f}% of gold)\n\n")
    f.write("By chunk length:\n")
    f.write(f"  {'span_len':>10s} {'missed (under)':>16s} {'extra (over)':>15s}\n")
    for length in sorted(len_buckets.keys()):
        b = len_buckets[length]
        f.write(f"  {length:>10d} {b['missed']:>16d} {b['extra']:>15d}\n")

# ──────────────────────────────────────────────────────────────────────
# E. Generation precision — can we regenerate a learned chunk?
# ──────────────────────────────────────────────────────────────────────
print(f"\n  --- E. Generation precision (chunk → tokens) ---")
# For each of K sampled training chunks, find the composite's content-tree
# leaf, then use webster.generate_sentence on a sentence template that
# anchors the composite at its original position and asks the model to
# regenerate it. We measure exact-token match.
GEN_K = 30
sampled_chunks = []
for h in hollow_corpus:
    gold = _gold_spans_from_merges(h["merges"], h["sentence"])
    if gold is None: continue
    tokens = h["sentence"].split()
    for s, e in gold:
        if e > s and not (s == 0 and e == len(tokens) - 1):
            sampled_chunks.append((h["sentence"], tokens, s, e))
random.shuffle(sampled_chunks)
sampled_chunks = sampled_chunks[:GEN_K]

gen_results = []
for sentence, tokens, s, e in sampled_chunks:
    gold_chunk = tokens[s:e+1]
    masked = " ".join(tokens[:s] + ["[mask]"] + tokens[e+1:])
    try:
        completed, _ = webster.generate_sentence(
            masked_sentence=masked, debug=False)
    except Exception:
        completed = ""
    comp_toks = completed.split() if completed else []
    # Extract inserted span (handles empty-suffix correctly).
    prefix = tokens[:s]; suffix = tokens[e+1:]
    inserted = []
    if len(comp_toks) >= len(prefix):
        ptr = len(prefix)
        if not suffix:
            inserted = comp_toks[ptr:]
        else:
            for j in range(ptr, len(comp_toks) - len(suffix) + 1):
                if comp_toks[j:j+len(suffix)] == suffix:
                    inserted = comp_toks[ptr:j]; break
            else:
                inserted = comp_toks[ptr:]
    exact = (inserted == gold_chunk)
    length_match = (len(inserted) == len(gold_chunk))
    first_match  = bool(inserted) and inserted[0] == gold_chunk[0]
    gen_results.append({
        "sentence": sentence, "span": (s, e),
        "gold": gold_chunk, "inserted": inserted,
        "exact": exact, "length_match": length_match,
        "first_match": first_match,
    })

if gen_results:
    n = len(gen_results)
    n_exact = sum(1 for r in gen_results if r["exact"])
    n_len   = sum(1 for r in gen_results if r["length_match"])
    n_first = sum(1 for r in gen_results if r["first_match"])
    print(f"    Sampled {n} training chunks. Regenerated via mask-completion.")
    print(f"      exact-token match : {n_exact}/{n} ({100*n_exact/n:.1f}%)")
    print(f"      length match      : {n_len}/{n}   ({100*n_len/n:.1f}%)")
    print(f"      first-token match : {n_first}/{n} ({100*n_first/n:.1f}%)")

    with open(f"{DIAG_DIR}/E_generation_precision.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "span_start", "span_end", "gold", "inserted",
                    "exact", "length_match", "first_match"])
        for r in gen_results:
            w.writerow([r["sentence"], r["span"][0], r["span"][1],
                        " ".join(r["gold"]), " ".join(r["inserted"]),
                        r["exact"], r["length_match"], r["first_match"]])

# ──────────────────────────────────────────────────────────────────────
# F. Unpack-from-leaf precision — bypass _basic_sample
# ──────────────────────────────────────────────────────────────────────
# Section E went through masked-completion (context-hierarchy
# categorize → ref → _basic_sample → bag → resolve). Section F
# pins the content-tree leaf the chunk landed at (via greedy
# descent on the chunk's content-instance) and calls the new
# ``start_content_leaf=`` generation mode. No _basic_sample, no
# leaf resampling — we read THAT leaf's left/right bags directly.
# If the chunk is truly memorised, this should regenerate the
# exact tokens; if even THIS fails the bags themselves are too
# diffuse for deterministic recall.
print(f"\n  --- F. Unpack-from-leaf precision (deterministic recall) ---")

f_results = []
for h in hollow_corpus:
    sentence = h["sentence"]
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold=THRESHOLD)
    fail = 0
    for m in h["merges"]:
        try:
            tree.apply_candidate(m["left"], m["right"])
        except Exception:
            fail += 1
    if fail:
        continue
    tokens = sentence.split()
    # Build a position → span map for the chunks in this sentence
    def _span_of(node):
        starts, ends = [], []
        def w(n):
            if isinstance(n, PrimitiveParseNode):
                pos = int(n.position_idx)
                starts.append(pos); ends.append(pos)
                return
            for _, c in getattr(n, "children", []):
                w(c)
        w(node)
        return (min(starts), max(ends)) if starts else (None, None)

    for comp in _walk_composites(tree.global_root_node):
        ci = comp.get_content_instance()
        if not ci:
            continue
        leaf = _greedy_descend(webster.ltm.content_hierarchy.root, ci)
        s, e = _span_of(comp)
        if s is None:
            continue
        gold_chunk = tokens[s:e+1]
        try:
            unpacked, _ = webster.generate_sentence(
                start_content_leaf=leaf, debug=False)
        except Exception as ex:
            unpacked = f"<failed: {ex}>"
        gen_tokens = unpacked.split() if unpacked and not unpacked.startswith("<failed") else []
        exact = (gen_tokens == gold_chunk)
        length_match = (len(gen_tokens) == len(gold_chunk))
        first_match = bool(gen_tokens) and gen_tokens[0] == gold_chunk[0]
        f_results.append({
            "sentence": sentence,
            "span": (s, e),
            "gold": gold_chunk,
            "unpacked": gen_tokens,
            "leaf_count": int(getattr(leaf, "count", 0)),
            "exact": exact,
            "length_match": length_match,
            "first_match": first_match,
        })

if f_results:
    n = len(f_results)
    n_exact = sum(1 for r in f_results if r["exact"])
    n_len   = sum(1 for r in f_results if r["length_match"])
    n_first = sum(1 for r in f_results if r["first_match"])
    print(f"    Probed {n} chunks via start_content_leaf.")
    print(f"      exact-token match : {n_exact}/{n} ({100*n_exact/n:.1f}%)")
    print(f"      length match      : {n_len}/{n}   ({100*n_len/n:.1f}%)")
    print(f"      first-token match : {n_first}/{n} ({100*n_first/n:.1f}%)")
    # Stratify by leaf count — leaves with more training count should
    # have tighter bags and recover more often.
    print(f"    Stratified by leaf count:")
    print(f"      {'leaf_count_range':>18s} {'n':>5s} {'exact':>8s}")
    for lo, hi in [(1, 2), (2, 5), (5, 20), (20, 100), (100, int(1e9))]:
        sub = [r for r in f_results if lo <= r["leaf_count"] < hi]
        if not sub:
            continue
        nx = sum(1 for r in sub if r["exact"])
        hi_str = "∞" if hi >= 1e9 else str(hi)
        print(f"      [{lo:>4}, {hi_str:>4}): {len(sub):>5d} {100*nx/len(sub):>7.1f}%")

    with open(f"{DIAG_DIR}/F_unpack_from_leaf.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "span_start", "span_end", "gold",
                    "unpacked", "leaf_count",
                    "exact", "length_match", "first_match"])
        for r in f_results:
            w.writerow([r["sentence"], r["span"][0], r["span"][1],
                        " ".join(r["gold"]), " ".join(r["unpacked"]),
                        r["leaf_count"],
                        r["exact"], r["length_match"], r["first_match"]])

print(f"\nDiagnostic outputs in {DIAG_DIR}/:")
print("  A_per_sentence.csv         — per-sentence re-parse stats")
print("  B_threshold_sweep.csv/.png — threshold vs P/R/F1")
print("  C_chunk_memorization.csv/.png — leaf-count distribution")
print("  D_generalization.txt       — over/under stats")
print("  E_generation_precision.csv — chunk regeneration (masked-completion)")
print("  F_unpack_from_leaf.csv     — chunk regeneration (start_content_leaf)")

# ── Test: auto-parse held-out sentences ────────────────────────────────────
print("\n=== AUTO-PARSE TEST SENTENCES ===")
num_test = 20
test_documents = [generate("S", grammar) for _ in range(num_test)]

for i, test in enumerate(test_documents):
    parse_tree = webster.parse_sentence(
        test,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=True,
    )
    parse_tree.visualize(f"{OUT_DIR}/test_trees/test_parse_tree{i}")
    print(f"  Test tree {i}: \"{test}\"")

# ── Test: random / fake sentences ──────────────────────────────────────────
print("\n=== FAKE SENTENCE PARSING ===")
fake_sentences = [
    " ".join([random.choice(corpus) for _ in range(random.randint(3, 8))])
    for _ in range(10)
]
for i, fake in enumerate(fake_sentences):
    parse_tree = webster.parse_sentence(
        fake,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=True,
    )
    parse_tree.visualize(f"{OUT_DIR}/fake_trees/fake_parse_tree{i}")
    print(f"  Fake tree {i}: \"{fake}\"")

# ============================================================================
# Generation tests — restored under the new frontier-categorize regime.
# WEBSTER's ``_resolve_bag`` (see src/parse_mh.py) now scores every
# candidate content-ref against the bag of K canonical context-tree
# nodes by *bag-weighted* mean log-prob and picks ``argmax``, instead
# of stochastically descending one canonical's subtree. Generation
# below uses the same path through ``webster.generate_sentence``.
# ============================================================================

# ── Generation: from-scratch ────────────────────────────────────────────────
print("\n=== FROM-SCRATCH GENERATION ===")
gen_path = f"{OUT_DIR}/generated_sentences.txt"
os.makedirs(os.path.dirname(gen_path), exist_ok=True)
with open(gen_path, "w") as gf:
    for i in range(10):
        try:
            sentence, parse = webster.generate_sentence(debug=False)
        except Exception as e:
            sentence, parse = f"<generation failed: {e}>", None
        print(f"  Generated [{i}]: \"{sentence}\"")
        gf.write(f"[{i}] {sentence}\n")
        if parse is not None and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/generated_trees/generated_parse_tree{i}")
print(f"Saved generated sentences to \"{gen_path}\"")

# ── Generation: single-token mask ───────────────────────────────────────────
print("\n=== MASKED COMPLETION (single token) ===")
single_mask_path = f"{OUT_DIR}/single_mask_results.txt"
os.makedirs(os.path.dirname(single_mask_path), exist_ok=True)
with open(single_mask_path, "w") as smf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 2:
            continue
        mask_idx = random.randint(1, len(tokens) - 1)
        original_token = tokens[mask_idx]
        masked_tokens = list(tokens); masked_tokens[mask_idx] = "[mask]"
        masked = " ".join(masked_tokens)
        try:
            completed, parse = webster.generate_sentence(
                masked_sentence=masked, debug=False)
        except Exception as e:
            completed, parse = f"<failed: {e}>", None
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"  (replaced \"{original_token}\" @ pos {mask_idx})")
        print(f"  Completed: \"{completed}\"\n")
        smf.write(f"[{i}] original: {test_documents[i]}\n")
        smf.write(f"    masked:   {masked}\n")
        smf.write(f"    replaced: \"{original_token}\" at pos {mask_idx}\n")
        smf.write(f"    completed: {completed}\n\n")
        if parse is not None and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/single_mask_trees/single_mask_tree{i}")
print(f"Saved single-mask results to \"{single_mask_path}\"")

# ── Generation: mid-sentence single mask ────────────────────────────────────
print("\n=== MID-SENTENCE MASKED COMPLETION ===")
mid_mask_path = f"{OUT_DIR}/mid_mask_results.txt"
os.makedirs(os.path.dirname(mid_mask_path), exist_ok=True)
with open(mid_mask_path, "w") as mf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 3:
            continue
        mid = len(tokens) // 2
        original_token = tokens[mid]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + 1:]
        masked = " ".join(masked_tokens)
        try:
            completed, parse = webster.generate_sentence(
                masked_sentence=masked, debug=False)
        except Exception as e:
            completed, parse = f"<failed: {e}>", None
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"  (replaced \"{original_token}\" @ pos {mid})")
        print(f"  Completed: \"{completed}\"\n")
        mf.write(f"[{i}] original: {test_documents[i]}\n")
        mf.write(f"    masked:   {masked}\n")
        mf.write(f"    replaced: \"{original_token}\" at pos {mid}\n")
        mf.write(f"    completed: {completed}\n\n")
        if parse is not None and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/mid_mask_trees/mid_mask_tree{i}")
print(f"Saved mid-sentence mask results to \"{mid_mask_path}\"")

# ── Generation: multi-token mid-sentence mask ───────────────────────────────
print("\n=== MULTI-TOKEN MID-SENTENCE MASK ===")
multi_mid_path = f"{OUT_DIR}/multi_mid_mask_results.txt"
os.makedirs(os.path.dirname(multi_mid_path), exist_ok=True)
with open(multi_mid_path, "w") as mmf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 5:
            continue
        max_remove = min(4, len(tokens) - 2)
        num_remove = random.randint(2, max(2, max_remove))
        mid = max(1, (len(tokens) - num_remove) // 2)
        removed = tokens[mid:mid + num_remove]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + num_remove:]
        masked = " ".join(masked_tokens)
        try:
            completed, parse = webster.generate_sentence(
                masked_sentence=masked, debug=False)
        except Exception as e:
            completed, parse = f"<failed: {e}>", None
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"  (removed {num_remove} tokens: "
              f"\"{' '.join(removed)}\" at pos {mid}-{mid+num_remove-1})")
        print(f"  Completed: \"{completed}\"\n")
        mmf.write(f"[{i}] original: {test_documents[i]}\n")
        mmf.write(f"    masked:   {masked}\n")
        mmf.write(f"    removed: \"{' '.join(removed)}\" ({num_remove} tokens) "
                  f"at pos {mid}-{mid+num_remove-1}\n")
        mmf.write(f"    completed: {completed}\n\n")
        if parse is not None and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/multi_mid_trees/multi_mid_tree{i}")
print(f"Saved multi-token mid mask results to \"{multi_mid_path}\"")

# ── Generation: prefix prediction (expand second half) ──────────────────────
print("\n=== MASKED PREDICTION (expand second half) ===")
mask_pred_path = f"{OUT_DIR}/masked_prediction_results.txt"
os.makedirs(os.path.dirname(mask_pred_path), exist_ok=True)
with open(mask_pred_path, "w") as mpf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 2:
            continue
        split_point = len(tokens) // 2
        prefix = tokens[:split_point]
        masked = " ".join(prefix + ["[mask]"])
        try:
            completed, parse = webster.generate_sentence(
                masked_sentence=masked, debug=False)
        except Exception as e:
            completed, parse = f"<failed: {e}>", None
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"")
        print(f"  Completed: \"{completed}\"\n")
        mpf.write(f"[{i}] original: {test_documents[i]}\n")
        mpf.write(f"    masked:   {masked}\n")
        mpf.write(f"    completed: {completed}\n\n")
        if parse is not None and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/masked_pred_trees/masked_pred_tree{i}")
print(f"Saved masked prediction results to \"{mask_pred_path}\"")
