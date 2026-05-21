"""
WEBSTER / Cobweb parameter sweep
================================

Sweep the cobweb hyperparameters that most plausibly affect parse quality
and measure their effect on hollow-learn-test parse F1, step-pick, and
exact-match. Uses the same 80/20 hollow corpus split as
unittests/hollow_learn_test_mh.py and tests/met5/grammar_threshold_test.py.

Sweeps (one knob at a time, others held at the SOTA config):

  alpha                       smoothing of the cobweb tree
  bl_alpha                    smoothing during EPMI evaluation (get_basic)
  weight_attr                 whether attribute weights are scaled by root frequency
  content_top_k               TopK-Pool encoder fan-out
  content_pool_depth          context-tree depth for pool selection
  threshold (climbing gate)   "ample count" threshold for chunk admission

For each config, we report:
  - Bracket P / R / F1 on held-out test fold
  - Exact-match parse rate
  - Step-pick accuracy (under the climbing-ancestor gate)

Outputs:
  tests/met5/grammar_param_sweep_output/sweep_results.csv
  tests/met5/grammar_param_sweep_output/sweep_summary.png
"""

import os
import sys
import csv
import json
import glob
import random
import shutil
import re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import WEBSTER, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR           = os.path.join(_HERE, "grammar_param_sweep_output")
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
CONTEXT_LENGTH    = 3
PRIMITIVES_FIRST  = 200
SEED              = 13

# Current SOTA config (the same as hollow_learn_test_mh.py uses).
SOTA = {
    "content_alpha":      1e-6,
    "context_alpha":      1e-6,
    "content_bl_alpha":   10,
    "context_bl_alpha":   10,
    "weight_attr":        False,
    "content_top_k":      7,
    "content_pool_depth": 4,
    "threshold":          30,
}

# Sweeps (one knob at a time).
SWEEPS = {
    "alpha": [
        ("alpha=1e-6 (current)", {"content_alpha": 1e-6, "context_alpha": 1e-6}),
        ("alpha=1e-4",           {"content_alpha": 1e-4, "context_alpha": 1e-4}),
        ("alpha=1e-3",           {"content_alpha": 1e-3, "context_alpha": 1e-3}),
        ("alpha=1e-2",           {"content_alpha": 1e-2, "context_alpha": 1e-2}),
        ("alpha=0.1",            {"content_alpha": 0.1,  "context_alpha": 0.1}),
    ],
    "bl_alpha": [
        ("bl_alpha=1",            {"content_bl_alpha": 1.0,  "context_bl_alpha": 1.0}),
        ("bl_alpha=10 (current)", {"content_bl_alpha": 10,   "context_bl_alpha": 10}),
        ("bl_alpha=100",          {"content_bl_alpha": 100,  "context_bl_alpha": 100}),
    ],
    "weight_attr": [
        ("weight_attr=False (current)", {"weight_attr": False}),
        ("weight_attr=True",            {"weight_attr": True}),
    ],
    "content_top_k": [
        ("top_k=3",          {"content_top_k": 3}),
        ("top_k=5",          {"content_top_k": 5}),
        ("top_k=7 (current)",{"content_top_k": 7}),
        ("top_k=10",         {"content_top_k": 10}),
    ],
    "content_pool_depth": [
        ("pool_depth=2",          {"content_pool_depth": 2}),
        ("pool_depth=3",          {"content_pool_depth": 3}),
        ("pool_depth=4 (current)",{"content_pool_depth": 4}),
        ("pool_depth=5",          {"content_pool_depth": 5}),
    ],
    "threshold": [
        ("thr=5",           {"threshold": 5}),
        ("thr=10",          {"threshold": 10}),
        ("thr=30 (current)",{"threshold": 30}),
        ("thr=50",          {"threshold": 50}),
        ("thr=100",         {"threshold": 100}),
    ],
}

# ── Setup ────────────────────────────────────────────────────────────────────
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared helpers ───────────────────────────────────────────────────────────
def _chunk_span(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            out.append(int(n.position_idx)); return
        for _, c in getattr(n, "children", []): w(c)
    w(node)
    if not out: return None, None
    return min(out), max(out)

def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode): return
    if not getattr(node, "is_global_root", False):
        yield node
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)

def _bracket_set(tree):
    brackets = set()
    for comp in _walk_composites(tree.global_root_node):
        s, e = _chunk_span(comp)
        if s is not None and e is not None and s != e:
            brackets.add((s, e))
    return brackets

# ── Load hollow corpus once ─────────────────────────────────────────────────
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try: data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
print(f"Loaded {len(hollow_corpus)} hollow trees")

# Same 80/20 split as hollow_learn_test (seeded the same way).
random.seed(SEED)
np.random.seed(SEED)
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
TRAIN_HOLLOW = hollow_corpus[:_split]
TEST_HOLLOW  = hollow_corpus[_split:]
print(f"  Train: {len(TRAIN_HOLLOW)}   Test: {len(TEST_HOLLOW)}")

# Same primitives generator seed.
PRIMITIVE_SENTS = []
random.seed(SEED + 1)   # avoid colliding with the corpus shuffle's state
for _ in range(PRIMITIVES_FIRST):
    PRIMITIVE_SENTS.append(generate("S", TEST_GRAMMAR1))
print(f"  Primitive sentences: {len(PRIMITIVE_SENTS)}")

# ── Eval functions ──────────────────────────────────────────────────────────
def _train(cfg):
    """Train a fresh WEBSTER with the given config; return webster."""
    # Reset ALL three rngs (Python, NumPy, cobweb C++) to the same seed
    # so the only difference between sweep rows is the config itself.
    random.seed(SEED + 999)
    np.random.seed(SEED + 999)
    cobweb_set_seed(SEED + 999)

    webster = WEBSTER(
        TEST_CORPUS1,
        context_length=CONTEXT_LENGTH,
        threshold=cfg["threshold"],
        content_alpha=cfg["content_alpha"],
        context_alpha=cfg["context_alpha"],
        content_bl_alpha=cfg["content_bl_alpha"],
        context_bl_alpha=cfg["context_bl_alpha"],
        bow=False,
        empty_weighting=True,
        chunk_context=False,
        weighting="binary",
        categorization_mode="dfs",
        depth_max_content=1000,
        depth_max_context=1000,
        branch_max_content=1000,
        branch_max_context=1000,
        content_top_k=cfg["content_top_k"],
        content_pool_depth=cfg["content_pool_depth"],
        content_weight_attr=cfg["weight_attr"],
        context_weight_attr=cfg["weight_attr"],
    )
    # Phase 1: primitives-only.
    for s in PRIMITIVE_SENTS:
        webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                               learning=True, debug=False)
    # Phase 2: hollow replay.
    for hollow in TRAIN_HOLLOW:
        tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        tree.build_primitives(hollow["sentence"], threshold=cfg["threshold"])
        for m in hollow["merges"]:
            try: tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
    return webster

def _evaluate(webster, cfg):
    """Compute F1, exact-match, step-pick on TEST_HOLLOW."""
    total_tp = total_fp = total_fn = 0
    exact = 0
    n_sents = 0
    n_step_total = n_step_correct = n_step_no_cand = 0

    for hollow in TEST_HOLLOW:
        sentence = hollow["sentence"]
        sent_len = len(re.findall(r"[\w']+|[.,!?;]", sentence))
        if sent_len < 2:
            continue

        # Gold brackets.
        gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = _bracket_set(gold_tree)

        # End-to-end parse (uses build()).
        pred_tree = webster.parse_sentence(
            sentence, threshold=cfg["threshold"],
            new_vocab=False, learning=False, debug=False)
        pred = _bracket_set(pred_tree)

        total_tp += len(gold & pred)
        total_fp += len(pred - gold)
        total_fn += len(gold - pred)
        if gold == pred and len(gold) > 0:
            exact += 1
        n_sents += 1

        # Step-pick on gold trajectory.
        if not gold:
            continue
        step_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        step_tree.build_primitives(sentence, threshold="converge")
        for step_idx, m in enumerate(hollow["merges"]):
            pairs = step_tree.get_parentless_pairs()
            if not pairs: break
            admitted = []
            for p in pairs:
                try:
                    res = step_tree.evaluate_pair(
                        p["left_word_index"], p["right_word_index"],
                        climb_count_threshold=cfg["threshold"])
                except Exception:
                    continue
                csd = res.get("content_score_data", {})
                if csd.get("climb_hit_root", True):
                    continue
                ln = step_tree._find_root_child_by_index(p["left_word_index"])
                rn = step_tree._find_root_child_by_index(p["right_word_index"])
                if ln is None or rn is None: continue
                ls, _ = _chunk_span(ln); _, re_ = _chunk_span(rn)
                admitted.append((csd.get("root_log_prob", -float("inf")),
                                  (int(ls), int(re_))))
            n_step_total += 1
            if not admitted:
                n_step_no_cand += 1
            else:
                admitted.sort(key=lambda x: x[0], reverse=True)
                if admitted[0][1] in gold:
                    n_step_correct += 1
            try: step_tree.apply_candidate(m["left"], m["right"])
            except Exception: break

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    em        = exact / max(n_sents, 1)
    sp        = n_step_correct / max(n_step_total, 1)
    gate_pass = (n_step_total - n_step_no_cand) / max(n_step_total, 1)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "exact_match": em, "step_pick": sp, "gate_pass": gate_pass,
        "n_sents": n_sents, "n_step_total": n_step_total,
    }

# ── Run sweep ───────────────────────────────────────────────────────────────
all_results = []

def _make_cfg(overrides):
    cfg = dict(SOTA)
    cfg.update(overrides)
    return cfg

print(f"\n=== Running parameter sweep ===\n")
for sweep_name, configs in SWEEPS.items():
    print(f"--- Sweeping {sweep_name} ---")
    for label, overrides in configs:
        cfg = _make_cfg(overrides)
        print(f"  [{label}]   cfg={overrides}")
        try:
            webster = _train(cfg)
            metrics = _evaluate(webster, cfg)
            row = {
                "sweep": sweep_name, "label": label, **overrides, **metrics,
            }
            all_results.append(row)
            print(f"    F1={100*metrics['f1']:.1f}%  "
                  f"EM={100*metrics['exact_match']:.1f}%  "
                  f"step-pick={100*metrics['step_pick']:.1f}%  "
                  f"gate-pass={100*metrics['gate_pass']:.1f}%")
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"sweep": sweep_name, "label": label,
                                **overrides, "error": str(e)})

# ── Save CSV ────────────────────────────────────────────────────────────────
all_keys = ["sweep", "label",
            "content_alpha", "context_alpha",
            "content_bl_alpha", "context_bl_alpha", "weight_attr",
            "content_top_k", "content_pool_depth", "threshold",
            "precision", "recall", "f1", "exact_match",
            "step_pick", "gate_pass", "n_sents", "n_step_total", "error"]
with open(os.path.join(OUT_DIR, "sweep_results.csv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader()
    for r in all_results:
        w.writerow(r)

# ── Plot ─────────────────────────────────────────────────────────────────────
sweeps_with_data = list(SWEEPS.keys())
n = len(sweeps_with_data)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("WEBSTER / Cobweb parameter sweep — hollow-learn test", fontsize=14)

for i, sname in enumerate(sweeps_with_data):
    ax = axes[i]
    rows = [r for r in all_results if r["sweep"] == sname and "error" not in r]
    if not rows: continue
    labels = [r["label"].split(" ")[0] for r in rows]
    f1s = [r["f1"] for r in rows]
    sps = [r["step_pick"] for r in rows]
    ems = [r["exact_match"] for r in rows]
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, f1s, w, label="F1",       color="#d62728")
    ax.bar(x,     sps, w, label="step-pick",color="#2ca02c")
    ax.bar(x + w, ems, w, label="exact-match", color="#9467bd")
    ax.set_title(sname, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# Hide unused axes
for i in range(len(sweeps_with_data), len(axes)):
    axes[i].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "sweep_summary.png"), dpi=140,
            bbox_inches="tight")
plt.close()

# ── Best-config report ──────────────────────────────────────────────────────
valid = [r for r in all_results if "error" not in r]
best_f1 = max(valid, key=lambda r: r["f1"]) if valid else None
best_em = max(valid, key=lambda r: r["exact_match"]) if valid else None
best_sp = max(valid, key=lambda r: r["step_pick"]) if valid else None

print()
print("=" * 70)
print("PARAMETER SWEEP RESULTS")
print("=" * 70)
print(f"  Best F1          : {best_f1['label']}  → {100*best_f1['f1']:.1f}%")
print(f"  Best Exact-match : {best_em['label']}  → {100*best_em['exact_match']:.1f}%")
print(f"  Best Step-pick   : {best_sp['label']}  → {100*best_sp['step_pick']:.1f}%")
print()
print(f"Artefacts in {OUT_DIR}/:")
print(f"  sweep_results.csv  sweep_summary.png")
