"""
Primitive Maturity Threshold — Research Loop (met6)
===================================================

Goal: find the best context-hierarchy heuristic (and its operating
threshold) for deciding whether a PRIMITIVE's representation is mature
enough to be used as a chunk in future parses. Replaces the current
``PRIMITIVES_FIRST=200`` warm-up + integer count gate (``cost > 30``)
with a per-primitive maturity decision that kicks in from sentence 1.

Why only the context hierarchy?
-------------------------------
Primitives are categorised against the context hierarchy
(``_categorize`` against ``self.ltm.context_hierarchy``) and only
interact with it. Content-tree heuristics are not in scope — a
primitive is one token, and its content "bag" is just its own identity.

Ground-truth maturity (POS-class recoverability)
------------------------------------------------
For each (label_path → POS) pair seen across all primitive observations,
compute ``majority_pos(label_path)`` = argmax POS frequency. A primitive
observation is **mature** iff its ``true_pos == majority_pos(label_path)``.

This is the classic cluster-purity criterion: the cluster the primitive
landed in is dominated by its own POS, so the representation IS
distinguishing categories.

Heuristics evaluated (all from ``_score_along_path``)
-----------------------------------------------------
    cost / basic_level_count   — count at basic-level (current gate)
    basic_level_log_prob       — log P(instance | BL cluster)
    basic_level_class_log_prob — log P(class | instance) at BL
    root_log_prob              — log P at root cluster
    leaf_log_prob              — log P at leaf cluster
    tree_log_prob              — root marginal
    tree_class_log_prob        — root P(class | instance)
    path_depth                 — depth of categorisation path
    leaf_count                 — count at categorised leaf
    max_path_count             — max count along path
    lp_gain                    — leaf_lp − root_lp (information gain)
    bl_depth                   — depth of basic-level on path

Pipeline
--------
0. Load hollow corpus + grammar POS map.
1. Train WEBSTER incrementally on the train fold, NO primitives-only
   warm-up; for each sentence, build primitives, log each primitive's
   score_data + true POS, then apply gold merges and update LTM.
2. Compute ground-truth maturity from cluster majority POS.
3. Per-heuristic discriminability: ROC + AUC, PR + AP.
4. Threshold sweep: for each heuristic, sweep τ over its observed
   range and report precision / recall / F1 of mature-prediction at
   each τ. Pick best τ per heuristic by F1 (more honest than AUC for
   the gate downstream).
5. Write a summary CSV ranking heuristics by AUC and best F1, and
   point to the winner — this becomes the production gate.

Outputs (``tests/primitive_threshold/output/``)
----------------------------------------------------
    primitive_log.csv            — every primitive observation + heuristics
    heuristic_summary.csv        — AUC + best τ + best-F1 per heuristic
    heuristic_histograms.png     — mature vs immature per heuristic
    heuristic_roc.png            — ROC per heuristic
    threshold_sweep.png          — F1 vs τ per heuristic
    winner.txt                   — chosen heuristic + threshold + rationale

Usage:
    python tests/primitive_threshold/primitive_threshold_test.py
"""
import os, sys, json, glob, random, csv
from collections import defaultdict, Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.cfg import TEST_GRAMMAR_MED, TEST_CORPUS_MED
from parse_mh import WEBSTER, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

# ───────────────────────────── Configuration ─────────────────────────
HOLLOW_DIR  = os.path.join(_ROOT, "data", "test_hollow_grammar_1")
OUT_DIR     = os.path.join(_HERE, "output")
SEED        = 13
CONTEXT_LEN = 3

# Heuristics to log + analyse. Order = display order in plots.
HEURISTICS = [
    "basic_level_count",       # cost (current gate target)
    "basic_level_log_prob",
    "basic_level_class_log_prob",
    "root_log_prob",
    "leaf_log_prob",
    "tree_log_prob",
    "tree_class_log_prob",
    "path_depth",
    "leaf_count",
    "max_path_count",
    "lp_gain",
    "bl_depth",
]

# ───────────────────────────── Setup ─────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)

# Derive POS map from grammar (matches hollow_learn_test_mh)
POS_LIST = []
for sym, prods in TEST_GRAMMAR_MED.items():
    if not prods: continue
    if all(len(p) == 1 and p[0] not in TEST_GRAMMAR_MED for p in prods):
        POS_LIST.append(sym)
WORD_TO_POS = {}
for pos in POS_LIST:
    for prod in TEST_GRAMMAR_MED[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
print(f"POS classes: {POS_LIST}  ({len(WORD_TO_POS)} words mapped)")

# Load hollow corpus
hollow = []
for p in sorted(glob.glob(os.path.join(HOLLOW_DIR, "*.json"))):
    with open(p) as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            continue
    if "sentence" in d and "merges" in d:
        hollow.append(d)
random.shuffle(hollow)
split = int(0.8 * len(hollow))
train, test = hollow[:split], hollow[split:]
print(f"Hollow corpus: {len(hollow)}  train={len(train)}  test={len(test)}")

# Initialise WEBSTER — same hyperparams as hollow_learn_test_mh
webster = WEBSTER(
    TEST_CORPUS_MED,
    context_length=CONTEXT_LEN,
    threshold=30,
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

# ───────────────────── Phase 1: Train fully (then re-score) ──────────
# Train fully against the train fold WITHOUT logging heuristics — we
# want the FINAL converged LTM as our reference. Then in Phase 1b we
# re-build primitives against that final LTM and log heuristics there.
# This is the apples-to-apples setup: heuristics and ground-truth
# (cluster purity) both come from the same fixed tree, so there's no
# temporal bias from a growing tree.
print(f"\n=== PHASE 1a: train WEBSTER on train fold (no logging yet) ===")
for sent_idx, h in enumerate(train):
    sentence = h["sentence"]
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LEN)
    tree.build_primitives(sentence, threshold=0)
    for m in h["merges"]:
        try:
            tree.apply_candidate(m["left"], m["right"])
        except Exception:
            pass
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
    if (sent_idx + 1) % 20 == 0:
        print(f"  trained [{sent_idx+1}/{len(train)}]")

print(f"\n=== PHASE 1b: re-score all train primitives against FINAL LTM ===")

def _extract_heuristics(score_data: dict) -> dict:
    """Pull a fixed set of context-tree heuristics out of score_data
    (the dict returned by `_score_along_path`)."""
    # Parse path arrays (stored as str(list))
    try:
        node_lps = eval(score_data.get("raw_node_log_probs", "[]"))
    except Exception:
        node_lps = []
    try:
        node_counts = eval(score_data.get("candidate_counts", "[]"))
    except Exception:
        node_counts = []

    root_lp  = float(score_data.get("root_log_prob", float("nan")))
    leaf_lp  = float(score_data.get("leaf_log_prob", float("nan")))
    bl_count = float(score_data.get("basic_level_count", -1))

    # Find basic-level depth (where bl_count lives on the path).
    bl_depth = -1
    if node_counts:
        for d, c in enumerate(node_counts):
            if float(c) == bl_count:
                bl_depth = d
                break

    return {
        "basic_level_count":        float(score_data.get("basic_level_count", -1)),
        "basic_level_log_prob":     float(score_data.get("basic_level_log_prob", float("nan"))),
        "basic_level_class_log_prob": float(score_data.get("basic_level_class_log_prob", float("nan"))),
        "root_log_prob":            root_lp,
        "leaf_log_prob":            leaf_lp,
        "tree_log_prob":            float(score_data.get("tree_log_prob", float("nan"))),
        "tree_class_log_prob":      float(score_data.get("tree_class_log_prob", float("nan"))),
        "path_depth":               float(len(node_lps)),
        "leaf_count":               float(node_counts[-1]) if node_counts else 0.0,
        "max_path_count":           float(max(node_counts)) if node_counts else 0.0,
        "lp_gain":                  (leaf_lp - root_lp) if np.isfinite(leaf_lp) and np.isfinite(root_lp) else float("nan"),
        "bl_depth":                 float(bl_depth),
    }


primitive_log: list[dict] = []   # one row per (sentence, position) observation

for sent_idx, h in enumerate(train):
    sentence = h["sentence"]
    words    = sentence.split()

    # Re-categorize against the FULLY-TRAINED tree. Each primitive's
    # score_data now reflects the converged LTM state, not whatever
    # snapshot existed when this sentence was first trained on.
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LEN)
    tree.build_primitives(sentence, threshold=0)

    for pos_idx, prim in enumerate(tree.nodes):
        word = words[pos_idx] if pos_idx < len(words) else "?"
        true_pos = WORD_TO_POS.get(word, "OTHER")
        heur = _extract_heuristics(prim.score_data)
        primitive_log.append({
            "sentence_idx": sent_idx,
            "position":     pos_idx,
            "word":         word,
            "true_pos":     true_pos,
            "label_path":   int(prim.label_path),
            **heur,
        })

    if (sent_idx + 1) % 20 == 0:
        print(f"  rescored [{sent_idx+1}/{len(train)}]  log size={len(primitive_log)}")

print(f"\nLogged {len(primitive_log)} primitive observations across {len(train)} sentences")

# ───────────────────── Phase 2: Ground-truth maturity ────────────────
print(f"\n=== PHASE 2: ground-truth maturity (final-tree cluster purity) ===")

# Both heuristics and ground truth are computed against the SAME
# fully-trained LTM (Phase 1b), so there's no temporal bias.
#
# A primitive is "mature" iff its label_path (final-tree leaf cluster) is
#   (a) populated by ≥ MIN_CLUSTER_OBS observations, AND
#   (b) ≥ PURITY_THRESHOLD of those observations share its POS class.
#
# (a) protects against trusting a cluster with too little evidence.
# (b) protects against polysemous clusters that mix POS classes.
MIN_CLUSTER_OBS  = 3
PURITY_THRESHOLD = 0.8

lp_pos_counts: dict[int, Counter] = defaultdict(Counter)
for row in primitive_log:
    lp_pos_counts[row["label_path"]].update([row["true_pos"]])

lp_total:    dict[int, int]   = {lp: sum(c.values()) for lp, c in lp_pos_counts.items()}
lp_purity:   dict[int, float] = {
    lp: c.most_common(1)[0][1] / max(sum(c.values()), 1)
    for lp, c in lp_pos_counts.items()
}
lp_majority: dict[int, str] = {
    lp: c.most_common(1)[0][0] for lp, c in lp_pos_counts.items()
}

for row in primitive_log:
    lp = row["label_path"]
    cluster_big_enough = lp_total[lp]  >= MIN_CLUSTER_OBS
    cluster_pure       = lp_purity[lp] >= PURITY_THRESHOLD
    pos_matches        = (row["true_pos"] == lp_majority[lp])
    row["mature"]        = int(cluster_big_enough and cluster_pure and pos_matches)
    row["cluster_size"]  = lp_total[lp]
    row["cluster_purity"] = lp_purity[lp]

n_mature  = sum(r["mature"] for r in primitive_log)
n_small   = sum(1 for r in primitive_log if r["cluster_size"] < MIN_CLUSTER_OBS)
n_impure  = sum(1 for r in primitive_log
                if r["cluster_size"] >= MIN_CLUSTER_OBS
                and r["cluster_purity"] < PURITY_THRESHOLD)
print(f"  Distinct label_paths : {len(lp_majority)}")
print(f"  Mature (cluster ≥{MIN_CLUSTER_OBS} obs + purity ≥{PURITY_THRESHOLD} + POS match) : "
      f"{n_mature}/{len(primitive_log)} "
      f"({100*n_mature/max(len(primitive_log),1):.1f}%)")
print(f"  Immature: cluster size < {MIN_CLUSTER_OBS} : {n_small} "
      f"({100*n_small/max(len(primitive_log),1):.1f}%)")
print(f"  Immature: cluster purity < {PURITY_THRESHOLD} : {n_impure} "
      f"({100*n_impure/max(len(primitive_log),1):.1f}%)")

# Write per-primitive log
log_csv = os.path.join(OUT_DIR, "primitive_log.csv")
with open(log_csv, "w", newline="") as f:
    cols = (["sentence_idx", "position", "word", "true_pos", "label_path",
             "cluster_size", "cluster_purity", "mature"]
            + HEURISTICS)
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in primitive_log:
        w.writerow({k: r.get(k, "") for k in cols})
print(f"  Log → {log_csv}")

# ───────────────────── Phase 3: Discriminability ─────────────────────
print(f"\n=== PHASE 3: per-heuristic discriminability (ROC + threshold sweep) ===")

def _roc_auc(y, s):
    """Return (fprs, tprs, auc) sorted by descending threshold.
    y: 0/1 labels, s: scores (higher = more positive)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]; s = s[mask]
    if len(y) == 0 or len(set(y)) < 2:
        return np.array([0,1]), np.array([0,1]), 0.5
    order = np.argsort(-s)
    y = y[order]; s = s[order]
    P = max(y.sum(), 1)
    N = max(len(y) - y.sum(), 1)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tprs = tps / P
    fprs = fps / N
    fprs = np.concatenate([[0.0], fprs])
    tprs = np.concatenate([[0.0], tprs])
    auc  = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc


def _threshold_sweep(y, s, n_thresholds=50):
    """For each τ in a sweep, compute P/R/F1 of (s > τ) → mature.
    Returns (taus, precisions, recalls, f1s, best_tau, best_f1)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]; s = s[mask]
    if len(y) == 0:
        return [], [], [], [], None, 0.0
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        taus = np.array([lo])
    else:
        taus = np.linspace(lo, hi, n_thresholds)
    Ps, Rs, Fs = [], [], []
    for t in taus:
        pred = (s > t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        Ps.append(P); Rs.append(R); Fs.append(F)
    Fs_arr = np.array(Fs)
    best = int(np.argmax(Fs_arr))
    return taus, Ps, Rs, Fs, float(taus[best]), float(Fs_arr[best])


# Compute discriminability per heuristic
results = []
y = np.array([r["mature"] for r in primitive_log])
for h in HEURISTICS:
    s = np.array([r[h] for r in primitive_log])
    _, _, auc = _roc_auc(y, s)
    taus, Ps, Rs, Fs, best_tau, best_f1 = _threshold_sweep(y, s)
    # Coverage at best τ = fraction of primitives admitted
    if best_tau is not None:
        cov = float(((s[np.isfinite(s)]) > best_tau).mean())
    else:
        cov = 0.0
    results.append({
        "heuristic": h, "auc": auc, "best_tau": best_tau,
        "best_f1": best_f1, "coverage_at_best": cov,
    })
    tau_s = f"{best_tau:.3g}" if best_tau is not None else "n/a"
    print(f"  {h:30s}  AUC={auc:.3f}  best τ={tau_s}  "
          f"F1={best_f1:.3f}  cov={cov:.2f}")

# Sort by AUC desc
results.sort(key=lambda r: -r["auc"])

# Write summary CSV
summary_csv = os.path.join(OUT_DIR, "heuristic_summary.csv")
with open(summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["heuristic", "auc", "best_tau", "best_f1", "coverage_at_best"])
    w.writeheader()
    for r in results: w.writerow(r)
print(f"\n  Summary → {summary_csv}")

# ───────────────────── Phase 4: Visuals ──────────────────────────────
print(f"\n=== PHASE 4: visuals ===")

# Histograms (mature vs immature per heuristic).
n_h   = len(HEURISTICS)
ncols = 4
nrows = (n_h + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
axes = axes.flatten()
for i, h in enumerate(HEURISTICS):
    ax = axes[i]
    mat_vals = np.array([r[h] for r in primitive_log if r["mature"]])
    imm_vals = np.array([r[h] for r in primitive_log if not r["mature"]])
    mat_vals = mat_vals[np.isfinite(mat_vals)]
    imm_vals = imm_vals[np.isfinite(imm_vals)]
    if len(mat_vals)+len(imm_vals) == 0:
        ax.set_title(f"{h}\n(no data)"); continue
    all_v = np.concatenate([mat_vals, imm_vals])
    lo, hi = float(all_v.min()), float(all_v.max())
    if lo == hi: lo -= 0.5; hi += 0.5
    bins = np.linspace(lo, hi, 30)
    ax.hist(imm_vals, bins=bins, alpha=0.5, color="#d62728",
            label=f"immature (n={len(imm_vals)})")
    ax.hist(mat_vals, bins=bins, alpha=0.7, color="#2ca02c",
            label=f"mature (n={len(mat_vals)})")
    # Cohen's d
    if len(mat_vals) > 1 and len(imm_vals) > 1:
        pooled = float(np.sqrt(
            (np.var(mat_vals, ddof=1) + np.var(imm_vals, ddof=1)) / 2))
        d = (float(mat_vals.mean()) - float(imm_vals.mean())) / max(pooled, 1e-9)
        ax.set_title(f"{h}  (d={d:+.2f})", fontsize=10)
    else:
        ax.set_title(h, fontsize=10)
    ax.set_ylabel("# primitives", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)
for j in range(n_h, len(axes)): axes[j].set_visible(False)
fig.suptitle("Primitive heuristics — mature vs immature distributions "
             "(d = Cohen's effect size)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR, "heuristic_histograms.png"),
            dpi=120, bbox_inches="tight")
plt.close()
print(f"  Histograms → {OUT_DIR}/heuristic_histograms.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
for r in results:
    s = np.array([row[r["heuristic"]] for row in primitive_log])
    fprs, tprs, auc = _roc_auc(y, s)
    ax.plot(fprs, tprs, linewidth=1.5,
            label=f"{r['heuristic']:30s} AUC={auc:.3f}")
ax.plot([0,1],[0,1], color="gray", linestyle="--", linewidth=0.7)
ax.set_xlabel("False positive rate (impure clusters)")
ax.set_ylabel("True positive rate (correctly classified)")
ax.set_title("ROC — primitive maturity from each context-tree heuristic",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "heuristic_roc.png"),
            dpi=120, bbox_inches="tight")
plt.close()
print(f"  ROC → {OUT_DIR}/heuristic_roc.png")

# Threshold sweep — F1 vs τ for each heuristic (normalised τ axis)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
axes = axes.flatten()
for i, h in enumerate(HEURISTICS):
    ax = axes[i]
    s = np.array([r[h] for r in primitive_log])
    taus, Ps, Rs, Fs, best_tau, best_f1 = _threshold_sweep(y, s)
    if len(taus) == 0: continue
    ax.plot(taus, Ps, color="#1f77b4", label="precision", linewidth=1.5)
    ax.plot(taus, Rs, color="#2ca02c", label="recall",    linewidth=1.5)
    ax.plot(taus, Fs, color="#d62728", label="F1",        linewidth=2)
    ax.axvline(best_tau, color="black", linestyle="--", linewidth=0.7,
               label=f"τ*={best_tau:.2g} F1={best_f1:.2f}")
    ax.set_title(h, fontsize=10)
    ax.set_xlabel("τ", fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
for j in range(n_h, len(axes)): axes[j].set_visible(False)
fig.suptitle("Threshold sweep — P / R / F1 of mature-prediction vs τ",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR, "threshold_sweep.png"),
            dpi=120, bbox_inches="tight")
plt.close()
print(f"  Threshold sweep → {OUT_DIR}/threshold_sweep.png")

# ───────────────────── Phase 5: Winner ───────────────────────────────
print(f"\n=== PHASE 5: choose winner ===")

# Composite score: AUC × F1 × min(coverage, 1−coverage)→0 penalises
# trivial gates that admit everything / nothing. Use plain (AUC, F1)
# instead — simpler and easier to defend.
ranked = sorted(results,
                key=lambda r: (round(r["auc"], 3), round(r["best_f1"], 3)),
                reverse=True)
winner = ranked[0]
print(f"\n  >>> Winner: {winner['heuristic']}")
print(f"      AUC               : {winner['auc']:.3f}")
print(f"      Best τ            : {winner['best_tau']}")
print(f"      F1 at best τ      : {winner['best_f1']:.3f}")
print(f"      Coverage at best τ: {winner['coverage_at_best']:.2f}")

# Coverage at a few common admission rates for the winner — useful for
# picking a less-restrictive operating point when downstream parsing
# prefers recall over precision.
ws    = np.array([row[winner["heuristic"]] for row in primitive_log])
ws    = ws[np.isfinite(ws)]
print(f"\n  Winner τ for admission rates (use these for a recall-leaning gate):")
for cov_target in [0.50, 0.70, 0.80, 0.90, 0.95]:
    # τ that admits ~cov_target fraction
    if len(ws) == 0:
        break
    tau = float(np.quantile(ws, 1 - cov_target))
    admitted = float((ws > tau).mean())
    pred = (ws > tau).astype(int)
    yarr = np.array([r["mature"] for r in primitive_log
                     if np.isfinite(r[winner["heuristic"]])])
    tp = int(((pred == 1) & (yarr == 1)).sum())
    fp = int(((pred == 1) & (yarr == 0)).sum())
    fn = int(((pred == 0) & (yarr == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    print(f"    cov≈{cov_target:.2f}  τ={tau:+.3f}  admit={admitted:.2f}  "
          f"prec={prec:.2f}  rec={rec:.2f}")

with open(os.path.join(OUT_DIR, "winner.txt"), "w") as f:
    f.write(f"heuristic = {winner['heuristic']}\n")
    f.write(f"threshold = {winner['best_tau']}\n")
    f.write(f"auc       = {winner['auc']:.4f}\n")
    f.write(f"f1        = {winner['best_f1']:.4f}\n")
    f.write(f"coverage  = {winner['coverage_at_best']:.4f}\n")
    f.write("\nFull ranking (by AUC, then F1):\n")
    for r in ranked:
        f.write(f"  {r['heuristic']:30s}  AUC={r['auc']:.3f}  "
                f"F1={r['best_f1']:.3f}  τ={r['best_tau']}\n")
    f.write("\nWinner thresholds for common admission rates:\n")
    for cov_target in [0.50, 0.70, 0.80, 0.90, 0.95]:
        if len(ws) == 0: break
        tau = float(np.quantile(ws, 1 - cov_target))
        f.write(f"  cov≈{cov_target:.2f}  τ={tau:+.3f}\n")
print(f"  winner.txt written")

print(f"\n=== DONE ===")
