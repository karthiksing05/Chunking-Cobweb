"""
Grammar-experiment experiment: holds the parser fixed and varies the
GRAMMAR's structural complexity (SMALL → MED → LARGE). For each
grammar we generate 200 CFG-derived sentences and run incremental
training with **5 seeds**, averaging the learning curves into
mean ± std for a stable comparison.

Hypothesis: parser F1 degrades monotonically with grammar complexity
(more chunk types per sentence → fewer training examples per pattern).

Variants (from ``util.cfg``):
    small — TEST_GRAMMAR_SMALL (=TEST_GRAMMAR2)
            S→NP VP; NP=Det N; VP=V (NP); 4 N, 4 V, 2 Det
    med   — TEST_GRAMMAR_MED   (=TEST_GRAMMAR1)
            + AdjP recursion, + PP, + ditransitive V NP PP
    large — TEST_GRAMMAR_LARGE (=TEST_GRAMMAR3)
            + RelClause + RelPro POS class

Output:
    data/cfg_grammar_experiment_{small,med,large}/   # 200 sentences each
    confs/acs-26/grammar_experiment/
        {small,med,large}/
            seed_{13,17,23,42,100}/learning_curves.{csv,png}
            aggregated.csv
        comparison.png                          # 3-variant overlay, ±1 std

Usage:
    python confs/acs-26/run_grammar_experiment.py
"""
import os, sys, json, random, hashlib, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL,
                      TEST_GRAMMAR_MED,   TEST_CORPUS_MED,
                      TEST_GRAMMAR_LARGE, TEST_CORPUS_LARGE,
                      generate_with_merges)
from experiment_harness import (run_multi_seed_learning_curves,
                      plot_overlay_with_bands, SEEDS)


# Each variant: (grammar, corpus, flatten_at_parent, max_words).
# flatten_at_parent=() keeps every grammar non-terminal as its own
# bracket in the gold hollow tree, so the parser learns the
# grammar's actual constituent structure rather than the flattened
# May-30 hollow convention (which dissolved VP and VPobj into the
# running S, producing spurious (NP V) intermediate brackets that
# have no constituent home in the grammar).
VARIANTS = {
    "small": (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL, (),  8),
    "med":   (TEST_GRAMMAR_MED,   TEST_CORPUS_MED,   (), 10),
    "large": (TEST_GRAMMAR_LARGE, TEST_CORPUS_LARGE, (), 10),
}

N_SENTENCES = 400
DATASET_SEED = 13

# Context-forward ranker weights (calibrated on grammar_large).
# score = w_ctx*ctx_leaf_lp + w_climb*log(1+climb_count) + w_cnt*root_log_prob
RANK_W_CTX   = 1.0
RANK_W_CLIMB = 3.0
RANK_W_CNT   = 0.05

DATA_BASE = os.path.join(_ROOT, "data")
OUT_BASE  = os.path.join(_HERE, "grammar_experiment")

COLORS = {"small": "#2ca02c", "med": "#1f77b4", "large": "#d62728"}
GRAMMAR_DESC = {
    "small": "S→NP VP; VP=V NP",
    "med":   "+AdjP, +PP (S→S PP)",
    "large": "+RelClause",
}


def make_dataset(variant: str, out_dir: str, seed: int = DATASET_SEED):
    """Generate ``N_SENTENCES`` unique sentences for ``variant``."""
    grammar, corpus, flatten, max_words = VARIANTS[variant]
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    seen: set[str] = set()
    n_written = 0
    n_too_long = 0
    max_attempts = N_SENTENCES * 60
    attempts = 0
    while n_written < N_SENTENCES and attempts < max_attempts:
        attempts += 1
        text, merges = generate_with_merges(
            "S", grammar, flatten_at_parent=flatten)
        text = text.strip()
        if not text or text in seen:
            continue
        if len(text.split()) > max_words:
            n_too_long += 1
            continue
        seen.add(text)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        with open(os.path.join(out_dir, f"cfg_{h}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"sentence": text, "merges": merges},
                      f, indent=2)
        n_written += 1
    print(f"  {variant}: wrote {n_written} sentences "
          f"(max_words={max_words}, {attempts} attempts, "
          f"{n_too_long} rejected for length) → {out_dir}")
    return grammar, corpus


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"=== Grammar experiment — generating datasets (one per variant) ===")
    grammars = {}
    corpora  = {}
    for v in ["small", "med", "large"]:
        ds_dir = os.path.join(DATA_BASE, f"cfg_grammar_experiment_{v}")
        if os.path.exists(ds_dir):
            shutil.rmtree(ds_dir)
        g, c = make_dataset(v, ds_dir)
        grammars[v] = g
        corpora[v]  = c

    print(f"\n=== Grammar experiment — {len(SEEDS)} seeds × {len(VARIANTS)} variants "
          f"= {len(SEEDS)*len(VARIANTS)} learning-curve runs ===")
    agg_by_variant = {}
    for v in ["small", "med", "large"]:
        ds_dir  = os.path.join(DATA_BASE, f"cfg_grammar_experiment_{v}")
        out_dir = os.path.join(OUT_BASE, v)
        print(f"\n--- variant: {v} ({GRAMMAR_DESC[v]}) ---")
        agg_by_variant[v] = run_multi_seed_learning_curves(
            corpus_dir=ds_dir,
            out_base=out_dir,
            grammar=grammars[v],
            corpus=corpora[v],
            seeds=SEEDS,
            eval_every=10,
            n_gen_per_eval=20,
            primitives_first=0,
            # Climbing-ancestor recognition threshold (paper τ=30).
            maturity_gate=("climb_ancestor_count", 30),
            gate_mode="skip",
            content_alpha=1e-4,
            content_bl_alpha=10,
            context_bl_alpha=10,
            # FAITHFUL representation (no hints): sharper + longer context
            # (α=1e-5, len 5) makes the two hierarchies cleanly separable, so
            # the paper's bag-of-concepts content instance + class-conditional
            # recognition score reach the F1 target without edge-word/seam/
            # child-class enrichments. See project_faithful_representation_push.
            context_alpha=1e-5,
            context_length=5,
            rank_mode="class_lp",
            parse_mode="greedy",
            # Paper-faithful content instance: bag-of-concepts (pool depth 4,
            # top-k 3) + VISIBLE child complexity (load-bearing structural axis
            # separating NP=Det+N from S=NP+VP). NO edge-word/seam/child-class
            # hints.
            content_pool_depth=4,
            content_top_k=3,
            content_drop_cplx=False,
            content_boundary_feat=None,
            content_seam_feat=None,
            content_child_class=False,
            # FAITHFUL generation: P9 anchor at a well-supported (maturity)
            # level rather than max-EPMI basic, and P8/P10 recombination from a
            # maturity-level intermediary GENERALIZATION pool conditioned on the
            # child's context class — grammatical AND genuinely novel.
            gen_anchor_mode="maturity",
            gen_anchor_tau=20,
            gen_pool_mode="mat",
            gen_pool_tau=50,
        )

    print(f"\n=== Building overlay comparison chart (mean ± std) ===")
    labels = {v: f"{v} — {GRAMMAR_DESC[v]}" for v in agg_by_variant}
    plot_overlay_with_bands(
        out_path=os.path.join(OUT_BASE, "comparison.png"),
        agg_by_variant=agg_by_variant,
        colors=COLORS,
        labels=labels,
        suptitle="TEST_GRAMMAR experiment — learning curves vs grammar complexity",
        n_seeds=len(SEEDS),
    )


if __name__ == "__main__":
    main()
