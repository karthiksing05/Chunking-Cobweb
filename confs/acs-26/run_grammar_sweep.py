"""
Grammar-sweep experiment: holds the parser fixed and varies the
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
    data/cfg_grammar_sweep_{small,med,large}/   # 200 sentences each
    confs/acs-26/grammar_sweep/
        {small,med,large}/
            seed_{13,17,23,42,100}/learning_curves.{csv,png}
            aggregated.csv
        comparison.png                          # 3-variant overlay, ±1 std

Usage:
    python confs/acs-26/run_grammar_sweep.py
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
from _helpers import (run_multi_seed_learning_curves,
                      plot_overlay_with_bands, SEEDS)


# Each variant: (grammar, corpus, flatten_at_parent, max_words).
VARIANTS = {
    "small": (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL, ("VP",), 8),
    "med":   (TEST_GRAMMAR_MED,   TEST_CORPUS_MED,   ("VP",), 10),
    "large": (TEST_GRAMMAR_LARGE, TEST_CORPUS_LARGE, ("VP",), 10),
}

N_SENTENCES = 200
DATASET_SEED = 13
DATA_BASE = os.path.join(_ROOT, "data")
OUT_BASE  = os.path.join(_HERE, "grammar_sweep")

COLORS = {"small": "#2ca02c", "med": "#1f77b4", "large": "#d62728"}
GRAMMAR_DESC = {
    "small": "S→NP VP; VP=V (NP)",
    "med":   "+AdjP, PP, V NP PP",
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
    print(f"=== Grammar sweep — generating datasets (one per variant) ===")
    grammars = {}
    corpora  = {}
    for v in ["small", "med", "large"]:
        ds_dir = os.path.join(DATA_BASE, f"cfg_grammar_sweep_{v}")
        if os.path.exists(ds_dir):
            shutil.rmtree(ds_dir)
        g, c = make_dataset(v, ds_dir)
        grammars[v] = g
        corpora[v]  = c

    print(f"\n=== Grammar sweep — {len(SEEDS)} seeds × {len(VARIANTS)} variants "
          f"= {len(SEEDS)*len(VARIANTS)} learning-curve runs ===")
    agg_by_variant = {}
    for v in ["small", "med", "large"]:
        ds_dir  = os.path.join(DATA_BASE, f"cfg_grammar_sweep_{v}")
        out_dir = os.path.join(OUT_BASE, v)
        print(f"\n--- variant: {v} ({GRAMMAR_DESC[v]}) ---")
        agg_by_variant[v] = run_multi_seed_learning_curves(
            corpus_dir=ds_dir,
            out_base=out_dir,
            grammar=grammars[v],
            corpus=corpora[v],
            seeds=SEEDS,
            eval_every=10,
        )

    print(f"\n=== Building overlay comparison chart (mean ± std) ===")
    labels = {v: f"{v} — {GRAMMAR_DESC[v]}" for v in agg_by_variant}
    plot_overlay_with_bands(
        out_path=os.path.join(OUT_BASE, "comparison.png"),
        agg_by_variant=agg_by_variant,
        colors=COLORS,
        labels=labels,
        suptitle="TEST_GRAMMAR sweep — learning curves vs grammar complexity",
        n_seeds=len(SEEDS),
    )


if __name__ == "__main__":
    main()
