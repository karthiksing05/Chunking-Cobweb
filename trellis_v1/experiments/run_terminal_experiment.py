"""
Terminal-experiment experiment: varies the lexicon size of TEST_GRAMMAR_MED
(low / med / high) while keeping non-terminal productions identical.
For each variant we generate 200 CFG-derived sentences and run
incremental training with **5 seeds**, averaging the learning curves
into mean ± std for a stable comparison.

Hypothesis: with more terminals, each surface form appears less often
in training, so the parser needs more sentences to converge on the
same parse-F1 level.

Variants (from ``util.cfg.LEXICON_VARIANTS``):
    low  — 11 terminals (2 Det, 3 N, 2 Adj, 2 V, 2 P)
    med  — 22 terminals (TEST_GRAMMAR_MED default)
    high — 39 terminals (4 Det, 12 N, 9 Adj, 8 V, 6 P)

Output:
    data/cfg_terminal_{low,med,high}/   # 200 sentences each
    confs/acs-26/terminal_experiment/
        {low,med,high}/
            seed_{13,17,23,42,100}/learning_curves.{csv,png}
            aggregated.csv
        comparison.png                  # 3-variant overlay, ±1 std

Usage:
    python confs/acs-26/run_terminal_experiment.py
"""
import os, sys, json, random, hashlib, shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import (make_grammar_variant, LEXICON_VARIANTS,
                      generate_with_merges)
from experiment_harness import (run_multi_seed_learning_curves,
                      plot_overlay_with_bands, SEEDS)

N_SENTENCES = 400
DATASET_SEED = 13   # only controls which sentences land in the corpus

# Context-forward ranker weights (calibrated on grammar_large).
# Same recipe as the grammar experiment.
RANK_W_CTX   = 1.0
RANK_W_CLIMB = 3.0
RANK_W_CNT   = 0.05

DATA_BASE = os.path.join(_ROOT, "data")
OUT_BASE  = os.path.join(_HERE, "terminal_experiment")

COLORS = {"low": "#2ca02c", "med": "#1f77b4", "high": "#d62728"}


def make_dataset(variant: str, out_dir: str, seed: int = DATASET_SEED):
    """Generate ``N_SENTENCES`` unique sentences for ``variant`` and
    write each as a hollow-style JSON. Returns ``(grammar, corpus)``."""
    sizes = LEXICON_VARIANTS[variant]
    grammar, corpus = make_grammar_variant(sizes)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    seen: set[str] = set()
    n_written = 0
    max_attempts = N_SENTENCES * 30
    attempts = 0
    while n_written < N_SENTENCES and attempts < max_attempts:
        attempts += 1
        text, merges = generate_with_merges(
            "S", grammar, flatten_at_parent=())
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        with open(os.path.join(out_dir, f"cfg_{h}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"sentence": text, "merges": merges},
                      f, indent=2)
        n_written += 1
    print(f"  {variant}: wrote {n_written} sentences "
          f"({attempts} attempts) into {out_dir}")
    return grammar, corpus


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"=== Terminal experiment — generating datasets (one per variant) ===")
    grammars = {}
    corpora  = {}
    n_terminals = {}
    for v in ["low", "med", "high"]:
        ds_dir = os.path.join(DATA_BASE, f"cfg_terminal_{v}")
        if os.path.exists(ds_dir):
            shutil.rmtree(ds_dir)
        g, c = make_dataset(v, ds_dir)
        grammars[v] = g
        corpora[v]  = c
        n_terminals[v] = len(c)

    print(f"\n=== Terminal experiment — {len(SEEDS)} seeds × {len(grammars)} variants "
          f"= {len(SEEDS)*len(grammars)} learning-curve runs ===")
    agg_by_variant = {}
    for v in ["low", "med", "high"]:
        ds_dir  = os.path.join(DATA_BASE, f"cfg_terminal_{v}")
        out_dir = os.path.join(OUT_BASE, v)
        print(f"\n--- variant: {v} ({n_terminals[v]} terminals) ---")
        agg_by_variant[v] = run_multi_seed_learning_curves(
            corpus_dir=ds_dir,
            out_base=out_dir,
            grammar=grammars[v],
            corpus=corpora[v],
            seeds=SEEDS,
            eval_every=10,
            n_gen_per_eval=20,
            primitives_first=0,
            maturity_gate=("climb_ancestor_count", 30),
            gate_mode="skip",
            content_alpha=1e-4,
            content_bl_alpha=10,
            context_bl_alpha=10,
            # FAITHFUL representation (no hints): sharper+longer context makes
            # the hierarchies separable; the sparse terminal lexicon inherits
            # POS support from distributional siblings in the context tree.
            context_alpha=1e-5,
            context_length=5,
            rank_mode="class_lp",
            parse_mode="greedy",
            # Paper-faithful content instance: bag-of-concepts + VISIBLE
            # complexity. NO edge-word/seam/child-class hints.
            content_pool_depth=4,
            content_top_k=3,
            content_drop_cplx=False,
            content_boundary_feat=None,
            content_seam_feat=None,
            content_child_class=False,
            # FAITHFUL generation: maturity-anchor + maturity-pool intermediary
            # generalization conditioned on context class.
            gen_anchor_mode="maturity",
            gen_anchor_tau=20,
            gen_pool_mode="mat",
            gen_pool_tau=50,
        )

    print(f"\n=== Building overlay comparison chart (mean ± std) ===")
    labels = {v: f"{v}  ({n_terminals[v]} terminals)" for v in agg_by_variant}
    plot_overlay_with_bands(
        out_path=os.path.join(OUT_BASE, "comparison.png"),
        agg_by_variant=agg_by_variant,
        colors=COLORS,
        labels=labels,
        suptitle="TEST_GRAMMAR_MED — learning curves vs lexicon size",
        n_seeds=len(SEEDS),
    )


if __name__ == "__main__":
    main()
