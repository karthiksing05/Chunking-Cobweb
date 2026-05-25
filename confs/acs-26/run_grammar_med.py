"""
confs/acs-26 / Grammar MED (TEST_GRAMMAR_MED)
=============================================

Run the full ACS-26 evaluation against the synthetic CFG-derived
dataset under ``data/cfg_grammar_med/`` (200 sentences whose merges
come straight from the CFG derivation via
``util.cfg.generate_with_merges``). Both the hollow_learn pipeline
and the learning-curves test are run with **5 seeds** (13, 17, 23,
42, 100) and aggregated as mean ± std so the reported numbers reflect
real model behaviour, not lucky train/test splits.

Outputs (all under ``confs/acs-26/grammar_med/``):

    hollow_learn/
        seed_{13,17,23,42,100}/         per-seed pipeline outputs
        hollow_learn_summary.csv        aggregated F1 / EM / step-pick
    learning_curves/
        seed_{13,17,23,42,100}/         per-seed learning curve runs
        aggregated.csv                  mean / std per training step
        learning_curves.png             chart with ±1 std bands

Usage:
    python confs/acs-26/run_grammar_med.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import TEST_GRAMMAR_MED, TEST_CORPUS_MED
from _helpers import (run_multi_seed_hollow_learn,
                      run_multi_seed_learning_curves,
                      plot_learning_curves_with_band, SEEDS)

CORPUS_DIR = os.path.join(_ROOT, "data", "cfg_grammar_med")
OUT_BASE   = os.path.join(_HERE, "grammar_med")
HL_DIR     = os.path.join(OUT_BASE, "hollow_learn")
LC_DIR     = os.path.join(OUT_BASE, "learning_curves")

print(f"=== ACS-26 / TEST_GRAMMAR_MED — 5-seed hollow_learn ===")
hl_agg = run_multi_seed_hollow_learn(
    corpus_dir=CORPUS_DIR,
    out_base=HL_DIR,
    grammar=TEST_GRAMMAR_MED,
    corpus=TEST_CORPUS_MED,
    seeds=SEEDS,
)

print(f"\n=== ACS-26 / TEST_GRAMMAR_MED — 5-seed learning curves ===")
lc_agg = run_multi_seed_learning_curves(
    corpus_dir=CORPUS_DIR,
    out_base=LC_DIR,
    grammar=TEST_GRAMMAR_MED,
    corpus=TEST_CORPUS_MED,
    seeds=SEEDS,
)
plot_learning_curves_with_band(
    out_path=os.path.join(LC_DIR, "learning_curves.png"),
    agg=lc_agg,
    title="TEST_GRAMMAR_MED — learning curves",
    n_seeds=len(SEEDS),
)
