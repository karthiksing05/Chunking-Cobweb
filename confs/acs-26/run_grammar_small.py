"""
confs/acs-26 / Grammar SMALL (TEST_GRAMMAR_SMALL)
=================================================

Same shape as ``run_grammar_med.py`` but for ``TEST_GRAMMAR_SMALL`` —
the minimal grammar (S→NP VP; VP=V (NP); 4 N, 4 V, 2 Det). Both the
hollow_learn pipeline and the learning-curves test are run with **5
seeds** and aggregated as mean ± std.

Outputs (all under ``confs/acs-26/grammar_small/``):

    hollow_learn/
        seed_{13,17,23,42,100}/         per-seed pipeline outputs
        hollow_learn_summary.csv        aggregated F1 / EM / step-pick
    learning_curves/
        seed_{13,17,23,42,100}/         per-seed learning curve runs
        aggregated.csv                  mean / std per training step
        learning_curves.png             chart with ±1 std bands

Usage:
    python confs/acs-26/run_grammar_small.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL
from _helpers import (run_multi_seed_hollow_learn,
                      run_multi_seed_learning_curves,
                      plot_learning_curves_with_band, SEEDS)

CORPUS_DIR = os.path.join(_ROOT, "data", "cfg_grammar_small")
OUT_BASE   = os.path.join(_HERE, "grammar_small")
HL_DIR     = os.path.join(OUT_BASE, "hollow_learn")
LC_DIR     = os.path.join(OUT_BASE, "learning_curves")

# Primitive-maturity gate (chosen via tests/primitive_threshold).
# Replaces the legacy ``primitives_first=200`` warm-up with a per-
# primitive decision: admit a primitive only if its context-tree
# root log-prob exceeds the threshold.
# τ=-12.0 from the τ sweeps on med (diag_tau_sweep) and large
# (diag_tau_sweep_large) — it's the single value that works across
# all three grammars: matches/beats baseline F1 on small/med/large
# while keeping admission rates >95%. τ=-10.0 was best on med but
# rejected too many primitives in the larger-vocab large grammar.
MATURITY_GATE = ("root_log_prob", -12.0)
GATE_MODE     = "skip"        # individual immature primitives are dropped

print(f"=== ACS-26 / TEST_GRAMMAR_SMALL — 5-seed hollow_learn ===")
print(f"    primitives_first=0, maturity_gate={MATURITY_GATE}, mode={GATE_MODE}")
hl_agg = run_multi_seed_hollow_learn(
    corpus_dir=CORPUS_DIR,
    out_base=HL_DIR,
    grammar=TEST_GRAMMAR_SMALL,
    corpus=TEST_CORPUS_SMALL,
    seeds=SEEDS,
    viz_dir=OUT_BASE,
    primitives_first=0,
    maturity_gate=MATURITY_GATE,
    gate_mode=GATE_MODE,
)

print(f"\n=== ACS-26 / TEST_GRAMMAR_SMALL — 5-seed learning curves ===")
lc_agg = run_multi_seed_learning_curves(
    corpus_dir=CORPUS_DIR,
    out_base=LC_DIR,
    grammar=TEST_GRAMMAR_SMALL,
    corpus=TEST_CORPUS_SMALL,
    seeds=SEEDS,
    primitives_first=0,
    maturity_gate=MATURITY_GATE,
    gate_mode=GATE_MODE,
)
plot_learning_curves_with_band(
    out_path=os.path.join(LC_DIR, "learning_curves.png"),
    agg=lc_agg,
    title="TEST_GRAMMAR_SMALL — learning curves",
    n_seeds=len(SEEDS),
)
