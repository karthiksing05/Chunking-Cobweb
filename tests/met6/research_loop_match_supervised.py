"""
met6 research loop — find unsupervised settings as good as supervised.

The supervised-gap probe (diag_supervised_gap.py) showed:
  - SMALL: unsupervised τ=5 already matches supervised (F1 98.6 vs 100,
    gen 100 vs 100).
  - generation grammaticality: unsupervised ≥ supervised on both grammars.
  - MED parse-F1-vs-gold lags (40–46 vs 84.6) and τ interacts strongly
    with grammar complexity (τ=5 helps SMALL, hurts MED).

This loop sweeps τ finely (and lets you vary n_train / epochs) per
grammar, prints each config against the supervised reference, and reports
the best-F1 and best-balanced unsupervised config — the settings to feed
gen_learn_test.

Usage:
    PYTHONHASHSEED=0 python tests/met6/research_loop_match_supervised.py
"""
import os, sys, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL,
                      TEST_GRAMMAR_MED, TEST_CORPUS_MED)
import diag_supervised_gap as G

# Per grammar: the τ grid to search. MED gets a finer/wider grid since it
# is the hard case; SMALL just confirms the τ=5 match.
SWEEP = {
    "SMALL": dict(grammar=TEST_GRAMMAR_SMALL, corpus=TEST_CORPUS_SMALL,
                  taus=[3, 5, 8], n_train=150, epochs=6, flatten=("VP",)),
    "MED":   dict(grammar=TEST_GRAMMAR_MED, corpus=TEST_CORPUS_MED,
                  taus=[2, 3, 4, 6, 10, 15], n_train=150, epochs=6, flatten=("VP",)),
}


def run_grammar(gname, cfg):
    grammar, corpus = cfg["grammar"], cfg["corpus"]
    recog, w2p = G._build_cyk_recognizer(grammar)
    data = G.gen_corpus(grammar, cfg["n_train"] + 40, seed=G.SEED, flatten=cfg["flatten"])
    random.seed(G.SEED); random.shuffle(data)
    train, test = data[:cfg["n_train"]], data[cfg["n_train"]:cfg["n_train"] + 40]
    train_set = {h["sentence"].strip() for h in train}

    scratch = G.make_webster(corpus)
    for h in data:
        for tok in __import__("re").findall(r"[\w']+|[.,!?;]", h["sentence"]):
            scratch.ltm.add_to_vocab(tok)
    gold = {h["sentence"]: G.gold_brackets(scratch, h) for h in test}

    web_s, trees_s = G.train_supervised(corpus, train)
    sup = G.eval_model(web_s, test, gold, 30, recog, w2p, train_set, trees_s)
    print(f"\n=== {gname} (n_train={cfg['n_train']}, epochs={cfg['epochs']}) ===")
    print(f"  {'SUPERVISED':>14} {'—':>3}  F1={100*sup['F1']:5.1f}  EM={100*sup['EM']:5.1f}  "
          f"gen={100*sup['gen_gram']:5.1f}  novel={100*sup['gen_novel']:5.1f}   [reference]")

    results = []
    for tau in cfg["taus"]:
        web_u, trees_u = G.train_unsupervised(corpus, train, tau, cfg["epochs"])
        r = G.eval_model(web_u, test, gold, tau, recog, w2p, train_set, trees_u)
        r["tau"] = tau
        results.append(r)
        # closeness to supervised: how much of supervised F1 + gen we reach
        f1_ratio = r["F1"] / max(sup["F1"], 1e-9)
        gen_ratio = r["gen_gram"] / max(sup["gen_gram"], 1e-9)
        flag = "  ← matches sup" if (f1_ratio >= 0.95 and gen_ratio >= 0.95) else ""
        print(f"  {'UNSUPERVISED':>14} {tau:>3}  F1={100*r['F1']:5.1f}  EM={100*r['EM']:5.1f}  "
              f"gen={100*r['gen_gram']:5.1f}  novel={100*r['gen_novel']:5.1f}   "
              f"(F1 {100*f1_ratio:4.0f}% of sup, gen {100*gen_ratio:4.0f}%){flag}", flush=True)

    best_f1 = max(results, key=lambda r: r["F1"])
    # balanced = harmonic-ish of F1 and gen relative to supervised
    best_bal = max(results, key=lambda r: min(r["F1"] / max(sup["F1"], 1e-9),
                                              r["gen_gram"] / max(sup["gen_gram"], 1e-9)))
    print(f"  → best F1:       τ={best_f1['tau']}  F1={100*best_f1['F1']:.1f}  gen={100*best_f1['gen_gram']:.1f}")
    print(f"  → best balanced: τ={best_bal['tau']}  F1={100*best_bal['F1']:.1f}  gen={100*best_bal['gen_gram']:.1f}")
    return sup, results, best_f1, best_bal


def main():
    print("=== met6 research loop — matching the supervised reference ===")
    for gname, cfg in SWEEP.items():
        run_grammar(gname, cfg)


if __name__ == "__main__":
    main()
