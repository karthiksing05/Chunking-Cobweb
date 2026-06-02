"""
met6 diagnostic — frequency-driven merge policy.

The convergence diag showed the climbing-ancestor gate is permissive
regardless of threshold (coverage stays 1.0), so the LOG-PROB RANKER sets
parse shape and drifts as counts grow → low convergence. This diag tests
the alternative: a DETERMINISTIC, frequency-driven merge
(FiniteParseTree.MERGE_POLICY = {"rank":"freq_basic","gate":"freq_basic",
"freq_min":k}) — merge the most-frequent recognizable chunk class first,
and only merge classes seen > k times. Frequency order is monotone, so
the grammar should stabilise (converge) while staying simple + recursive.

Sweeps freq_min, traces convergence per epoch, reports coverage /
recursion / simplicity (#chunk types).

Usage:
    PYTHONHASHSEED=0 python tests/met6/diag_freq_policy.py
"""
import os, sys, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
import unsupervised_grammar_formation as M
from parse_mh import FiniteParseTree, CompositeParseNode
from diag_convergence import distinct_chunk_types

FREQ_MINS = [1, 2, 4, 8]
N_TRAIN  = 40
N_EPOCHS = 8
SEED     = 13


def run_policy(freq_min, sents):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    FiniteParseTree.MERGE_POLICY = {
        "rank": "freq_basic", "gate": "freq_basic", "freq_min": freq_min}
    try:
        web = M.make_webster()
        prev = None
        conv = []
        # gate value passed to parse_sentence is ignored by the freq policy,
        # but still flows to build_primitives; keep it nominal.
        for epoch in range(N_EPOCHS):
            brk = {}
            for s in sents:
                pt = web.parse_sentence(
                    s, threshold=30, climb_count_threshold=30,
                    new_vocab=(epoch == 0), learning=True,
                    maturity_gate=M.MATURITY_GATE, gate_mode=M.GATE_MODE)
                brk[s] = M._bracket_set(pt)
            if prev is not None:
                conv.append(round(sum(1 for s in sents if brk[s] == prev[s]) / len(sents), 2))
            prev = brk
        cov = 0; depths = []; emb = 0
        for s in sents:
            pt = M.parse_no_learn(web, s, 30)
            kids = list(pt.global_root_node.children)
            if M.n_frontier_roots(pt) == 1 and kids and isinstance(kids[0][1], CompositeParseNode):
                cov += 1
            d, nc, se = M.tree_depth_and_selfembed(pt)
            depths.append(d)
            if se: emb += 1
        n_types = distinct_chunk_types(web, sents, 30)
        return {
            "freq_min": freq_min,
            "conv_traj": conv,
            "conv_final": conv[-1] if conv else 0.0,
            "coverage": round(cov / len(sents), 2),
            "mean_depth": round(float(np.mean(depths)), 1),
            "self_embed": round(emb / len(sents), 2),
            "n_types": n_types,
        }
    finally:
        FiniteParseTree.MERGE_POLICY = None   # restore legacy


def main():
    sents = M.gen_unique(N_TRAIN, seed=SEED, min_words=3)
    print(f"=== frequency-policy diagnostic — {len(sents)} sentences, {N_EPOCHS} epochs ===")
    print("    policy: rank=freq_basic gate=freq_basic (merge most-frequent chunk class first)\n")
    print(f"{'freqMin':>7} {'convFinal':>9} {'coverage':>8} {'meanD':>6} "
          f"{'selfEmb':>7} {'#types':>6}   conv-trajectory")
    for k in FREQ_MINS:
        r = run_policy(k, sents)
        print(f"{r['freq_min']:>7} {r['conv_final']:>9.2f} {r['coverage']:>8.2f} "
              f"{r['mean_depth']:>6.1f} {r['self_embed']:>7.2f} {r['n_types']:>6d}   "
              f"{r['conv_traj']}", flush=True)


if __name__ == "__main__":
    main()
