"""
met6 diagnostic — WHY won't the unsupervised grammar converge?

Baseline finding: coverage / determinism / recursion / round-trip are all
healthy across τ, but CONVERGENCE (fraction of corpus whose bracketing
stops changing between epochs) is low and noisy (0.13–0.50). The grammar
parses fully and recursively but never *stabilises*.

Hypothesis: an ABSOLUTE count gate (node.count > τ) is never actually
selective once candidate counts accumulate (coverage stays 1.0 even at
τ=80), so the parse SHAPE is decided entirely by the ranker, whose
log-probs drift as counts grow → churn. A RELATIVE-support gate
(node.count / root.count > ρ) keeps selectivity constant as the tree
grows, so the committed structure reflects settled statistics and should
converge.

This diag traces convergence PER EPOCH over more epochs for a mix of
absolute (τ≥1) and relative (0<ρ<1) gate values, and reports a
simplicity measure (# distinct committed chunk types).

Usage:
    PYTHONHASHSEED=0 python tests/met6/diag_convergence.py
"""
import os, sys, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
import unsupervised_grammar_formation as M
from parse_mh import CompositeParseNode

# gate values: absolute ints AND relative fractions
GATES = [3, 30, 0.30, 0.15, 0.07, 0.03]
N_TRAIN  = 40
N_EPOCHS = 8
SEED     = 13


def label_of(n):
    lab = getattr(n, "label", None)
    if isinstance(lab, dict) and lab:
        return next(iter(lab.keys()))
    return None


def distinct_chunk_types(web, sents, gate):
    """# distinct committed chunk types (context-concept labels) across the
    corpus — a simplicity proxy. Fewer = simpler grammar."""
    types = set()
    for s in sents:
        pt = M.parse_no_learn(web, s, gate)
        def rec(n):
            if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
                lab = label_of(n)
                if lab is not None:
                    types.add(lab)
            for _, ch in getattr(n, "children", []):
                rec(ch)
        for _, ch in pt.global_root_node.children:
            rec(ch)
    return len(types)


def run_gate(gate, sents):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = M.make_webster()
    prev = None
    conv_traj = []
    for epoch in range(N_EPOCHS):
        brk = {}
        for s in sents:
            pt = web.parse_sentence(
                s, threshold=30, climb_count_threshold=gate,
                new_vocab=(epoch == 0), learning=True,
                maturity_gate=M.MATURITY_GATE, gate_mode=M.GATE_MODE)
            brk[s] = M._bracket_set(pt)
        if prev is not None:
            same = sum(1 for s in sents if brk[s] == prev[s])
            conv_traj.append(round(same / len(sents), 2))
        prev = brk
    # final-state metrics
    cov = 0; depths = []; emb = 0
    for s in sents:
        pt = M.parse_no_learn(web, s, gate)
        if M.n_frontier_roots(pt) == 1:
            kids = list(pt.global_root_node.children)
            if kids and isinstance(kids[0][1], CompositeParseNode):
                cov += 1
        d, nc, se = M.tree_depth_and_selfembed(pt)
        depths.append(d)
        if se: emb += 1
    n_types = distinct_chunk_types(web, sents, gate)
    return {
        "gate": gate,
        "kind": "rel" if 0 < gate < 1 else "abs",
        "conv_traj": conv_traj,
        "conv_final": conv_traj[-1] if conv_traj else 0.0,
        "coverage": round(cov / len(sents), 2),
        "mean_depth": round(float(np.mean(depths)), 1),
        "self_embed": round(emb / len(sents), 2),
        "n_chunk_types": n_types,
        "root_count": round(float(web.ltm.content_hierarchy.root.count)),
    }


def main():
    sents = M.gen_unique(N_TRAIN, seed=SEED, min_words=3)
    print(f"=== convergence diagnostic — {len(sents)} sentences, {N_EPOCHS} epochs ===\n")
    print(f"{'gate':>7} {'kind':>4} {'convFinal':>9} {'coverage':>8} "
          f"{'meanD':>6} {'selfEmb':>7} {'#types':>6} {'rootCnt':>8}   conv-trajectory")
    for g in GATES:
        r = run_gate(g, sents)
        print(f"{str(r['gate']):>7} {r['kind']:>4} {r['conv_final']:>9.2f} "
              f"{r['coverage']:>8.2f} {r['mean_depth']:>6.1f} {r['self_embed']:>7.2f} "
              f"{r['n_chunk_types']:>6d} {r['root_count']:>8d}   {r['conv_traj']}",
              flush=True)


if __name__ == "__main__":
    main()
