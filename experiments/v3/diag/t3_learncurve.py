"""Learning curves for TRELLIS v2: experiences are learned one at a time; at
each checkpoint, generation from the two hierarchies alone (generate_tree) is
scored (grammatical, grammatical and new against the experiences seen so far,
distinct), with held-out bracket omission as a side number and the size of the
two trees. Evaluation only: the target grammar scores, it never teaches.

Usage: t3_learncurve.py <tag> <cfg json> <conds> <seed> [checkpoints]
"""
import os, sys, json, random, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG, BracketTally
from trellis2.trellis3 import Trellis3

tag, cfg = sys.argv[1], json.loads(sys.argv[2])
conds = sys.argv[3].split(",")
seed = int(sys.argv[4])
checks = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [10, 20, 40, 80, 160, 320]
n_gen = 300


def size(tree):
    nodes, leaves, stack = 0, 0, [tree.root]
    while stack:
        x = stack.pop(); nodes += 1
        leaves += not x.children
        stack.extend(x.children)
    return nodes, leaves


for cond in conds:
    train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    g = CFG(target_grammar(cond))
    L = Trellis3(seed=seed, **cfg)
    t0, done = time.time(), 0
    for n in checks:
        if n > len(train):
            break
        for e in train[done:n]:
            L.learn(e.tokens)
        done = n
        learn_s = time.time() - t0
        seen = {tuple(e.tokens) for e in train[:n]}
        rng = random.Random(seed)
        ix = L.concept_index()
        S = [s for s in (L.generate_tree(rng, index=ix) for _ in range(n_gen)) if s]
        gram = [s for s in S if g.recognizes(s)]
        te = BracketTally()
        for e in test[:60]:
            te.add(e.tree, L.parse(e.tokens)["tree"])
        print(json.dumps({
            "tag": tag, "cond": cond, "seed": seed, "experiences": n,
            "generated": len(S),
            "grammatical": round(len(gram) / max(len(S), 1), 3),
            "grammatical and new": round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3),
            "distinct": round(len(set(map(tuple, S))) / max(len(S), 1), 3),
            "mean length": round(sum(map(len, S)) / max(len(S), 1), 2),
            "omission (side)": round(te.omission, 3),
            "context tree nodes/leaves": size(L.ctx), "content tree nodes/leaves": size(L.cnt),
            "learn seconds": round(learn_s)}), flush=True)
