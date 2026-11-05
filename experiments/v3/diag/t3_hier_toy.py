"""Hierarchy ablation on the toy grammars: inside-outside generation after 10 to 320
experiences under hierarchy two / shared / joint. The grammar scores, it never teaches.

Usage: t3_hier_toy.py <cfg json> <conds, comma separated> <seed> [checkpoints]
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.chart_gen import ChartGen  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, conds, seed = json.loads(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
checks = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [40, 80, 160, 320]
N_GEN = 200
for cond in conds:
    train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    g = CFG(target_grammar(cond))
    lens = sorted(len(e.tokens) for e in train)
    length = (lens[0], lens[-1])
    for mode in ("two", "shared", "joint"):
        L = Trellis3(seed=seed, hierarchy=mode, **cfg)
        done, t0 = 0, time.time()
        for n in checks:
            if n > len(train):
                break
            for e in train[done:n]:
                L.learn(e.tokens)
            done = n
            learn_s = time.time() - t0
            L._quiet = True
            CG = ChartGen(L, cut=64, ctx_cut=12, lam=4, word_sym="context", cut_by="bits", fill="both", length=length)
            L._quiet = False
            rng = random.Random(seed)
            S = [s for s in (CG.generate(["*"], rng) for _ in range(N_GEN)) if s]
            seen = {tuple(e.tokens) for e in train[:n]}
            gram = [s for s in S if g.recognizes(s)]
            print(json.dumps({"cond": cond, "seed": seed, "mode": mode, "n": n, "learn s": round(learn_s), "generated": len(S),
                              "grammatical": round(len(gram) / max(len(S), 1), 3),
                              "grammatical and new": round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3),
                              "examples": [" ".join(s) for s in S[:4]]}), flush=True)
