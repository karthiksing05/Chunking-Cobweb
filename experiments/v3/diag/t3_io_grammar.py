"""Where does inside-outside generation lose grammar on the toy grammars? One memory per grammar and
seed; each arm changes one mechanism against the base (whole tree, words within 12 bits, lam 4, fill
both). The grammar scores, it never teaches.

Usage: t3_io_grammar.py <cfg json> <conds, comma separated> <seed> <n experiences> [n generated]
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

cfg, conds, seed, n = json.loads(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
n_gen = int(sys.argv[5]) if len(sys.argv) > 5 else 150
BASE = {"cut": 64, "ctx_cut": 12, "lam": 4, "word_sym": "context", "cut_by": "bits", "fill": "both"}
ARMS = [("base", {}),
        ("lam 0.25", {"lam": 0.25}), ("lam 1", {"lam": 1}), ("lam 16", {"lam": 16}),
        ("words 8 bits", {"ctx_cut": 8}), ("words 10 bits", {"ctx_cut": 10}), ("words 16 bits", {"ctx_cut": 16}), ("words whole tree", {"ctx_cut": 99}),
        ("content 0 bits", {"cut": 0}), ("content 8 bits", {"cut": 8}), ("content 16 bits", {"cut": 16}),
        ("marginal only", {"fill": "chart"})]
for cond in conds:
    train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    g = CFG(target_grammar(cond))
    lens = sorted(len(e.tokens) for e in train)
    length = (lens[0], lens[-1])
    L = Trellis3(seed=seed, **cfg)
    t0 = time.time()
    for e in train[:n]:
        L.learn(e.tokens)
    L._quiet = True
    learn_s = round(time.time() - t0)
    seen = {tuple(e.tokens) for e in train[:n]}
    for name, change in ARMS:
        t0 = time.time()
        CG = ChartGen(L, length=length, **{**BASE, **change})
        rng = random.Random(seed)
        S = [s for s in (CG.generate(["*"], rng) for _ in range(n_gen)) if s]
        gram = [s for s in S if g.recognizes(s)]
        bad = [" ".join(s) for s in S if not g.recognizes(s)]
        print(json.dumps({"cond": cond, "seed": seed, "n": n, "arm": name, "learn s": learn_s, "gen s": round(time.time() - t0),
                          "generated": len(S), "grammatical": round(len(gram) / max(len(S), 1), 3),
                          "grammatical and new": round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3),
                          "distinct": round(len({tuple(s) for s in S}) / max(len(S), 1), 3),
                          "ungrammatical examples": bad[:6], "stats": CG.stats}), flush=True)
