"""Round 215: train the paper-recipe learner on the first n kept sentences of a seed's stream, then parse every control sentence of that seed
(r215_sets.py) and store the parse scores and the chosen tree. Parsing does not change the learner.

Scores per sentence (L.parse):
  logZ      the log inside weight over all binary analyses, factors 2^(gain/tau) (the quantity generate_best's critic and r182 use)
  vit       the best analysis' total gain in bits (the chart's Viterbi value, _chart best[(0, n)])
  chosen    the chosen (MBR) tree's total gain, and its smallest and mean join gain
Usage (repo root, compute node, env trellis):  r215_parse.py <seed> <n>
"""
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402


def bracket(chosen, n):
    split = {(i, j): k for i, k, j in chosen}

    def f(i, j):
        if j - i == 1:
            return str(i)
        k = split.get((i, j))
        if k is None:
            return "(" + " ".join(f(x, x + 1) for x in range(i, j)) + ")"
        return "(" + f(i, k) + " " + f(k, j) + ")"
    return f(0, n)


def main(seed, n):
    train, _, _, _ = C.stream(seed, n)
    cfg = dict(C.CFG, window=C.WINDOW[n])
    L = Trellis3(seed=seed, **cfg)
    t0 = time.time()
    for s, fol in train:
        L.learn(s, follows=fol)
    learn_s = time.time() - t0
    print(json.dumps({"seed": seed, "n": n, "window": C.WINDOW[n], "learn s": round(learn_s), "records": len(L.records)}), flush=True)
    L._quiet = True
    S = C.load_sets(seed)
    whole = {tuple(s) for s, _ in train}
    res, t1 = {}, time.time()
    for name, sents in C.all_sentences(S).items():
        for line in sents:
            if line in res:
                continue
            w = line.split()
            r = L.parse(w)
            ch = r["chosen"]
            gains = [r["comps"][c].gain for c in ch]
            res[line] = {"len": len(w), "logZ": r["logZ"], "vit": L._best[(0, len(w))][0], "chosen_sum": sum(gains),
                         "chosen_min": min(gains) if gains else 0.0, "chosen_mean": sum(gains) / len(gains) if gains else 0.0,
                         "tree": bracket(ch, len(w)), "stored": tuple(w) in whole}
    print(json.dumps({"parsed": len(res), "parse s": round(time.time() - t1)}), flush=True)
    os.makedirs(C.RAW, exist_ok=True)
    json.dump({"seed": seed, "n": n, "window": C.WINDOW[n], "learn_s": learn_s, "tau": getattr(L, "tau", None), "res": res},
              open(os.path.join(C.RAW, "trellis_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
