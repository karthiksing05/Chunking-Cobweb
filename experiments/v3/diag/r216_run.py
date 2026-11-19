"""Round 216: one (seed, n) job. Retrains the r215 learner, then for every r215 control sentence (and, for the cross-seed tree comparison, seed 13's
C1 sentences) stores
  default parse   logZ and chosen tree (reproduces r215: checked against r215_raw/trellis_s*_n*.json)
  gen steps       r215b's gated generator score (Gen.score, edge-exact, pair gate): per step (p, rank, support, fallback)
  split / chunk   logZ and chosen tree of Trellis3R216(r216="split" | "chunk", mu=1): the parser with the generator's conditionals in its weights
Usage (repo root, compute node, env trellis):  r216_run.py <seed> <n>
"""
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
import r215_parse as P  # noqa: E402
from r215b_gen import Gen  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402
from trellis2.trellis3_r216 import Trellis3R216  # noqa: E402

OUT = os.path.join(C.OUT, "r216_raw")


def main(seed, n):
    train, _, _, vocab = C.stream(seed, n)
    cfg = dict(C.CFG, window=C.WINDOW[n])
    L = Trellis3(seed=seed, **cfg)
    t0 = time.time()
    for s, fol in train:
        L.learn(s, follows=fol)
    print(json.dumps({"seed": seed, "n": n, "window": C.WINDOW[n], "learn s": round(time.time() - t0), "records": len(L.records)}), flush=True)
    L._quiet = True
    S = C.load_sets(seed)
    lines = []
    for ss in C.all_sentences(S).values():
        for l in ss:
            if l not in lines:
                lines.append(l)
    xl = [l for l in C.load_sets(13)["C1"] if l not in lines]      # the common set for cross-seed agreement (parse only)
    old = json.load(open(os.path.join(C.RAW, "trellis_s%d_n%d.json" % (seed, n)), encoding="utf-8"))["res"]
    res = {l: {"len": len(l.split())} for l in lines + xl}
    bad = 0
    t1 = time.time()
    for l in lines + xl:
        r = L.parse(l.split())
        res[l]["chosen"] = [list(c) for c in r["chosen"]]
        res[l]["logZ"] = r["logZ"]
        if l in old:
            bad += (abs(r["logZ"] - old[l]["logZ"]) > 0) or (P.bracket(r["chosen"], len(l.split())) != old[l]["tree"])
    print(json.dumps({"default parse s": round(time.time() - t1), "sentences": len(res), "r215 mismatches (logZ or tree)": int(bad)}), flush=True)
    L.__class__ = Trellis3R216
    L.edge_exact, L.r216, L.r216_mu = True, None, 1.0
    t1 = time.time()
    G = Gen(L, True, {})
    for l in lines:
        res[l]["gate"] = G.score(l.split())
    print(json.dumps({"gen scored": len(lines), "s": round(time.time() - t1), "contexts": len(G.cache)}), flush=True)
    del G
    for mode in ("split", "chunk"):
        L.r216 = mode
        t1 = time.time()
        for l in lines + xl:
            r = L.parse(l.split())
            res[l][mode] = {"chosen": [list(c) for c in r["chosen"]], "logZ": r["logZ"]}
        print(json.dumps({"mode": mode, "s": round(time.time() - t1), "word contexts": len(L.__dict__.get("_r216_cache", {}))}), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"seed": seed, "n": n, "window": C.WINDOW[n], "V": len(vocab), "lines": len(lines), "xlines": xl, "res": res},
              open(os.path.join(OUT, "run_s%d_n%d.json" % (seed, n)), "w", encoding="utf-8"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
