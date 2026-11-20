"""Round 225 (TRELLIS arms): train one learner arm on the first n sentences of a seed's stream and
  (always) score the dev and test sentences with the generator, per step (p, rank, support, fallback) as r215b / r216 (Gen.score, gate arm);
  (mode "parse") also parse every r215 control sentence of the seed (default parse: logZ, chosen tree) and generator-score it, for the
  r216-style parse metrics (r225_analyse.py).
Arms: full (the paper recipe, Trellis3R204 edge-exact reads), nochunk (store="none": no composite stored), noconcept (generator reads and gate at the leaf only; learning and parsing as full),
neither (nochunk + noconcept), noconceptp (noconcept + the parser reads gains at the leaf only, so learning stores different chunks). See src/trellis2/trellis3_r225.py.
Usage (repo root, compute node, env trellis):  r225_trellis.py <seed> <n> <arm> [bits|parse]
"""
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
import r225_common as D  # noqa: E402
from r215b_gen import Gen  # noqa: E402
from trellis2.trellis3_r225 import Trellis3R225  # noqa: E402

ARMS = {"full": ({}, False, False), "nochunk": ({"store": "none"}, False, False), "noconcept": ({}, True, False),
        "neither": ({"store": "none"}, True, False), "noconceptp": ({}, True, True)}


def main(seed, n, arm, mode):
    kw, off, offp = ARMS[arm]
    sents, dev, test, vocab = D.data(seed, n)
    train, _, _, _ = C.stream(seed, n)
    cfg = dict(C.CFG, window=C.WINDOW[n], **kw)
    L = Trellis3R225(seed=seed, concepts_off=off, concepts_off_parse=offp, **cfg)
    t0 = time.time()
    for s, fol in train:
        L.learn(s, follows=fol)
    comp = sum(1 for e in L.records if e.token is None and e.leaf is not None)
    print(json.dumps({"seed": seed, "n": n, "arm": arm, "window": C.WINDOW[n], "learn s": round(time.time() - t0), "records": len(L.records),
                      "stored composites": comp}), flush=True)
    L._quiet = True
    L.edge_exact = True
    whole = {tuple(s) for s in sents}
    G = Gen(L, True, {})
    t1 = time.time()
    out = {"seed": seed, "n": n, "arm": arm, "mode": mode, "window": C.WINDOW[n], "V": len(vocab), "records": len(L.records), "stored composites": comp,
           "train distinct": len(whole), "dev": [], "test": [], "test_seen": [tuple(s) in whole for s in test],
           "test_words": [len(s) for s in test], "dev_words": [len(s) for s in dev]}
    for s in dev:
        out["dev"].append(G.score(s))
    print(json.dumps({"dev scored": len(dev), "s": round(time.time() - t1)}), flush=True)
    for s in test:
        out["test"].append(G.score(s))
    print(json.dumps({"test scored": len(test), "s": round(time.time() - t1), "contexts": len(G.cache)}), flush=True)
    os.makedirs(D.RAW, exist_ok=True)
    f = os.path.join(D.RAW, "trellis_%s_s%d_n%d_%s.json" % (arm, seed, n, mode))
    if mode == "parse":
        S = C.load_sets(seed)
        lines = []
        for k, ss in C.all_sentences(S).items():
            if k == "C4":
                continue
            for l in ss:
                if l not in lines:
                    lines.append(l)
        res = {l: {"len": len(l.split())} for l in lines}
        t2 = time.time()
        for l in lines:
            r = L.parse(l.split())
            res[l]["chosen"] = [list(c) for c in r["chosen"]]
            res[l]["logZ"] = r["logZ"]
        print(json.dumps({"parsed": len(lines), "s": round(time.time() - t2)}), flush=True)
        t2 = time.time()
        for l in lines:
            res[l]["gate"] = G.score(l.split())
        print(json.dumps({"gen scored": len(lines), "s": round(time.time() - t2)}), flush=True)
        out["res"] = res
    json.dump(out, open(f, "w", encoding="utf-8"))
    print("wrote", f, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "bits")
