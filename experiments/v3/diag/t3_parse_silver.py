"""Parsing TinyStories against silver trees (Stanza; silver_parse.py): unlabelled
bracket precision / recall / F1 over constituents of 2+ words other than the whole
sentence, on held-out sentences, at checkpoints of the training stream; right- and
left-branching baselines.

Usage: t3_parse_silver.py <cfg json> <seed> <checkpoints> <tag>
"""
import json
import random
import sys
import time

sys.path.insert(0, "src")
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4] if len(sys.argv) > 4 else ""
corpus = simple_sentences(read_sentences(), vocab_size=250)
random.Random(seed).shuffle(corpus)
silver = json.load(open(f"experiments/v3/results/silver/silver_s{seed}.json", encoding="utf-8"))


def inner(n, spans):
    return {tuple(s) for s in spans if s[1] - s[0] >= 2 and tuple(s) != (0, n)}


def prf(pairs):
    m = sum(len(g & p) for g, p in pairs)
    G = sum(len(g) for g, _ in pairs)
    P = sum(len(p) for _, p in pairs)
    pr, rc = m / max(P, 1), m / max(G, 1)
    return {"P": round(pr, 3), "R": round(rc, 3), "F1": round(2 * pr * rc / max(pr + rc, 1e-9), 3)}


rb = prf([(inner(len(x["tokens"]), x["spans"]), {(i, len(x["tokens"])) for i in range(1, len(x["tokens"]) - 1)}) for x in silver])
lb = prf([(inner(len(x["tokens"]), x["spans"]), {(0, j) for j in range(2, len(x["tokens"]))}) for x in silver])
print(json.dumps({"baselines": {"right-branching": rb, "left-branching": lb}}), flush=True)
L = Trellis3(seed=seed, **cfg)
t0, done = time.time(), 0
for c in checks:
    for s in corpus[done:c]:
        L.learn(s)
    done = c
    learn_s = time.time() - t0
    pairs, ex = [], []
    for x in silver:
        toks = x["tokens"]
        t = L.parse(toks)["tree"]
        p = inner(len(toks), t.brackets())
        pairs.append((inner(len(toks), x["spans"]), p))
        if len(ex) < 6:
            ex.append(t.to_string(toks) + "    silver " + x["tree"])
    print(json.dumps({"tag": tag, "checkpoint": c, "learn s": round(learn_s), **prf(pairs), "examples": ex}), flush=True)
