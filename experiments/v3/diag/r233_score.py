"""Round 233 offline critic test: can the memory's parse (composition) tell the window generator's correct sentences from its
errors? usage: r233_score.py <seed> <n> <dump.jsonl> [store]   (store = posterior (default) or none = chunks off)
Scores every generated sentence by (a) parse logZ per word, (b) -regeneration bits of the chosen analysis per word
(L._deriv_bits), (c) an interpolated trigram over the same training sentences (mean log prob per transition, baseline), and
reports AUC of each score for R-sem against R-syn and against R-non (exact judge), plus AUC on R-sem against object violations."""
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, i, sum_r = 1, 0, 0.0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        sum_r += avg * sum(1 for k in range(i, j) if s[k][1])
        i = j
    return (sum_r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def trigram(train):
    c3, c2, c1 = Counter(), Counter(), Counter()
    for s in train:
        t = ["<s>", "<s>"] + list(s) + ["</s>"]
        for i in range(2, len(t)):
            c3[tuple(t[i - 2:i + 1])] += 1
            c2[tuple(t[i - 1:i + 1])] += 1
            c1[t[i]] += 1
    h2, h1 = Counter(), Counter()
    for (a, b, c), v in c3.items():
        h2[(a, b)] += v
    for (b, c), v in c2.items():
        h1[b] += v
    N, V = sum(c1.values()), len(c1) + 1

    def score(s):
        t = ["<s>", "<s>"] + list(s) + ["</s>"]
        lp = 0.0
        for i in range(2, len(t)):
            a, b, c = t[i - 2], t[i - 1], t[i]
            p1 = (c1[c] + 1) / (N + V)
            p2 = c2[(b, c)] / h1[b] if h1[b] else 0.0
            p3 = c3[(a, b, c)] / h2[(a, b)] if h2[(a, b)] else 0.0
            lp += math.log(0.6 * p3 + 0.3 * p2 + 0.1 * p1)
        return lp / (len(t) - 2)
    return score


def main():
    seed, n, dump = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    store = sys.argv[4] if len(sys.argv) > 4 else "posterior"
    cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
    cfg["store"] = store
    train = [s for s, _ in G.load_corpus(seed, n)["train"]]
    L = Trellis3R223(seed=seed, **cfg)
    for s in train:
        L.learn(s)
    tri = trigram(train)
    rows = [json.loads(x) for x in open(dump, encoding="utf-8")][1:]
    gens = [r["gen"].split() for r in rows if r.get("gen")]
    seen = {tuple(s) for s in train}
    res = {"sem": [], "syn": [], "non": [], "obj": []}
    sc = {"logZ": {}, "regen": {}, "trigram": {}}
    L._quiet = True
    for g in gens:
        j, why = G.judge(g)
        k = tuple(g)
        if k not in sc["logZ"]:
            r = L.parse(g)
            sc["logZ"][k] = r["logZ"] / len(g)
            sc["regen"][k] = -sum(L._deriv_bits(r["comps"][c]) for c in r["chosen"] if c in r["comps"]) / len(g)
            sc["trigram"][k] = tri(g)
        res[j].append(k)
        if j == "syn" and "object" in why:
            res["obj"].append(k)
    out = {"seed": seed, "n": n, "store": store, "dump": dump, "counts": {k: len(v) for k, v in res.items()}}
    for name, d in sc.items():
        out[name] = {f"sem vs {b}": round(auc([d[k] for k in res["sem"]], [d[k] for k in res[b]]), 3) for b in ("syn", "obj", "non")}
    # what a critic of 4 would do: expected R-sem of the best-scored of 4 random draws, per score
    import random
    rng = random.Random(seed)
    lab = {tuple(g): G.judge(g)[0] for g in gens}
    for name, d in sc.items():
        hits = 0
        T = 4000
        for _ in range(T):
            c = [tuple(rng.choice(gens)) for _ in range(4)]
            hits += lab[max(c, key=lambda x: d[x])] == "sem"
        out[name]["best of 4 R-sem"] = round(hits / T, 3)
    out["random R-sem"] = round(len(res["sem"]) / len(gens), 3)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
