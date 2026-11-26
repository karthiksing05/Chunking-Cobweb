"""Round 233 diagnostic: how pure are TRELLIS's own concepts at each level, on the controlled grammar?
usage: r233_purity.py <seed> <n> [cfg json]   (default cfg = r214p01 cfg_w5)
Groups the stored word occurrences (context taxonomy) and the committed composites (content taxonomy) by their ancestor at
level b bits (-log2 share of the root), and reports token-pair purity: P(two occurrences in a group share the syntactic
category), P(same semantic type | same kind of content word), and the group count. For composites: P(same head category
signature) where a composite's signature is the sequence of syntactic categories of its words."""
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402


def lift(node, b, root_c):
    while node.parent is not None and -math.log2(max(node.count, 1e-9) / root_c) > b:
        node = node.parent
    return node


def pair_rate(groups, same):
    num = den = 0.0
    for items in groups.values():
        c = Counter(items)
        tot = sum(c.values())
        den += tot * (tot - 1)
        for k in c:
            pass
        # pairs of distinct occurrences that agree on ``same``
        by = Counter()
        for k, v in c.items():
            by[same(k)] += v
        num += sum(v * (v - 1) for v in by.values())
    return num / den if den else float("nan")


def words_of(e):
    if e.token is not None:
        return [e.token]
    return words_of(e.left) + words_of(e.right)


def main():
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
    C = G.load_corpus(seed, n)
    L = Trellis3R223(seed=seed, **cfg)
    for s, _ in C["train"]:
        L.learn(s)
    words = [e for e in L.records if e.token is not None and e.leaf is not None]
    comps = [e for e in L.records if e.token is None and e.cleaf is not None and e.used > 0]
    rc, cc = L.ctx.root.count, max(L.cnt.root.count, 1e-9)
    syn = lambda w: G.syncat(w)
    sem = lambda w: G.semtype(w)
    out = {"seed": seed, "n": n, "word occurrences": len(words), "committed composites": len(comps), "ctx": {}, "cnt": {}}
    chance_syn = pair_rate({0: [e.token for e in words]}, syn)
    out["chance syn (word occurrences)"] = round(chance_syn, 3)
    for b in [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 99]:
        g = defaultdict(list)
        for e in words:
            g[lift(e.leaf, b, rc).id].append(e.token)
        gs = defaultdict(list)
        for k, ws in g.items():
            for w in ws:
                if sem(w) is not None:
                    gs[(k, sem(w)[0])].append(w)
        out["ctx"][b] = {"groups": len(g), "syn": round(pair_rate(g, syn), 3), "sem|kind": round(pair_rate(gs, sem), 3),
                         "words per group": round(len(words) / len(g), 1), "types per group": round(sum(len(set(v)) for v in g.values()) / len(g), 2)}
        h = defaultdict(list)
        for e in comps:
            h[lift(e.cleaf, b, cc).id].append(tuple(syn(w) for w in words_of(e)))
        out["cnt"][b] = {"groups": len(h), "sig": round(pair_rate(h, lambda x: x), 3)}
        print(b, json.dumps(out["ctx"][b]), json.dumps(out["cnt"][b]), flush=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ("ctx", "cnt")}))
    os.makedirs("experiments/v3/results/story_gen/r233", exist_ok=True)
    json.dump(out, open(f"experiments/v3/results/story_gen/r233/purity_s{seed}_n{n}_w{cfg.get('window')}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
