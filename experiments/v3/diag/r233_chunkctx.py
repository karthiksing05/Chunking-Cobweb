"""Round 233: do learned chunks supply the context that word concepts need? (controlled grammar)
usage: r233_chunkctx.py <seed> <n>
Learns TRELLIS on the first n sentences (cfg_w5), parses each training sentence (chosen MBR tree, learning off), and builds
type-level Cobweb word taxonomies (one instance per word type, as WordTree) from different context descriptions:
  win1   : L, R              (immediate neighbours; WordTree "all")
  win2   : L, R, L2, R2      (fixed window of width 2)
  win3   : + L3, R3
  chunk  : L, R, PL, PR      (PL/PR = the words just outside the smallest chunk of size >= 2 that contains the word, in the
                              learned tree)
  rb     : as chunk, brackets from the right-branching tree
  rand   : as chunk, brackets from a random binary tree per sentence
Reports, at bit levels b, syntactic purity and semantic purity (r230_grammar.purity definitions: pairs of class mates) and class
count; writes results/story_gen/r233/chunkctx_s<seed>_n<n>.json and the class names per level (for generation)."""
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402
from trellis2.cu_cobweb import CUCobwebTree  # noqa: E402
from trellis2.trellis3 import BOS, EOS  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402


def spans_of(chosen):
    return [(i, j) for (i, k, j) in chosen if j - i >= 2]


def rb_spans(n):
    return [(i, n) for i in range(n - 1)]


def rand_spans(n, rng):
    out = []

    def split(i, j):
        if j - i < 2:
            return
        out.append((i, j))
        k = rng.randint(i + 1, j - 1)
        split(i, k)
        split(k, j)
    split(0, n)
    return out


def features(sent, spans, kind):
    t = [BOS] + list(sent) + [EOS]
    n = len(sent)
    out = []
    for i, w in enumerate(sent):
        f = {"L": t[i], "R": t[i + 2]}
        if kind in ("win2", "win3"):
            f["L2"] = t[i - 1] if i >= 1 else BOS
            f["R2"] = t[i + 3] if i + 3 < len(t) else EOS
        if kind == "win3":
            f["L3"] = t[i - 2] if i >= 2 else BOS
            f["R3"] = t[i + 4] if i + 4 < len(t) else EOS
        if kind in ("chunk", "rb", "rand"):
            best = None
            for (a, b) in spans:
                if a <= i < b and (best is None or b - a < best[1] - best[0]):
                    best = (a, b)
            a, b = best if best else (0, n)
            f["PL"] = t[a]            # word before the chunk (t is offset by one)
            f["PR"] = t[b + 1]
        out.append((w, f))
    return out


def build(per_sent, attrs, seed):
    bags = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    order = []
    for items in per_sent:
        for w, f in items:
            if w not in bags:
                order.append(w)
            for a in attrs:
                bags[w][a][f[a]] += 1.0
    tree = CUCobwebTree(list(attrs), seed=seed)
    tree.exact_alone = True
    leaf = {}
    for w in order:
        x = {a: {k: v / sum(bags[w][a].values()) for k, v in bags[w][a].items()} for a in attrs}
        leaf[w] = tree.ifit(x)
    return tree, leaf, order


def class_of(tree, leaf, w, b):
    path, n = [], leaf[w]
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    out = path[0]
    for x in path:
        if -math.log2(max(x.count, 1e-9) / tree.root.count) <= b + 1e-12:
            out = x
        else:
            break
    return out.id


def purity(names):
    cls = defaultdict(list)
    for w, c in names.items():
        if w in G.LEX:
            cls[c].append(w)
    tot = ss = sp = sm = 0
    for ws in cls.values():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                tot += 1
                ss += G.syncat(a) == G.syncat(b)
                ta, tb = G.semtype(a), G.semtype(b)
                if ta is not None and tb is not None and ta[0] == tb[0]:
                    sp += 1
                    sm += ta == tb
    return {"classes": len(cls), "syn": round(ss / tot, 3) if tot else None, "sem": round(sm / sp, 3) if sp else None}


def main():
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
    train = [s for s, _ in G.load_corpus(seed, n)["train"]]
    L = Trellis3R223(seed=seed, **cfg)
    for s in train:
        L.learn(s)
    learned = [spans_of(L.parse(s)["chosen"]) for s in train]
    # how often the learned tree brackets the object NP (Det (Adj) N after the verb) as one chunk
    np_hit = np_tot = 0
    for s, sp in zip(train, learned):
        for i, w in enumerate(s):
            if G.LEX[w][0] == "v" and i + 1 < len(s):
                j = i + 1 + (s[i + 1] in G.PREPS)
                if j < len(s):
                    np_tot += 1
                    np_hit += (j, len(s)) in set(sp)
    rng = random.Random(seed)
    kinds = {"win1": ("L", "R"), "win2": ("L", "R", "L2", "R2"), "win3": ("L", "R", "L2", "R2", "L3", "R3"),
             "chunk": ("L", "R", "PL", "PR"), "rb": ("L", "R", "PL", "PR"), "rand": ("L", "R", "PL", "PR")}
    res = {"seed": seed, "n": n, "object NP bracketed": round(np_hit / max(np_tot, 1), 3), "kinds": {}, "names": {}}
    print(json.dumps({"object NP bracketed": res["object NP bracketed"]}), flush=True)
    for kind, attrs in kinds.items():
        if kind == "chunk":
            sps = learned
        elif kind == "rb":
            sps = [rb_spans(len(s)) for s in train]
        elif kind == "rand":
            sps = [rand_spans(len(s), rng) for s in train]
        else:
            sps = [[] for _ in train]
        per = [features(s, sp, kind) for s, sp in zip(train, sps)]
        tree, leaf, order = build(per, attrs, seed)
        res["kinds"][kind] = {}
        res["names"][kind] = {}
        for b in (3, 4, 5, 6, 7, 8):
            nm = {w: "k%d" % class_of(tree, leaf, w, b) for w in order}
            res["kinds"][kind][b] = purity(nm)
            res["names"][kind][b] = nm
        print(kind, json.dumps(res["kinds"][kind]), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233", exist_ok=True)
    json.dump(res, open(f"experiments/v3/results/story_gen/r233/chunkctx_s{seed}_n{n}.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
