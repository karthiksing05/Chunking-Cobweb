"""Do merged units (chunks) help the trigram + licence coder?

Loading, escape and vocabulary conventions are those of mix_check.py (copied, not imported). Byte-pair-style merges
are learned on the training sequences only (most frequent adjacent pair of units, ties by smallest repr, stop when
the count is below 2); a unit is a tuple of base tokens. For each K the first K merges re-tokenize train and test,
and four coders (bigram, bigram + licence, trigram, trigram + licence; licence weight 0.5) code the test set.

Usage: merge_licence.py <chars|chess-abs|chess-rel> <seed> <n_learn> [K list, default 0,25,50,100,200,400,800,1600]
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.type_trees import SideTrees  # noqa: E402

domain, seed, n_learn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,25,50,100,200,400,800,1600").split(",")]
tag = ""

if domain == "chars":
    from trellis2.characters import default_ids_path, load_characters
    N_TEST = int(os.environ.get("N_TEST", 300))
    chars, _ = load_characters(default_ids_path(), seed=seed)
    test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
    learn_all = [c.tokens for c in chars]
    inventory_size = len({t for c in chars for t in c.tokens})
else:
    from trellis2 import chess as C
    kind = domain.split("-")[1]
    N_CODE = int(os.environ.get("N_CODE", 1000))
    rd = lambda f: [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    if os.environ.get("LEARN_FILE") and os.environ.get("CODE_FILE"):
        learn_fens, code_fens = rd(os.environ["LEARN_FILE"]), rd(os.environ["CODE_FILE"])
    else:
        fens = rd(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen"))
        random.Random(seed).shuffle(fens)
        learn_fens, code_fens = fens, fens[4000:]
        if os.environ.get("LEARN_FILE"):
            learn_fens = rd(os.environ["LEARN_FILE"])
        if os.environ.get("CODE_FILE"):
            code_fens = rd(os.environ["CODE_FILE"])
    if os.environ.get("CODE_FILE"):
        tag = os.path.splitext(os.path.basename(os.environ["CODE_FILE"]))[0]

    def tokens(fen):
        pos = C.parse_fen(fen)
        out, prev = [], -1
        for idx, p, f, r in sorted((((7 - r) * 8 + f), p, f, r) for (f, r), p in pos.items()):
            out.append((p, f, r) if kind == "abs" else (p, idx - prev))
            prev = idx
        return out

    learn_all = [tokens(x) for x in learn_fens[:n_learn]]
    test = [tokens(x) for x in code_fens[:N_CODE]]
    inventory_size = 64 * 12 if kind == "abs" else 12 * 127

ESC = math.log2(inventory_size)
mean = lambda xs: sum(xs) / len(xs)
train0 = [[(t,) for t in s] for s in learn_all[:n_learn]]
test0 = [[(t,) for t in s] for s in test]


def apply_merge(seqs, pair):
    """Replace every non-overlapping left-to-right occurrence of pair by one merged unit, in place."""
    a, b = pair
    new = a + b
    for si, s in enumerate(seqs):
        if len(s) < 2:
            continue
        out, i, n, hit = [], 0, len(s), False
        while i < n:
            if i + 1 < n and s[i] == a and s[i + 1] == b:
                out.append(new)
                i += 2
                hit = True
            else:
                out.append(s[i])
                i += 1
        if hit:
            seqs[si] = out


def learn_merges(seqs, kmax):
    seqs = [list(s) for s in seqs]
    merges = []
    while len(merges) < kmax:
        cnt = Counter()
        for s in seqs:
            cnt.update(zip(s, s[1:]))
        if not cnt:
            break
        pair = min(cnt, key=lambda p: (-cnt[p], repr(p)))
        if cnt[pair] < 2:
            break
        merges.append(pair)
        apply_merge(seqs, pair)
    return merges


ORDER = 3


class Model:
    def __init__(self, train, V, order):
        self.V, self.order = V, order
        self.counts = [defaultdict(Counter) for _ in range(order)]
        for s in train:
            seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
            for i in range(order - 1, len(seq)):
                for k in range(order):
                    self.counts[k][tuple(seq[i - k:i])][seq[i]] += 1
        self.seen = {t for s in train for t in s}
        self.trees = SideTrees(seed)
        for s in train:
            self.trees.observe(s)
        self.cache = {}

    def q_of(self, prev, t):
        if prev not in self.seen:
            return None
        if prev not in self.cache:
            self.cache[prev] = self.trees.next_distribution(prev)
        d = self.cache[prev]
        return d.get(t, 0.0) if d else None

    def p(self, ctx, t, w):
        pr = 1.0 / self.V
        for k in range(self.order):
            if k == 1 and w > 0:
                q = self.q_of(ctx[-1], t)
                if q is not None:
                    pr = (1 - w) * pr + w * q
            c = self.counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot, types = sum(c.values()), len(c)
            pr = (c.get(t, 0) + types * pr) / (tot + types)
        return pr

    def bits(self, s, w):
        o = self.order
        seq = ["<s>"] * (o - 1) + list(s) + ["</s>"]
        return sum(-math.log2(self.p(seq[i - o + 1:i], seq[i], w)) for i in range(o - 1, len(seq))) \
            + sum(ESC for t in s if len(t) == 1 and t not in self.seen)


merges = learn_merges(train0, max(KS))
cur_train, cur_test, applied = [list(s) for s in train0], [list(s) for s in test0], 0
for K in sorted(KS):
    t0 = time.time()
    while applied < min(K, len(merges)):
        apply_merge(cur_train, merges[applied])
        apply_merge(cur_test, merges[applied])
        applied += 1
    V = inventory_size + K + 1
    row = {"domain": domain, "tag": tag, "seed": seed, "learn": n_learn, "K": K, "merges applied": applied,
           "units per sequence": round(mean([len(s) for s in cur_test]), 3)}
    m3 = Model(cur_train, V, 3)
    m2 = Model(cur_train, V, 2)
    row["bigram"] = round(mean([m2.bits(s, 0.0) for s in cur_test]), 3)
    row["bigram + licence"] = round(mean([m2.bits(s, 0.5) for s in cur_test]), 3)
    row["trigram"] = round(mean([m3.bits(s, 0.0) for s in cur_test]), 3)
    row["trigram + licence"] = round(mean([m3.bits(s, 0.5) for s in cur_test]), 3)
    row["types"] = len(m3.trees.types())
    row["build s"] = round(time.time() - t0, 1)
    print(json.dumps(row), flush=True)
print(json.dumps({"first merges": ["+".join(map(repr, a + b)) if False else
                                   f"{''.join(map(str, a))} | {''.join(map(str, b))}" for a, b in merges[:15]]},
                 ensure_ascii=False), flush=True)
