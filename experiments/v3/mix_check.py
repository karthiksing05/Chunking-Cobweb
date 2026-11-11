"""How much does the trigram + licence coder depend on the 0.5 mix weight, and does a learned weight do as well?

Same data, test set, vocabulary and escape conventions as chars_licence.py (domain chars) and
chess_licence.py (domains chess-abs / chess-rel, env LEARN_FILE / CODE_FILE). At the order-2 step the
back-off probability p handed up from the unigram step becomes (1 - w) * p + w * q,
q = SideTrees.next_distribution(prev)[t], when prev is a known token with a non-empty distribution.

Learned w (training data only): the model is built on the head (all but the last T = max(50, n // 10)
training sequences); the (p, q) pairs of the tail's order-2 steps feed EM for a two-component mixture
with fixed components (uniform prior on w, 50 iterations from 0.5); the final model is built on all
n_learn sequences and codes the test set with that w.

Usage: mix_check.py <chars|chess-abs|chess-rel> <seed> <n_learn>
"""
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.type_trees import SideTrees  # noqa: E402

domain, seed, n_learn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ORDER = 3
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
V = inventory_size + 1
mean = lambda xs: sum(xs) / len(xs)


class Model:
    def __init__(self, train):
        self.counts = [defaultdict(Counter) for _ in range(ORDER)]
        for s in train:
            seq = ["<s>"] * (ORDER - 1) + list(s) + ["</s>"]
            for i in range(ORDER - 1, len(seq)):
                for k in range(ORDER):
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

    def p(self, ctx, t, w, pairs=None):
        pr = 1.0 / V
        for k in range(ORDER):
            if k == 1 and w > 0:
                q = self.q_of(ctx[-1], t)
                if q is not None:
                    if pairs is not None:
                        pairs.append((pr, q))
                    pr = (1 - w) * pr + w * q
            c = self.counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot, types = sum(c.values()), len(c)
            pr = (c.get(t, 0) + types * pr) / (tot + types)
        return pr

    def bits(self, s, w, pairs=None):
        seq = ["<s>"] * (ORDER - 1) + list(s) + ["</s>"]
        return sum(-math.log2(self.p(seq[i - ORDER + 1:i], seq[i], w, pairs)) for i in range(ORDER - 1, len(seq))) \
            + sum(ESC for t in s if t not in self.seen)


train = learn_all[:n_learn]
T = max(50, n_learn // 10)
head, tail = train[:n_learn - T], train[n_learn - T:]
hm, pairs = Model(head), []
for s in tail:
    hm.bits(s, 0.5, pairs)                      # w > 0 only so the licence step is visited
w = 0.5
for _ in range(50):
    r = [w * q / (w * q + (1 - w) * p) for p, q in pairs if w * q + (1 - w) * p > 0]
    w = (1 + sum(r)) / (2 + len(pairs))

fm = Model(train)
row = {"domain": domain, "seed": seed, "learn": n_learn, "tag": tag}
for name, wt in (("trigram", 0.0), ("w=0.1", 0.1), ("w=0.25", 0.25), ("w=0.5", 0.5), ("w=0.75", 0.75), ("w=0.9", 0.9)):
    row[name] = round(mean([fm.bits(s, wt) for s in test]), 3)
row["learned w"] = round(w, 3)
row["learned"] = round(mean([fm.bits(s, w) for s in test]), 3)
row["tail pairs"] = len(pairs)
print(json.dumps(row), flush=True)
