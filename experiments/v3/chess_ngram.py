"""Reference coders for held-out chess positions: Witten-Bell n-grams in a fixed reading order.

TRELLIS's chess coder must beat these. A position becomes one sequence of its pieces in
reading order: rank 8 down to rank 1, file a to h within a rank (FEN order). The order is
fixed, not learned. Two tokenizations:
  abs  token = (piece, file, rank)
  rel  token = (piece, square_index - previous_square_index), square_index = 0..63 in
       reading order, previous index -1 for the first piece (a position shifted by one
       file keeps most of its tokens)
Coders (bits per coded position, mean): unigram, bigram, trigram (Witten-Bell). Every
sequence ends with the end mark </s>; a token never seen in training costs
log2(inventory size) extra bits (64 * 12 for abs, 12 * 127 for rel).

Data as chess_compress_bpe.py: LEARN_FILE / CODE_FILE (else the default fen file shuffled
with the seed: learn on the first n, code the n_rep-th position onwards). The first
N_CODE (1000) positions of the code file are coded.

Usage: chess_ngram.py <seed> <n_learn checkpoints, e.g. 1000,2000,4000> <tag>
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
from trellis2 import chess as C  # noqa: E402

seed, checkpoints, tag = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
N_CODE = int(os.environ.get("N_CODE", 1000))
N_REP = 4000

if os.environ.get("LEARN_FILE") and os.environ.get("CODE_FILE"):
    learn_fens = [l.strip() for l in open(os.environ["LEARN_FILE"], encoding="utf-8") if l.strip()]
    code_fens = [l.strip() for l in open(os.environ["CODE_FILE"], encoding="utf-8") if l.strip()]
else:
    fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen"),
                                    encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(fens)
    learn_fens, code_fens = fens, fens[N_REP:]
    if os.environ.get("LEARN_FILE"):
        learn_fens = [l.strip() for l in open(os.environ["LEARN_FILE"], encoding="utf-8") if l.strip()]
    if os.environ.get("CODE_FILE"):
        code_fens = [l.strip() for l in open(os.environ["CODE_FILE"], encoding="utf-8") if l.strip()]
code_fens = code_fens[:N_CODE]


def tokens(fen, kind):
    pos = C.parse_fen(fen)                                      # {(file, rank 0..7): piece}
    items = sorted((((7 - r) * 8 + f), p, f, r) for (f, r), p in pos.items())
    out, prev = [], -1
    for idx, p, f, r in items:
        out.append((p, f, r) if kind == "abs" else (p, idx - prev))
        prev = idx
    return out


mean = lambda xs: sum(xs) / len(xs)


def make_coders(kind, train):
    inventory_size = 64 * 12 if kind == "abs" else 12 * 127
    ESC = math.log2(inventory_size)
    V = inventory_size + 1
    seen = {t for s in train for t in s}

    def unigram():
        n = Counter(t for s in train for t in s)
        tot, Vu = sum(n.values()) + len(train), len(n) + 2      # + the end mark, + escape
        def bits(s):
            b = 0.0
            for t in list(s) + ["</s>"]:
                c = len(train) if t == "</s>" else n.get(t, 0)
                b += -math.log2((c + 0.5) / (tot + 0.5 * Vu)) + (ESC if t != "</s>" and c == 0 else 0)
            return b
        return bits

    def ngram(order):
        counts = [defaultdict(Counter) for _ in range(order)]
        for s in train:
            seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
            for i in range(order - 1, len(seq)):
                for k in range(order):
                    counts[k][tuple(seq[i - k:i])][seq[i]] += 1

        def p(ctx, t):
            pr = 1.0 / V
            for k in range(order):                              # Witten-Bell, low to high order
                c = counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
                if not c:
                    continue
                tot, types = sum(c.values()), len(c)
                pr = (c.get(t, 0) + types * pr) / (tot + types)
            return pr

        def bits(s):
            seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
            return sum(-math.log2(p(seq[i - order + 1:i], seq[i])) for i in range(order - 1, len(seq))) \
                + sum(ESC for t in s if t not in seen)
        return bits
    return unigram, ngram


for kind in ("abs", "rel"):
    learn_all = [tokens(x, kind) for x in learn_fens[:max(checkpoints)]]
    test = [tokens(x, kind) for x in code_fens]
    for n in checkpoints:
        train = learn_all[:n]
        unigram, ngram = make_coders(kind, train)
        row = {"tag": tag, "seed": seed, "learn": n, "coded": len(test), "tokens": kind,
               "unigram": round(mean([unigram()(s) for s in test]), 3)}
        for name, order in (("bigram", 2), ("trigram", 3)):
            f = ngram(order)
            row[name] = round(mean([f(s) for s in test]), 3)
        print(json.dumps(row), flush=True)
