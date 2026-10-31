"""Coders on Chinese characters (S5), the counterpart of the chess coder study (S6).

Each coder learns the first n characters (IDS decompositions as token sequences, prefix
order) and codes held-out characters; bits per character, the model not charged. A
token no coder has seen costs log2 of the whole token inventory on top of its escape.

  trellis    the chart's own probability of the sequence: the start over composite
             concepts times the inside score (chart_gen.ChartGen.logZ), the grammar
             TRELLIS generates from, so compression reads the same machinery
  unigram    each token by its frequency, then an end mark (the component list)
  bigram     interpolated Witten-Bell n-gram with start/end marks
  trigram    the same, order 3
  bpe        pair-merge units learned on the training sequences, units coded by their
             frequency with an escape to single tokens, then an end mark
  xz         xz(train + one test character) - xz(train), averaged over the first 200

Usage: chars_coders.py <seed> <checkpoints, e.g. 250,500,1000> <out.json> [config json] [charts json]
"""
import json
import lzma
import math
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402
from trellis2.chart_gen import ChartGen  # noqa: E402

seed, checkpoints, out = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
N_TEST = int(os.environ.get("N_TEST", 300))
CHARTS = [{"cut": 2, "lam": 4, "word_sym": "context", "ctx_cut": 2},
          {"cut": 3, "lam": 4, "word_sym": "context", "ctx_cut": 3},
          {"cut": 2, "lam": 4, "word_sym": "category"},
          {"cut": 3, "lam": 4, "word_sym": "token"}]
CHARTS = json.loads(sys.argv[5]) if len(sys.argv) > 5 else CHARTS   # optional chart list

chars, _ = load_characters(default_ids_path(), seed=seed)
test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
inventory = {t for c in chars for t in c.tokens}
ESC = math.log2(len(inventory))
LN2 = math.log(2)


def unigram(train):
    n = Counter(t for s in train for t in s)
    tot, V = sum(n.values()) + len(train), len(n) + 2           # + the end mark, + escape
    def bits(s):
        b = 0.0
        for t in list(s) + ["</s>"]:
            c = len(train) if t == "</s>" else n.get(t, 0)
            b += -math.log2((c + 0.5) / (tot + 0.5 * V)) + (ESC if t != "</s>" and c == 0 else 0)
        return b
    return bits


def ngram(train, order):
    counts = [defaultdict(Counter) for _ in range(order)]      # context tuple -> next counts
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            for k in range(order):
                counts[k][tuple(seq[i - k:i])][seq[i]] += 1
    V = len(inventory) + 1
    def p(ctx, t):
        pr = 1.0 / V
        for k in range(order):                                  # Witten-Bell, low to high order
            c = counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot, types = sum(c.values()), len(c)
            pr = (c.get(t, 0) + types * pr) / (tot + types)
        return pr
    seen = {t for s in train for t in s}
    def bits(s):
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        return sum(-math.log2(p(seq[i - order + 1:i], seq[i])) for i in range(order - 1, len(seq))) \
            + sum(ESC for t in s if t not in seen)
    return bits


def bpe(train, n_merges=2000, min_count=2):
    seqs = [tuple(s) for s in train]
    merges = []
    for _ in range(n_merges):
        pc = Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]):
                pc[(a, b)] += 1
        if not pc:
            break
        (a, b), c = pc.most_common(1)[0]
        if c < min_count:
            break
        merges.append((a, b))
        ab = a + " " + b
        new = []
        for s in seqs:
            o, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    o.append(ab); i += 2
                else:
                    o.append(s[i]); i += 1
            new.append(tuple(o))
        seqs = new
    rank = {m: r for r, m in enumerate(merges)}
    units = Counter(u for s in seqs for u in s)
    base = Counter(t for s in train for t in s)
    tot, V = sum(units.values()) + len(seqs), len(units) + 2
    btot = sum(base.values())
    def encode(s):
        s = list(s)
        while len(s) > 1:
            best = min(((rank.get((s[i], s[i + 1]), math.inf), i) for i in range(len(s) - 1)))
            if best[0] == math.inf:
                break
            i = best[1]
            s[i:i + 2] = [s[i] + " " + s[i + 1]]
        return s
    def bits(s):
        b = -math.log2((len(seqs) + 0.5) / (tot + 0.5 * V))   # the end mark
        for u in encode(s):
            c = units.get(u, 0)
            if c:
                b += -math.log2((c + 0.5) / (tot + 0.5 * V))
            else:                                               # escape, then the single token
                b += -math.log2(0.5 / (tot + 0.5 * V))
                b += -math.log2((base.get(u, 0) + 0.5) / (btot + 0.5 * (len(base) + 1)))
                b += ESC if u not in base else 0
        return b
    return bits, len(merges)


def xz_bits(train, items):
    blob = "\n".join(" ".join(s) for s in train).encode("utf-8")
    z0 = len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))
    return [8 * (len(lzma.compress(blob + b"\n" + " ".join(s).encode("utf-8"),
                                   preset=9 | lzma.PRESET_EXTREME)) - z0) for s in items]


def chart_bits(CG, seen, s):
    ins = CG.inside(list(s))
    z = CG.logZ(ins, len(s))
    if z == -math.inf:
        return None
    return -z / LN2 + sum(ESC for t in s if t not in seen)


L = Trellis3(seed=seed, **cfg)
rows, done, t0 = [], 0, time.time()
for n in checkpoints:
    for c in chars[done:n]:
        L.learn(c.tokens)
    done = n
    train = [c.tokens for c in chars[:n]]
    seen = {t for s in train for t in s}
    row = {"seed": seed, "train": n, "test": len(test), "learn s": round(time.time() - t0),
           "unseen-token share": sum(t not in seen for s in test for t in s) / sum(len(s) for s in test),
           "tokens per character": sum(len(s) for s in test) / len(test)}
    mean = lambda xs: sum(xs) / len(xs)
    row["unigram"] = mean([unigram(train)(s) for s in test])
    row["bigram"] = mean([ngram(train, 2)(s) for s in test])
    row["trigram"] = mean([ngram(train, 3)(s) for s in test])
    bb, nm = bpe(train)
    row["bpe"], row["bpe merges"] = mean([bb(s) for s in test]), nm
    row["xz"] = mean(xz_bits(train, test[:200]))
    for k, ch in enumerate(CHARTS):
        t1 = time.time()
        CG = ChartGen(L, **ch)
        bs = [chart_bits(CG, seen, s) for s in test]
        ok = [b for b in bs if b is not None]
        row[f"trellis {k}"] = {"chart": ch, "bits": mean(ok) if ok else None, "coded": len(ok) / len(bs),
                               "stats": CG.stats, "s": round(time.time() - t1)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
