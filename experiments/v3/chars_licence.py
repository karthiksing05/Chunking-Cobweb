"""Does the two-tree type licence shorten the code for held-out Chinese characters?

Same data, test set and Witten-Bell n-gram coder as chars_coders.py (trigram column); the
"+ licence" coders replace, at the order-2 step, the back-off probability handed up from the
unigram step by 0.5 * p_unigram + 0.5 * q, q = SideTrees.next_distribution(prev)[t], when the
previous token is a real token known to the trees and its distribution is non-empty.

Usage: chars_licence.py <seed> <checkpoints, e.g. 500,1000,2000,4000>
"""
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa: E402
from trellis2.type_trees import SideTrees  # noqa: E402

seed, checkpoints = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
N_TEST = int(os.environ.get("N_TEST", 300))

chars, _ = load_characters(default_ids_path(), seed=seed)
test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
inventory = {t for c in chars for t in c.tokens}
ESC = math.log2(len(inventory))


def ngram(train, order, trees=None):
    counts = [defaultdict(Counter) for _ in range(order)]      # context tuple -> next counts
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            for k in range(order):
                counts[k][tuple(seq[i - k:i])][seq[i]] += 1
    V = len(inventory) + 1
    seen = {t for s in train for t in s}
    cache = {}

    def q_of(prev, t):
        if prev not in seen:                                    # boundaries and unknown tokens
            return None
        if prev not in cache:
            cache[prev] = trees.next_distribution(prev)
        d = cache[prev]
        return d.get(t, 0.0) if d else None

    def p(ctx, t):
        pr = 1.0 / V
        for k in range(order):                                  # Witten-Bell, low to high order
            if k == 1 and trees is not None:
                q = q_of(ctx[-1], t)
                if q is not None:
                    pr = 0.5 * pr + 0.5 * q
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


mean = lambda xs: sum(xs) / len(xs)
trees, done, t0 = SideTrees(seed), 0, time.time()
for n in checkpoints:
    for c in chars[done:n]:
        trees.observe(c.tokens)
    done = n
    trees.licence(chars[0].tokens[0], chars[0].tokens[0])      # force the index so build time counts
    build = round(time.time() - t0)
    train = [c.tokens for c in chars[:n]]
    row = {"seed": seed, "train": n, "test": len(test), "types": len(trees.types()), "build s": build}
    for name, order, tr in (("bigram", 2, None), ("bigram + licence", 2, trees),
                            ("trigram", 3, None), ("trigram + licence", 3, trees)):
        row[name] = round(mean([ngram(train, order, tr)(s) for s in test]), 3)
    print(json.dumps(row), flush=True)
