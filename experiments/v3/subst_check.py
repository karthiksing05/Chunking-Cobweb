"""Coder check: SubstitutionLearner reproduces chars_licence.py's "trigram + licence" number.

Usage: subst_check.py <seed>     (first 1,000 characters, N_TEST test characters)
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa: E402
from trellis2.substitution import SubstitutionLearner  # noqa: E402

seed = int(sys.argv[1])
N_TEST, N_TRAIN = int(os.environ.get("N_TEST", 300)), 1000
chars, _ = load_characters(default_ids_path(), seed=seed)
test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
inventory = {t for c in chars for t in c.tokens}
ESC, V = math.log2(len(inventory)), len(inventory) + 1
mean = lambda xs: sum(xs) / len(xs)
train = [c.tokens for c in chars[:N_TRAIN]]


def plain_trigram(train, order=3):          # copied from chars_licence.py (trees=None)
    counts = [defaultdict(Counter) for _ in range(order)]
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            for k in range(order):
                counts[k][tuple(seq[i - k:i])][seq[i]] += 1
    seen = {t for s in train for t in s}

    def p(ctx, t):
        pr = 1.0 / V
        for k in range(order):
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


row = {"seed": seed, "train": N_TRAIN, "test": len(test), "trigram": round(mean([plain_trigram(train)(s) for s in test]), 3)}
for refit in ("always", "set"):
    L = SubstitutionLearner(seed, order=3, refit=refit)
    for s in train:
        L.learn(s)
    row[f"trigram + licence (refit={refit})"] = round(mean([L.bits(s, escape=ESC, vocab_size=V) for s in test]), 3)
print(json.dumps(row), flush=True)
