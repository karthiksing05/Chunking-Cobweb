"""Does the substitution learner's generator work on Chinese characters?

A character is a prefix IDS token sequence. A generated sequence is RIGHT when it is the
decomposition of a real (loaded) character that the learner was not taught.

Well-formed = the loader's own notion: trellis2.characters.parse_prefix(tokens) is not None
(one complete prefix expression; ARITY: U+2FF2 and U+2FF3 take 3 parts, the other ten operators 2).

Usage: chars_generate.py <seed> <checkpoints, e.g. 1000,4000> [G=2000]
"""
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.characters import default_ids_path, load_characters, parse_prefix  # noqa: E402
from trellis2.substitution import SubstitutionLearner  # noqa: E402

BOS, EOS = "<s>", "</s>"
seed = int(sys.argv[1])
checkpoints = [int(x) for x in sys.argv[2].split(",")]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

chars, _ = load_characters(default_ids_path(), seed=seed)
real = {tuple(c.tokens) for c in chars}


def well_formed(s):
    return parse_prefix(list(s)) is not None


def ref_swap(train, rng):
    frame = defaultdict(Counter)                      # (left, right) -> token counts
    for s in train:
        u = [BOS] + list(s) + [EOS]
        for i in range(1, len(u) - 1):
            frame[(u[i - 1], u[i + 1])][u[i]] += 1

    def gen():
        for _ in range(200):
            s = list(rng.choice(train))
            i = rng.randrange(len(s))
            l = s[i - 1] if i > 0 else BOS
            r = s[i + 1] if i + 1 < len(s) else EOS
            c = {y: n for y, n in frame[(l, r)].items() if y != s[i]}
            if not c:
                continue
            ys = sorted(c)
            y = rng.choices(ys, weights=[c[k] for k in ys])[0]
            return s[:i] + [y] + s[i + 1:]
        return None
    return gen


def ref_trigram(train, rng):
    nxt = defaultdict(Counter)
    for s in train:
        u = [BOS, BOS] + list(s) + [EOS]
        for i in range(2, len(u)):
            nxt[(u[i - 2], u[i - 1])][u[i]] += 1
    lo, hi = min(len(s) for s in train), max(len(s) for s in train)

    def gen():
        for _ in range(200):
            out, ctx = [], (BOS, BOS)
            while len(out) <= hi:
                c = nxt[ctx]
                ys = sorted(c)
                y = rng.choices(ys, weights=[c[k] for k in ys])[0]
                if y == EOS:
                    break
                out.append(y)
                ctx = (ctx[1], y)
            else:
                continue
            if lo <= len(out) <= hi:
                return out
        return None
    return gen


def report(seed_, n, name, outs, train_set, pairs):
    outs = [tuple(o) for o in outs if o is not None]
    k = len(outs)
    new = [o for o in outs if o not in train_set]
    rn = [o for o in new if o in real]

    def unstored(o):
        u = [BOS] + list(o) + [EOS]
        return any(p not in pairs for p in zip(u[1:-1], u[2:-1]))
    div = lambda a, b: round(a / b, 4) if b else None
    ex = [{"seq": " ".join(o), "real": o in real} for o in outs[:12]]
    return {"seed": seed_, "train": n, "generator": name, "generated": k,
            "new": div(len(new), k), "real and new": div(len(rn), k),
            "real among new": div(len(rn), len(new)),
            "well-formed": div(sum(well_formed(o) for o in outs), k),
            "distinct": len(set(outs)), "with an unstored pair": div(sum(unstored(o) for o in outs), k),
            "examples": ex}


t0 = time.time()
for n in checkpoints:
    train = [list(c.tokens) for c in chars[:n]]
    train_set = {tuple(s) for s in train}
    pairs = set()
    for s in train:
        pairs.update(zip(s, s[1:]))
    lens = [len(s) for s in train]
    print(json.dumps({"seed": seed, "train": n, "total characters": len(chars),
                      "distinct real sequences": len(real), "vocabulary": len({t for s in train for t in s}),
                      "training length range": [min(lens), max(lens)], "G": G,
                      "stored well-formed": round(sum(well_formed(s) for s in train) / n, 4)}), flush=True)
    L = SubstitutionLearner(seed)
    for s in train:
        L.learn(s, follows=False)
    gens = [
        ("learner, an unstored pair required", lambda r: L.generate(prev=None, rng=r, require_new=True)),
        ("learner, no requirement", lambda r: L.generate(prev=None, rng=r, require_new=False)),
        ("learner, no requirement, votes multiplied",
         lambda r: L.generate(prev=None, rng=r, require_new=False, vote="product")),
        ("reference: one token swapped in the same frame", None),
        ("reference: trigram sampler", None),
        ("reference: stored character", lambda r: list(r.choice(train))),
    ]
    for name, f in gens:
        rng = random.Random(seed + 7)
        if name.startswith("reference: one token"):
            g = ref_swap(train, rng)
            f = lambda r, g=g: g()
        elif name.startswith("reference: trigram"):
            g = ref_trigram(train, rng)
            f = lambda r, g=g: g()
        outs = [f(rng) for _ in range(G)]
        row = report(seed, n, name, outs, train_set, pairs)
        row["elapsed s"] = round(time.time() - t0)
        print(json.dumps(row, ensure_ascii=False), flush=True)
