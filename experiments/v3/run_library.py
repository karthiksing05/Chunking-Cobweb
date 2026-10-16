"""The chunk library (trellis2.library) on any domain: learns every candidate of
each training experience, then reports what the user cares about:

* compression: bits per held-out experience under the library's shortest code,
  against a unigram and a bigram code (Laplace, with an end marker);
* the library: how many distinct chunk concepts the training reads use, and how
  consistently held-out reads reuse them (share of held-out composites coded
  through a concept used in at least 2 / 5 training experiences);
* generation: samples from the same code (synthetic: accepted by the generating
  grammar; TinyStories: every word bigram / trigram attested in the corpus;
  characters: well formed), and novel;
* a side diagnostic: held-out bracket omission against gold, and right-branching.

Usage: python experiments/v3/run_library.py synthetic --conditions small,med --seeds 13 --out ...
       python experiments/v3/run_library.py stories --train 2000 --out ...
       python experiments/v3/run_library.py characters --train 1000 --out ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from trellis2.data import Tree  # noqa: E402
from trellis2.evaluation import BracketTally  # noqa: E402
from trellis2.library import ChunkLibrary  # noqa: E402

LN2 = math.log(2)


def right_branching(n: int) -> Tree:
    return Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)])


def trigram_bits(train, test):
    """A Witten-Bell interpolated trigram (with an end marker), bits per test experience."""
    from collections import defaultdict
    c1, c2, c3 = Counter(), defaultdict(Counter), defaultdict(Counter)
    for s in train:
        h = ["<s>", "<s>"] + list(s) + ["</s>"]
        for t in range(2, len(h)):
            c1[h[t]] += 1
            c2[h[t - 1]][h[t]] += 1
            c3[(h[t - 2], h[t - 1])][h[t]] += 1
    V = len(c1) + 1
    N = sum(c1.values())

    def wb(cnt, lower):
        n, types = sum(cnt.values()), len(cnt)
        return lambda w: (cnt.get(w, 0) + types * lower(w)) / (n + types) if n else lower(w)
    bits = 0.0
    for s in test:
        h = ["<s>", "<s>"] + list(s) + ["</s>"]
        for t in range(2, len(h)):
            w = h[t]
            p1 = lambda x: (c1.get(x, 0) + 1) / (N + V)
            p2 = wb(c2.get(h[t - 1], Counter()), p1)
            p3 = wb(c3.get((h[t - 2], h[t - 1]), Counter()), p2)
            bits -= math.log2(p3(w))
    return bits / len(test)


def ngram_bits(train, test):
    """Unigram and bigram (Laplace, end marker) bits per test experience."""
    uni, big, ctx = Counter(), Counter(), Counter()
    for s in train:
        prev = "<s>"
        for w in list(s) + ["</s>"]:
            uni[w] += 1
            big[(prev, w)] += 1
            ctx[prev] += 1
            prev = w
    V = len(uni) + 1
    N = sum(uni.values())
    u_bits = b_bits = 0.0
    for s in test:
        prev = "<s>"
        for w in list(s) + ["</s>"]:
            u_bits -= math.log2((uni[w] + 1) / (N + V))
            b_bits -= math.log2((big[(prev, w)] + 1) / (ctx[prev] + V))
            prev = w
    return u_bits / len(test), b_bits / len(test)


def load(args, seed):
    """(train items, test items, gold or None, judge(tokens) -> bool or None, name)."""
    if args.domain == "synthetic":
        from trellis2 import load_corpus, v1_split
        from trellis2.data import CONDITIONS, default_data_root, target_grammar
        from trellis2.evaluation import CFG
        out = []
        for cond in args.conditions.split(","):
            train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
            g = CFG(target_grammar(cond))
            out.append((cond, [e.tokens for e in train], [e.tokens for e in test],
                        [e.tree for e in test], g.recognizes))
        return out
    if args.domain == "characters":
        from trellis2.characters import default_ids_path, load_characters, parse_prefix
        chars, _ = load_characters(default_ids_path(), seed=seed)
        train, test = chars[:args.train], chars[-args.test:]
        return [("characters", [c.tokens for c in train], [c.tokens for c in test], [c.tree for c in test],
                 lambda t: parse_prefix(t) is not None)]
    from trellis2.stories import read_sentences, simple_sentences
    import random
    corpus = read_sentences()
    simple = simple_sentences(corpus, args.vocab, 3, args.max_len)
    rng = random.Random(seed)
    rng.shuffle(simple)
    test, train = simple[:args.test], simple[args.test:args.test + args.train]
    bigrams = {tuple(s[i:i + 2]) for s in corpus for i in range(len(s) - 1)}
    trigrams = {tuple(s[i:i + 3]) for s in corpus for i in range(len(s) - 2)}

    def attested(t):
        return (all(tuple(t[i:i + 2]) in bigrams for i in range(len(t) - 1))
                and all(tuple(t[i:i + 3]) in trigrams for i in range(len(t) - 2)))
    return [("stories", train, test, None, attested)]


SKIP_READ = {"on": False}


def args_skip_read():
    return SKIP_READ["on"]


def evaluate(name, seed, train, test, gold, judge, cfg, n_gen, n_reads):
    t0 = time.time()
    L = ChunkLibrary(seed=seed, **cfg)
    for x in train:
        L.learn(x)
    learn_s = time.time() - t0
    # the library as the training reads use it
    used = Counter()
    for x in ([] if args_skip_read() else train[:n_reads]):
        used.update(set(L.read(x)["concepts"]))
    reads = [L.read(x) for x in test] if not args_skip_read() else []
    pred_bits = sum(L.code_bits(x) for x in test) / len(test)
    units = [L.unit_code(x) for x in test] if cfg.get("prediction") == "mixture" else []
    unit_bits = sum(u["bits"] for u in units) / len(units) if units else float("nan")
    chunk_use = Counter(u for r in units for u in r["units"] if len(u) > 1)
    tri = trigram_bits(train, test)
    bits = sum(r["bits"] for r in reads) / len(reads) if reads else float("nan")
    uni, big = ngram_bits(train, test)
    held = [c for r in reads for c in r["concepts"]]
    reuse2 = sum(used[c] >= 2 for c in held) / max(len(held), 1)
    reuse5 = sum(used[c] >= 5 for c in held) / max(len(held), 1)
    row = {"domain": name, "seed": seed, "config": cfg, "train": len(train),
           "bits per experience": bits, "predictive bits": pred_bits, "unit bits": unit_bits,
           "trigram bits": tri, "unigram bits": uni, "bigram bits": big,
           "chunks used in held-out units (distinct, uses)": [len(chunk_use), sum(chunk_use.values())],
           "top chunks": [" ".join(k) for k, _ in chunk_use.most_common(15)],
           "library concepts (training reads)": len(used),
           "held-out composites through concepts reused in >=2 / >=5 training reads": [reuse2, reuse5],
           "learn seconds": learn_s, "stats": L.stats}
    if gold is not None:
        te, rb = BracketTally(), BracketTally()
        for r, g in zip(reads, gold):
            te.add(g, r["tree"])
            rb.add(g, right_branching(g.n))
        row.update({"omission (side)": te.omission, "right-branching omission (side)": rb.omission})
    samples = [s for s in (L.generate() for _ in range(n_gen)) if s]
    seen = {tuple(x) for x in train}
    row.update({"generated": len(samples) / n_gen,
                "coherent (judge)": sum(judge(s) for s in samples) / max(len(samples), 1),
                "novel": sum(tuple(s) not in seen for s in samples) / max(len(samples), 1),
                "coherent and novel": sum(judge(s) and tuple(s) not in seen for s in samples) / max(len(samples), 1),
                "mean length": sum(map(len, samples)) / max(len(samples), 1),
                "examples": [" ".join(s) + ("  ✓" if judge(s) else "  ✗") for s in samples[:15]],
                "reads": [r["tree"].to_string(x) for r, x in zip(reads[:8], test[:8])],
                "seconds": time.time() - t0})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=("synthetic", "characters", "stories"))
    ap.add_argument("--conditions", default="small,med,large,term_low,term_med,term_high")
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--n-reads", type=int, default=300)
    ap.add_argument("--skip-read", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(args.config)
    SKIP_READ["on"] = args.skip_read
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, train, test, gold, judge in load(args, seed):
            r = evaluate(name, seed, train, test, gold, judge, cfg, args.n_gen, args.n_reads)
            rows.append(r)
            side = (f" | omission {r['omission (side)']:.2f} (RB {r['right-branching omission (side)']:.2f})"
                    if "omission (side)" in r else "")
            print(f"[{name} s{seed}] predictive {r['predictive bits']:.1f} units {r['unit bits']:.1f} "
                  f"chunks {r['chunks used in held-out units (distinct, uses)']} (trigram {r['trigram bits']:.1f}, unigram {r['unigram bits']:.1f}, bigram "
                  f"{r['bigram bits']:.1f}) | library {r['library concepts (training reads)']} concepts, held-out reuse "
                  f"{r['held-out composites through concepts reused in >=2 / >=5 training reads'][0]:.2f}/"
                  f"{r['held-out composites through concepts reused in >=2 / >=5 training reads'][1]:.2f} | gen coherent "
                  f"{r['coherent (judge)']:.2f} novel {r['novel']:.2f} len {r['mean length']:.1f}{side} | "
                  f"{r['seconds']:.0f}s", flush=True)
            for x in r["reads"][:3]:
                print("    read:", x, flush=True)
            for x in r["examples"][:5]:
                print("    gen:", x, flush=True)
            with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
