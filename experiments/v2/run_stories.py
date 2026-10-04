"""TRELLIS v2 on simple English: sentences of TinyStories (see
``trellis2/stories.py``), learned from the sentences alone.

The sentences are those of 3 to 8 words that use only the 250 most frequent
words of the TinyStories validation file (about 54,000 sentences). Each seed
shuffles them, holds out 500 and learns from the next ``--train``.

Reported, with word unigram, bigram and trigram models as references:

* held-out bits per sentence (summed over all analyses);
* the grammar: categories, chunk types, how often a sentence is analysed as
  one whole tree;
* generated sentences (1,000): new (not among the training sentences), real
  (the sentence occurs, word for word, somewhere among the 497,000
  sentences of TinyStories), and the share of their word pairs and triples
  that occur in TinyStories;
* consistency: a generated sentence is perceived again (Viterbi analysis)
  with the analysis it was generated from.

Usage:
    python experiments/v2/run_stories.py --train 5000 --out experiments/v2/results/stories
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

LN2 = math.log(2)


def ngram_model(train, order, alpha=0.1):
    vocab = sorted({w for s in train for w in s} | {"</s>"})
    counts, ctx = defaultdict(Counter), Counter()
    for s in train:
        seq = ["<s>"] * (order - 1) + s + ["</s>"]
        for i in range(order - 1, len(seq)):
            h = tuple(seq[i - order + 1:i])
            counts[h][seq[i]] += 1
            ctx[h] += 1
    return vocab, counts, ctx


def ngram_bits(model, test, order, alpha=0.1):
    vocab, counts, ctx = model
    total = 0.0
    for s in test:
        seq = ["<s>"] * (order - 1) + s + ["</s>"]
        for i in range(order - 1, len(seq)):
            h = tuple(seq[i - order + 1:i])
            total -= math.log2((counts[h][seq[i]] + alpha) / (ctx[h] + alpha * len(vocab)))
    return total / len(test)


def ngram_sample(model, order, rng, max_len=20):
    vocab, counts, _ = model
    seq = ["<s>"] * (order - 1)
    out = []
    while len(out) < max_len:
        c = counts[tuple(seq[len(seq) - order + 1:])] if order > 1 else counts[()]
        words = list(c)
        w = words[rng.choice(len(words), p=np.array([c[x] for x in words], float) / sum(c.values()))]
        if w == "</s>":
            break
        out.append(w)
        seq.append(w)
    return out


def generation_measures(samples, train_set, corpus_set, corpus_grams):
    n = len(samples)
    new = [s for s in samples if tuple(s) not in train_set]
    pairs = [tuple(s[i:i + 2]) in corpus_grams for s in samples for i in range(len(s) - 1)]
    triples = [tuple(s[i:i + 3]) in corpus_grams for s in samples for i in range(len(s) - 2)]
    return {"new": len(new) / n,
            "real": sum(tuple(s) in corpus_set for s in samples) / n,
            "new and real": sum(tuple(s) in corpus_set for s in new) / n,
            "word pairs in TinyStories": float(np.mean(pairs)) if pairs else 0.0,
            "word triples in TinyStories": float(np.mean(triples)) if triples else 0.0,
            "mean length": float(np.mean([len(s) for s in samples]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--vocab", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "stories"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    corpus = read_sentences()
    corpus_set = {tuple(s) for s in corpus}
    corpus_grams = {tuple(s[i:i + n]) for s in corpus for n in (2, 3) for i in range(len(s) - n + 1)}
    simple = simple_sentences(corpus, args.vocab, 3, args.max_len)
    order = np.random.default_rng(args.seed).permutation(len(simple))
    test = [simple[i] for i in order[:args.test]]
    train = [simple[i] for i in order[args.test:args.test + args.train]]
    train_set = {tuple(s) for s in train}
    print(f"{len(corpus)} sentences in TinyStories, {len(simple)} simple ones; learning from {len(train)}, "
          f"holding out {len(test)}", flush=True)

    t0 = time.time()
    learner = UnsupervisedLearner(seed=args.seed)
    for s in train:
        learner.observe(s)
    g = learner.sleep()
    seconds = time.time() - t0
    tops = [len(t.roots) for t in learner.trees]
    held = sum(-learner.chart(s).log_prob / LN2 for s in test) / len(test)
    models = {k: ngram_model(train, k) for k in (1, 2, 3)}
    baselines = {f"{k}-gram": ngram_bits(models[k], test, k) for k in (1, 2, 3)}
    print(f"night {seconds:.0f}s: {g.K} symbols, {g.info['chunk types']} chunk types, "
          f"{np.mean(tops):.2f} top-level chunks per sentence, whole trees {np.mean([t == 1 for t in tops]):.1%}; "
          f"held out {held:.1f} bits/sentence vs {baselines}", flush=True)

    rng = np.random.default_rng(args.seed)
    gen, _ = learner.generate(args.n_gen, rng, max_len=20)
    samples = [tokens for tokens, _ in gen]
    consistent = 0
    for tokens, tree in gen:
        seen = learner.analyse(tokens)
        consistent += seen.brackets() == tree.brackets() and seen.roots == tree.roots
    measures = {"TRELLIS v2": generation_measures(samples, train_set, corpus_set, corpus_grams)}
    measures["TRELLIS v2"]["perceived with the analysis it was generated from"] = consistent / len(gen)
    for k in (2, 3):
        smp = [ngram_sample(models[k], k, rng) for _ in range(args.n_gen)]
        measures[f"word {k}-gram"] = generation_measures([s for s in smp if s] or [["-"]], train_set,
                                                           corpus_set, corpus_grams)
    print(json.dumps(measures, indent=1), flush=True)

    # The grammar: categories and chunk types.
    by_size = np.argsort([-sum(y.values()) for y in g.symbol_yields])
    categories = [{"symbol": int(a), "count": float(sum(g.symbol_yields[a].values())),
                   "yields": g.symbol_yields[a].most_common(8)} for a in by_size]
    chunks = Counter()
    for tokens, tree in zip(learner.sentences, learner.trees):
        for (i, j) in tree.brackets():
            if j - i >= 2 and (i, j) not in tree.roots:
                chunks[" ".join(tokens[i:j])] += 1
    examples = {"held-out analyses": [learner.parse(s).to_string(s) for s in test[:25]],
                "generated": [tree.to_string(tokens) + ("" if tuple(tokens) in train_set else "  (new)")
                              for tokens, tree in gen[:40]]}
    results = {"train": len(train), "test": len(test), "vocabulary": args.vocab, "seconds": seconds,
               "symbols": g.K, "rule classes": g.M, "chunk types": g.info["chunk types"],
               "training bits": g.info["total bits"], "top-level chunks per sentence": float(np.mean(tops)),
               "whole trees": float(np.mean([t == 1 for t in tops])),
               "held-out bits per sentence": held, "baselines": baselines, "generation": measures,
               "categories": categories, "frequent chunks": chunks.most_common(60),
               "examples": examples, "history": learner.history}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    lines = [f"TinyStories, sentences of 3–{args.max_len} words over the {args.vocab} most frequent words: "
             f"{len(train)} learned from sentences alone, {len(test)} held out (seed {args.seed}).", "",
             "| Model | Held-out bits per sentence |", "|---|---|"]
    lines += [f"| word {k} model | {v:.1f} |" for k, v in baselines.items()]
    lines += [f"| TRELLIS v2 | {held:.1f} |", "",
              f"Grammar: {g.K} categories, {g.info['chunk types']} chunk types; "
              f"{np.mean([t == 1 for t in tops]):.0%} of training sentences analysed as one tree "
              f"({np.mean(tops):.2f} top-level chunks per sentence).", "",
              f"| Generated sentences ({args.n_gen:,}) | " + " | ".join(measures) + " |",
              "|---|" + "---|" * len(measures)]
    for k in measures["TRELLIS v2"]:
        lines.append(f"| {k} | " + " | ".join(
            (f"{m[k]:.1%}" if k != "mean length" else f"{m[k]:.1f}") if k in m else "–"
            for m in measures.values()) + " |")
    lines += ["", "Largest categories:", ""]
    for c in categories[:12]:
        lines.append(f"- S{c['symbol']} ({c['count']:.0f}): " + ", ".join(y for y, _ in c["yields"]))
    lines += ["", "Most frequent chunks inside training analyses: "
              + ", ".join(f"*{c}* ({n})" for c, n in chunks.most_common(20)), ""]
    lines += ["", "Generated sentences (analysis as generated):", ""]
    lines += [f"    {x}" for x in examples["generated"][:25]]
    lines += ["", "Held-out sentences (minimum-risk analysis):", ""]
    lines += [f"    {x}" for x in examples["held-out analyses"][:15]]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
