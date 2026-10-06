"""TRELLIS v2 on real text: Penn Treebank WSJ10 (gold part-of-speech tags).

The NLTK sample of the treebank (see ``trellis2/treebank.py``) gives 542
sentences of at most ten tags; each seed splits them 80/20. Reported per
seed:

* unsupervised TRELLIS v2 (tag sequences only, one night = batch learning);
* supervised TRELLIS v2 trained on the right-binarized gold trees;
* right- and left-branching trees, the classic baselines;
* unigram and bigram tag models (add-1/2), as references for held-out bits;
* generation (1,000 tag sequences each, of the training length): how many
  occur in the treebank sample, and how many have every tag triple, the
  sequence's edges included, somewhere in it, for TRELLIS v2's own
  sequences (those it derives as one tree), all its samples, and a tag
  bigram's.

Parsing is scored with unlabelled brackets (omission = gold brackets missed,
commission = predicted brackets not in the gold tree), ignoring single tags
and the whole sentence (Klein & Manning 2002). Gold trees are n-ary, so every
binary parse carries some unavoidable commission.

Usage:
    python experiments/v2/run_treebank.py --seeds 13,17 --out experiments/v2/results/treebank
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2  # noqa: E402
from trellis2.treebank import default_ptb_root, evaluable, load_wsj, split  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

LN2 = math.log(2)


class Tally:
    def __init__(self):
        self.hit = self.gold = self.pred = self.base_hit = self.base = 0

    def add(self, sentence, pred):
        n = len(sentence.tags)
        g, p = evaluable(sentence.brackets, n), evaluable(pred, n)
        self.hit += len(g & p)
        self.gold += len(g)
        self.pred += len(p)
        base = evaluable(sentence.base, n)
        self.base_hit += len(base & p)
        self.base += len(base)

    def result(self):
        return {"omission": 1 - self.hit / max(self.gold, 1),
                "commission": 1 - self.hit / max(self.pred, 1),
                "base_phrase_omission": 1 - self.base_hit / max(self.base, 1)}


def baseline(test, brackets_of) -> dict:
    t = Tally()
    for s in test:
        t.add(s, brackets_of(len(s.tags)))
    return t.result()


def ngram_bits(train, test, order: int, alpha: float = 0.5) -> float:
    """Held-out bits per sentence of an add-alpha tag n-gram model (with an
    end-of-sentence symbol)."""
    vocab = sorted({t for s in train for t in s.tags} | {"</s>", "<unk>"})
    counts, ctx = Counter(), Counter()
    for s in train:
        seq = ["<s>"] + s.tags + ["</s>"]
        for i in range(1, len(seq)):
            h = tuple(seq[max(0, i - order + 1):i]) if order > 1 else ()
            counts[(h, seq[i])] += 1
            ctx[h] += 1
    total = 0.0
    for s in test:
        seq = ["<s>"] + [t if t in vocab else "<unk>" for t in s.tags] + ["</s>"]
        for i in range(1, len(seq)):
            h = tuple(seq[max(0, i - order + 1):i]) if order > 1 else ()
            total -= math.log2((counts[(h, seq[i])] + alpha) / (ctx[h] + alpha * len(vocab)))
    return total / len(test)


def generation(samples, attested, length) -> dict:
    """Of the samples of the training length: real (the tag sequence occurs
    in the treebank sample) and every tag triple attested (edges included)."""
    lo, hi = length
    ins = [tuple(t) for t in samples if lo <= len(t) <= hi]
    seqs, triples = attested
    def every(t):
        p = ("<s>",) + t + ("</s>",)
        return all(p[i:i + 3] in triples for i in range(len(p) - 2))
    return {"of the training length": len(ins) / max(len(samples), 1),
            "real": float(np.mean([t in seqs for t in ins])) if ins else 0.0,
            "every triple attested": float(np.mean([every(t) for t in ins])) if ins else 0.0}


def bigram_samples(train, n: int, rng: np.random.Generator, max_len: int = 40) -> list:
    """Tag sequences from the maximum-likelihood tag bigram of the training sentences."""
    nxt = {}
    for s in train:
        p = ["<s>"] + list(s.tags) + ["</s>"]
        for a, b in zip(p, p[1:]):
            nxt.setdefault(a, Counter())[b] += 1
    out = []
    while len(out) < n:
        t, cur = [], "<s>"
        while len(t) <= max_len:
            c = nxt[cur]
            keys = list(c)
            cur = keys[int(rng.choice(len(keys), p=np.array([c[k] for k in keys], dtype=float) / sum(c.values())))]
            if cur == "</s>":
                break
            t.append(cur)
        if len(t) <= max_len:
            out.append(t)
    return out


def evaluate_model(chart_of, test) -> dict:
    t, bits, tops = Tally(), 0.0, []
    for s in test:
        chart = chart_of(s.tags)
        t.add(s, chart.mbr_tree().brackets())
        bits -= chart.log_prob / LN2
        tops.append(len(chart.viterbi_tree().roots))
    out = t.result()
    out.update(test_bits_per_sentence=bits / len(test), test_top_level_chunks=float(np.mean(tops)))
    return out


def run_seed(seed: int, root: str, train_max_len: int = 10) -> dict:
    """Test on held-out WSJ10 sentences; train on the rest of WSJ10 or, with
    ``train_max_len`` > 10, on every other sentence of up to that many tags."""
    sentences = load_wsj(root)
    train, test = split(sentences, seed)
    if train_max_len > 10:
        held_out = {tuple(s.words) for s in test}
        train = [s for s in load_wsj(root, max_len=train_max_len) if tuple(s.words) not in held_out]
    everything = load_wsj(root, max_len=1000)
    attested = ({tuple(s.tags) for s in everything},
                {p[i:i + 3] for p in (("<s>",) + tuple(s.tags) + ("</s>",) for s in everything)
                 for i in range(len(p) - 2)})
    length = (min(len(s.tags) for s in train), max(len(s.tags) for s in train))
    rng = np.random.default_rng(seed)
    row = {"seed": seed, "train": len(train), "test": len(test), "train_max_len": train_max_len,
           "baselines": {
               "right-branching": baseline(test, lambda n: {(i, n) for i in range(n - 1)}),
               "left-branching": baseline(test, lambda n: {(0, j) for j in range(2, n + 1)}),
               "unigram bits/sentence": ngram_bits(train, test, 1),
               "bigram bits/sentence": ngram_bits(train, test, 2),
               "bigram generation": generation(bigram_samples(train, 1000, rng), attested, length)}}
    t0 = time.time()
    learner = UnsupervisedLearner(seed=seed)
    for s in train:
        learner.observe(s.tags)
    g = learner.sleep()
    unsup = evaluate_model(learner.chart, test)
    unsup.update(total_bits=g.info["total bits"], symbols=g.K, rule_classes=g.M,
                 chunk_types=g.info["chunk types"], seconds=time.time() - t0,
                 history=[{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in h.items()}
                          for h in learner.history],
                 train_top_level_chunks=float(np.mean([len(t.roots) for t in learner.trees])),
                 own_generation=generation([t for t, _ in learner.generate(1000, rng, whole_only=True)[0]],
                                           attested, length),
                 all_generation=generation([t for t, _ in learner.generate(1000, rng)[0]], attested, length))
    row["unsupervised"] = unsup
    print(f"[seed {seed}] unsupervised: omission {unsup['omission']:.3f}, commission "
          f"{unsup['commission']:.3f}, {unsup['test_bits_per_sentence']:.1f} b/s "
          f"({unsup['seconds']:.0f}s)", flush=True)
    t0 = time.time()
    model = Trellis2(seed=seed)
    for s in train:
        model.learn(s.tags, s.tree)
    sg = model.consolidate()
    sup = evaluate_model(model.chart, test)
    sup.update(total_bits=sg.info["total bits"], symbols=sg.K, rule_classes=sg.M,
               chunk_types=sg.info["chunk types"], seconds=time.time() - t0,
               own_generation=generation([t for t, _ in model.generate(1000, rng, whole_only=True)[0]],
                                         attested, length))
    row["supervised"] = sup
    print(f"[seed {seed}] supervised: omission {sup['omission']:.3f}, commission "
          f"{sup['commission']:.3f}, {sup['test_bits_per_sentence']:.1f} b/s ({sup['seconds']:.0f}s)",
          flush=True)
    return row


def summarise(rows) -> str:
    def m(get, fmt="{:.1f}", pct=False):
        vals = [get(r) for r in rows]
        v = float(np.mean(vals))
        return f"{100 * v:.1f}%" if pct else fmt.format(v)
    lines = ["| Model | Bracket omission | Bracket commission | Base-phrase omission | "
             "Held-out bits/sentence | Symbols | Chunk types |",
             "|---|---|---|---|---|---|---|"]
    for name in ("right-branching", "left-branching"):
        b = lambda key: m(lambda r: r['baselines'][name][key], pct=True)
        lines.append(f"| {name} | {b('omission')} | {b('commission')} | {b('base_phrase_omission')} "
                     f"| – | – | – |")
    for name, key in (("unigram tag model", "unigram bits/sentence"), ("bigram tag model", "bigram bits/sentence")):
        lines.append(f"| {name} | – | – | – | {m(lambda r: r['baselines'][key])} | – | – |")
    for name, side in (("TRELLIS v2, tags only (unsupervised)", "unsupervised"),
                       ("TRELLIS v2, binarized gold trees (supervised)", "supervised")):
        lines.append(f"| {name} | {m(lambda r: r[side]['omission'], pct=True)} | "
                     f"{m(lambda r: r[side]['commission'], pct=True)} | "
                     f"{m(lambda r: r[side]['base_phrase_omission'], pct=True)} | "
                     f"{m(lambda r: r[side]['test_bits_per_sentence'])} | "
                     f"{m(lambda r: r[side]['symbols'])} | {m(lambda r: r[side]['chunk_types'])} |")
    lines += ["", "| Generated tag sequences (1,000 each) | Of the training length | Real, among those | "
              "Every tag triple attested, among those |", "|---|---|---|---|"]
    for name, get in (("tag bigram", lambda r: r["baselines"]["bigram generation"]),
                      ("TRELLIS v2, tags only: its own sequences", lambda r: r["unsupervised"]["own_generation"]),
                      ("TRELLIS v2, tags only: all samples", lambda r: r["unsupervised"]["all_generation"]),
                      ("TRELLIS v2, gold trees: its own sequences", lambda r: r["supervised"]["own_generation"])):
        lines.append(f"| {name} | " + " | ".join(m(lambda r, k=k: get(r)[k], pct=True)
                     for k in ("of the training length", "real", "every triple attested")) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="13,17")
    ap.add_argument("--root", default=default_ptb_root())
    ap.add_argument("--train-max-len", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "treebank"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        rows = list(pool.map(run_seed, seeds, [args.root] * len(seeds),
                             [args.train_max_len] * len(seeds)))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    table = summarise(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Penn Treebank, NLTK sample: trained on {rows[0]['train']} sentences of up to "
                f"{args.train_max_len} tags, tested on {rows[0]['test']} held-out WSJ10 sentences "
                f"(mean over seeds {args.seeds}).\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
