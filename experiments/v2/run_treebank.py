"""TRELLIS v2 on real text: Penn Treebank WSJ10 (gold part-of-speech tags).

The NLTK sample of the treebank (see ``trellis2/treebank.py``) gives 542
sentences of at most ten tags; each seed splits them 80/20. Reported per
seed:

* unsupervised TRELLIS v2 (tag sequences only, one night = batch learning);
* supervised TRELLIS v2 trained on the right-binarized gold trees;
* right- and left-branching trees, the classic baselines;
* unigram and bigram tag models (add-1/2), as references for held-out bits.

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
        self.hit = self.gold = self.pred = 0

    def add(self, gold, pred, n):
        g, p = evaluable(gold, n), evaluable(pred, n)
        self.hit += len(g & p)
        self.gold += len(g)
        self.pred += len(p)

    def result(self):
        return {"omission": 1 - self.hit / max(self.gold, 1),
                "commission": 1 - self.hit / max(self.pred, 1)}


def baseline(test, brackets_of) -> dict:
    t = Tally()
    for s in test:
        t.add(s.brackets, brackets_of(len(s.tags)), len(s.tags))
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


def evaluate_model(chart_of, test) -> dict:
    t, bits, tops = Tally(), 0.0, []
    for s in test:
        chart = chart_of(s.tags)
        t.add(s.brackets, chart.mbr_tree().brackets(), len(s.tags))
        bits -= chart.log_prob / LN2
        tops.append(len(chart.viterbi_tree().roots))
    out = t.result()
    out.update(test_bits_per_sentence=bits / len(test), test_top_level_chunks=float(np.mean(tops)))
    return out


def run_seed(seed: int, root: str) -> dict:
    sentences = load_wsj(root)
    train, test = split(sentences, seed)
    row = {"seed": seed, "train": len(train), "test": len(test),
           "baselines": {
               "right-branching": baseline(test, lambda n: {(i, n) for i in range(n - 1)}),
               "left-branching": baseline(test, lambda n: {(0, j) for j in range(2, n + 1)}),
               "unigram bits/sentence": ngram_bits(train, test, 1),
               "bigram bits/sentence": ngram_bits(train, test, 2)}}
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
                 train_top_level_chunks=float(np.mean([len(t.roots) for t in learner.trees])))
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
               chunk_types=sg.info["chunk types"], seconds=time.time() - t0)
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
    lines = ["| Model | Bracket omission | Bracket commission | Held-out bits/sentence | Symbols | Chunk types |",
             "|---|---|---|---|---|---|"]
    for name in ("right-branching", "left-branching"):
        lines.append(f"| {name} | {m(lambda r: r['baselines'][name]['omission'], pct=True)} | "
                     f"{m(lambda r: r['baselines'][name]['commission'], pct=True)} | – | – | – |")
    for name, key in (("unigram tag model", "unigram bits/sentence"), ("bigram tag model", "bigram bits/sentence")):
        lines.append(f"| {name} | – | – | {m(lambda r: r['baselines'][key])} | – | – |")
    for name, side in (("TRELLIS v2, tags only (unsupervised)", "unsupervised"),
                       ("TRELLIS v2, binarized gold trees (supervised)", "supervised")):
        lines.append(f"| {name} | {m(lambda r: r[side]['omission'], pct=True)} | "
                     f"{m(lambda r: r[side]['commission'], pct=True)} | "
                     f"{m(lambda r: r[side]['test_bits_per_sentence'])} | "
                     f"{m(lambda r: r[side]['symbols'])} | {m(lambda r: r[side]['chunk_types'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="13,17")
    ap.add_argument("--root", default=default_ptb_root())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "treebank"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        rows = list(pool.map(run_seed, seeds, [args.root] * len(seeds)))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    table = summarise(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Penn Treebank WSJ10 (NLTK sample, {rows[0]['train']} training / {rows[0]['test']} test "
                f"sentences per seed; mean over seeds {args.seeds}).\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
