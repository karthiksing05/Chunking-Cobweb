"""Unsupervised TRELLIS v2 (sentences only) on the paper's synthetic conditions.

For each condition and seed (v1 splits: 320 training sentences, 40 held-out),
the learner starts from flat sentences and builds chunks by minimum
description length. Reported, next to the supervised model trained on the
gold trees of the same sentences:

* description length of the training corpus: total, model and data bits;
* chunk inventory: symbols, rule classes, chunk types (distinct productions);
* held-out compression: bits per test sentence, -log2 P(sentence);
* generation: commission (share the target grammar rejects) and novelty;
* structure, for reference only: bracket omission/commission of the
  minimum-risk parse against gold, and the share of predicted brackets that
  cross no gold bracket. The learner is not asked to reproduce linguists'
  binarizations; these numbers show how close it comes.

Usage:
    python experiments/v2/run_unsupervised.py --out experiments/v2/results/unsupervised
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2, load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG, BracketTally  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402


def crosses(a, b) -> bool:
    (i, j), (k, l) = a, b
    return i < k < j < l or k < i < l < j


def evaluate(learner, grammar, test, cfg, train_sentences, seed, n_gen):
    tally, noncross, total, bits = BracketTally(), 0, 0, 0.0
    tops = []
    for e in test:
        chart = learner.chart(e.tokens)
        tree = chart.mbr_tree()
        tally.add(e.tree, tree)
        bits += -chart.log_prob / np.log(2)
        tops.append(len(chart.viterbi_tree().roots))
        gold = e.tree.brackets()
        for s in tree.brackets():
            total += 1
            noncross += not any(crosses(s, g) for g in gold)
    samples, _ = learner.generate(n_gen, np.random.default_rng(seed))
    gram = [cfg.recognizes(toks) for toks, _ in samples]
    novel = [" ".join(toks) not in train_sentences for toks, _ in samples]
    info = grammar.info
    return {
        "total_bits": info["total bits"], "model_bits": info["model bits"],
        "data_bits": info["data bits"], "symbols": grammar.K,
        "rule_classes": grammar.M, "chunk_types": info["chunk types"],
        "test_bits_per_sentence": bits / len(test),
        "test_top_level_chunks": float(np.mean(tops)),
        "omission": tally.omission, "parse_commission": tally.commission,
        "non_crossing": noncross / max(total, 1),
        "gen_commission": 1.0 - float(np.mean(gram)),
        "novelty": float(np.mean(novel)),
    }


def run_one(condition: str, seed: int, n_gen: int, data_root: str, search: dict) -> dict:
    t0 = time.time()
    examples = load_corpus(os.path.join(data_root, CONDITIONS[condition]))
    train, test = v1_split(examples, seed)
    cfg = CFG(target_grammar(condition))
    sentences = {e.sentence for e in train}

    learner = UnsupervisedLearner(seed=seed, **search)
    for e in train:
        learner.observe(e.tokens)
    g = learner.sleep()
    unsup = evaluate(learner, g, test, cfg, sentences, seed, n_gen)
    unsup["steps"] = len(learner.history) - 1
    unsup["history"] = [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in h.items()} for h in learner.history]

    sup_model = Trellis2(seed=seed)
    for e in train:
        sup_model.learn(e.tokens, e.tree)
    sg = sup_model.consolidate()
    sup = evaluate(sup_model, sg, test, cfg, sentences, seed, n_gen)

    print(f"[{condition} s{seed}] unsupervised: {unsup['total_bits']:.0f} bits, "
          f"{unsup['chunk_types']} chunk types, gen-commission {unsup['gen_commission']:.3f}, "
          f"test {unsup['test_bits_per_sentence']:.1f} b/s | gold trees: {sup['total_bits']:.0f} bits, "
          f"{sup['chunk_types']} chunk types ({time.time() - t0:.0f}s)", flush=True)
    return {"condition": condition, "seed": seed, "unsupervised": unsup, "supervised": sup,
            "seconds": time.time() - t0}


def summarise(rows) -> str:
    lines = ["| Condition | Train bits (unsup / gold trees) | Chunk types | Symbols | "
             "Test bits/sentence | Gen. commission | Novelty | Bracket omission | Non-crossing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        def m(side, key, pct=False, fmt="{:.0f}"):
            v = np.mean([r[side][key] for r in rs])
            return f"{100 * v:.1f}%" if pct else fmt.format(v)
        lines.append(
            f"| {cond} | {m('unsupervised', 'total_bits')} / {m('supervised', 'total_bits')} | "
            f"{m('unsupervised', 'chunk_types', fmt='{:.1f}')} / {m('supervised', 'chunk_types', fmt='{:.1f}')} | "
            f"{m('unsupervised', 'symbols', fmt='{:.1f}')} / {m('supervised', 'symbols', fmt='{:.1f}')} | "
            f"{m('unsupervised', 'test_bits_per_sentence', fmt='{:.1f}')} / {m('supervised', 'test_bits_per_sentence', fmt='{:.1f}')} | "
            f"{m('unsupervised', 'gen_commission', pct=True)} / {m('supervised', 'gen_commission', pct=True)} | "
            f"{m('unsupervised', 'novelty', pct=True)} | "
            f"{m('unsupervised', 'omission', pct=True)} | {m('unsupervised', 'non_crossing', pct=True)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--levels", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "unsupervised"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, int(s)) for c in args.conditions.split(",") for s in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        search = {"beam": args.beam, "patience": args.patience, "levels": args.levels}
        rows = [f.result() for f in [pool.submit(run_one, c, s, args.n_gen, args.data_root, search)
                                     for c, s in jobs]]
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    table = summarise(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("Each cell: unsupervised / supervised on gold trees, where both apply.\n\n" + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
