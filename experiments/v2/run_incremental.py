"""Learning by day and by night versus batch learning, on the synthetic conditions.

Incremental: the learner perceives the training sentences one at a time with
its current grammar and sleeps when the number of sentences reaches a
checkpoint (10, 20, 40, ...), each night starting from the stored analyses.
Batch: at every checkpoint a fresh learner observes the same prefix and
sleeps once. After every night both are evaluated as in
``run_unsupervised.py``; the incremental learner also reports how well each
day's sentences were perceived before the night (bits per sentence under the
grammar of the morning, and top-level chunks per sentence).

Usage:
    python experiments/v2/run_incremental.py --out experiments/v2/results/incremental
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
sys.path.insert(0, HERE)

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
from run_unsupervised import evaluate  # noqa: E402

CHECKPOINTS = [10, 20, 40, 80, 160, 320]
LN2 = np.log(2)


def night_record(learner) -> dict:
    """Which starting point won the structure search, and how long it took."""
    night = [h for h in learner.history if h["night"] == learner.nights - 1]
    structure = next(h for h in night if h["stage"] == "structure")
    return {"night_start": structure["move"].split(" from ")[-1],
            "search_seconds": structure["seconds"]}


def _setup(condition, seed, data_root):
    examples = load_corpus(os.path.join(data_root, CONDITIONS[condition]))
    train, test = v1_split(examples, seed)
    return train, test, CFG(target_grammar(condition))


def run_incremental(condition, seed, n_gen, data_root, search):
    train, test, cfg = _setup(condition, seed, data_root)
    learner = UnsupervisedLearner(seed=seed, **search)
    rows, day_bits, day_tops = [], [], []
    for i, e in enumerate(train[:CHECKPOINTS[-1]]):
        if learner.model is not None:
            day_bits.append(-learner.chart(e.tokens).log_prob / LN2)
        tree = learner.observe(e.tokens)
        if tree is not None:
            day_tops.append(len(tree.roots))
        if i + 1 in CHECKPOINTS:
            t0 = time.time()
            g = learner.sleep()
            seconds = time.time() - t0
            sentences = {x.sentence for x in train[:i + 1]}
            row = evaluate(learner, g, test, cfg, sentences, seed, n_gen)
            row.update(mode="incremental", condition=condition, seed=seed, sentences=i + 1,
                       night_seconds=seconds, **night_record(learner),
                       day_bits_per_sentence=float(np.mean(day_bits)) if day_bits else None,
                       day_top_level_chunks=float(np.mean(day_tops)) if day_tops else None)
            rows.append(row)
            print(f"[{condition} s{seed} incremental {i + 1}] {row['total_bits']:.0f} bits, "
                  f"gen-commission {row['gen_commission']:.3f}, test {row['test_bits_per_sentence']:.1f} b/s, "
                  f"night {seconds:.0f}s", flush=True)
            day_bits, day_tops = [], []
    return rows


def run_batch(condition, seed, n_gen, data_root, search, n):
    train, test, cfg = _setup(condition, seed, data_root)
    learner = UnsupervisedLearner(seed=seed, **search)
    for e in train[:n]:
        learner.observe(e.tokens)
    t0 = time.time()
    g = learner.sleep()
    seconds = time.time() - t0
    row = evaluate(learner, g, test, cfg, {x.sentence for x in train[:n]}, seed, n_gen)
    row.update(mode="batch", condition=condition, seed=seed, sentences=n, night_seconds=seconds,
               **night_record(learner))
    print(f"[{condition} s{seed} batch {n}] {row['total_bits']:.0f} bits, "
          f"gen-commission {row['gen_commission']:.3f}, test {row['test_bits_per_sentence']:.1f} b/s, "
          f"{seconds:.0f}s", flush=True)
    return [row]


def summarise(rows) -> str:
    lines = ["| Condition | Sentences | Train bits/sentence (incr / batch) | "
             "Test bits/sentence (incr / batch) | Gen. commission (incr / batch) | "
             "Chunk types (incr / batch) | Night seconds (incr / batch) | "
             "Day: bits/sentence before the night | Day: top-level chunks |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for n in CHECKPOINTS:
            sel = {m: [r for r in rows if r["condition"] == cond and r["sentences"] == n
                       and r["mode"] == m] for m in ("incremental", "batch")}
            if not sel["incremental"] or not sel["batch"]:
                continue

            def m(key, mode, fmt="{:.1f}", scale=1.0):
                vals = [r[key] * scale / (r["sentences"] if key == "total_bits" else 1)
                        for r in sel[mode] if r.get(key) is not None]
                return fmt.format(np.mean(vals)) if vals else "–"

            pair = lambda key, fmt="{:.1f}", scale=1.0: (f"{m(key, 'incremental', fmt, scale)} / "
                                                         f"{m(key, 'batch', fmt, scale)}")
            lines.append(
                f"| {cond} | {n} | {pair('total_bits')} | {pair('test_bits_per_sentence')} | "
                f"{pair('gen_commission', '{:.1f}%', 100)} | {pair('chunk_types')} | "
                f"{pair('night_seconds', '{:.0f}')} | {m('day_bits_per_sentence', 'incremental')} | "
                f"{m('day_top_level_chunks', 'incremental', '{:.2f}')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--levels", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "incremental"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    search = {"beam": args.beam, "patience": args.patience, "levels": args.levels}
    jobs = []
    for c in args.conditions.split(","):
        for s in map(int, args.seeds.split(",")):
            jobs.append((run_incremental, (c, s, args.n_gen, args.data_root, search)))
            jobs += [(run_batch, (c, s, args.n_gen, args.data_root, search, n)) for n in CHECKPOINTS]
    # Longest jobs first.
    jobs.sort(key=lambda j: -(1000 if j[0] is run_incremental else j[1][-1]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for f in [pool.submit(fn, *a) for fn, a in jobs] for r in f.result()]
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    table = summarise(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("Incremental (sleep at each checkpoint) / batch (one sleep over the prefix).\n\n"
                + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
