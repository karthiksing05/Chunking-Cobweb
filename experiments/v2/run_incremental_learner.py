"""The continuously incremental learner (no night) on the synthetic corpora.

For each condition and seed (v1 splits: 320 training sentences, 40 held
out), ``trellis2.incremental.IncrementalLearner`` perceives the training
sentences one at a time; at checkpoints along the stream it is evaluated as
in ``run_unsupervised.py`` (total code of the stored analyses, held-out bits,
generation commission and novelty, bracket omission and commission of the
minimum-risk parse). ``--learner night`` runs the night learner
(``UnsupervisedLearner`` with its defaults, one night after the stream) on
the same sentences, as the baseline.

Usage:
    python experiments/v2/run_incremental_learner.py --conditions small,med --seeds 13 \
        --out experiments/v2/results/incremental_learner/proto
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
from trellis2.incremental import HedgedBeam, IncrementalBeam, IncrementalLearner  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
from run_unsupervised import evaluate  # noqa: E402


def run_one(job) -> dict:
    condition, seed, args = job
    t0 = time.time()
    examples = load_corpus(os.path.join(args["data_root"], CONDITIONS[condition]))
    train, test = v1_split(examples, seed)
    train = train[:args["n_train"]]
    cfg = CFG(target_grammar(condition))
    sentences = {e.sentence for e in train}
    row = {"condition": condition, "seed": seed, "learner": args["learner"], "config": args["config"]}
    if args["learner"] == "night":
        learner = UnsupervisedLearner(seed=seed, workers=1)
        for e in train:
            learner.observe(e.tokens)
        g = learner.sleep()
        row["final"] = evaluate(learner, g, test, cfg, sentences, seed, args["n_gen"])
        row["seconds"] = time.time() - t0
    else:
        if args["hypotheses"]:
            learner = IncrementalBeam(seed=seed, configs=[dict(args["config"], **h) for h in args["hypotheses"]])
        elif args["beam_width"] > 1:
            learner = HedgedBeam(seed=seed, width=args["beam_width"], config=args["config"])
        else:
            learner = IncrementalLearner(seed=seed, config=args["config"])
        checkpoints = set(args["checkpoints"])
        curve, spent = [], 0.0
        for k, e in enumerate(train, 1):
            s0 = time.time()
            learner.observe(e.tokens)
            spent += time.time() - s0
            if k in checkpoints or k == len(train):
                point = evaluate(learner, learner.grammar, test, cfg,
                                 {x.sentence for x in train[:k]}, seed, args["n_gen"])
                point.update({"sentences": k, "learn_seconds": spent,
                              "stats": dict(learner.stats)})
                curve.append(point)
                print(f"[{condition} s{seed}] {k}: {point['total_bits']:.1f} bits, K={point['symbols']} "
                      f"M={point['rule_classes']}, gen-comm {point['gen_commission']:.3f}, "
                      f"omission {point['omission']:.3f}, test {point['test_bits_per_sentence']:.2f} "
                      f"({spent:.0f}s)", flush=True)
        row["curve"] = curve
        row["final"] = curve[-1]
        row["step_seconds"] = [h["seconds"] for h in learner.history]
        row["seconds"] = time.time() - t0
        row["stats"] = dict(learner.stats)
        row["memory_elements_total"] = len(learner.memory)
        row["live_elements"] = sum(len(x) for x in learner.elems)
        if args.get("pickle"):
            import pickle
            sys.setrecursionlimit(100000)
            # Pickles go to logs/ (gitignored), not to the results.
            pk = os.path.join(ROOT, "logs", "inc_pickles", os.path.basename(os.path.normpath(args["out"])))
            os.makedirs(pk, exist_ok=True)
            with open(os.path.join(pk, f"learner_{condition}_s{seed}.pkl"), "wb") as fh:
                pickle.dump(learner, fh)
        if args.get("save_trees"):
            row["trees"] = [[sorted(t.split.items()), t.roots, sorted(t.label.items())]
                            for t in learner.stored_trees()]
    f = row["final"]
    print(f"[{condition} s{seed} {args['learner']}] final {f['total_bits']:.1f} bits, gen-comm "
          f"{f['gen_commission']:.3f}, omission {f['omission']:.3f}, commission {f['parse_commission']:.3f} "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="13,17")
    ap.add_argument("--n-train", type=int, default=320)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--checkpoints", default="10,20,40,80,160,320")
    ap.add_argument("--learner", default="incremental", choices=["incremental", "night"])
    ap.add_argument("--config", default="{}", help="JSON overrides of incremental.DEFAULT_CONFIG")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--save-trees", action="store_true")
    ap.add_argument("--beam-width", type=int, default=1, help="hypotheses of a HedgedBeam (1: one learner)")
    ap.add_argument("--hypothesis", action="append", default=[],
                    help="JSON config overrides of one hypothesis of an IncrementalBeam (give several)")
    ap.add_argument("--pickle", action="store_true", help="save each final learner (gitignored .pkl)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    shared = {"learner": args.learner, "config": json.loads(args.config), "n_train": args.n_train,
              "n_gen": args.n_gen, "checkpoints": [int(c) for c in args.checkpoints.split(",") if c],
              "data_root": args.data_root, "save_trees": args.save_trees, "pickle": args.pickle, "beam_width": args.beam_width,
              "hypotheses": [json.loads(h) for h in args.hypothesis],
              "out": args.out}
    jobs = [(c, int(s), shared) for c in args.conditions.split(",") for s in args.seeds.split(",")]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_one, jobs))
    else:
        rows = [run_one(j) for j in jobs]
    with open(os.path.join(args.out, f"results_{args.learner}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n| Condition | Seed | Total bits | Gen. commission | Omission | Commission | Test bits | Seconds |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["final"]
        print(f"| {r['condition']} | {r['seed']} | {f['total_bits']:.1f} | {100 * f['gen_commission']:.1f}% | "
              f"{100 * f['omission']:.1f}% | {100 * f['parse_commission']:.1f}% | "
              f"{f['test_bits_per_sentence']:.2f} | {r['seconds']:.0f} |")


if __name__ == "__main__":
    main()
