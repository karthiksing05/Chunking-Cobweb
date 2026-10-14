"""Nightly learning (``run_complex_curriculum.py``'s schedule) with or without
split moves at the end of each night (``UnsupervisedLearner(joint_splits)``;
joint chunks on, the ``src`` default).

The learner perceives the training sentences in order (by size, or the seed's
order) and sleeps at each checkpoint, each night starting from the stored
analyses (in the categories of the last night, split ones included). After
every night the grammar is evaluated as in ``run_complex.py`` (number-blind
rates included; novelty against the sentences seen so far).

    python experiments/v2/run_curriculum_splits.py --grammar relative --order size --seed 13 --splits
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "src"), HERE]

import complex_grammars as CG  # noqa: E402
import run_complex as RC  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

OUT = os.path.join(HERE, "results", "joint_splits", "curriculum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="relative", choices=CG.GRAMMARS)
    ap.add_argument("--order", default="size", choices=["size", "random"])
    ap.add_argument("--checkpoints", default="250:500:1000:2000")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--spine-depth", type=int, default=1)
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    checkpoints = [int(c) for c in args.checkpoints.split(":")]
    full, test = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(full))
    train = [full[i] for i in order[:checkpoints[-1]]]
    if args.order == "size":
        train = sorted(train, key=lambda e: len(e.tokens))
    test = test[:300]
    cfg = CFG(CG.build(args.grammar))
    RC.BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    out_dir = os.path.join(OUT, f"{args.grammar}_{args.order}_d{args.spine_depth}_s{args.seed}_"
                                f"{'splits' if args.splits else 'nosplits'}")
    os.makedirs(out_dir, exist_ok=True)
    learner = UnsupervisedLearner(seed=args.seed, workers=args.workers, spine_depth=args.spine_depth,
                                  joint_splits=args.splits)
    results, seen = [], 0
    for cp in checkpoints:
        for e in train[seen:cp]:
            learner.observe(e.tokens)
        seen = cp
        t0 = time.time()
        g = learner.sleep()
        night = time.time() - t0
        train_set = {tuple(e.tokens) for e in train[:cp]}
        r = {"sentences": cp, "night seconds": night, "total_bits": g.info["total bits"], "symbols": g.K,
             "chunk types": g.info["chunk types"],
             "train whole trees": float(np.mean([len(t.roots) == 1 for t in learner.trees])),
             "splits accepted": g.info.get("splits: accepted"), "joint splits accepted": g.info.get("splits: joint accepted"),
             "unsplit total bits": g.info.get("splits: unsplit total bits")}
        r.update(RC.evaluate(learner, test, cfg, train_set, args.seed, 1000, train[:cp], learner.trees))
        r.pop("samples", None)
        results.append(r)
        gen = r["generation"]
        print(f"[{args.grammar} {args.order} s{args.seed} {'splits' if args.splits else 'nosplits'}] {cp}: "
              f"night {night:.0f}s, {r['total_bits']:.0f} bits, whole {r['train whole trees']:.1%}, "
              f"held-out {r['test_bits_per_sentence']:.2f}, grammatical {gen['grammatical']}, "
              f"blind {gen.get('grammatical ignoring number')}, splits {r['splits accepted']}", flush=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "checkpoints": results}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
