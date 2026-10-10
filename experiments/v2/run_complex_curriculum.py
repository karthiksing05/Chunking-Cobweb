"""Idea 3, starting small: learning by day and by night with experiences
presented in order of size, against the same schedule in random order.

The learner perceives the training sentences one at a time with its current
grammar and sleeps at each checkpoint, every night starting from the stored
analyses (as ``run_incremental.py``). ``--order size`` presents the shortest
experiences first (ties in the seed's order); ``--order random`` uses the
seed's order. After every night the grammar is evaluated as in
``run_complex.py``; whole-tree rates of the training analyses are reported.

    python experiments/v2/run_complex_curriculum.py --grammar deep --order size --seed 13
"""
import argparse
import functools
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

OUT = os.path.join(HERE, "results", "complex_curriculum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="deep", choices=CG.GRAMMARS)
    ap.add_argument("--order", default="size", choices=["size", "random"])
    ap.add_argument("--checkpoints", default="250:500:1000:2000")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--spine-depth", type=int, default=1)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    checkpoints = [int(c) for c in args.checkpoints.split(":")]

    full, test = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(full))
    train = [full[i] for i in order[:checkpoints[-1]]]
    if args.order == "size":
        train = sorted(train, key=lambda e: len(e.tokens))      # stable: ties keep the seed's order
    test = test[:args.heldout]
    cfg = CFG(CG.build(args.grammar))

    out_dir = os.path.join(OUT, f"{args.grammar}_{args.order}_d{args.spine_depth}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    learner = UnsupervisedLearner(seed=args.seed, workers=args.workers, spine_depth=args.spine_depth)
    results = []
    seen = 0
    for cp in checkpoints:
        for e in train[seen:cp]:
            learner.observe(e.tokens)
        seen = cp
        t0 = time.time()
        g = learner.sleep()
        night = time.time() - t0
        train_set = {" ".join(e.tokens) for e in train[:cp]}
        whole = float(np.mean([len(t.roots) == 1 for t in learner.trees]))
        r = {"sentences": cp, "mean length so far": float(np.mean([len(e.tokens) for e in train[:cp]])),
             "night seconds": night, "total_bits": g.info["total bits"], "symbols": g.K,
             "chunk types": g.info["chunk types"], "train whole trees": whole}
        r.update(RC.evaluate(learner, test, cfg, train_set, args.seed, args.n_gen))
        r.pop("samples", None)
        results.append(r)
        gen = r.get("generation", {})
        print(f"[{args.order} s{args.seed}] {cp} sentences (mean length {r['mean length so far']:.1f}): "
              f"night {night:.0f}s, {r['total_bits']:.0f} bits, whole trees {whole:.1%}, "
              f"held-out {r.get('test_bits_per_sentence', float('nan')):.2f} b/s, "
              f"grammatical {gen.get('grammatical')}, whole samples {gen.get('whole trees')}", flush=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "checkpoints": results}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
