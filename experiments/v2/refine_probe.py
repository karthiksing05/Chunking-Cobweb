"""``trellis2.splits.refine`` on a stored night (its analyses refitted), with a
given split configuration; evaluated as ``run_complex.py`` does.

Usage:
    python experiments/v2/refine_probe.py --pickle PATH --grammar agree --seed 13 --config '{"joint_max_seeds": 40}'
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from trellis2.splits import refine  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--name", default="default")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "joint_splits", "refine_probe"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import complex_grammars as CG
    import run_complex as RC
    from trellis2.evaluation import CFG
    cfg = CFG(CG.build(args.grammar))
    RC.BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    train_all, held = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:args.train]]
    with open(args.pickle, "rb") as f:
        saved = pickle.load(f)
    learner = UnsupervisedLearner(seed=args.seed)
    for e in train:
        learner.observe(e.tokens)
    model = learner.fit(saved["trees"])
    learner.model, learner.trees = model, saved["trees"]
    g0 = model.grammar
    t0 = time.time()
    g = refine(model.memory, g0, alpha=learner.alpha, seed=args.seed, search=model.search,
               config=json.loads(args.config))
    seconds = time.time() - t0
    model._grammar = g
    res = {"pickle": args.pickle, "config": json.loads(args.config), "seconds": seconds,
           "unsplit bits": g0.info["total bits"], "total bits": g.info["total bits"],
           "accepted": g.info.get("splits: accepted"), "joint accepted": g.info.get("splits: joint accepted")}
    ev = RC.evaluate(learner, held[:300], cfg, {tuple(e.tokens) for e in train}, args.seed, 1000)
    res["generation"] = {k: v for k, v in ev["generation"].items()}
    res["test bits"] = ev["test_bits_per_sentence"]
    print(json.dumps(res), flush=True)
    with open(os.path.join(args.out, f"{args.grammar}_s{args.seed}_{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
