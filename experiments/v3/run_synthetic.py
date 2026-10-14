"""TRELLIS v3 (candidate trees) on the paper's synthetic corpora.

Learns the training sentences one at a time (greedy parsing that stores every
candidate), then reports bracket omission/commission of held-out parses and of
the training analyses, and generation: the share of 1,000 samples the target
grammar accepts (grammatical) and of those not seen in training (novel).

Usage: python experiments/v3/run_synthetic.py --conditions small,med --seeds 13 --out experiments/v3/results/synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.candidates import CandidateLearner  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.data import Tree  # noqa: E402
from trellis2.evaluation import CFG, BracketTally  # noqa: E402


def right_branching(n: int) -> Tree:
    return Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)])


def run(condition: str, seed: int, cfg: dict, n_gen: int) -> dict:
    t0 = time.time()
    train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[condition])), seed)
    L = CandidateLearner(seed=seed, **cfg)
    trees = [L.learn(e.tokens) for e in train]
    learn_s = time.time() - t0
    tr = BracketTally()
    for e, t in zip(train, trees):
        tr.add(e.tree, t)
    te, rb = BracketTally(), BracketTally()
    whole = 0
    for e in test:
        t, _ = L.parse(e.tokens)
        te.add(e.tree, t)
        rb.add(e.tree, right_branching(len(e.tokens)))
        whole += len(t.roots) == 1
    g = CFG(target_grammar(condition))
    seen = {tuple(e.tokens) for e in train}
    samples = [L.generate() for _ in range(n_gen)]
    ok = [s for s in samples if s is not None]
    gram = [s for s in ok if g.recognizes(s)]
    row = {"condition": condition, "seed": seed, "config": cfg,
           "held-out omission": te.omission, "held-out commission": te.commission,
           "held-out exact": te.exact_match, "right-branching omission": rb.omission, "held-out whole trees": whole / max(len(test), 1),
           "train omission": tr.omission, "train commission": tr.commission,
           "generated": len(ok) / n_gen, "grammatical": len(gram) / max(len(ok), 1),
           "novel": sum(tuple(s) not in seen for s in ok) / max(len(ok), 1),
           "grammatical and novel": sum(tuple(s) not in seen for s in gram) / max(len(ok), 1),
           "examples": [" ".join(s) + ("  ✓" if g.recognizes(s) else "  ✗") for s in ok[:12]],
           "parses": [e_t.to_string(e.tokens, labels=False) for e, e_t in
                      [(e, L.parse(e.tokens)[0]) for e in test[:8]]],
           "stats": L.stats, "content nodes": sum(1 for _ in L.cnt.nodes()),
           "learn seconds": learn_s, "seconds": time.time() - t0}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="small,med,large,term_low,term_med,term_high")
    ap.add_argument("--seeds", default="13,17")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(args.config)
    rows = []
    for cond in args.conditions.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = run(cond, seed, cfg, args.n_gen)
            rows.append(r)
            print(f"[{cond} s{seed}] held-out omission {r['held-out omission']:.3f} commission "
                  f"{r['held-out commission']:.3f} (right-branching {r['right-branching omission']:.3f}) whole {r['held-out whole trees']:.2f} | train omission "
                  f"{r['train omission']:.3f} | gen grammatical {r['grammatical']:.3f} novel {r['novel']:.3f} "
                  f"(generated {r['generated']:.2f}) | {r['seconds']:.0f}s", flush=True)
            for p in r["parses"][:4]:
                print("    ", p, flush=True)
            for x in r["examples"][:6]:
                print("    gen:", x, flush=True)
            with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
