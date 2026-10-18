"""TRELLIS v3 (candidate trees) on Chinese characters as token sequences.

The same learner as the sentences, nothing domain-specific: each character's
IDS decomposition is a sequence of tokens (operators and components, prefix
order). Reports held-out bracket omission against the gold decomposition (and
right-branching), and generation: well formed (the sequence reads back into a
structure), positions attested (every component in a slot it fills in some real
character), real (an existing character) and novel.

Usage: python experiments/v3/run_characters.py --train 250,1000,2000 --seeds 13 --out experiments/v3/results/characters
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

from trellis2.candidates import CandidateLearner  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402
from trellis2.characters import default_ids_path, load_characters, parse_prefix, placements  # noqa: E402
from trellis2.data import Tree  # noqa: E402
from trellis2.evaluation import BracketTally  # noqa: E402


def right_branching(n: int) -> Tree:
    return Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="250,1000,2000")
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--learner", default="candidates", choices=["candidates", "trellis3"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(args.config)
    checkpoints = sorted(int(x) for x in args.train.split(","))
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        chars, _ = load_characters(default_ids_path(), seed=seed)
        test = chars[-args.test:]
        real = {parse_prefix(c.tokens) for c in chars}
        attested = set()
        for c in chars:
            placements(parse_prefix(c.tokens), attested)
        L = (Trellis3 if args.learner == "trellis3" else CandidateLearner)(seed=seed, **cfg)
        parse = (lambda toks: (L.parse(toks)["tree"], None)) if args.learner == "trellis3" else L.parse
        if args.learner == "trellis3":
            import random as _random
            _rng = _random.Random(seed)
            def _gen(max_len=15, _mem=[None]):
                if _mem[0] is None or _mem[0][0] != len(L.records):
                    _mem[0] = (len(L.records), (L._members(L.ctx, "ctx"), L._members(L.cnt, "cnt")))
                s = L.generate_sub(_rng, via="ctx", up=1, members=_mem[0][1])
                return s if s is not None and len(s) <= max_len else None
            L.generate = _gen
        t0, done = time.time(), 0
        for n in checkpoints:
            for c in chars[done:n]:
                L.learn(c.tokens)
            done = n
            learn_s = time.time() - t0
            te, rb, whole = BracketTally(), BracketTally(), 0
            for c in test:
                t, _ = parse(c.tokens)
                te.add(c.tree, t)
                rb.add(c.tree, right_branching(len(c.tokens)))
                whole += len(t.roots) == 1
            seen = {tuple(c.tokens) for c in chars[:n]}
            samples = [s for s in (L.generate(max_len=15) for _ in range(args.n_gen)) if s is not None]
            structs = [parse_prefix(s) for s in samples]
            good = [st for st in structs if st is not None]
            attest = [st for st in good if placements(st) <= attested]
            row = {"seed": seed, "train": n, "config": cfg,
                   "held-out omission": te.omission, "held-out commission": te.commission,
                   "right-branching omission": rb.omission, "held-out whole": whole / len(test),
                   "generated": len(samples) / args.n_gen,
                   "well formed": len(good) / max(len(samples), 1),
                   "positions attested": len(attest) / max(len(samples), 1),
                   "real": sum(st in real for st in good) / max(len(samples), 1),
                   "novel": sum(tuple(s) not in seen for s in samples) / max(len(samples), 1),
                   "examples": [" ".join(s) for s in samples[:12]],
                   "parses": [parse(c.tokens)[0].to_string(c.tokens) for c in test[:8]],
                   "gold": [c.tree.to_string(c.tokens) for c in test[:8]],
                   "stats": {k: v for k, v in L.stats.items()}, "learn seconds": learn_s}
            rows.append(row)
            print(f"[characters s{seed} {n}] omission {te.omission:.3f} commission {te.commission:.3f} "
                  f"(right-branching {rb.omission:.3f}) whole {row['held-out whole']:.2f} | well formed "
                  f"{row['well formed']:.3f} attested {row['positions attested']:.3f} real {row['real']:.3f} "
                  f"novel {row['novel']:.3f} | {learn_s:.0f}s", flush=True)
            for p, g in zip(row["parses"][:4], row["gold"][:4]):
                print("    ", p, "   gold", g, flush=True)
            with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
