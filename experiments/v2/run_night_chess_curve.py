"""The night learner (``ChessLearner``, label context, read context off) on
the first N positions of the seed-13 split, evaluated on the held-out set of
a run with ``--split`` training positions (default 4000: positions
4000..4499), so that its learning curve is comparable with the checkpoints
of run_incremental_chess.py. A baseline only.

    python run_night_chess_curve.py --train 1000 [--split 4000]

Results: experiments/v2/results/chess_incremental/night/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.environ.get("TRELLIS_REPO", "/coc/flash10/ksingara3/Chunking-Cobweb")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_hier import generation  # noqa: E402
from run_chess_memory import fen_to_board, randomize, to_position  # noqa: E402
from run_incremental_chess import chunk_report  # noqa: E402
from trellis2.chess import ChessLearner  # noqa: E402

OUT = os.path.join(REPO, "experiments", "v2", "results", "chess_incremental", "night")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1800")
    ap.add_argument("--ply", type=int, default=16)
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--split", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--kind", default="real", choices=["real", "random"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-gen", type=int, default=200)
    args = ap.parse_args()
    name = f"e{args.elo}_p{args.ply}_n{args.train}_split{args.split}_a{args.alpha:g}_{args.kind}_night"
    with open(positions_path(args.elo, args.ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    print(f"{name}: {len(fens)} positions", flush=True)
    order = np.random.default_rng(args.seed).permutation(len(fens))
    boards = [fen_to_board(fens[i]) for i in order[:args.split + args.test]]
    if args.kind == "random":
        rng = np.random.default_rng(2026)
        boards = [randomize(b, rng) for b in boards]
    train = [to_position(b) for b in boards[:args.train]]
    held = [to_position(b) for b in boards[args.split:]]
    t0 = time.time()
    learner = ChessLearner(seed=args.seed, alpha=args.alpha, context=True)
    for p in train:
        learner.observe(p)
    g = learner.sleep()
    seconds = time.time() - t0
    n = len(train)
    flat = learner.history[0]["bits"] / n
    held_search, held_flat = learner.search.held_out_bits(held)
    held_grammar = -sum(learner.log_prob(p) for p in held) / len(held) / math.log(2)
    res = {"name": name, "N": n, "split": args.split, "seconds": seconds,
           "train bits/pos": {"read alone": flat, "after the search": learner.search.bits() / n,
                              "TRELLIS total": g.info["total bits"] / n},
           "held-out bits/pos": {"read alone": held_flat, "search code": held_search, "TRELLIS grammar": held_grammar},
           "chunk gain, held out (read alone - grammar)": held_flat - held_grammar,
           "symbols": g.K, "rule classes": g.M,
           "search moves": [h["move"] for h in learner.history if h["stage"] == "chunk"]}
    res.update(chunk_report(train, learner.analyses))
    res["generation"] = generation(learner, train, held, args.n_gen, args.seed)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{name}: {seconds:.0f}s; total {res['train bits/pos']['TRELLIS total']:.3f}; held grammar {held_grammar:.3f} "
          f"(read {held_flat:.3f}); moves {res['search moves']}; gen pass "
          f"{res['generation']['checks'].get('passes every check', 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
