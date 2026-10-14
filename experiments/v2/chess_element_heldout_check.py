"""Held-out top-level symbols under a context made of elements: the
commonest training symbol of each search label (the first version) against
the symbol that codes each element shortest, chosen in reading order (the
fix). Prints both held-out codes and how the labels' symbols are spread."""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_memory import fen_to_board, to_position  # noqa: E402
from trellis2.chess import ChessLearner  # noqa: E402


def relabel(node, sym_of):
    lab, body = node
    if len(body) == 3:
        return (sym_of(lab), (relabel(body[0], lambda l: l), body[1], relabel(body[2], lambda l: l)))
    return (sym_of(lab), body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1500")
    ap.add_argument("--ply", type=int, default=30)
    ap.add_argument("--train", type=int, default=2000)
    args = ap.parse_args()
    with open(positions_path(args.elo, args.ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    order = np.random.default_rng(13).permutation(len(fens))
    boards = [to_position(fen_to_board(fens[i])) for i in order[:args.train + 500]]
    train, held = boards[:args.train], boards[args.train:]
    learner = ChessLearner(seed=13, alpha=1e-3, context_of="elements")
    for p in train:
        learner.observe(p)
    g = learner.sleep()
    mem = learner.model.memory
    s = np.asarray(g.elem_symbol)
    votes = defaultdict(Counter)
    for e in np.flatnonzero(mem.is_root):
        votes[mem.label[e]][int(s[e])] += 1
    print("symbols per top-level label:", {str(k): len(v) for k, v in votes.items()})
    print("share of the commonest symbol:", {str(k): round(v.most_common(1)[0][1] / sum(v.values()), 3)
                                              for k, v in votes.items()})
    majority = {lab: v.most_common(1)[0][0] for lab, v in votes.items()}
    best = maj = 0.0
    for p in held:
        a = learner.analyse(p)
        best -= learner.model.log_prob(p, a)
        maj -= learner.model.log_prob(p, [relabel(n, lambda l: majority[l]) for n in a])
    n = len(held) * math.log(2)
    print(f"held-out grammar code: commonest symbol per label {maj / n:.3f}, best symbol per element {best / n:.3f}; "
          f"held-out search code {learner.search.held_out_bits(held)[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
