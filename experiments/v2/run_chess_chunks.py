"""In what setting do TRELLIS's chess chunks pay, and does their gain grow with N?

A setting is data (minimum rating, ply of the position; positions from
extract_chess_settings.py), the read (counts of each kind on earlier squares,
or squares on their own) and the Dirichlet concentration alpha. The mechanism
(src/trellis2/chess.py) is unchanged. For each run (the split of
run_chess.py: seed-13 permutation, first N learned, next 500 held out):

  chunk gain, training  = read alone - after the search (the search's own code;
                          chunks are its only moves) and read alone - TRELLIS total
  chunk gain, held out  = read alone - the grammar's code (and - the search's code)

plus the chunk types (count, most frequent anchors) and the share of pieces
inside a chunk. Positive gain = chunks help.

    python run_chess_chunks.py --sweep 1800 16            # the first-pass grid for one data setting
    python run_chess_chunks.py --elo 1800 --ply 16 --train 16000 [--no-context] [--alpha 3e-4] [--kind random]

Results: experiments/v2/results/chess_chunks/runs/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "chess_chunks", "runs")
SETTINGS = os.path.join(ROOT, "data", "chess", "settings")

from trellis2.chess import BoardSearch, ChessLearner, is_chunk, square_name  # noqa: E402
from run_chess import node_anchor, pieces_of, render_relative  # noqa: E402  (read-only import)
from run_chess_memory import fen_to_board, randomize, to_position  # noqa: E402


FAMILIES = {  # tag: (description, test on a games.tsv row); all games have both players 1500+
    "open4": ("1.e4 e5 2.Nf3 Nc6 (open games)", lambda r: r["opening6"].split()[:4] == ["e2e4", "e7e5", "g1f3", "b8c6"]),
    "ecoC": ("ECO C (1.e4 e5 and the French)", lambda r: r["eco"].startswith("C")),
}


def write_families():
    """positions_<tag>_ply<p>.fen for each opening family, in game order."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(SETTINGS, "games.tsv")), delimiter="\t"))
    for tag, (desc, keep) in FAMILIES.items():
        sel = [r for r in rows if keep(r)]
        for p in (12, 16, 20, 30):
            with open(positions_path(tag, p), "w") as f:
                f.writelines(r[f"ply{p}"] + "\n" for r in sel)
        print(f"{tag} ({desc}): {len(sel)} games")


def positions_path(elo, ply):
    """elo: a minimum rating (1500, 1800, 2000) or an opening-family tag."""
    return os.path.join(SETTINGS, f"positions_{elo}_ply{ply}.fen")


def name_of(elo, ply, n, counts, alpha, kind):
    return f"e{elo}_p{ply}_n{n}_{'counts' if counts else 'plain'}_a{alpha:g}_{kind}"


def run(elo, ply, n, counts, alpha, kind, test=500, seed=13, force=False):
    name = name_of(elo, ply, n, counts, alpha, kind)
    path = os.path.join(OUT, name + ".json")
    if os.path.exists(path) and not force:
        print("done already:", name, flush=True)
        return json.load(open(path))
    with open(positions_path(elo, ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    if len(fens) < n + test:
        print(f"skip {name}: {len(fens)} positions < {n + test}", flush=True)
        return None
    order = np.random.default_rng(seed).permutation(len(fens))
    boards = [fen_to_board(fens[i]) for i in order[:n + test]]
    if kind == "random":
        rng = np.random.default_rng(2026)
        boards = [randomize(b, rng) for b in boards]
    train = [to_position(b) for b in boards[:n]]
    held = [to_position(b) for b in boards[n:n + test]]

    t0 = time.time()
    learner = ChessLearner(seed=seed, alpha=alpha, counts=counts, context=False)
    for p in train:
        learner.observe(p)
    g = learner.sleep()
    seconds = time.time() - t0
    flat = learner.history[0]["bits"] / n
    search = [h for h in learner.history if h["stage"] in ("flat", "chunk")][-1]["bits"] / n
    total = g.info["total bits"] / n
    held_chunks, held_flat = learner.search.held_out_bits(held)
    held_grammar = -sum(learner.log_prob(p) for p in held) / len(held) / math.log(2)

    found, anchors = Counter(), defaultdict(Counter)
    covered = pieces = 0
    for position, tops in zip(train, learner.analyses):
        pieces += len(position)
        for node in tops:
            if is_chunk(node):
                key = render_relative(node, position)
                found[key] += 1
                anchors[key][square_name(node_anchor(node))] += 1
                covered += len(pieces_of(node, position))
    res = {"name": name, "elo": elo, "ply": ply, "N": n, "test": len(held), "counts": counts, "alpha": alpha,
           "kind": kind, "positions available": len(fens), "seconds": seconds,
           "train bits/pos": {"read alone": flat, "after the search": search, "TRELLIS v2 total": total},
           "held-out bits/pos": {"read alone": held_flat, "search code with chunks": held_chunks,
                                 "TRELLIS v2 grammar": held_grammar},
           "chunk gain, train (read alone - search)": flat - search,
           "chunk gain, train (read alone - TRELLIS total)": flat - total,
           "chunk gain, held out (read alone - grammar)": held_flat - held_grammar,
           "chunk gain, held out (read alone - search code)": held_flat - held_chunks,
           "model bits": g.info["model bits"], "data bits": g.info["data bits"],
           "symbols": g.K, "rule classes": g.M, "chunk types (grammar)": g.info["chunk types"],
           "share of pieces inside a chunk": covered / max(pieces, 1),
           "chunk types": [{"chunk": k, "count": c, "anchors": dict(anchors[k].most_common(3))}
                           for k, c in found.most_common()],
           "search moves": [h["move"] for h in learner.history if h["stage"] == "chunk"]}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{name}: {seconds:.0f}s; train read {flat:.2f} search {search:.2f} total {total:.2f}; held read "
          f"{held_flat:.2f} grammar {held_grammar:.2f}; gain held {held_flat - held_grammar:+.2f}; "
          f"{len(found)} chunk types {[k for k, _ in found.most_common(5)]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", action="store_true", help="write the opening-family position files")
    ap.add_argument("--sweep", nargs=2, metavar=("ELO", "PLY"),
                    help="first pass: N 1000 and 4000 x counts on/off x alpha 1e-4, 3e-4, 1e-3 (real)")
    ap.add_argument("--elo", default="1500")
    ap.add_argument("--ply", type=int, default=30)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--kind", default="real", choices=["real", "random"])
    args = ap.parse_args()
    if args.families:
        write_families()
        return
    if args.sweep:
        elo, ply = args.sweep[0], int(args.sweep[1])
        for n in (1000, 4000):
            for counts in (True, False):
                for alpha in (1e-3, 3e-4, 1e-4):
                    run(elo, ply, n, counts, alpha, "real")
    else:
        run(args.elo, args.ply, args.train, not args.no_context, args.alpha, args.kind)


if __name__ == "__main__":
    main()
