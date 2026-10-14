"""Do attack/defence (typed) chunk candidates reach the search, and what do
they score? At the flat start and at the converged state of a typed run:
the number of typed candidates per type, their instances, and the best
score of each type against the current code and the best geometric move."""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_memory import fen_to_board, to_position  # noqa: E402
from trellis2.chess import TACTICAL, ContextBoardSearch  # noqa: E402


def report(search, label):
    t0 = time.time()
    cur = search.bits()
    n = len(search.pos)
    cands = search.candidates()
    typed = {k: v for k, v in cands.items() if k[1] in TACTICAL}
    out = {"state": label, "bits/pos": cur / n, "candidate keys": len(cands),
           "typed keys": {t: sum(1 for k in typed if k[1] == t) for t in TACTICAL},
           "typed keys with >= 2 instances": {t: sum(1 for k, v in typed.items() if k[1] == t
                                                     and len(search.select(v)) >= 2) for t in TACTICAL}}
    best = {}
    base = {}
    for key, pairs in cands.items():
        chosen = search.select(pairs)
        if len(chosen) < 2:
            continue
        b = search.score(key, chosen, base)
        kind = key[1] if key[1] in TACTICAL else "geometric"
        if kind not in best or b < best[kind][0]:
            best[kind] = (b, key, len(chosen))
    out["best per type (bits change, move, instances)"] = {
        k: [round(v[0] - cur, 2), f"[{v[1][0]} {v[1][1]} {v[1][2]}]", v[2]] for k, v in best.items()}
    # The most frequent typed candidates and their scores.
    top = sorted(typed.items(), key=lambda kv: -len(search.select(kv[1])))[:10]
    out["most frequent typed"] = [[f"[{k[0]} {k[1]} {k[2]}]", len(search.select(v)),
                                   round(search.score(k, search.select(v), base) - cur, 2),
                                   dict(Counter(search._relations(k, search.select(v))).most_common(4))]
                                  for k, v in top]
    out["seconds"] = time.time() - t0
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1800")
    ap.add_argument("--ply", type=int, default=16)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--run", default="")
    args = ap.parse_args()
    with open(positions_path(args.elo, args.ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    order = np.random.default_rng(13).permutation(len(fens))
    train = [to_position(fen_to_board(fens[i])) for i in order[:args.train]]
    s = ContextBoardSearch(train, args.alpha, typed=True)
    res = [report(s, "flat")]
    if args.run:
        run = json.load(open(args.run))
        print("typed run's moves:", run["search moves"], flush=True)
        g = ContextBoardSearch(train, args.alpha, typed=True)
        g.run(500)
        print("rerun moves:", g.moves, flush=True)
        res.append(report(g, "converged"))
    out = os.path.join(HERE, "results", "chess_chunk_structure", f"typed_check_e{args.elo}_p{args.ply}_n{args.train}.json")
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
