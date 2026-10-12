"""Step 2: the board search with merge moves (chess_search.HierBoardSearch)
against the current ChessLearner (chunk moves only), on one setting.

Same split as run_chess_chunks.py (seed-13 permutation, first N learned,
next 500 held out). Reports chunk gain, chunk types and their levels,
categories made by merges, held-out bits, total code, generation checks and
(with --kind random) the random-position control.

    python run_chess_hier.py --elo 1800 --ply 16 --train 4000 --alpha 1e-3 --method merge [--patience 3]
    python run_chess_hier.py ... --method current

Results: experiments/v2/results/chess_hier/runs/<name>.json
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
sys.path.insert(0, HERE)
from chess_search import HierBoardSearch  # noqa: E402
from run_chess import node_anchor, pieces_of, render_relative  # noqa: E402
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_memory import fen_to_board, randomize, to_position  # noqa: E402
from trellis2.chess import (KINDS, BoardMemory, BoardSearch, ChessLearner, is_chunk,  # noqa: E402
                            plausibility, read_row, square_name)

OUT = os.path.join(HERE, "results", "chess_hier", "runs")


class HierChessLearner(ChessLearner):
    """ChessLearner whose night search also merges labels into categories."""

    def __init__(self, seed=0, alpha=0.001, max_steps=500, counts=True, merges=True, patience=0,
                 merge_kinds="any"):
        super().__init__(seed=seed, alpha=alpha, max_steps=max_steps, counts=counts)
        self.merges, self.patience, self.merge_kinds = merges, patience, merge_kinds

    def sleep(self):
        t0 = time.time()
        search = HierBoardSearch(self.experiences, self.alpha, self.counts, self.merges, self.merge_kinds)
        self.history.append({"stage": "flat", "bits": search.bits(), "seconds": 0.0})

        def log(step, move, bits):
            if move[0] == "chunk":
                k = move[1]
                desc = f"[{k[0]} {k[1]} {k[2]}] x{len(move[2])}"
            else:
                desc = f"merge({move[1]}, {move[2]})"
            self.history.append({"stage": "chunk" if move[0] == "chunk" else "merge", "move": desc,
                                 "bits": bits, "seconds": time.time() - t0})
            print(f"  step {step}: {desc} -> {bits / len(self.experiences):.4f} bits/pos "
                  f"({time.time() - t0:.0f}s)", flush=True)
        best = search.run(self.max_steps, log, self.patience)
        self.history.append({"stage": "search", "bits": best, "seconds": time.time() - t0})
        self.search = search
        self.analyses = search.analyses()
        self.model = self.fit(self.analyses, BoardMemory(counts=self.counts, context=False))
        g = self.model.grammar
        self.history.append({"stage": "consolidate", "bits": g.info["total bits"], "seconds": time.time() - t0})
        return g

    def analyse(self, position):
        search = HierBoardSearch([position], self.alpha, self.counts, self.merges, self.merge_kinds)
        search.replay(self.search.moves)
        return search.analyses()[0]


def depth(node):
    return 0 if not is_chunk(node) else 1 + max(depth(node[1][0]), depth(node[1][2]))


def generation(learner, train, held, n_gen, seed):
    rng = np.random.default_rng(seed)
    test_sets = [dict(p) for p in held]
    samples, failed = learner.generate(n_gen, rng)
    checks = Counter()
    gen_chunks = exact = 0
    depths = Counter()
    for position, tops in samples:
        ok = plausibility(position)
        for k, v in ok.items():
            checks[k] += v
        checks["passes every check"] += all(ok.values())
        for node in tops:
            if is_chunk(node):
                gen_chunks += 1
                depths[depth(node)] += 1
                pcs = pieces_of(node, position)
                exact += any(all(p.get(sq) == t for sq, t in pcs.items()) for p in test_sets)
    return {"attempts": n_gen + failed, "collision or off the board": failed / (n_gen + failed),
            "checks": {k: v / n_gen for k, v in checks.items()},
            "generated chunks": gen_chunks, "generated chunk depths": dict(sorted(depths.items())),
            "generated chunks found in a held-out position": exact / max(gen_chunks, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1800")
    ap.add_argument("--ply", type=int, default=16)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--kind", default="real", choices=["real", "random"])
    ap.add_argument("--method", default="merge", choices=["merge", "current", "context", "context_merge", "read_context"])
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--merge-kinds", default="any", choices=["any", "same"])
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="", help="appended to the result's name")
    args = ap.parse_args()
    tag = args.method if args.method in ("current", "context", "context_merge", "read_context") else (
        f"merge_p{args.patience}" + ("" if args.merge_kinds == "any" else "_same"))
    out_dir = (OUT if args.method in ("current", "merge") else
               os.path.join(HERE, "results", "chess_read_context" if args.method == "read_context" else "chess_context",
                            "runs"))
    name = f"e{args.elo}_p{args.ply}_n{args.train}_a{args.alpha:g}_{args.kind}_{tag}{args.tag}"
    with open(positions_path(args.elo, args.ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    if len(fens) < args.train + args.test:
        sys.exit(f"{name}: only {len(fens)} positions")
    order = np.random.default_rng(args.seed).permutation(len(fens))
    boards = [fen_to_board(fens[i]) for i in order[:args.train + args.test]]
    if args.kind == "random":
        rng = np.random.default_rng(2026)
        boards = [randomize(b, rng) for b in boards]
    train = [to_position(b) for b in boards[:args.train]]
    held = [to_position(b) for b in boards[args.train:]]

    t0 = time.time()
    if args.method == "current":
        learner = ChessLearner(seed=args.seed, alpha=args.alpha, context=False)
    elif args.method == "read_context":
        learner = ChessLearner(seed=args.seed, alpha=args.alpha, context=True, read_context=True)
    elif args.method.startswith("context"):
        from chess_context import ContextChessLearner
        learner = ContextChessLearner(seed=args.seed, alpha=args.alpha, merges=args.method == "context_merge",
                                      patience=args.patience)
    else:
        learner = HierChessLearner(seed=args.seed, alpha=args.alpha, patience=args.patience,
                                   merge_kinds=args.merge_kinds)
    for p in train:
        learner.observe(p)
    g = learner.sleep()
    seconds = time.time() - t0
    n = len(train)
    flat = learner.history[0]["bits"] / n
    search_bits = learner.search.bits() / n
    total = g.info["total bits"] / n
    held_chunks, held_flat = learner.search.held_out_bits(held)
    held_grammar = -sum(learner.log_prob(p) for p in held) / len(held) / math.log(2)

    found, anchors, levels = Counter(), defaultdict(Counter), Counter()
    covered = pieces = 0
    for position, tops in zip(train, learner.analyses):
        pieces += len(position)
        for node in tops:
            if is_chunk(node):
                key = render_relative(node, position)
                found[key] += 1
                anchors[key][square_name(node_anchor(node))] += 1
                covered += len(pieces_of(node, position))
                levels[depth(node)] += 1
    s = learner.search
    chunk_labels = [lab for lab, row in s.rows.items() if any(o[0] == "p" for o in row)]
    categories = []
    for lab, row in s.rows.items():
        if len(row) > 1:
            categories.append({"label": str(lab), "outcomes": [(str(o), c) for o, c in row.most_common(8)],
                               "occurrences": sum(row.values())})
    gen = generation(learner, train, held, args.n_gen, args.seed)
    res = {"name": name, "method": tag, "elo": args.elo, "ply": args.ply, "N": n, "test": len(held),
           "alpha": args.alpha, "kind": args.kind, "seconds": seconds,
           "train bits/pos": {"read alone": flat, "after the search": search_bits, "TRELLIS v2 total": total},
           "held-out bits/pos": {"read alone": held_flat, "search code": held_chunks,
                                 "TRELLIS v2 grammar": held_grammar},
           "chunk gain, train (read alone - search)": flat - search_bits,
           "chunk gain, train (read alone - TRELLIS total)": flat - total,
           "chunk gain, held out (read alone - grammar)": held_flat - held_grammar,
           "chunk gain, held out (read alone - search code)": held_flat - held_chunks,
           "model bits": g.info["model bits"], "data bits": g.info["data bits"],
           "symbols": g.K, "rule classes": g.M, "chunk types (grammar)": g.info["chunk types"],
           "search chunk labels": len(chunk_labels), "search labels": len(s.rows),
           "merges": sum(1 for m in s.moves if m[0] == "merge"),
           "chunk moves": sum(1 for m in s.moves if m[0] != "merge"),
           "categories (rows with several outcomes)": categories,
           "top-level chunk levels (depth: count)": dict(sorted(levels.items())),
           "chunk shapes": len(found),
           "share of pieces inside a chunk": covered / max(pieces, 1),
           "chunk types": [{"chunk": k, "count": c, "anchors": dict(anchors[k].most_common(3))}
                           for k, c in found.most_common(40)],
           "search moves": [h["move"] for h in learner.history if h["stage"] in ("chunk", "merge")],
           "generation": gen}
    if args.method.startswith("context") or args.method == "read_context":
        res["context option (depth, beta)"] = list(learner.search.option)
        res["grammar context option"] = list(learner.model.memory._predictor(g)[:2])
    if args.method == "read_context":
        learner.search._read_nats()
        res["read context option (depth, beta)"] = list(learner.search.read_option)
        res["grammar read context option"] = list(learner.model.memory.answers_nats(args.alpha)[1])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{name}: {seconds:.0f}s; train read {flat:.3f} search {search_bits:.3f} total {total:.3f}; "
          f"held read {held_flat:.3f} grammar {held_grammar:.3f} search {held_chunks:.3f}; "
          f"{len(chunk_labels)} chunk labels, {res['merges']} merges, levels {dict(levels)}; "
          f"gen pass {gen['checks'].get('passes every check', 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
