"""The continuously incremental chess learner (``trellis2.incremental_chess``)
on the split of run_chess_chunks.py (seed-13 permutation, first N learned
one position at a time, the next 500 held out), with checkpoints along the
stream. At each checkpoint, evaluation only (the learner's state is not
changed): the training code (read alone, the state's search code, the total
of the grammar read off the state), the held-out code (read alone, search
code, grammar), chunk types, levels, generation checks, seconds per
position. With ``--night-readout`` the final analyses are also consolidated
as the night consolidates its own (a diagnostic of the readout).

    python run_incremental_chess.py --elo 1800 --ply 16 --train 4000 --checkpoints 250,500,1000,2000,4000
    python run_incremental_chess.py ... --kind random

Results: experiments/v2/results/chess_incremental/runs/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.environ.get("TRELLIS_REPO", "/coc/flash10/ksingara3/Chunking-Cobweb")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from run_chess import node_anchor, pieces_of, render_relative  # noqa: E402
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_hier import depth, generation  # noqa: E402
from run_chess_memory import fen_to_board, randomize, to_position  # noqa: E402
from trellis2.chess import BoardMemory, ContextBoardSearch, is_chunk, square_name  # noqa: E402
from trellis2.incremental_chess import IncrementalChessLearner  # noqa: E402
from trellis2.model import Learner  # noqa: E402

OUT = os.path.join(REPO, "experiments", "v2", "results", "chess_incremental", "runs")


def chunk_report(train, analyses):
    found, anchors, levels = Counter(), defaultdict(Counter), Counter()
    covered = pieces = 0
    for position, tops in zip(train, analyses):
        pieces += len(position)
        for node in tops:
            if is_chunk(node):
                key = render_relative(node, position)
                found[key] += 1
                anchors[key][square_name(node_anchor(node))] += 1
                covered += len(pieces_of(node, position))
                levels[depth(node)] += 1
    return {"top-level chunk levels (depth: count)": dict(sorted(levels.items())),
            "share of pieces inside a chunk": covered / max(pieces, 1),
            "chunk shapes": len(found),
            "chunk types": [{"chunk": k, "count": c, "anchors": dict(anchors[k].most_common(3))}
                            for k, c in found.most_common(40)]}


def evaluate(learner, train, held, n_gen, seed, night_readout=False):
    t0 = time.time()
    st = learner.state
    n = len(st.pos)
    seen = train[:n]
    flat = ContextBoardSearch(seen, learner.alpha, learner.counts).bits() / n
    ref = st.reference()
    assert abs(ref.bits() - st.bits()) < 1e-6 * max(1.0, st.bits()), (ref.bits(), st.bits())
    held_search, held_flat = ref.held_out_bits(held)
    g = learner.grammar
    held_grammar = -sum(learner.log_prob(p) for p in held) / len(held) / math.log(2)
    secs = np.array(learner.seconds)
    res = {"N": n,
           "train bits/pos": {"read alone": flat, "after the search": st.bits() / n,
                              "TRELLIS total": g.info["total bits"] / n},
           "held-out bits/pos": {"read alone": held_flat, "search code": held_search, "TRELLIS grammar": held_grammar},
           "chunk gain, train (read alone - total)": flat - g.info["total bits"] / n,
           "chunk gain, held out (read alone - grammar)": held_flat - held_grammar,
           "symbols": g.K, "rule classes": g.M, "chunk types (grammar)": g.info["chunk types"],
           "chunk labels": len(st.moves), "moves": [f"[{m[0]} {m[1]} {m[2]}] -> {m[3]}" for m in st.moves],
           "context option": list(st.top_nats()[1]),
           "seconds per position": {"mean": float(secs.mean()), "median": float(np.median(secs)),
                                    "p99": float(np.percentile(secs, 99)), "max": float(secs.max()),
                                    "total": float(secs.sum()),
                                    "mean, last 10%": float(secs[-max(1, n // 10):].mean())},
           "scores": dict(st.stats)}
    res.update(chunk_report(seen, st.analyses()))
    if n_gen:
        res["generation"] = generation(learner, seen, held, n_gen, seed)
    if night_readout:
        # Diagnostic: the night's consolidation of these analyses.
        night = Learner(seed=seed, alpha=learner.alpha)
        night.experiences = list(seen)
        model = night.fit(st.analyses(), BoardMemory(counts=learner.counts, context=True))
        lp = [model.log_prob(p, learner.analyse(p)) for p in held]
        res["night consolidation of these analyses"] = {
            "TRELLIS total": model.grammar.info["total bits"] / n,
            "held-out grammar": -sum(lp) / len(held) / math.log(2),
            "symbols": model.grammar.K, "rule classes": model.grammar.M}
    res["evaluation seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1800")
    ap.add_argument("--ply", type=int, default=16)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--kind", default="real", choices=["real", "random"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--checkpoints", default="250,500,1000,2000,4000")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--n-gen-mid", type=int, default=200)
    ap.add_argument("--config", default="{}", help="JSON overrides of IncrementalChessLearner.DEFAULTS")
    ap.add_argument("--night-readout", action="store_true")
    ap.add_argument("--extra-held", type=int, default=0,
                    help="also evaluate each checkpoint on the held-out set of a run with this many training positions")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    config = json.loads(args.config)
    name = f"e{args.elo}_p{args.ply}_n{args.train}_a{args.alpha:g}_{args.kind}_inc{args.tag}"
    with open(positions_path(args.elo, args.ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    print(f"{name}: {len(fens)} positions in {positions_path(args.elo, args.ply)}", flush=True)
    if len(fens) < args.train + args.test:
        sys.exit(f"{name}: only {len(fens)} positions")
    order = np.random.default_rng(args.seed).permutation(len(fens))
    boards = [fen_to_board(fens[i]) for i in order[:args.train + args.test]]
    if args.kind == "random":
        rng = np.random.default_rng(2026)
        boards = [randomize(b, rng) for b in boards]
    train = [to_position(b) for b in boards[:args.train]]
    held = [to_position(b) for b in boards[args.train:]]
    extra = None
    if args.extra_held:
        b2 = [fen_to_board(fens[i]) for i in order[args.extra_held:args.extra_held + args.test]]
        extra = [to_position(b) for b in b2]
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",") if c} | {args.train})
    learner = IncrementalChessLearner(seed=args.seed, alpha=args.alpha, config=config)
    out = {"name": name, "elo": args.elo, "ply": args.ply, "N": args.train, "test": len(held), "alpha": args.alpha,
           "kind": args.kind, "config": learner.config, "checkpoints": [], "history": learner.history}
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(train):
        learner.observe(p)
        n = i + 1
        if n % 100 == 0:
            print(f"  {n}: {learner.state.bits() / n:.4f} bits/pos, {len(learner.state.moves)} chunk types, "
                  f"{np.mean(learner.seconds[-100:]):.3f} s/pos ({time.time() - t0:.0f}s)", flush=True)
        if n in checkpoints:
            final = n == args.train
            res = evaluate(learner, train, held, args.n_gen if final else args.n_gen_mid, args.seed,
                           night_readout=final and args.night_readout)
            if extra is not None:
                ex_search, ex_flat = learner.state.reference().held_out_bits(extra)
                res[f"held-out of the N={args.extra_held} split"] = {
                    "read alone": ex_flat, "search code": ex_search,
                    "TRELLIS grammar": -sum(learner.log_prob(p) for p in extra) / len(extra) / math.log(2)}
            res["learning seconds so far"] = float(np.sum(learner.seconds))
            out["checkpoints"].append(res)
            print(f"checkpoint {n}: train total {res['train bits/pos']['TRELLIS total']:.3f} "
                  f"(search {res['train bits/pos']['after the search']:.3f}, read {res['train bits/pos']['read alone']:.3f}); "
                  f"held grammar {res['held-out bits/pos']['TRELLIS grammar']:.3f} "
                  f"(read {res['held-out bits/pos']['read alone']:.3f}, gain {res['chunk gain, held out (read alone - grammar)']:+.2f}); "
                  f"{res['chunk labels']} chunk types; {res['seconds per position']['mean']:.3f} s/pos"
                  + (f"; gen pass {res['generation']['checks'].get('passes every check', 0):.3f}" if "generation" in res else ""),
                  flush=True)
            for c in res["chunk types"][:12]:
                print(f"    {c['chunk']} x{c['count']} {c['anchors']}", flush=True)
            with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1, default=str)
            model, learner.model, learner._grammar_at = learner.model, None, None
            with open(os.path.join(OUT, name + f"_state{n}.pkl"), "wb") as f:
                pickle.dump(learner, f)
    print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
