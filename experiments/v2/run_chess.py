"""TRELLIS v2 on chess positions: chunks joined by typed relations.

The domain is described in ``trellis2/chess.py``: a piece's context window is
the star (the eight queen rays to any distance, the eight knight jumps), a
chunk joins two elements whose anchors see each other along a star relation,
and a board is read square by square.

Data: middlegame positions from the Lichess database of January 2013 (CC0):
games in which both players are rated at least 1800 and that last at least
40 plies, the position after ply 30. ``--extract`` writes them to
``data/chess/positions_1800_ply30.fen`` (needs python-chess and zstd):

    curl -L -o data/chess/lichess_db_standard_rated_2013-01.pgn.zst \\
        https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
    python experiments/v2/run_chess.py --extract
    python experiments/v2/run_chess.py --train 4000 --test 500 --out experiments/v2/results/chess

Reported: the context the square-by-square read learns (features of the
pieces on earlier squares that pay for themselves); training and held-out
bits per position against the code that reads each square on its own (with
the same context, and with none); the chunk types the search finds and the
categories the representation hierarchy forms; and for positions generated
from the grammar, how many place every piece on a free square, simple chess
checks, and how many of their chunks occur in held-out games.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.chess import (EMPTY, BoardSearch, ChessLearner, default_positions_path,  # noqa: E402
                            is_chunk, load_positions, plausibility, scan_rows, square_name)

GLYPH = {"wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
         "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟"}
SURFACE, INK, INK2, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#2a78d6", "#eb6834"
LIGHT, DARK = "#efece6", "#d6d1c7"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def extract(src: str, out: str, min_elo: int = 1800, ply: int = 30, min_plies: int = 40) -> None:
    import chess.pgn
    elo = re.compile(r'\[(White|Black)Elo "(\d+)"\]')

    def games(stream):
        buf = []
        for line in stream:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)

    proc = subprocess.Popen(["zstd", "-dc", src], stdout=subprocess.PIPE)
    n = kept = 0
    with open(out, "w") as f:
        for text in games(io.TextIOWrapper(proc.stdout, encoding="utf-8")):
            n += 1
            ratings = dict(elo.findall(text))
            if len(ratings) < 2 or min(int(v) for v in ratings.values()) < min_elo:
                continue
            game = chess.pgn.read_game(io.StringIO(text))
            moves = list(game.mainline_moves())
            if len(moves) < min_plies:
                continue
            board = game.board()
            for m in moves[:ply]:
                board.push(m)
            f.write(board.board_fen() + "\n")
            kept += 1
    print(f"{n} games, {kept} positions written to {out}")


# --------------------------------------------------------------------------- #
# Chunks as sets of pieces
# --------------------------------------------------------------------------- #
def pieces_of(node, position=None):
    """{square: token} of a chunk (a node of an analysis or of a sample)."""
    out = {}
    stack = [node]
    while stack:
        n = stack.pop()
        if is_chunk(n):
            stack.extend([n[1][0], n[1][2]])
        else:
            out[n[1]] = position[n[1]]
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_board(ax, pieces, title=None, focus=None, crop=None):
    files = range(8) if crop is None else range(crop[0], crop[2] + 1)
    ranks = range(8) if crop is None else range(crop[1], crop[3] + 1)
    for f in files:
        for r in ranks:
            ax.add_patch(Rectangle((f, r), 1, 1, facecolor=DARK if (f + r) % 2 == 0 else LIGHT, edgecolor="none"))
            if focus and (f, r) in focus:
                ax.add_patch(Rectangle((f + 0.04, r + 0.04), 0.92, 0.92, facecolor="none",
                                       edgecolor=BLUE, linewidth=1.6))
    for (f, r), t in pieces.items():
        if f in files and r in ranks:
            ax.text(f + 0.5, r + 0.47, GLYPH.get(t, "?"), ha="center", va="center", fontsize=15 if crop else 13,
                    color=INK, family="DejaVu Sans")
    ax.set_xlim(min(files), max(files) + 1)
    ax.set_ylim(min(ranks), max(ranks) + 1)
    ax.set_aspect("equal")
    ax.set_xticks([f + 0.5 for f in files])
    ax.set_xticklabels(["abcdefgh"[f] for f in files], fontsize=7, color=MUTED)
    ax.set_yticks([r + 0.5 for r in ranks])
    ax.set_yticklabels([str(r + 1) for r in ranks], fontsize=7, color=MUTED)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, color=INK2)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="Dirichlet concentration of every table")
    ap.add_argument("--no-context", action="store_true",
                    help="read every square on its own (no learned context)")
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "chess"))
    args = ap.parse_args()
    if args.extract:
        src = os.path.join(os.path.dirname(default_positions_path()), "lichess_db_standard_rated_2013-01.pgn.zst")
        extract(src, default_positions_path())
        return
    os.makedirs(args.out, exist_ok=True)
    positions = load_positions()
    order = np.random.default_rng(args.seed).permutation(len(positions))
    train = [positions[i] for i in order[:args.train]]
    test = [positions[i] for i in order[args.train:args.train + args.test]]

    t0 = time.time()
    learner = ChessLearner(seed=args.seed, alpha=args.alpha, context=not args.no_context)
    g = learner.sleep(train)
    seconds = time.time() - t0
    flat_bits = learner.history[0]["bits"] / len(train)
    search_bits = [h for h in learner.history if h["stage"] in ("flat", "chunk")][-1]["bits"] / len(train)
    held_chunks, held_flat = learner.search.held_out_bits(test)
    held_plain = BoardSearch(train, learner.alpha).code_of(BoardSearch(test, learner.alpha)) / len(test)
    context = [f"at least {m} {kind}" for kind, m in learner.features]
    print(f"context of the read: {', '.join(context)}; held out without it {held_plain:.2f} bits/position")
    print(f"{len(train)} positions, night {seconds:.0f}s: flat {flat_bits:.2f}, search {search_bits:.2f}, "
          f"consolidated {g.info['total bits'] / len(train):.2f} bits/position; held out: chunks {held_chunks:.2f}, "
          f"flat {held_flat:.2f}; {g.K} symbols, {g.info['chunk types']} chunk types", flush=True)

    # Chunk types in the training analyses.
    found = Counter()
    anchors = defaultdict(Counter)
    example = {}
    for position, tops in zip(train, learner.analyses):
        for node in tops:
            if is_chunk(node):
                pcs = pieces_of(node, position)
                key = render_relative(node, position)
                found[key] += 1
                anchors[key][node_anchor(node)] += 1
                example.setdefault((key, node_anchor(node)), pcs)
    chunk_types = [{"chunk": k, "count": c, "anchors": {square_name(a): n for a, n in anchors[k].most_common(3)}}
                   for k, c in found.most_common()]

    # Categories of the representation hierarchy: yields and where they stand.
    mem = learner.model.memory
    where = defaultdict(Counter)
    for e, sym in enumerate(g.elem_symbol):
        where[int(sym)][square_name(mem.anchor[e])] += mem.weight[e]
    categories = [{"symbol": a, "count": float(sum(g.symbol_yields[a].values())),
                   "yields": g.symbol_yields[a].most_common(5), "squares": where[a].most_common(5)}
                  for a in np.argsort([-sum(y.values()) for y in g.symbol_yields])]

    # Generation.
    rng = np.random.default_rng(args.seed)
    test_sets = [dict(p) for p in test]
    samples, failed = [], 0
    while len(samples) < args.n_gen:
        s = learner.sample(rng)
        if s is None:
            failed += 1
            continue
        samples.append(s)
    checks = Counter()
    gen_chunks = exact = 0
    for position, tops in samples:
        ok = plausibility(position)
        for k, v in ok.items():
            checks[k] += v
        checks["passes every check"] += all(ok.values())
        for node in tops:
            if is_chunk(node):
                gen_chunks += 1
                pcs = pieces_of(node, position)
                exact += any(all(p.get(sq) == t for sq, t in pcs.items()) for p in test_sets)
    # The same checks for positions whose squares are drawn on their own
    # (each in the same context: the pieces on earlier squares).
    flat = BoardSearch(train, learner.alpha, learner.features)
    flat_checks = Counter()
    for _ in range(args.n_gen):
        pos = {}
        for i in range(64):
            c = flat.sq.get(scan_rows(pos, learner.features)[i]) or Counter({EMPTY: 1})
            o = rng.choice(list(c), p=np.array(list(c.values())) / sum(c.values()))
            if o != EMPTY:
                pos[(i % 8, i // 8)] = o
        ok = plausibility(pos)
        for k, v in ok.items():
            flat_checks[k] += v
        flat_checks["passes every check"] += all(ok.values())
    gen = {"attempts": args.n_gen + failed, "collision or off the board": failed / (args.n_gen + failed),
           "checks": {k: v / args.n_gen for k, v in checks.items()},
           "checks, squares drawn on their own": {k: v / args.n_gen for k, v in flat_checks.items()},
           "generated chunks": gen_chunks,
           "generated chunks found in a held-out position": exact / max(gen_chunks, 1)}
    print(json.dumps(gen, indent=1), flush=True)

    results = {"train": len(train), "test": len(test), "seconds": seconds,
               "bits per position": {"training, squares on their own": flat_bits,
                                     "training, after the search": search_bits,
                                     "training, TRELLIS v2 (consolidated)": g.info["total bits"] / len(train),
                                     "held out, squares on their own": held_flat,
                                     "held out, squares on their own, no context": held_plain,
                                     "held out, learned chunks": held_chunks},
               "context of the read": context, "alpha": learner.alpha,
               "symbols": g.K, "rule classes": g.M, "chunk types (grammar)": g.info["chunk types"],
               "model bits": g.info["model bits"], "data bits": g.info["data bits"],
               "search moves": [h["move"] for h in learner.history if h["stage"] == "chunk"],
               "chunk types": chunk_types, "categories": categories[:20], "generation": gen}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    # Figure 1: the chunk types, each at its most frequent anchor.
    top = chunk_types[:12]
    if top:
        cols = 4
        rows = (len(top) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), facecolor=SURFACE)
        for ax, ct in zip(np.atleast_1d(axes).ravel(), top):
            pcs = example[(ct["chunk"], anchors[ct["chunk"]].most_common(1)[0][0])]
            fs = [sq[0] for sq in pcs]
            rs = [sq[1] for sq in pcs]
            crop = (max(0, min(fs) - 1), max(0, min(rs) - 1), min(7, max(fs) + 1), min(7, max(rs) + 1))
            draw_board(ax, pcs, f"{ct['chunk']}\n×{ct['count']}", focus=set(pcs), crop=crop)
        for ax in np.atleast_1d(axes).ravel()[len(top):]:
            ax.axis("off")
        fig.suptitle("Chunks TRELLIS v2 finds in middlegame positions (each at its most frequent place)",
                     fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "chunk_types.png"), dpi=170, facecolor=SURFACE)
        plt.close(fig)
    # Figure 2: generated positions, chunks outlined.
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), facecolor=SURFACE)
    for ax, (position, tops) in zip(axes.ravel(), samples[:8]):
        focus = set()
        for node in tops:
            if is_chunk(node):
                focus |= set(pieces_of(node, position))
        ok = plausibility(position)
        short = {"one king each": "kings", "no pawn on a back rank": "back-rank pawn",
                 "at most 8 pawns each": "pawns", "at most 16 pieces each": "pieces",
                 "no more of any kind than at the start": "material"}
        draw_board(ax, position, "passes every check" if all(ok.values()) else
                   "fails: " + ", ".join(short[k] for k, v in ok.items() if not v), focus=focus)
    fig.suptitle("Positions generated from the grammar (chunks outlined)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "generated_positions.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    lines = [f"Chess positions (Lichess, both players 1800+, after ply 30): {len(train)} learned, {len(test)} held out.", "",
             (f"The context of the square-by-square read (features of the pieces on earlier squares that pay for "
              f"themselves): {', '.join(context)}. Dirichlet concentration "
              f"α = {learner.alpha:g}. Without the context, squares on their own: {held_plain:.2f} held-out bits "
              f"per position." if context else
              f"The read has no context: every square is read on its own (α = {learner.alpha:g})."), "",
             "| Bits per position | Squares on their own | TRELLIS v2 |", "|---|---|---|",
             f"| training | {flat_bits:.2f} | {g.info['total bits'] / len(train):.2f} (search alone: {search_bits:.2f}) |",
             f"| held out | {held_flat:.2f} | {held_chunks:.2f} (learned chunks) |", "",
             f"Symbols: {g.K}; rule classes: {g.M}; chunk types: {g.info['chunk types']}. The grammar's size: "
             f"{g.info['model bits']:,.0f} model bits (and {g.info['data bits']:,.0f} data bits for the training positions).", "",
             "| Chunk type | Count | Most frequent anchors |", "|---|---|---|"]
    for ct in chunk_types[:15]:
        lines.append(f"| `{ct['chunk']}` | {ct['count']} | {', '.join(f'{a} ({n})' for a, n in ct['anchors'].items())} |")
    lines += ["", "| Generated positions (1,000) | TRELLIS v2 | Squares on their own |", "|---|---|---|"]
    for k in gen["checks"]:
        lines.append(f"| {k} | {gen['checks'][k]:.1%} | {gen['checks, squares drawn on their own'][k]:.1%} |")
    lines.append(f"| samples rejected (a chunk off the board or on an occupied square) | "
                 f"{gen['collision or off the board']:.1%} | – |")
    lines.append(f"| generated chunks found, piece for piece, in a held-out position | "
                 f"{gen['generated chunks found in a held-out position']:.1%} | – |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def node_anchor(node):
    while is_chunk(node):
        node = node[1][0]
    return node[1]


def render_relative(node, position):
    if not is_chunk(node):
        return position[node[1]]
    x, rel, y = node[1]
    return f"[{render_relative(x, position)} {rel} {render_relative(y, position)}]"


if __name__ == "__main__":
    main()
