"""Chess positions as a storage test (Track 2 of HANDOFF_TRELLIS_V2.md).

How many bits does each method need to store N positions, on real positions
and on random positions with the same pieces (the same multiset per position,
placed uniformly on distinct squares, pawns off the back ranks)?

Modes (each writes JSON under experiments/v2/results/chess_memory/):

  --export              the split of run_chess.py (seed 13 permutation of
                        data/chess/positions_1500_ply30.fen) as int8 boards,
                        with each position's random counterpart, to
                        data/chess/memory/boards.npz (read by chess_nets.py)
  --trellis N --kind K  TRELLIS v2 on the first N positions (real or random):
                        two-part description length (grammar model bits +
                        data bits), held-out bits (the grammar's code), and
                        the read alone, as run_chess.py measures them
  --baselines           fixed baselines: 64 log2 13; FEN strings under gzip,
                        xz and zstd, per position and as a corpus; an
                        independent per-square categorical model
  --oracle              the random positions' generator as a code (a floor)
  --summarize           tables (summary.md) and figures from all JSON files

Boards: square i = rank * 8 + file (a1 = 0, the read's order); 0 = empty,
1 + KIND_INDEX[token] otherwise (KINDS sorted: bB bK bN bP bQ bR wB wK wN wP
wQ wR). Only --trellis imports trellis2; the rest runs in any env with numpy
(--baselines needs the python ``zstandard`` package for zstd).
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "chess_memory")
DATA = os.path.join(ROOT, "data", "chess", "positions_1500_ply30.fen")
BOARDS = os.path.join(ROOT, "data", "chess", "memory", "boards.npz")
KINDS = ["bB", "bK", "bN", "bP", "bQ", "bR", "wB", "wK", "wN", "wP", "wQ", "wR"]
PAWNS = (KINDS.index("bP") + 1, KINDS.index("wP") + 1)
NS = (1000, 4000, 16000, 32000)
TEST = 500
SEED = 13


# --------------------------------------------------------------------------- #
# Boards
# --------------------------------------------------------------------------- #
def fen_to_board(fen: str) -> np.ndarray:
    b = np.zeros(64, dtype=np.int8)
    for r, row in enumerate(fen.split()[0].split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                b[(7 - r) * 8 + f] = KINDS.index(("w" if ch.isupper() else "b") + ch.upper()) + 1
                f += 1
    return b


def board_to_fen(b: np.ndarray) -> str:
    rows = []
    for r in range(7, -1, -1):
        s, empty = "", 0
        for f in range(8):
            v = int(b[r * 8 + f])
            if v == 0:
                empty += 1
                continue
            if empty:
                s += str(empty)
                empty = 0
            t = KINDS[v - 1]
            s += t[1] if t[0] == "w" else t[1].lower()
        rows.append(s + (str(empty) if empty else ""))
    return "/".join(rows)


def randomize(b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same pieces on uniformly random distinct squares, pawns on ranks
    2-7. Pawns first on the 48 inner squares, then the rest on what is left:
    every valid placement is equally likely."""
    pieces = b[b > 0]
    pawns = pieces[np.isin(pieces, PAWNS)]
    others = pieces[~np.isin(pieces, PAWNS)]
    out = np.zeros(64, dtype=np.int8)
    inner = np.arange(8, 56)
    sq = rng.choice(inner, size=len(pawns), replace=False)
    out[sq] = pawns
    free = np.flatnonzero(out == 0)
    sq = rng.choice(free, size=len(others), replace=False)
    out[sq] = others
    return out


def export():
    with open(DATA) as f:
        fens = [line.strip() for line in f if line.strip()]
    order = np.random.default_rng(SEED).permutation(len(fens))
    real = np.stack([fen_to_board(fens[i]) for i in order])
    rng = np.random.default_rng(2026)
    rand = np.stack([randomize(b, rng) for b in real])
    assert all(board_to_fen(real[i]) == fens[order[i]].split()[0] for i in range(200))
    assert np.array_equal(np.sort(real, axis=1), np.sort(rand, axis=1))
    assert not np.isin(rand[:, :8], PAWNS).any() and not np.isin(rand[:, 56:], PAWNS).any()
    os.makedirs(os.path.dirname(BOARDS), exist_ok=True)
    np.savez_compressed(BOARDS, real=real, random=rand, order=order)
    print(f"{len(real)} positions (permuted with seed {SEED}) and their random counterparts -> {BOARDS}")


def load(kind: str) -> np.ndarray:
    return np.load(BOARDS)[kind]


def to_position(b: np.ndarray) -> dict:
    return {(i % 8, i // 8): KINDS[v - 1] for i, v in enumerate(b) if v}


# --------------------------------------------------------------------------- #
# TRELLIS v2
# --------------------------------------------------------------------------- #
def run_trellis(n: int, kind: str, alpha: float = 0.001):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from trellis2.chess import BoardSearch, ChessLearner

    boards = load(kind)
    train = [to_position(b) for b in boards[:n]]
    test = [to_position(b) for b in boards[n:n + TEST]]
    t0 = time.time()
    learner = ChessLearner(seed=SEED, alpha=alpha, context=False)
    for p in train:
        learner.observe(p)
    g = learner.sleep()
    seconds = time.time() - t0
    held_chunks, held_flat = learner.search.held_out_bits(test)
    held_plain = (BoardSearch(train, alpha, counts=False).code_of(BoardSearch(test, alpha, counts=False)) / len(test))
    held_grammar = -sum(learner.log_prob(p) for p in test) / len(test) / math.log(2)
    flat_plain_train = BoardSearch(train, alpha, counts=False).bits()
    res = {"method": "trellis", "kind": kind, "N": n, "test": len(test), "alpha": alpha, "seconds": seconds,
           "total bits": g.info["total bits"], "model bits": g.info["model bits"], "data bits": g.info["data bits"],
           "train bits/pos, TRELLIS v2 (two-part: model + data)": g.info["total bits"] / n,
           "train bits/pos, the read alone (adaptive code)": learner.history[0]["bits"] / n,
           "train bits/pos, after the search": [h for h in learner.history if h["stage"] in ("flat", "chunk")][-1]["bits"] / n,
           "train bits/pos, squares on their own (adaptive code)": flat_plain_train / n,
           "held-out bits/pos, TRELLIS v2 (the grammar)": held_grammar,
           "held-out bits/pos, learned chunks (search code)": held_chunks,
           "held-out bits/pos, the read alone": held_flat,
           "held-out bits/pos, squares on their own": held_plain,
           "symbols": g.K, "rule classes": g.M, "chunk types": g.info["chunk types"],
           "search moves": [h["move"] for h in learner.history if h["stage"] == "chunk"]}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"trellis_{kind}_{n}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- #
# Fixed baselines
# --------------------------------------------------------------------------- #
def compressors():
    import zstandard
    zc = zstandard.ZstdCompressor(level=19)
    return {"gzip -9": lambda s: gzip.compress(s, compresslevel=9, mtime=0),
            "xz -9e": lambda s: lzma.compress(s, preset=9 | lzma.PRESET_EXTREME),
            "zstd -19": zc.compress}


def dm_bits(counts: np.ndarray, a: float) -> float:
    """Adaptive (Dirichlet-multinomial mixture) code of count rows, bits."""
    from scipy.special import gammaln
    K = counts.shape[-1]
    return float(np.sum(gammaln(counts.sum(-1) + K * a) - gammaln(K * a))
                 - np.sum(gammaln(counts + a) - gammaln(a))) / math.log(2)


def per_square(train: np.ndarray, test: np.ndarray, a: float = 0.5):
    c = np.zeros((64, 13))
    np.add.at(c, (np.tile(np.arange(64), len(train)), train.reshape(-1).astype(int)), 1)
    n = len(train)
    adaptive = dm_bits(c, a) / n
    p = (c + a) / (c.sum(1, keepdims=True) + 13 * a)
    held = -np.log2(p[np.arange(64), test.astype(int)]).sum(1).mean()
    pml = c / c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ml = -np.nansum(np.where(c > 0, c * np.log2(pml), 0)) / n
    free = int(((c > 0).sum(1) - 1).clip(0).sum())
    return {"train bits/pos, adaptive code (KT, a=0.5)": adaptive,
            "train bits/pos, ML data bits": ml,
            "train bits/pos, ML + (free/2) log2 N (BIC)": ml + free / 2 * math.log2(n) / n,
            "held-out bits/pos": float(held), "free parameters": free}


def baselines():
    comp = compressors()
    out = []
    for kind in ("real", "random"):
        boards = load(kind)
        for n in NS:
            train, test = boards[:n], boards[n:n + TEST]
            fens = [board_to_fen(b).encode() for b in train]
            tfens = [board_to_fen(b).encode() for b in test]
            res = {"method": "baselines", "kind": kind, "N": n,
                   "uniform 64 log2 13": 64 * math.log2(13),
                   "FEN raw (8 bits/char, + newline)": 8 * np.mean([len(s) + 1 for s in fens])}
            corpus = b"\n".join(fens) + b"\n"
            both = corpus + b"\n".join(tfens) + b"\n"
            for name, fn in comp.items():
                res[f"{name}, per position"] = 8 * float(np.mean([len(fn(s)) for s in fens]))
                c_train = len(fn(corpus))
                res[f"{name}, corpus / N"] = 8 * c_train / n
                # Held out: what the 500 test positions add to the compressed corpus.
                res[f"{name}, held out (corpus+test minus corpus) / 500"] = 8 * (len(fn(both)) - c_train) / len(test)
            res["per-square categorical"] = per_square(train, test)
            out.append(res)
            print(kind, n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "baselines.json"), "w") as f:
        json.dump(out, f, indent=1)


def oracle():
    """The random positions' own generator as a code: the material (which
    pieces, coded with its empirical distribution over all 57,884 positions,
    a plug-in entropy, so slightly optimistic) plus log2 of the number of
    distinct placements (pawns on the 48 inner squares, the rest anywhere
    free). No model of random positions can beat it on average."""
    from collections import Counter
    rand = load("random")
    mats = [tuple(np.bincount(b, minlength=13)[1:]) for b in rand]
    freq = Counter(mats)
    lg = lambda n: math.lgamma(n + 1) / math.log(2)
    res = {}
    for n in NS:
        bits = []
        for m in mats[n:n + TEST]:
            m = np.array(m)
            pw = m[[p - 1 for p in PAWNS]]
            npw, no = int(pw.sum()), int(m.sum() - pw.sum())
            place = (lg(48) - lg(48 - npw) - sum(lg(int(c)) for c in pw)
                     + lg(64 - npw) - lg(64 - npw - no)
                     - sum(lg(int(c)) for i, c in enumerate(m) if i + 1 not in PAWNS))
            bits.append(place - math.log2(freq[tuple(m)] / len(mats)))
        res[n] = float(np.mean(bits))
    with open(os.path.join(OUT, "oracle_random.json"), "w") as f:
        json.dump({"held-out bits/pos, the generator's own code (random positions)": res}, f, indent=1)
    print(res)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--trellis", type=int)
    ap.add_argument("--kind", default="real", choices=["real", "random"])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    if args.export:
        export()
    if args.trellis:
        run_trellis(args.trellis, args.kind)
    if args.baselines:
        baselines()
    if args.oracle:
        oracle()
    if args.summarize:
        from summarize_chess_memory import summarize  # noqa: E402
        summarize(OUT)


if __name__ == "__main__":
    main()
