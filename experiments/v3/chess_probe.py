"""Chess puzzle themes as a downstream test of TRELLIS representations (docs/GRAPH.md).

Learning sees positions only (the puzzle positions of a learning pool, no
labels). Each position of a probe set is then parsed (memory unchanged) and
described by the bag of concepts it reaches: for every piece, the information-band
concepts of its context; for every chunk of the stored analysis, those of its
context and its content. A logistic-regression probe per theme (Lichess
puzzle themes) is trained on n labelled positions and scored by ROC AUC on a
held-out set. Baselines see the same position without learning: a
piece-square bag, and a piece-relation bag (each piece with each piece of its
star, the relations the learner uses).

Usage: chess_probe.py <cfg json> <n learn> <seed> <out json> [max_size]
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import ChessDomain, GraphTrellis  # noqa: E402

PUZZLES = os.path.join(ROOT, "data", "chess", "puzzles", "puzzles_400k.csv")   # zstd -dc ... | head -400001
THEMES = ["fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "backRankMate",
          "mateIn1", "mateIn2", "sacrifice", "deflection", "attraction", "kingsideAttack",
          "promotion", "advancedPawn", "trappedPiece", "exposedKing"]


def apply_uci(pos: C.Position, uci: str) -> C.Position:
    """The position after a move (castling, en passant and promotion included)."""
    p = dict(pos)
    f = (ord(uci[0]) - 97, int(uci[1]) - 1)
    t = (ord(uci[2]) - 97, int(uci[3]) - 1)
    piece = p.pop(f)
    if piece[1] == "K" and abs(t[0] - f[0]) == 2:                       # castling: the rook too
        rf, rt = ((7, f[1]), (5, f[1])) if t[0] > f[0] else ((0, f[1]), (3, f[1]))
        if rf in p:
            p[rt] = p.pop(rf)
    if piece[1] == "P" and f[0] != t[0] and t not in p:                  # en passant
        p.pop((t[0], f[1]), None)
    if len(uci) == 5:
        piece = piece[0] + uci[4].upper()
    p[t] = piece
    return p


def load_puzzles(n: int, seed: int):
    rows = []
    with open(PUZZLES, encoding="utf-8") as text:
        for k, row in enumerate(csv.DictReader(text)):
            rows.append(row)
            if len(rows) >= 20 * n:
                break
    random.Random(seed).shuffle(rows)
    out = []
    for row in rows[:n]:
        pos = apply_uci(C.parse_fen(row["FEN"]), row["Moves"].split()[0])
        out.append((pos, set(row["Themes"].split())))
    return out


def piece_square(pos):
    return Counter(f"{t}@{C.square_name(s)}" for s, t in pos.items())


def piece_relation(pos):
    out = Counter()
    for s, t in pos.items():
        st = C.star(pos, s)
        for d, v in st.items():
            if v not in (C.EMPTY, C.EDGE):
                out[f"{t}|{d}:{v}"] += 1
    return out


def concepts(L: GraphTrellis, pos):
    """The bag of band concepts a position reaches, parsed against memory."""
    r = L.parse(pos)
    out = Counter()
    for e in r["words"]:
        for v in L._path_ids(L.ctx.categorize(e.ctx), "bands"):
            out["x" + v] += 1
    stack = [e for e in r["top"] if e.token is None]
    while stack:
        e = stack.pop()
        for v in L._path_ids(L.ctx.categorize(e.ctx), "bands"):
            out["x" + v] += 1
        for v in L._path_ids(L.cnt.categorize(e.cnt), "bands"):
            out["n" + v] += 1
        stack += [p for p in (e.left, e.right) if p.token is None]
    return out


def _auc(y, s):
    """ROC AUC by the rank statistic (ties averaged)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _vectorise(feats_tr, feats_te):
    vocab = {}
    for f in feats_tr:
        for k in f:
            vocab.setdefault(k, len(vocab))
    def mat(F):
        X = np.zeros((len(F), len(vocab)), dtype=np.float32)
        for i, f in enumerate(F):
            for k, v in f.items():
                j = vocab.get(k)
                if j is not None:
                    X[i, j] = np.log1p(v)
        return X
    return mat(feats_tr), mat(feats_te)


def probe(feats_tr, y_tr, feats_te, y_te, l2: float = 1.0, steps: int = 300):
    """L2 logistic regression by gradient descent (full batch, Adam), AUC on the test set."""
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return None
    Xtr, Xte = _vectorise(feats_tr, feats_te)
    y = np.asarray(y_tr, np.float32)
    w, b = np.zeros(Xtr.shape[1], np.float32), 0.0
    m, v, mb, vb = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    n = len(y)
    for t in range(1, steps + 1):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        gw = Xtr.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        m = 0.9 * m + 0.1 * gw; v = 0.999 * v + 0.001 * gw * gw
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
        w -= 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        b -= 0.05 * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return float(_auc(y_te, Xte @ w + b))


def main():
    cfg, n_learn, seed, out = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    max_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    sizes, n_test = [100, 300, 1000, 3000], 2000
    if len(sys.argv) > 6 and sys.argv[6] == "smoke":
        sizes, n_test = [50, 100], 200
    data = load_puzzles(n_learn + max(sizes) + n_test, seed)
    learn = data[:n_learn]
    lab = data[n_learn:n_learn + max(sizes)]
    test = data[n_learn + max(sizes):]
    L = GraphTrellis(ChessDomain(), max_size=max_size, seed=seed, **cfg)
    t0 = time.time()
    for k, (pos, _) in enumerate(learn, 1):
        L.learn(pos)
        if k % 100 == 0:
            print("learned", k, round(time.time() - t0), flush=True)
    learn_s = time.time() - t0
    t0 = time.time()
    feats = {"piece-square": [], "piece-relation": [], "trellis": [], "trellis+piece-square": []}
    for pos, _ in lab + test:
        ps, tc = piece_square(pos), concepts(L, pos)
        feats["piece-square"].append(ps)
        feats["piece-relation"].append(piece_relation(pos))
        feats["trellis"].append(tc)
        feats["trellis+piece-square"].append(ps + tc)
    feat_s = time.time() - t0
    res = {"cfg": cfg, "n learn": n_learn, "seed": seed, "max size": max_size,
           "learn s": round(learn_s), "featurise s": round(feat_s), "stats": L.stats,
           "theme rate (test)": {t: round(sum(t in th for _, th in test) / len(test), 3) for t in THEMES}, "auc": {}}
    nl = len(lab)
    for name, F in feats.items():
        res["auc"][name] = {}
        for n in sizes:
            aucs = {}
            for t in THEMES:
                a = probe(F[:n], [t in th for _, th in lab[:n]], F[nl:], [t in th for _, th in test])
                if a is not None:
                    aucs[t] = round(a, 3)
            res["auc"][name][n] = {"mean": round(sum(aucs.values()) / max(len(aucs), 1), 3), "by theme": aucs}
        print(name, {n: res["auc"][name][n]["mean"] for n in sizes}, flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
