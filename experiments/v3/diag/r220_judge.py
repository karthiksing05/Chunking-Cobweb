"""Round 220: judge for the chess POSITION regimes (defined before any TRELLIS output was generated).

Boards are {(file,rank): token} dicts as trellis2.chess; side to move is assumed White, no castling rights, no en passant.
  R-non    illegal: python-chess Board.is_valid() is False (a king count other than one each, pawns on rank 1/8, the side not to
           move in check, too many pieces / pawns / promoted pieces, ...)
  R-real   legal and realism score >= TAU
  R-legal  legal and realism score < TAU (legal syntax, not game-like)
Realism score = P(real) of an L2 logistic regression (numpy) on: 768 piece-square indicators, 12 piece-kind counts, 16 pawn-per-file
counts, the material difference, the total number of pieces. Classes: real = positions of data/chess/cs1973/train_long.fen
lines [19459:25459] (train), TAU = the 5th percentile of the score of lines [26459:27459] (validation, so 95% of real pass by
construction); real held-out control = lines [25459:26459]. The generator trains only on lines [0:19459] (disjoint games).
Negatives for training the score: 50% real positions with the pieces put on random squares (no pawn on rank 1/8), legal ones kept;
50% positions reached by random play from the start (20-70 plies), legal by construction.
Controls (reported): real held-out; random legal (shuffled-legal + random-play, fresh seeds); random illegal (pieces on random
squares, illegal kept, plus the real position with one extra king / a missing king).
Usage (compute node, repo root): r220_judge.py  -> writes experiments/v3/results/chess_regimes/r220_judge.json (weights, TAU, controls)
Import: Judge().classify(board_dict) -> "real" | "legal" | "non"
"""
import json
import os
import random
import sys

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2 import chess as C  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes", "r220_judge.json")
LINES = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "cs1973", "train_long.fen"), encoding="utf-8") if l.strip()]
KINDS = sorted(C.TOKENS)
VAL = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
SYM = {"w": lambda c: c.upper(), "b": lambda c: c.lower()}


def to_fen(pos):
    rows = []
    for r in range(7, -1, -1):
        s, e = "", 0
        for f in range(8):
            t = pos.get((f, r))
            if t is None:
                e += 1
            else:
                if e:
                    s += str(e)
                    e = 0
                s += SYM[t[0]](t[1])
        rows.append(s + (str(e) if e else ""))
    return "/".join(rows) + " w - - 0 1"


def legal(pos):
    try:
        return chess.Board(to_fen(pos)).is_valid()
    except Exception:
        return False


def feats(pos):
    x = np.zeros(768 + 12 + 16 + 2)
    mat = 0
    pf = {}
    for (f, r), t in pos.items():
        k = KINDS.index(t)
        x[k * 64 + r * 8 + f] = 1
        x[768 + k] += 1
        if t[1] == "P":
            x[780 + (0 if t[0] == "w" else 8) + f] += 1
        mat += VAL[t[1]] * (1 if t[0] == "w" else -1)
    x[796], x[797] = mat / 10.0, len(pos) / 32.0
    return x


def shuffled(pos, rng):
    toks = list(pos.values())
    while True:
        sqs = rng.sample([(f, r) for f in range(8) for r in range(8)], len(toks))
        q = dict(zip(sqs, toks))
        if not any(t[1] == "P" and s[1] in (0, 7) for s, t in q.items()):
            return q


def random_play(rng):
    b = chess.Board()
    for _ in range(rng.randint(20, 70)):
        mvs = list(b.legal_moves)
        if not mvs:
            break
        b.push(rng.choice(mvs))
    pos = {(chess.square_file(s), chess.square_rank(s)): ("w" if p.color else "b") + p.symbol().upper() for s, p in b.piece_map().items()}
    return pos


def damaged(pos, rng):
    q = dict(pos)
    if rng.random() < 0.5:   # drop a king
        q = {s: t for s, t in q.items() if t != rng.choice(["wK", "bK"])} if False else q
        ks = [s for s, t in q.items() if t[1] == "K"]
        del q[rng.choice(ks)]
    else:                    # add a second king of one colour
        free = [(f, r) for f in range(8) for r in range(8) if (f, r) not in q]
        q[rng.choice(free)] = rng.choice(["wK", "bK"])
    return q


def real(i):
    return C.parse_fen(LINES[i])


class Judge:
    def __init__(self, path=OUT):
        d = json.load(open(path, encoding="utf-8"))
        self.w, self.b, self.mu, self.sd, self.tau = np.array(d["w"]), d["b"], np.array(d["mu"]), np.array(d["sd"]), d["tau"]

    def score(self, pos):
        z = ((feats(pos) - self.mu) / self.sd) @ self.w + self.b
        return 1 / (1 + np.exp(-z))

    def classify(self, pos):
        if not legal(pos):
            return "non"
        return "real" if self.score(pos) >= self.tau else "legal"


def fit(X, y, lam=1e-2, iters=400, lr=0.5):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = X.T @ (p - y) / len(y) + lam * w
        w -= lr * g
        b -= lr * float(np.mean(p - y))
    return w, b


def main():
    rng = random.Random(220)
    tr = [real(i) for i in range(19459, 25459)]
    neg = []
    for p in tr:
        while True:
            q = shuffled(p, rng) if rng.random() < 0.5 else random_play(rng)
            if legal(q):
                neg.append(q)
                break
    X = np.array([feats(p) for p in tr + neg])
    y = np.array([1.0] * len(tr) + [0.0] * len(neg))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w, b = fit((X - mu) / sd, y)
    J = Judge.__new__(Judge)
    J.w, J.b, J.mu, J.sd = w, b, mu, sd
    val = sorted(float(J.score(real(i))) for i in range(26459, 27459))
    J.tau = val[int(0.05 * len(val))]
    json.dump({"w": w.tolist(), "b": b, "mu": mu.tolist(), "sd": sd.tolist(), "tau": J.tau}, open(OUT, "w"))
    rng = random.Random(2200)
    ctrl = {}

    def tally(name, ps):
        c = {"real": 0, "legal": 0, "non": 0}
        for p in ps:
            c[J.classify(p)] += 1
        ctrl[name] = {k: round(v / len(ps), 3) for k, v in c.items()} | {"n": len(ps)}
        print(name, ctrl[name], flush=True)

    held = [real(i) for i in range(25459, 26459)]
    tally("real held-out", held)
    tally("real validation (sets TAU)", [real(i) for i in range(26459, 27459)])
    sl = []
    for p in held[:500]:
        while True:
            q = shuffled(p, rng)
            if legal(q):
                sl.append(q)
                break
    tally("random legal: pieces shuffled", sl)
    tally("random legal: random play", [q for q in (random_play(rng) for _ in range(500))])
    tally("random illegal: pieces on random squares", [q for q in (shuffled(p, rng) for p in held[:2000]) if not legal(q)][:500])
    tally("illegal: real with king added / removed", [q for q in (damaged(p, rng) for p in held[:500])])
    tr_s = [float(J.score(p)) for p in tr[:500]]
    ctrl["tau"] = J.tau
    ctrl["train real score mean"] = float(np.mean(tr_s))
    ctrl["train acc"] = float(np.mean(((X @ np.zeros(1)[0:0].sum() if False else ((X - mu) / sd) @ w + b) > 0) == (y > 0.5)))
    d = json.load(open(OUT, encoding="utf-8"))
    d["controls"] = ctrl
    json.dump(d, open(OUT, "w"))
    print(json.dumps(ctrl))


if __name__ == "__main__":
    main()
