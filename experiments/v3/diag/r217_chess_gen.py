"""Round 217: TRELLIS chess move choice under its existing dials, judged into regimes by r217_chess_judge.py.

TRELLIS scores positions, not moves: the existing player (chess_play.py) plays the legal move after which memory recognises the
position best (summed recognition bits of the parsed top units, content hierarchy + context hierarchy). Here every candidate
move m in position P is scored by the bits of the position after m, split into its content part (c) and context part (x).
Arms are then different readings of the same scores (the dials):
  training size   memory learned from the first N positions of the cs73L stream (seed 13): N in NS
  read            argmax of c + x (the recipe), argmax c (content hierarchy only), argmax x (context hierarchy only)
  temperature     softmax(bits / T) over candidates, T in TEMPS bits (T = inf is a uniform draw = random legal control)
  inverse         argmin of c + x
  safe            argmax among the legal moves that do not lose material to the best 1-ply reply (chess_play's +safe)
  candidate set   "legal" (python-chess legal moves, as chess_play) or "free": legal moves plus NFREE random piece displacements
                  (a piece of the side to move to any square not holding its own piece), i.e. nothing tells the model the rules
Config = the r213 / cs73L recipe (chess_flat, chess_join any), max_size 4, one seed.

Usage (compute node, repo root): r217_chess_gen.py <seed> <N list> <positions> <nfree> <tag>
"""
import json
import math
import os
import random
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402
import r217_chess_judge as J  # noqa: E402

seed = int(sys.argv[1])
NS = [int(x) for x in sys.argv[2].split(",")]
NPOS = int(sys.argv[3])
NFREE = int(sys.argv[4])
tag = sys.argv[5]
TEMPS = [2.0, 8.0]
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
rng = random.Random(seed)


def position(board):
    return {(chess.square_file(sq), chess.square_rank(sq)): ("w" if p.color else "b") + p.symbol().upper()
            for sq, p in board.piece_map().items()}


def after_pos(pos, mv):
    q = dict(pos)
    q[(chess.square_file(mv.to_square), chess.square_rank(mv.to_square))] = q.pop(
        (chess.square_file(mv.from_square), chess.square_rank(mv.from_square)))
    return q


def read_bits(L, pos):
    r = L.parse(pos)
    c = x = 0.0
    for e in r["top"]:
        if e.token is None:
            c += L._climb(L.cnt, L._content(e))
            x += L._climb(L.ctx, e.ctx)
    return c, x


def free_moves(board, k):
    own = [s for s, p in board.piece_map().items() if p.color == board.turn]
    out, seen = [], set()
    while len(out) < k:
        f, t = rng.choice(own), rng.randrange(64)
        piece_t = board.piece_at(t)
        if t == f or (piece_t and piece_t.color == board.turn) or (f, t) in seen:
            continue
        seen.add((f, t))
        out.append(chess.Move(f, t))
    return out


def safe_ok(board, mv):
    me = board.turn
    b = board.copy(stack=False)
    b.push(mv)
    base = J.material(b, me)
    loss = 0
    for r in list(b.legal_moves):
        b.push(r)
        loss = max(loss, base - J.material(b, me))
        b.pop()
    return loss <= 0


def pick(cands, key):
    best = max(cands, key=lambda c: (key(c), rng.random()))
    return best


def softmax_pick(cands, key, T):
    ks = [key(c) for c in cands]
    m = max(ks)
    w = [math.exp((k - m) / T) for k in ks]
    return rng.choices(cands, weights=w)[0]


rows = J.load_testset()[:NPOS]
train = [C.parse_fen(l.strip()) for l in open(os.path.join(ROOT, "data", "chess", "cs1973", "train_long.fen"), encoding="utf-8") if l.strip()]
random.Random(seed).shuffle(train)
L = GraphTrellis(max_size=4, seed=seed, **CFG)
done, t0 = 0, time.time()
res = {"seed": seed, "NS": NS, "positions": len(rows), "nfree": NFREE, "arms": {}}
vals_cache = {}


def record(arm, i, b, c):
    r = rows[i]
    if i not in vals_cache:
        vals_cache[i] = J.values(b)
    j = J.judge(b, c["uci"], vals_cache[i])
    res["arms"].setdefault(arm, []).append({"i": i, "fen": r["fen"], "move": c["uci"], "real": r["move"], **j})


for N in NS:
    for p in train[done:N]:
        L.learn(p)
    done = N
    L.samples = 0
    L._quiet = True
    print(json.dumps({"N": N, "learned": done, "s": round(time.time() - t0)}), flush=True)
    for i, r in enumerate(rows):
        b = chess.Board(r["fen"])
        pos = position(b)
        legal = list(b.legal_moves)
        free = free_moves(b, NFREE) if N == NS[-1] else []
        cands = []
        for mv in legal + free:
            a = b.copy(stack=False)
            is_legal = mv in b.legal_moves
            if is_legal:
                a.push(mv)
                q = position(a)
            else:
                q = after_pos(pos, mv)
            if is_legal and a.is_checkmate():
                c = x = 1e9 / 2
            else:
                c, x = read_bits(L, q)
            cands.append({"uci": mv.uci(), "c": c, "x": x, "legal": is_legal})
        leg = [c for c in cands if c["legal"]]
        tot = lambda c: c["c"] + c["x"]  # noqa: E731
        sets = {"legal": leg}
        if free:
            sets["free"] = cands
        for sname, cs in sets.items():
            pre = f"N{N}/{sname}/"
            record(pre + "argmax c+x", i, b, pick(cs, tot))
            if sname == "legal" or True:
                record(pre + "argmax c", i, b, pick(cs, lambda c: c["c"]))
                record(pre + "argmax x", i, b, pick(cs, lambda c: c["x"]))
                record(pre + "argmin c+x", i, b, pick(cs, lambda c: -tot(c)))
                for T in TEMPS:
                    record(pre + f"softmax T={T:g}", i, b, softmax_pick(cs, tot, T))
                record(pre + "uniform", i, b, rng.choice(cs))
            if sname == "legal":
                ok = [c for c in cs if safe_ok(b, chess.Move.from_uci(c["uci"]))] or cs
                record(pre + "argmax c+x, safe", i, b, pick(ok, tot))
        if i % 25 == 0:
            print(json.dumps({"N": N, "pos": i, "s": round(time.time() - t0)}), flush=True)
    os.makedirs(J.OUT, exist_ok=True)
    with open(os.path.join(J.OUT, f"r217_chess_gen_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh)
summ = {}
for arm, l in res["arms"].items():
    n = len(l)
    summ[arm] = {k: round(sum(1 for x in l if x["regime"] == k) / n, 3) for k in ("R-sem", "R-legal", "R-illegal")}
    summ[arm]["real-move match"] = round(sum(1 for x in l if x["move"] == x["real"]) / n, 3)
print(json.dumps(summ, indent=1))
print("done", round(time.time() - t0))
