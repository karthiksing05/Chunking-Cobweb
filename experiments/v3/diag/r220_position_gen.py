"""Round 220: TRELLIS chess POSITION regimes. The level of abstraction in the existing concept tree is the only dial.

NO new generation machinery. The sampler runs forward the distributions chess_compress_gen.py already uses to code a position by
the choices generation makes (its Model class, docstring "root / anchor / parts / copy / absolute / escape"):
  top unit      draw from the TOP alphabet at level kc: a stop item (n positions), a single piece-square (single_tops counts), or a
                chunk's content concept kc levels above its leaf (share of stored top chunks under the concept)
  anchor        the chunk's anchor square drawn from the anchor squares of the stored top chunks under the concept (renormalised to
                where the shape fits the board, as the code's `fits`)
  body          copy by offsets: a stored instance (pieces at offsets from the anchor) drawn from the stored top chunks under the concept
  abs0 (arm)    the code's absolute copy route at kc=0: a stored instance on its own squares, anchor free
  list (arm)    the code's escape route: a piece-square list, pieces drawn from the training piece-square frequencies (raw_use)
  occupied squares are excluded (the code's context head leaves them out): an unit that does not fit is redrawn up to 30 times, else
  dropped (count reported). Representation unchanged: the cs73L learner (GraphTrellis, chess_flat, chess_join any, max_size 4).
Dial kc in KCS (0 = the leaf concept, 99 = the root): which ancestor concept the unit's content, anchor and body are drawn from.
Memory learns lines [0:19459] of data/chess/cs1973/train_long.fen shuffled by the seed (disjoint from the judge's real positions).
Usage (compute node, repo root): r220_position_gen.py <seed> <N list> <n samples> <tag>
"""
import bisect
import chess
import itertools
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402
import r220_judge as J  # noqa: E402

seed = int(sys.argv[1])
NS = [int(x) for x in sys.argv[2].split(",")]
NS_ = int(sys.argv[3])
tag = sys.argv[4]
KCS = [0, 1, 2, 3, 4, 6, 99]
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
KINDS = sorted(C.TOKENS)
OUTF = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes", f"r220_gen_{tag}_s{seed}.json")
judge = J.Judge()
rng = random.Random(seed)

lines = J.LINES[:19459]
random.Random(seed).shuffle(lines)
L = GraphTrellis(max_size=4, seed=seed, **CFG)
singles, chunks, raw_use, sizes, boards = Counter(), [], Counter(), [], set()   # chunks: (rid, anchor, rel, abs)


def learn_one(f):
    pos = C.parse_fen(f)
    boards.add(frozenset(pos.items()))
    raw_use.update((t, sq) for sq, t in pos.items())
    sizes.append(len(pos))
    r = L.learn(pos)
    sqs, toks = r["exp"].data[1], r["exp"].tokens
    n = 0
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            singles[(e.token, sqs[i])] += 1
        elif e.rid is not None:
            a = sqs[min(e.span)]
            chunks.append((e.rid, a, frozenset((toks[i], sqs[i][0] - a[0], sqs[i][1] - a[1]) for i in e.span),
                           frozenset((sqs[i], toks[i]) for i in e.span)))
        n += 1
    return n


def wchoice(items, cum):
    return items[bisect.bisect(cum, rng.random() * cum[-1])]


class Level:
    """The distributions at content level kc (the code's Model, restricted to what generation samples)."""

    def __init__(self, kc, paths):
        self.kc = kc
        self.top, self.anc, self.rel, self.abs = Counter(), {}, {}, {}
        for rid, a, rel, ab in chunks:
            if rid not in paths:
                continue
            p = paths[rid]
            cid = p[max(0, len(p) - 1 - kc)].id
            self.top[cid] += 1
            self.anc.setdefault(cid, Counter())[a] += 1
            self.rel.setdefault(cid, Counter())[rel] += 1
            self.abs.setdefault(cid, Counter())[ab] += 1
        self.units = [("stop", None)] + [("s", k) for k in singles] + [("c", c) for c in self.top]
        w = [len(sizes)] + list(singles.values()) + list(self.top.values())
        self.cum = list(itertools.accumulate(w))
        self.cache = {}

    def cum_of(self, cid, which):
        k = (cid, which)
        if k not in self.cache:
            d = getattr(self, which)[cid]
            items = list(d)
            self.cache[k] = (items, list(itertools.accumulate(d[i] for i in items)))
        return self.cache[k]

    def sample(self, absolute=False):
        board, drop = {}, 0
        for _ in range(60):
            kind, v = wchoice(self.units, self.cum)
            if kind == "stop":
                break
            ok = False
            for _ in range(30):
                if kind == "s":
                    pieces = {v[1]: v[0]}
                elif absolute:
                    its, cu = self.cum_of(v, "abs")
                    pieces = {sq: t for sq, t in wchoice(its, cu)}
                else:
                    its, cu = self.cum_of(v, "anc")
                    a = wchoice(its, cu)
                    its, cu = self.cum_of(v, "rel")
                    pieces = {(a[0] + dx, a[1] + dy): t for t, dx, dy in wchoice(its, cu)}
                if all(0 <= x < 8 and 0 <= y < 8 and (x, y) not in board for x, y in pieces):
                    board.update(pieces)
                    ok = True
                    break
            drop += not ok
        return board, drop


def sample_list():
    items = list(raw_use)
    cum = list(itertools.accumulate(raw_use[i] for i in items))
    n, board = rng.choice(sizes), {}
    for _ in range(n):
        for _ in range(30):
            t, sq = wchoice(items, cum)
            if sq not in board:
                board[sq] = t
                break
    return board, 0


def run(name, fn, N, res):
    causes = Counter()
    c, ex, drops, kings, npc, copies, uniq = Counter(), {"real": [], "legal": [], "non": []}, 0, 0, 0, 0, set()
    for _ in range(NS_):
        b, d = fn()
        r = judge.classify(b) if b else "non"
        c[r] += 1
        if r == "non" and b:                     # why illegal: python-chess status flags
            st = chess.Board(J.to_fen(b)).status()
            for fl in chess.Status:
                if fl and st & fl and fl != chess.STATUS_VALID:
                    causes[fl.name] += 1
        drops += d
        npc += len(b)
        copies += frozenset(b.items()) in boards
        uniq.add(frozenset(b.items()))
        if len(ex[r]) < 4:
            ex[r].append(J.to_fen(b).split()[0])
    row = {k: round(c[k] / NS_, 3) for k in ("real", "legal", "non")}
    row |= {"drop_per_board": round(drops / NS_, 3), "mean_pieces": round(npc / NS_, 1), "exact_train_copy": round(copies / NS_, 3),
            "unique": round(len(uniq) / NS_, 3), "illegal_causes": {k: round(v / NS_, 3) for k, v in causes.items()}, "examples": ex}
    res[f"N{N}"][name] = row
    print(json.dumps({"N": N, "arm": name, **{k: v for k, v in row.items() if k != "examples"}}), flush=True)


res = {"seed": seed, "NS": NS, "n": NS_, "tau": judge.tau, "kcs": KCS}
done, t0 = 0, time.time()
for N in NS:
    for f in lines[done:N]:
        learn_one(f)
    done = N
    L._quiet = True
    paths = {}
    for rid in {c[0] for c in chunks}:
        try:
            paths[rid] = L.cnt.categorize(L.records[rid].cnt)
        except Exception:
            pass
    depth = Counter(len(p) for p in paths.values())
    res[f"N{N}"] = {"learned": N, "learn_s": round(time.time() - t0), "chunk units": len(chunks), "path depth": dict(depth)}
    print(json.dumps({"N": N, "learned": N, "chunks": len(chunks), "s": round(time.time() - t0), "depth": dict(depth)}), flush=True)
    for kc in KCS:
        lv = Level(kc, paths)
        run(f"kc={kc}", lv.sample, N, res)
        if kc == 0:
            run("abs0", lambda: lv.sample(absolute=True), N, res)
    run("list", sample_list, N, res)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done", round(time.time() - t0), flush=True)
