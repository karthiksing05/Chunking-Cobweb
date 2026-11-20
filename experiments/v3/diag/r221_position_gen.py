"""Round 221: chess position regimes with (1) a whole-board count facet and (2) the context-tree level kx as a second dial,
on top of r220's content dial kc. Judge unchanged (r220_judge.py). Learner unchanged (cs73L recipe).

(1) count facet. Each learned position stores its whole-board piece counts (12 piece kinds, kings included) as one more stored
    attribute of the position, learned by counting exactly as TOP counts and single_tops are (the "read counts" of the old v2
    grammar, FRAMEWORK.md: how many of each kind the board holds). Generation draws the board's count profile from that stored
    distribution (the distribution over the stored positions' profiles), then draws units as before, renormalised over the units
    whose pieces the profile still has (the code's existing renormalisation over what stays feasible, as for gaps that keep a unit on
    the board); the stop item is replaced by the profile being used up. counts=0 is r220's route.
(2) kx dial. The code's context head (CTXR in chess_compress_gen.py): a unit's anchor is drawn from the anchor squares of the stored
    top chunks (occupied squares left out); the star seen from the anchor over the units decoded so far (rays, jumps, square;
    no own-content attribute) is sorted in the context tree L.ctx; the unit's content is drawn by FILL at the concept kx levels
    above the sorted leaf: a chunk's content concept at level kc by the share of FILL chunks under it, then a stored instance
    under that concept that fits at the anchor; a single piece by FILL's piece fillers at the concept (its square from the
    single-piece squares). kx=None is r220's concept-first route (no context).
Usage (compute node, repo root): r221_position_gen.py <seed> <N list> <n samples> <tag>
"""
import bisect
import itertools
import json
import os
import random
import sys
import time
from collections import Counter

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from trellis2 import chess as C  # noqa: E402
from trellis2.trellis3 import FILL  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402
import r220_judge as J  # noqa: E402

seed = int(sys.argv[1])
NS = [int(x) for x in sys.argv[2].split(",")]
NS_ = int(sys.argv[3])
tag = sys.argv[4]
KCS, KXS = [0, 2, 99], [None, 0, 2, 99]
CFG = json.loads('{"store":"best","naming":"bands","restructure":"sample","type_names":"sample","type_key":"concept","children":false,'
                 '"spine":false,"heights":false,"tree":"cu","refresh":"verify","refresh_budget":300,"chess_flat":true,"chess_join":"any"}')
KINDS = sorted(C.TOKENS)
OUTF = os.path.join(ROOT, "experiments", "v3", "results", "chess_regimes", f"r221_gen_{tag}_s{seed}.json")
judge = J.Judge()
rng = random.Random(seed)
lines = J.LINES[:19459]
random.Random(seed).shuffle(lines)
L = GraphTrellis(max_size=4, seed=seed, **CFG)
singles, chunks, raw_use, sizes, boards, profiles, sq_single, anchor_all = Counter(), [], Counter(), [], set(), Counter(), Counter(), Counter()


def learn_one(f):
    pos = C.parse_fen(f)
    boards.add(frozenset(pos.items()))
    raw_use.update((t, sq) for sq, t in pos.items())
    sizes.append(len(pos))
    cnt = Counter(pos.values())
    profiles[tuple(cnt.get(k, 0) for k in KINDS)] += 1            # the count facet
    r = L.learn(pos)
    sqs, toks = r["exp"].data[1], r["exp"].tokens
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            singles[(e.token, sqs[i])] += 1
            sq_single[sqs[i]] += 1
        elif e.rid is not None:
            a = sqs[min(e.span)]
            anchor_all[a] += 1
            chunks.append((e.rid, a, frozenset((toks[i], sqs[i][0] - a[0], sqs[i][1] - a[1]) for i in e.span)))


def wchoice(items, cum):
    return items[bisect.bisect(cum, rng.random() * cum[-1])]


def wdraw(d):
    items = list(d)
    return wchoice(items, list(itertools.accumulate(d[i] for i in items))) if items else None


class Level:
    def __init__(self, kc, paths):
        self.kc = kc
        self.top, self.anc, self.rel = Counter(), {}, {}
        self.cpath = {}
        for rid, a, rel in chunks:
            if rid not in paths:
                continue
            p = paths[rid]
            cid = p[max(0, len(p) - 1 - kc)].id
            self.cpath[rid] = cid
            self.top[cid] += 1
            self.anc.setdefault(cid, Counter())[a] += 1
            self.rel.setdefault(cid, Counter())[rel] += 1
        self.units_all = [("stop", None)] + [("s", k) for k in singles] + [("c", c) for c in self.top]
        self.cum_all = list(itertools.accumulate([len(sizes)] + list(singles.values()) + list(self.top.values())))
        self.units_nostop = self.units_all[1:]
        self.cum_nostop = self.cum_all[1:]
        self.cum_nostop = [c - self.cum_all[0] for c in self.cum_nostop]
        self.cc = {}
        self.fills = {}

    def cum_of(self, cid, which):
        k = (cid, which)
        if k not in self.cc:
            d = getattr(self, which)[cid]
            items = list(d)
            self.cc[k] = (items, list(itertools.accumulate(d[i] for i in items)))
        return self.cc[k]

    def node_fills(self, node):
        """FILL at a context concept: (piece fillers, chunk fillers aggregated to the content concept at level kc)."""
        if node.id not in self.fills:
            w, c = Counter(), Counter()
            for f, n in L.ctx.passive(node, FILL).items():
                if f[0] == "w":
                    w[f[1]] += n
                elif f[1] in self.cpath:
                    c[self.cpath[f[1]]] += n
            self.fills[node.id] = (w, c)
        return self.fills[node.id]


def ctx_desc(board, sq):
    st = C.star(board, sq)
    return {"rays": {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.RAYS}, "jumps": {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.JUMPS},
            "sq": C.square_name(sq)}


def free_sq(board, d):
    return Counter({s: n for s, n in d.items() if s not in board})


def sample(lv, kx, counts):
    board, drop = {}, 0
    remaining = Counter(dict(zip(KINDS, wdraw(profiles)))) if counts else None

    def feasible(pieces):
        if remaining is None:
            return True
        need = Counter(pieces.values())
        return all(remaining[t] >= n for t, n in need.items())

    def place(pieces):
        board.update(pieces)
        if remaining is not None:
            remaining.subtract(Counter(pieces.values()))

    fails = 0
    for _ in range(80):
        if remaining is not None and sum(remaining.values()) <= 0:
            break
        if fails >= 12:                                              # 12 unit draws in a row did not fit: the rest of the profile stays unplaced
            break
        kind, v = wchoice(lv.units_nostop, lv.cum_nostop) if counts else wchoice(lv.units_all, lv.cum_all)
        if kind == "stop":
            break
        ok = False
        if kx is None:                                               # r220: concept-first
            for _ in range(30):
                if kind == "s":
                    pieces = {v[1]: v[0]}
                else:
                    its, cu = lv.cum_of(v, "anc")
                    a = wchoice(its, cu)
                    its, cu = lv.cum_of(v, "rel")
                    pieces = {(a[0] + dx, a[1] + dy): t for t, dx, dy in wchoice(its, cu)}
                if all(0 <= x < 8 and 0 <= y < 8 and (x, y) not in board for x, y in pieces) and feasible(pieces):
                    place(pieces)
                    ok = True
                    break
        else:                                                        # context head
            for _ in range(10):
                fa = free_sq(board, sq_single if kind == "s" else anchor_all)
                a = wdraw(fa)
                if a is None:
                    break
                path = L.ctx.categorize(ctx_desc(board, a))
                node = path[max(0, len(path) - 1 - kx)]
                w, c = lv.node_fills(node)
                if kind == "s":
                    ww = Counter({t: n for t, n in w.items() if remaining is None or remaining[t] > 0})
                    t = wdraw(ww)
                    if t is not None:
                        place({a: t})
                        ok = True
                        break
                else:
                    for _ in range(10):
                        cid = wdraw(c)
                        if cid is None:
                            break
                        its, cu = lv.cum_of(cid, "rel")         # a stored instance under the concept, redrawn until it fits at the anchor
                        for _ in range(25):
                            rel = wchoice(its, cu)
                            pieces = {(a[0] + dx, a[1] + dy): t for t, dx, dy in rel}
                            if all(0 <= x < 8 and 0 <= y < 8 and (x, y) not in board for x, y in pieces) and feasible(pieces):
                                place(pieces)
                                ok = True
                                break
                        if ok:
                            break
                    if ok:
                        break
        drop += not ok
        fails = 0 if ok else fails + 1
    return board, drop


def sample_list(counts):
    items = list(raw_use)
    cum = list(itertools.accumulate(raw_use[i] for i in items))
    board = {}
    if counts:                       # the stored profile's pieces, squares from the piece-square frequencies of their kind
        prof = dict(zip(KINDS, wdraw(profiles)))
        for t, n in prof.items():
            for _ in range(n):
                for _ in range(30):
                    tt, sq = wchoice(items, cum)
                    if tt == t and sq not in board:
                        board[sq] = t
                        break
        return board, 0
    for _ in range(rng.choice(sizes)):
        for _ in range(30):
            t, sq = wchoice(items, cum)
            if sq not in board:
                board[sq] = t
                break
    return board, 0


def run(name, fn, N, res):
    causes, c, drops, npc, copies, uniq = Counter(), Counter(), 0, 0, 0, set()
    ex, keep = {"real": [], "legal": [], "non": []}, {"real": [], "legal": []}
    for _ in range(NS_):
        b, d = fn()
        r = judge.classify(b) if b else "non"
        c[r] += 1
        if r == "non" and b:
            st = chess.Board(J.to_fen(b)).status()
            for fl in chess.Status:
                if fl and st & fl and fl != chess.STATUS_VALID:
                    causes[fl.name] += 1
        drops += d
        npc += len(b)
        copies += frozenset(b.items()) in boards
        uniq.add(frozenset(b.items()))
        fen = J.to_fen(b).split()[0] if b else ""
        if len(ex[r]) < 4:
            ex[r].append(fen)
        if r in keep and len(keep[r]) < 100:
            keep[r].append(fen)
    row = {k: round(c[k] / NS_, 3) for k in ("real", "legal", "non")}
    row |= {"drop_per_board": round(drops / NS_, 3), "mean_pieces": round(npc / NS_, 1), "exact_train_copy": round(copies / NS_, 3),
            "unique": round(len(uniq) / NS_, 3), "illegal_causes": {k: round(v / NS_, 3) for k, v in causes.items()},
            "examples": ex, "kept": keep}
    res[f"N{N}"][name] = row
    print(json.dumps({"N": N, "arm": name, **{k: v for k, v in row.items() if k not in ("examples", "kept", "illegal_causes")}}), flush=True)


res = {"seed": seed, "NS": NS, "n": NS_, "tau": judge.tau, "kcs": KCS, "kxs": KXS}
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
    res[f"N{N}"] = {"learned": N, "learn_s": round(time.time() - t0), "chunk units": len(chunks), "profiles": len(profiles)}
    print(json.dumps({"N": N, "learned": N, "chunks": len(chunks), "profiles": len(profiles), "s": round(time.time() - t0)}), flush=True)
    for counts in (0, 1):
        for kc in KCS:
            lv = Level(kc, paths)
            for kx in KXS:
                run(f"counts={counts} kc={kc} kx={kx}", lambda: sample(lv, kx, counts), N, res)
        run(f"counts={counts} list", lambda: sample_list(counts), N, res)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done", round(time.time() - t0), flush=True)
