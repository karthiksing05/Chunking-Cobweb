"""Chase & Simon (1973), "Perception in chess": an exact replication with TRELLIS
(GraphTrellis) memories, recall as generation. The spec is docs/CS1973_PROTOCOL.md.

Stimuli (protocol section 2). Per simulated subject: 10 middle-game positions (after
White's 21st move, 24-26 pieces) and 10 end-game positions (after White's 41st,
12-15 pieces) drawn from the cached Lichess 1800+ pools (data/chess/cs1973), and
4 random middle + 4 random end positions (the pieces of 4 + 4 further real
positions put on random squares, no pawn on rank 1 or 8): 28 positions. Each task
gets 5 middle + 5 end + 2 random middle + 2 random end (the 10 + 10 game positions
are split 5/5 between the copy and the memory task; protocol 2 notes the split is
an inference). ``reps`` subjects per seed each draw their own 28 positions; every
number is reported with the paper's per-position SE (mean over subjects) and the
SE over subjects.

Players (section 1): one memory learning the training positions in order, read at
a learning curve (0, 100, 300, 1000, 2000, 4000, 8000 positions; the cached
train.fen holds 4,660, so the last point is clipped to all of them). The headline
B / A / M are 300 / 1000 / 4000 (not tuned to the human numbers).

Labels. A 5-s view (memory task) or a glance (copy task) is one parse of the
position with learning off (L.parse, L._quiet). Its units are found as in
chess_inspect.py: each top element is walked down until it is a piece or is
recognised (content + context recognition bits >= theta, default 16). A unit's
label is its content concept and its context concept: on each hierarchy the
climbing ancestor that recognition is measured at (Trellis3._climb: the deepest
concept on the element's sorting path holding >= ``support`` instances), found by
sorting the element without learning. A piece's label is its kind (its content)
and its context concept.

Decoding a label (recall as generation; the choice the brief left open). A label
names two concepts; what memory holds under both is its extension, and the label
decodes to the extension's most frequent stored instance (pieces on squares):
  * identical chunk: a committed chunk with exactly the unit's pieces on exactly
    its squares exists in memory -> it decodes perfectly (the unit's own pieces);
  * composite: the committed chunks stored under the content concept whose context
    leaf lies under the context concept; their most frequent instance (absolute
    pieces on squares) is placed. When none lies under the context concept, the
    context concept backs off to its parent, up to the root (whose extension is
    every chunk of that content concept: "whatever the root predicts");
  * piece: the stored pieces of that kind whose context leaf lies under the
    context concept, backing off the same way; it decodes to their most frequent
    square.
The mode of stored instances is the most probable instance of a concept that the
API supports without a decoder (CU trees hold bag counts, from which an
instance's size cannot be read back; records keep their instances exactly).
Untrained memory has no records, so every label decodes to nothing (recall 0).

Memory task (section 4). Each trial is a 5-s view, then recall. STM takes K = 7
labels (C&S's 7 +/- 2; K = 4 reported as a sensitivity row), largest unit first.
The paper clears the board between trials, but the subject's knowledge carries
over; modelled as: the reconstruction after a trial is the union of every label
decoded so far (a later placement overrides an earlier one on its square), and each
trial's K labels prefer units (i) not yet tried and failed, (ii) with the most
pieces not correctly recalled in the previous trial's reconstruction, (iii) largest,
(iv) earliest anchor. A composite unit whose decoding misses any of its own pieces
is replaced by its parts (each walked down to recognised units again) for later
trials; a piece that fails stays (it is retried last). Game positions run until
perfect, cap 15 trials (B took up to 14); random positions run 7 trials (not to
criterion). Score = pieces on the exact square, with errors of commission (placed
pieces wrong in kind or square) and omissions.

Chunks (Table 8; section 6.3): on trial 1 each decoded label that places a piece
is one chunk; reported: average size (pieces placed) of chunks 1..8 in recall
order and chunks per trial. Relations (Tables 4-7; section 5): C&S's A/D/P/C/S
codes (16 combinations, D implies C) between SUCCESSIVELY placed pieces: pieces of
a chunk in the left-to-right leaf order of the learner's own structure (the stored
record's tree for a decoded chunk, the unit's for an identical one), chunks in
recall order; within-chunk = successive pieces of one chunk, between-chunk = last
piece of a chunk and first of the next; coded on the board of placed pieces.
A priori = all pairs of pieces of the subject's 20 game (8 random) positions.

Copy task (section 3): the stimulus stays visible. Each glance = the parse (one per
position) gives ONE label, the largest unit with a piece not yet correctly placed;
a composite label is decoded from memory and placed; a single piece is placed on
the square it is seen on (the board is visible). A composite that misses one of its
pieces is replaced by its parts, so a wrong piece is fixed by a later glance taking
it alone. Glances continue until every piece is correct. Reported: pieces per
glance (paper, middle games: M 2.0, A 2.8, B 2.0), glances per position and the
within- vs between-glance relations. The model has no timing: no latencies, and
chunk boundaries are labels (memory) or glances (copy), not 2-s pauses.

Ideas (p. 79): every multi-piece chunk recalled on trial 1 (and copied per glance)
is labelled with chess_ideas.py's ideas() (copied here) plus "same-colour cluster"
(all its pieces one colour); C&S: > 75% of M's chunks are pawn chains, castled-king
structures or same-colour clusters.

Writes experiments/v3/results/chess_cs1973_exact/<tag>_s<seed>.json (after every
curve point) and prints a human-vs-model table.

Usage: chess_cs1973_exact.py <cfg json> <seed> <tag> [curve e.g. 0,100,300] [reps] [theta]
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
curve = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,100,300,1000,2000,4000,8000").split(",")]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
THETA = float(sys.argv[6]) if len(sys.argv) > 6 else 16.0
max_size = cfg.pop("max_size", 4)
HEADLINE = {"B": 300, "A": 1000, "M": 4000}
KS = (7, 4)
CAP_GAME, CAP_RANDOM = 15, 7
D = os.path.join(ROOT, "data", "chess", "cs1973")
TYPES = ("middle", "end", "random middle", "random end")
CODES = ["-", "A", "P", "C", "S", "AP", "AS", "DC", "PC", "PS", "CS", "APS", "DPC", "DCS", "PCS", "DPCS"]
OUT_DIR = os.path.join(ROOT, "experiments", "v3", "results", "chess_cs1973_exact")


def load(name):
    with open(os.path.join(D, f"{name}.fen"), encoding="utf-8") as fh:
        return [C.parse_fen(l.strip()) for l in fh if l.strip()]


train = load("train")
random.Random(seed).shuffle(train)
curve = sorted({min(c, len(train)) for c in curve})
pools = {"middle": load("middle"), "end": load("end")}


def randomize(pos, rng):
    """C&S p. 58: an actual position's pieces put on random squares (no pawn on rank 1/8)."""
    pieces = list(pos.values())
    squares = [(f, r) for f in range(8) for r in range(8)]
    while True:
        rng.shuffle(squares)
        out = dict(zip(squares, pieces))
        if all(not (p[1] == "P" and sq[1] in (0, 7)) for sq, p in out.items()):
            return out


def draw_subject(rep):
    """One subject's 28 positions and their split between the two tasks."""
    rng = random.Random(seed * 1000 + rep)
    sets = {"copy": {}, "memory": {}}
    for kind in ("middle", "end"):
        picks = rng.sample(pools[kind], 14)
        game, rnd = picks[:10], [randomize(p, rng) for p in picks[10:]]
        sets["copy"][kind], sets["memory"][kind] = game[:5], game[5:]
        sets["copy"]["random " + kind], sets["memory"]["random " + kind] = rnd[:2], rnd[2:]
    return sets


subjects = [draw_subject(rep) for rep in range(REPS)]


# -------------------------------------------------------------- relations (from chess_cs1973.py)
def relations(pos, a, b):
    """C&S's code for two pieces: A attack, D defense, P proximity, C colour, S type."""
    pa, pb = pos[a], pos[b]
    same = pa[0] == pb[0]
    hit = any(t == b and (fwd != "sees" or back) for t, fwd, back in C.interactions(pos, a))
    code = ""
    if hit and not same:
        code += "A"
    if hit and same:
        code += "D"
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1:
        code += "P"
    if same:
        code += "C"
    if pa[1] == pb[1]:
        code += "S"
    return code or "-"


# -------------------------------------------------------------- ideas (from chess_ideas.py)
HOME = {"R": {(0, 0), (7, 0)}, "N": {(1, 0), (6, 0)}, "B": {(2, 0), (5, 0)}, "Q": {(3, 0)}, "K": {(4, 0)}}
IDEAS = ["pawn chain", "pawn phalanx", "doubled pawns", "pawn ram", "king shelter", "castled king", "fianchetto",
         "pawn-guarded", "battery", "undeveloped", "attack", "same-colour cluster"]
CS_TYPES = "C&S types (pawn chain, castled king or same-colour cluster)"
IDEAS.append(CS_TYPES)


def rel(pieces, colour):
    out = []
    for (f, r), tok in pieces:
        rr = r if colour == "w" else 7 - r
        out.append((f, rr, tok[1], tok[0] == colour))
    return out


def clear(pcs, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(abs(dx), abs(dy))
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    occ = {(f, r) for f, r, _, _ in pcs}
    return all((a[0] + sx * k, a[1] + sy * k) not in occ for k in range(1, n))


def ideas(pieces):
    """chess_ideas.ideas(), plus "same-colour cluster" (all pieces one colour)."""
    found = set()
    for colour in "wb":
        pcs = rel(pieces, colour)
        own = [(f, r, k) for f, r, k, o in pcs if o]
        foe = [(f, r, k) for f, r, k, o in pcs if not o]
        pawns = {(f, r) for f, r, k in own if k == "P"}
        fpawns = {(f, r) for f, r, k in foe if k == "P"}
        for f, r in pawns:
            if (f + 1, r + 1) in pawns or (f - 1, r + 1) in pawns:
                found.add("pawn chain")
            if (f + 1, r) in pawns:
                found.add("pawn phalanx")
            if any(g == f and s != r for g, s in pawns):
                found.add("doubled pawns")
            if (f, r + 1) in fpawns:
                found.add("pawn ram")
        for f, r, k in own:
            if k == "K":
                shelter = [p for p in pawns if abs(p[0] - f) <= 1 and 1 <= p[1] - r <= 2]
                if len(shelter) >= 2:
                    found.add("king shelter")
                if r == 0 and f in (6, 7) and any(g >= 5 and s <= 1 and kk in "RP" for g, s, kk in own if kk != "K"):
                    found.add("castled king")
                if r == 0 and f in (0, 1, 2) and any(g <= 2 and s <= 1 and kk in "RP" for g, s, kk in own if kk != "K"):
                    found.add("castled king")
            if k == "B" and (f, r) in ((1, 1), (6, 1)) and (f, 2) in pawns:
                found.add("fianchetto")
            if k != "P" and ((f - 1, r - 1) in pawns or (f + 1, r - 1) in pawns):
                found.add("pawn-guarded")
        heavy = [(f, r, k) for f, r, k in own if k in "QRB"]
        for i, (f, r, k) in enumerate(heavy):
            for g, s, kk in heavy[i + 1:]:
                line = f == g or r == s
                diag = abs(f - g) == abs(r - s)
                if ((line and {k, kk} <= {"Q", "R"}) or (diag and {k, kk} == {"Q", "B"})) and clear(pcs, (f, r), (g, s)):
                    found.add("battery")
        if sum(1 for f, r, k in own if k in HOME and (f, r) in HOME[k]) >= 2:
            found.add("undeveloped")
    for (sa, ta) in pieces:
        for (sb, tb) in pieces:
            if ta[0] != tb[0] and C.hits(ta, sb[0] - sa[0], sb[1] - sa[1]):
                pcs = rel(pieces, "w")
                if ta[1] in "NKP" or clear(pcs, sa, sb):
                    found.add("attack")
    if len({t[0] for _, t in pieces}) == 1:
        found.add("same-colour cluster")
    if found & {"pawn chain", "castled king", "same-colour cluster"}:
        found.add(CS_TYPES)
    return found


# -------------------------------------------------------------- memory
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
inst = {}          # record id -> its pieces on squares, left-to-right leaves of its tree
store = set()      # committed chunks as frozensets of (square, piece)


def leaves(e):
    out, stack = [], [e]
    while stack:
        x = stack.pop()
        if x.token is not None:
            out.append(min(x.span))
        else:
            stack.extend((x.right, x.left))
    return out


def learn(p):
    r = L.learn(p)
    exp = r["exp"]
    sqs = exp.data[1]
    for e in r["words"]:
        if e.rid is not None:
            inst[e.rid] = ((sqs[min(e.span)], e.token),)
    for e in r["comps"].values():
        if e.rid is not None and e.token is None:
            pcs = tuple((sqs[i], exp.tokens[i]) for i in leaves(e))
            inst[e.rid] = pcs
            store.add(frozenset(pcs))


def euler(tree):
    """Entry/exit times of every concept (node id -> (tin, tout)), so that 'leaf under
    concept' is an interval test."""
    tin, tout, t = {}, {}, 0
    cu = getattr(tree, "tree", None)
    if cu is not None and hasattr(cu, "children_of"):
        kids = cu.children_of
        stack = [(cu.root_id, False)]
    else:
        kids = None
        stack = [(tree.root, False)]
    while stack:
        n, done = stack.pop()
        i = n if kids is not None else n.id
        if done:
            tout[i] = t
            continue
        tin[i] = t
        t += 1
        stack.append((n, True))
        stack.extend((c, False) for c in (kids(n) if kids is not None else n.children))
    return tin, tout


class Index:
    """What memory holds, for decoding: each record's place in both hierarchies."""

    def __init__(self):
        self.ctin, self.ctout = euler(L.cnt)
        self.xtin, self.xtout = euler(L.ctx)
        comps, words = [], defaultdict(list)
        for e in L.records:
            if e.rid not in inst or e.leaf is None or e.leaf.id not in self.xtin:
                continue
            xt = self.xtin[e.leaf.id]
            if e.token is not None:
                words[e.token].append((xt, inst[e.rid][0][0]))
            elif e.cleaf is not None and e.cleaf.id in self.ctin:
                comps.append((self.ctin[e.cleaf.id], xt, e.rid))
        comps.sort()
        self.comps, self.comp_keys = comps, [c[0] for c in comps]
        self.words = {k: sorted(v) for k, v in words.items()}
        self.word_keys = {k: [w[0] for w in v] for k, v in self.words.items()}
        self.cache = {}

    def _deepest(self, path, xt):
        """The deepest concept of a context path (root first) that holds ctx time xt."""
        lo, hi = 0, len(path) - 1          # path[0] (root) holds everything
        while lo < hi:
            mid = (lo + hi + 1) // 2
            i = path[mid].id
            if self.xtin.get(i, -1) <= xt <= self.xtout.get(i, -2):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def composite(self, nc, px):
        key = ("c", nc.id, tuple(n.id for n in px))
        if key in self.cache:
            return self.cache[key]
        out = (None, None)
        if nc.id in self.ctin:
            lo = bisect.bisect_left(self.comp_keys, self.ctin[nc.id])
            hi = bisect.bisect_right(self.comp_keys, self.ctout[nc.id])
            best, cnt, first = -1, Counter(), {}
            for _, xt, rid in self.comps[lo:hi]:
                d = self._deepest(px, xt)
                if d < best:
                    continue
                if d > best:
                    best, cnt, first = d, Counter(), {}
                k = frozenset(inst[rid])
                cnt[k] += 1
                first.setdefault(k, inst[rid])
            if cnt:
                k = min(cnt, key=lambda k: (-cnt[k], sorted(first[k])))
                out = (first[k], "concept" if best == len(px) - 1 else "backoff")
        self.cache[key] = out
        return out

    def piece(self, kind, px):
        key = ("w", kind, px[-1].id)
        if key in self.cache:
            return self.cache[key]
        out = (None, None)
        keys = self.word_keys.get(kind)
        if keys:
            for d in range(len(px) - 1, -1, -1):
                i = px[d].id
                if i not in self.xtin:
                    continue
                lo = bisect.bisect_left(keys, self.xtin[i])
                hi = bisect.bisect_right(keys, self.xtout[i])
                if hi > lo:
                    cnt = Counter(sq for _, sq in self.words[kind][lo:hi])
                    sq = min(cnt, key=lambda s: (-cnt[s], s))
                    out = (((sq, kind),), "concept" if d == len(px) - 1 else "backoff")
                    break
        self.cache[key] = out
        return out


def climb_path(tree, x):
    """x's sorting path, root first, up to its climbing ancestor (Trellis3._climb)."""
    path = tree.categorize(x)
    k = 0
    for j, n in enumerate(path[1:], 1):
        if n.count >= L.support:
            k = j
    return path[:k + 1]


class View:
    """One parse of a position (learning off) and its labels, decoded."""

    def __init__(self, pos, idx):
        self.pos, self.idx = pos, idx
        self.r = L.parse(pos)
        self.exp = self.r["exp"]
        self.sqs = self.exp.data[1]
        self.dec = {}
        self.units = sorted([u for top in self.r["top"] for u in self.walk(top)], key=self.order)

    def bits(self, e):
        if "climb" not in e.memo:
            L._exp = self.exp
            e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
        return e.memo["climb"]

    def walk(self, e):
        if e.token is not None or self.bits(e) >= THETA:
            return [e]
        return self.walk(e.left) + self.walk(e.right)

    def order(self, u):
        return (-len(u.span), min(u.span))

    def own(self, u):
        return [(self.sqs[i], self.exp.tokens[i]) for i in leaves(u)]

    def parts(self, u):
        return self.walk(u.left) + self.walk(u.right)

    def decode(self, u):
        """The label of unit u decoded from memory: (pieces in placement order, how)."""
        if id(u) in self.dec:
            return self.dec[id(u)]
        own = self.own(u)
        if u.token is not None:
            px = climb_path(L.ctx, u.ctx)
            pcs, how = self.idx.piece(u.token, px)
        elif frozenset(own) in store:
            pcs, how = tuple(own), "identical"
        else:
            L._exp = self.exp
            pc = climb_path(L.cnt, L._content(u))
            px = climb_path(L.ctx, u.ctx)
            pcs, how = self.idx.composite(pc[-1], px)
        out = (tuple(pcs or ()), how or "none")
        self.dec[id(u)] = out
        return out


def score(pos, board):
    ok = sum(1 for sq, p in pos.items() if board.get(sq) == p)
    return ok, sum(1 for sq, p in board.items() if pos.get(sq) != p), len(pos) - ok


def code_seq(board, chunks, within, between):
    """Relations between successively placed pieces on ``board`` (chunks of (sq, piece))."""
    def on(x):
        return board.get(x[0]) == x[1]
    prev = None
    for ch in chunks:
        for a, b in zip(ch, ch[1:]):
            if on(a) and on(b):
                within[relations(board, a[0], b[0])] += 1
        if prev is not None and ch and on(prev[-1]) and on(ch[0]):
            between[relations(board, prev[-1][0], ch[0][0])] += 1
        if ch:
            prev = ch


def memory_task(v, K, cap):
    pos, avail, fails, board = v.pos, list(v.units), Counter(), {}
    curve, comm, chunks1, hows, n_trials, reached = [], [], [], Counter(), 0, False
    for t in range(cap):
        def wrong(u):
            return sum(1 for sq, p in v.own(u) if board.get(sq) != p)
        ranked = sorted((u for u in avail if wrong(u) > 0), key=lambda u: (fails[id(u)] > 0, -wrong(u)) + v.order(u))
        take = ranked[:K]
        placed_chunks = []
        for u in take:
            pcs, how = v.decode(u)
            hows[how] += 1
            for sq, p in pcs:
                board[sq] = p
            if pcs:
                placed_chunks.append(list(pcs))
            if any(board.get(sq) != p for sq, p in v.own(u)):
                if u.token is None:
                    avail = [x for x in avail if x is not u] + v.parts(u)
                else:
                    fails[id(u)] += 1
        if t == 0:
            chunks1 = placed_chunks
            board1 = dict(board)
        ok, c, _ = score(pos, board)
        curve.append(ok)
        comm.append(c)
        n_trials = t + 1
        if ok == len(pos):
            reached = True
            break
    full = curve + [curve[-1]] * (cap - len(curve))
    return {"curve": full, "commission": comm + [comm[-1]] * (cap - len(comm)), "trials": n_trials,
            "reached": reached, "n": len(pos), "chunks": chunks1, "board1": board1, "hows": hows,
            "gain": (curve[-1] - curve[0]) / (len(curve) - 1) if len(curve) > 1 else None}


def copy_task(v, cap=400):
    pos, avail, board, glances, hows = v.pos, list(v.units), {}, [], Counter()
    while len(glances) < cap:
        def wrong(u):
            return sum(1 for sq, p in v.own(u) if board.get(sq) != p)
        cand = [u for u in avail if wrong(u) > 0]
        if not cand:
            break
        u = min(cand, key=v.order)
        if u.token is not None:
            pcs, how = tuple(v.own(u)), "seen"
        else:
            pcs, how = v.decode(u)
        hows[how] += 1
        for sq, p in pcs:
            board[sq] = p
        glances.append(list(pcs))
        if u.token is None and any(board.get(sq) != p for sq, p in v.own(u)):
            avail = [x for x in avail if x is not u] + v.parts(u)
    ok, c, _ = score(pos, board)
    return {"glances": glances, "n_glances": len(glances), "placed": sum(map(len, glances)), "correct": ok,
            "commission": c, "n": len(pos), "hows": hows, "board": board}


# -------------------------------------------------------------- statistics
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def se(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def rnd(x, k=3):
    return None if x is None else round(x, k)


def agg(per_rep):
    """per_rep: one list of per-position values per subject -> mean, per-position SE
    (mean over subjects of the SE over positions), SE over subjects (of their means)."""
    means = [mean(v) for v in per_rep]
    return {"mean": rnd(mean(means)), "se_pos": rnd(mean([se(v) for v in per_rep])), "se_rep": rnd(se(means))}


def probs(cnt):
    n = sum(cnt.values())
    return {"N": n, **{k: rnd(cnt[k] / n if n else 0.0, 4) for k in CODES}}


def corr(a, b):
    xa, xb = [a[k] for k in CODES], [b[k] for k in CODES]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    va = math.sqrt(sum((x - ma) ** 2 for x in xa))
    vb = math.sqrt(sum((x - mb) ** 2 for x in xb))
    return rnd(sum((x - ma) * (y - mb) for x, y in zip(xa, xb)) / (va * vb)) if va and vb else None


# a priori relations: all pairs of each subject's 20 game (8 random) positions
prior = {"game": Counter(), "random": Counter()}
for s in subjects:
    for task in s.values():
        for typ, ps in task.items():
            for p in ps:
                sq = sorted(p)
                for i in range(len(sq)):
                    for j in range(i + 1, len(sq)):
                        prior["random" if typ.startswith("random") else "game"][relations(p, sq[i], sq[j])] += 1
prior_p = {k: probs(v) for k, v in prior.items()}


def evaluate(n_learned):
    idx = Index()
    mem = {K: {t: defaultdict(lambda: [[] for _ in range(REPS)]) for t in TYPES} for K in KS}
    t8 = {t: [[] for _ in range(8)] for t in TYPES}
    rel_mem = {g: (Counter(), Counter()) for g in ("game", "random")}
    rel_copy = {g: (Counter(), Counter()) for g in ("game", "random")}
    idea_cnt = {task: {g: Counter() for g in ("game", "random")} for task in ("memory", "copy")}
    idea_n = {task: {g: 0 for g in ("game", "random")} for task in ("memory", "copy")}
    hows = {"memory": Counter(), "copy": Counter()}
    cp = {t: defaultdict(lambda: [[] for _ in range(REPS)]) for t in TYPES}
    examples = []
    for rep, s in enumerate(subjects):
        for typ in TYPES:
            g = "random" if typ.startswith("random") else "game"
            cap = CAP_RANDOM if g == "random" else CAP_GAME
            for p in s["memory"][typ]:
                v = View(p, idx)
                for K in KS:
                    m = memory_task(v, K, cap)
                    row = mem[K][typ]
                    row["trial 1 correct"][rep].append(m["curve"][0])
                    row["trial 1 commission"][rep].append(m["commission"][0])
                    row["trial 1 omission"][rep].append(m["n"] - m["curve"][0])
                    row["trial 1 percent"][rep].append(100.0 * m["curve"][0] / m["n"])
                    row["chunks trial 1"][rep].append(len(m["chunks"]))
                    row["multi-piece chunks trial 1"][rep].append(sum(1 for c in m["chunks"] if len(c) > 1))
                    row["mean chunk size trial 1"][rep].append(mean([len(c) for c in m["chunks"]]))
                    row["first chunk size"][rep].append(len(m["chunks"][0]) if m["chunks"] else None)
                    row["pieces added per later trial"][rep].append(m["gain"])
                    row["reached criterion"][rep].append(1.0 if m["reached"] else 0.0)
                    row["trials (to criterion or cap)"][rep].append(m["trials"])
                    for t in range(cap):
                        row[f"curve {t + 1}"][rep].append(m["curve"][t])
                    if K != KS[0]:
                        continue
                    hows["memory"].update(m["hows"])
                    for i, c in enumerate(m["chunks"][:8]):
                        t8[typ][i].append(len(c))
                    code_seq(m["board1"], m["chunks"], *rel_mem[g])
                    for c in m["chunks"]:
                        if len(c) > 1:
                            idea_n["memory"][g] += 1
                            idea_cnt["memory"][g].update(ideas(c))
                    if rep == 0 and len(examples) < 4 and g == "game":
                        examples.append({"type": typ, "n": m["n"], "curve": m["curve"][:m["trials"]],
                                         "chunks": [[C.square_name(sq) + ":" + pc for sq, pc in c] for c in m["chunks"]]})
            for p in s["copy"][typ]:
                v = View(p, idx)
                c = copy_task(v)
                row = cp[typ]
                row["glances"][rep].append(c["n_glances"])
                row["pieces per glance"][rep].append(c["placed"] / max(1, c["n_glances"]))
                row["correct at end"][rep].append(c["correct"])
                row["commission at end"][rep].append(c["commission"])
                row["multi-piece glances"][rep].append(sum(1 for x in c["glances"] if len(x) > 1))
                hows["copy"].update(c["hows"])
                code_seq(p, c["glances"], *rel_copy[g])
                for x in c["glances"]:
                    if len(x) > 1:
                        idea_n["copy"][g] += 1
                        idea_cnt["copy"][g].update(ideas(x))
    out = {"positions learned": n_learned, "records": len(L.records), "stored chunks": len(store),
           "memory": {f"K={K}": {t: {k: agg(v) for k, v in mem[K][t].items()} for t in TYPES} for K in KS},
           "table 8 (K=7)": {t: [rnd(mean(x), 2) for x in t8[t]] for t in TYPES},
           "copy": {t: {k: agg(v) for k, v in cp[t].items()} for t in TYPES},
           "decoding": {k: dict(v) for k, v in hows.items()}, "examples": examples}
    rel = {"copy within random": probs(rel_copy["random"][0]), "copy within game": probs(rel_copy["game"][0]),
           "memory within game": probs(rel_mem["game"][0]), "copy between random": probs(rel_copy["random"][1]),
           "copy between game": probs(rel_copy["game"][1]), "memory between game": probs(rel_mem["game"][1]),
           "a priori game": prior_p["game"], "a priori random": prior_p["random"],
           "memory within random": probs(rel_mem["random"][0]), "memory between random": probs(rel_mem["random"][1])}
    cols = list(rel)[:7]
    out["relations"] = rel
    out["relations correlations (Table 7)"] = {f"{a} | {b}": corr(rel[a], rel[b])
                                              for i, a in enumerate(cols) for b in cols[i + 1:]}
    ide = {}
    for task in idea_cnt:
        for g in idea_cnt[task]:
            n = idea_n[task][g]
            sh = {k: rnd(idea_cnt[task][g][k] / n if n else None) for k in IDEAS}
            ide[f"{task} {g}"] = {"chunks": n, **sh}
    out["ideas"] = ide
    return out


# -------------------------------------------------------------- run
res = {"tag": tag, "seed": seed, "theta": THETA, "K": list(KS), "reps": REPS, "curve": curve,
       "train positions available": len(train), "headline": HEADLINE, "max size": max_size,
       "note": "no timing in the model: no latencies; chunk boundaries are labels (memory) or glances (copy)",
       "points": {}}
os.makedirs(OUT_DIR, exist_ok=True)
t0, done = time.time(), 0
for c in curve:
    for p in train[done:c]:
        learn(p)
    done = c
    t_learn = time.time() - t0
    L._quiet = True
    pt = evaluate(c)
    L._quiet = False
    pt["seconds"] = round(time.time() - t0)
    res["points"][str(c)] = pt
    m = pt["memory"]["K=7"]
    print(json.dumps({"learned": c, "s": pt["seconds"], "learn s": round(t_learn), "stored chunks": len(store),
                      "trial1": {t: m[t]["trial 1 correct"]["mean"] for t in TYPES},
                      "chunks": {t: m[t]["chunks trial 1"]["mean"] for t in ("middle", "end")},
                      "ppg middle": pt["copy"]["middle"]["pieces per glance"]["mean"],
                      "decoding": pt["decoding"]["memory"]}), flush=True)
    with open(os.path.join(OUT_DIR, f"{tag}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


# -------------------------------------------------------------- human vs model
HUMAN = {"trial 1 middle": (16, 8, 4), "trial 1 end": (8, 7, 4), "trial 1 random": (2.0, 2.5, 3.0),
         "chunks/trial middle": (7.7, 5.7, 5.3), "chunks/trial end": (7.6, 6.4, 4.2),
         "first chunk middle": (3.8, 2.6, 2.4), "pieces/glance middle": (2.0, 2.8, 2.0)}


def model_row(pt):
    m, cpy = pt["memory"]["K=7"], pt["copy"]
    rnd_t1 = mean([m["random middle"]["trial 1 correct"]["mean"], m["random end"]["trial 1 correct"]["mean"]])
    return {"trial 1 middle": m["middle"]["trial 1 correct"]["mean"], "trial 1 end": m["end"]["trial 1 correct"]["mean"],
            "trial 1 random": rnd(rnd_t1, 2), "chunks/trial middle": m["middle"]["chunks trial 1"]["mean"],
            "chunks/trial end": m["end"]["chunks trial 1"]["mean"],
            "first chunk middle": m["middle"]["first chunk size"]["mean"],
            "pieces/glance middle": cpy["middle"]["pieces per glance"]["mean"]}


def nearest(n):
    return str(min(curve, key=lambda c: abs(c - n)))


rows = {name: model_row(res["points"][nearest(n)]) for name, n in HEADLINE.items()}
print("\nHuman (C&S 1973; random = GS96a estimate from graph) vs model (K=7, theta %g, %d subjects)" % (THETA, REPS))
print("%-22s %16s %22s" % ("", "human M / A / B", "model M / A / B"))
for k, h in HUMAN.items():
    mv = [rows[x][k] for x in ("M", "A", "B")]
    print("%-22s %16s %22s" % (k, " / ".join(str(x) for x in h), " / ".join("-" if x is None else f"{x:.2f}" for x in mv)))
print("headline players: " + ", ".join(f"{k}={nearest(n)}" for k, n in HEADLINE.items()) + "; no latencies (no timing)")
print("\nLearning curve, K=7 trial-1 correct (middle / end / random middle / random end), K=4 middle, pieces/glance middle")
for c in curve:
    pt = res["points"][str(c)]
    m, m4 = pt["memory"]["K=7"], pt["memory"]["K=4"]
    print("%6d  %s  K4 %s  ppg %s" % (c, " / ".join(f"{m[t]['trial 1 correct']['mean']:.2f}" for t in TYPES),
                                      f"{m4['middle']['trial 1 correct']['mean']:.2f}",
                                      f"{pt['copy']['middle']['pieces per glance']['mean']:.2f}"))
print("wrote", os.path.join(OUT_DIR, f"{tag}_s{seed}.json"))
