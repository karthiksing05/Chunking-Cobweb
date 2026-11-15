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
a learning curve (default 0, 100, 300, 1000, 2000, 4000, 4660). The cap is 4,660
because the cached data/chess/cs1973/train.fen holds 4,660 positions; any larger
curve point is clipped to it. The headline B / A / M are 300 / 1000 / 4000 (not
tuned to the human numbers).

Labels. A 5-s view (memory task) or a glance (copy task) is one parse of the
position with learning off (L.parse, L._quiet). Its units are found as in
chess_inspect.py: each top element is walked down until it is a piece or is
recognised (content + context recognition bits >= theta, default 16). A unit's
label is its content concept and its context concept: on each hierarchy the
climbing ancestor that recognition is measured at (Trellis3._climb: the deepest
concept on the element's sorting path holding >= ``support`` instances), found by
sorting the element without learning. A piece's label is its kind (its content)
and its context concept.

Decoding a label (recall as generation). A label is recalled only when memory can
produce it from the label's own concept pair; there is no back-off to parents:
  * identical instance: memory holds a stored chunk with exactly the unit's pieces
    on exactly its squares (a committed chunk, from training or familiarised on an
    earlier trial of this position), or the unit was familiarised on an earlier
    trial (a single piece) -> it decodes perfectly;
  * composite: the committed chunks stored under the content concept whose context
    leaf lies under the context concept; their most frequent instance (absolute
    pieces on squares) is placed;
  * piece: the stored pieces of that kind whose context leaf lies under the context
    concept; it decodes to their most frequent square.
If the concept pair holds no matching stored instance, the label is an omission:
nothing is placed. A label at the root (either concept of a composite, the context
concept of a piece) is unrecognised and always an omission. The mode of stored
instances is the most probable instance of a concept the API supports without a
decoder (CU trees hold bag counts, from which an instance's size cannot be read
back; records keep their instances exactly). Untrained memory recalls nothing.

Memory task (section 4). Each trial is a 5-s view (a fresh parse with the current
memory), then recall. STM takes K = 7 labels (C&S's 7 +/- 2; K = 4 reported as a
sensitivity row): units with the most pieces not correctly recalled in the
previous trial's reconstruction first, then the largest, then the earliest anchor.
The paper clears the board between trials; the subject's knowledge carries over in
two ways: (a) the reconstruction after a trial is the union of every label decoded
so far (a later placement overrides an earlier one on its square), and (b) trials
are learning episodes: after each trial memory learns the K attended units by
GraphTrellis.learn on the sub-position made of those units' pieces with ``top`` =
the units (so each attended multi-piece unit is stored as exactly that chunk), and
the attended units are familiarised: later trials decode them as identical
instances. Learning the whole position would be too strong (criterion by trial 2).
Trial learning is confined to one position: each test position runs in a forked
child process, so its learning never reaches other positions or the curve
(copy.deepcopy cannot copy the compiled CU trees, which have no copy or pickle
support; fork gives an exact copy-on-write snapshot). Concept-mode decoding reads
memory as it stands at each trial (re-indexed every trial). Game positions run
until perfect, cap 15 trials (B took up to 14); random positions run 7 trials (not
to criterion). Score = pieces on the exact square, with errors of commission
(placed pieces wrong in kind or square) and omissions.

Chunks (Table 8; section 6.3): on trial 1 each decoded, non-omitted label is one
chunk; reported: average size (pieces placed) of chunks 1..8 in recall order and
chunks per trial. Relations (Tables 4-7; section 5): C&S's A/D/P/C/S codes (16
combinations, D implies C) between SUCCESSIVELY placed pieces: pieces of a chunk in
the left-to-right leaf order of the learner's own structure (the stored record's
tree for a decoded chunk, the unit's for an identical one), chunks in recall order;
within-chunk = successive pieces of one chunk, between-chunk = last piece of a
chunk and first of the next; coded on the board of placed pieces. A priori = all
pairs of pieces of the subject's 20 game (8 random) positions.

Copy task (section 3): the stimulus stays visible. One parse per position (forked
child, as above). Each glance gives ONE label, the largest unit with a piece not
yet correctly placed; a composite label is decoded from memory (an omission places
nothing) and a single piece is placed on the square it is seen on (the board is
visible). After each glance memory learns the attended unit as in the memory task,
so a composite that failed decodes perfectly when a later glance takes it again.
Glances continue until every piece is correct. Reported: pieces per glance (all
glances; paper, middle games: M 2.0, A 2.8, B 2.0), glances per position and the
within- vs between-glance relations. The model has no timing: no latencies, and
chunk boundaries are labels (memory) or glances (copy), not 2-s pauses.

Ideas (p. 79): every multi-piece chunk recalled on trial 1 (and copied per glance)
is labelled with chess_ideas.py's ideas() (copied here) and with C&S's three
categories: pawn chain, castled king, and same-colour cluster = 2+ pieces of one
colour that are not a pawn chain or castled king, with sub-labels back-rank group
(all on their own first rank) and connected rooks / queen + rooks (only R and Q,
2+ of them, a battery). C&S: > 75% of M's chunks fall in the three.

Writes experiments/v3/results/chess_cs1973_exact/<tag>_s<seed>.json (after every
curve point) and prints a human-vs-model table.

Usage: chess_cs1973_exact.py <cfg json> <seed> <tag> [curve e.g. 0,100,300] [reps] [theta]
"""
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
curve = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,100,300,1000,2000,4000,4660").split(",")]
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


train = load("train")                     # 4,660 positions: the cap of the curve
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
        game, rnd_ = picks[:10], [randomize(p, rng) for p in picks[10:]]
        sets["copy"][kind], sets["memory"][kind] = game[:5], game[5:]
        sets["copy"]["random " + kind], sets["memory"]["random " + kind] = rnd_[:2], rnd_[2:]
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
CS_CLUSTER, CS_BACK, CS_ROOKS = "same-colour cluster", "  back-rank group", "  connected rooks / queen+rooks"
CS_TYPES = "C&S types (pawn chain, castled king or same-colour cluster)"
IDEAS = ["pawn chain", "pawn phalanx", "doubled pawns", "pawn ram", "king shelter", "castled king", "fianchetto",
         "pawn-guarded", "battery", "undeveloped", "attack", CS_CLUSTER, CS_BACK, CS_ROOKS, CS_TYPES]


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
    """chess_ideas.ideas(), plus C&S's same-colour cluster (2+ pieces of one colour, not
    a pawn chain or castled king) with its two sub-labels, and the union of the three."""
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
    colours = {t[0] for _, t in pieces}
    if len(pieces) >= 2 and len(colours) == 1 and not found & {"pawn chain", "castled king"}:
        found.add(CS_CLUSTER)
        col = colours.pop()
        own = rel(pieces, col)
        if all(r == 0 for _, r, _, _ in own):
            found.add(CS_BACK)
        if all(k in "RQ" for _, _, k, _ in own) and any(k == "R" for _, _, k, _ in own) and "battery" in found:
            found.add(CS_ROOKS)
    if found & {"pawn chain", "castled king", CS_CLUSTER}:
        found.add(CS_TYPES)
    return found


# -------------------------------------------------------------- memory
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
inst = {}          # record id -> its pieces on squares, left-to-right leaves of its tree
store = set()      # committed chunks as frozensets of (square, piece)


def leaves(e):
    """An element's primitives in the left-to-right leaf order of its tree."""
    out, stack = [], [e]
    while stack:
        x = stack.pop()
        if x.token is not None:
            out.append(min(x.span))
        elif x.left is None:               # a unit stored by learn(top=...) without a parse tree
            out.extend(sorted(x.span))
        else:
            stack.extend((x.right, x.left))
    return out


def learn(p, top=None):
    r = L.learn(p, top=top)
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


def leaf_ids(tree, node):
    """Ids of the leaves under a concept."""
    cu = getattr(tree, "tree", None)
    out = []
    if cu is not None and hasattr(cu, "children_of"):
        stack = [node.id]
        while stack:
            i = stack.pop()
            kids = cu.children_of(i)
            if kids:
                stack.extend(kids)
            else:
                out.append(i)
        return out
    stack = [node]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            out.append(n.id)
    return out


class Mem:
    """What memory holds now, for decoding: the records stored at each leaf."""

    def __init__(self):
        self.cnt_at, self.ctx_at = defaultdict(list), defaultdict(list)
        for e in L.records:
            if e.rid not in inst or e.leaf is None:
                continue
            self.ctx_at[e.leaf.id].append(e)
            if e.token is None and e.cleaf is not None:
                self.cnt_at[e.cleaf.id].append(e)
        self.cache = {}

    def composite(self, nc, nx):
        """Most frequent stored chunk under content concept nc whose context leaf is under nx."""
        key = ("c", nc.id, nx.id)
        if key not in self.cache:
            xs = set(leaf_ids(L.ctx, nx))
            cnt, first = Counter(), {}
            for lid in leaf_ids(L.cnt, nc):
                for e in self.cnt_at.get(lid, ()):
                    if e.leaf.id in xs:
                        k = frozenset(inst[e.rid])
                        cnt[k] += 1
                        first.setdefault(k, inst[e.rid])
            self.cache[key] = first[min(cnt, key=lambda k: (-cnt[k], sorted(first[k])))] if cnt else None
        return self.cache[key]

    def piece(self, kind, nx):
        """Most frequent square of the stored pieces of ``kind`` under context concept nx."""
        key = ("w", kind, nx.id)
        if key not in self.cache:
            cnt = Counter(inst[e.rid][0][0] for lid in leaf_ids(L.ctx, nx)
                          for e in self.ctx_at.get(lid, ()) if e.token == kind)
            self.cache[key] = ((min(cnt, key=lambda s: (-cnt[s], s)), kind),) if cnt else None
        return self.cache[key]


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

    def __init__(self, pos, mem, familiar):
        self.pos, self.mem, self.familiar = pos, mem, familiar
        L._quiet = True
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

    def decode(self, u):
        """The label of unit u decoded from memory: (pieces in placement order, how)."""
        own = self.own(u)
        key = frozenset(own)
        if key in self.familiar or (u.token is None and key in store):
            return tuple(own), "identical"
        if id(u) in self.dec:
            return self.dec[id(u)]
        L._quiet = True
        px = climb_path(L.ctx, u.ctx)
        if u.token is not None:
            pcs = self.mem.piece(u.token, px[-1]) if len(px) > 1 else None
        else:
            L._exp = self.exp
            pc = climb_path(L.cnt, L._content(u))
            pcs = self.mem.composite(pc[-1], px[-1]) if len(pc) > 1 and len(px) > 1 else None
        out = (tuple(pcs), "concept") if pcs else ((), "omission")
        self.dec[id(u)] = out
        return out


def attend(v, units):
    """Learn the attended units: GraphTrellis.learn on the sub-position of their pieces,
    each multi-piece unit stored as exactly that chunk; all are familiarised."""
    pcs = [x for u in units for x in v.own(u)]
    if not pcs:
        return
    sub = dict(pcs)
    index = {sq: i for i, sq in enumerate(sorted(sub, key=lambda s: (s[1], s[0])))}
    top = [frozenset(index[sq] for sq, _ in v.own(u)) for u in units if len(u.span) > 1]
    L._quiet = False
    learn(sub, top=top)
    L._quiet = True
    for u in units:
        v.familiar.add(frozenset(v.own(u)))


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


def memory_task(pos, K, cap):
    """Trials of one position (run in a forked child: its learning dies with it)."""
    board, familiar = {}, set()
    curve, comm, chunks1, board1, hows, reached, taken1 = [], [], [], {}, Counter(), False, 0
    for t in range(cap):
        v = View(pos, Mem(), familiar)

        def wrong(u):
            return sum(1 for sq, p in v.own(u) if board.get(sq) != p)
        take = sorted((u for u in v.units if wrong(u) > 0), key=lambda u: (-wrong(u),) + v.order(u))[:K]
        chunks = []
        for u in take:
            pcs, how = v.decode(u)
            hows[how] += 1
            for sq, p in pcs:
                board[sq] = p
            if pcs:
                chunks.append(list(pcs))
        if t == 0:
            chunks1, board1, taken1 = chunks, dict(board), len(take)
        ok, c, _ = score(pos, board)
        curve.append(ok)
        comm.append(c)
        if ok == len(pos):
            reached = True
            break
        attend(v, take)
    n_trials = len(curve)
    return {"curve": curve + [curve[-1]] * (cap - n_trials), "commission": comm + [comm[-1]] * (cap - n_trials),
            "trials": n_trials, "reached": reached, "n": len(pos), "chunks": chunks1, "board1": board1,
            "taken1": taken1, "hows": hows,
            "gain": (curve[-1] - curve[0]) / (n_trials - 1) if n_trials > 1 else None}


def copy_task(pos, cap=400):
    v = View(pos, Mem(), set())
    board, glances, hows = {}, [], Counter()
    while len(glances) < cap:
        cand = [u for u in v.units if any(board.get(sq) != p for sq, p in v.own(u))]
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
        attend(v, [u])
    ok, c, _ = score(pos, board)
    return {"glances": glances, "n_glances": len(glances), "placed": sum(map(len, glances)), "correct": ok,
            "commission": c, "n": len(pos), "hows": hows}


def isolated(fn, *args):
    """Run fn in a forked child (an exact copy-on-write snapshot of memory) and return
    its result; whatever the child learns is discarded with it."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        gc.disable()
        try:
            data = pickle.dumps(("ok", fn(*args)))
        except BaseException:
            data = pickle.dumps(("err", traceback.format_exc()))
        with os.fdopen(w, "wb") as fh:
            fh.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as fh:
        data = fh.read()
    os.waitpid(pid, 0)
    status, val = pickle.loads(data)
    if status != "ok":
        raise RuntimeError("child failed:\n" + val)
    return val


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
    gc.collect()
    gc.freeze()                           # children's collections then leave parent pages alone
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
                for K in KS:
                    m = isolated(memory_task, p, K, cap)
                    row = mem[K][typ]
                    row["trial 1 correct"][rep].append(m["curve"][0])
                    row["trial 1 commission"][rep].append(m["commission"][0])
                    row["trial 1 omission"][rep].append(m["n"] - m["curve"][0])
                    row["trial 1 percent"][rep].append(100.0 * m["curve"][0] / m["n"])
                    row["chunks trial 1"][rep].append(len(m["chunks"]))
                    row["omitted labels trial 1"][rep].append(m["taken1"] - len(m["chunks"]))
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
                c = isolated(copy_task, p)
                row = cp[typ]
                row["glances"][rep].append(c["n_glances"])
                row["pieces per glance"][rep].append(c["placed"] / max(1, c["n_glances"]))
                row["pieces per productive glance"][rep].append(
                    c["placed"] / max(1, sum(1 for x in c["glances"] if x)))
                row["correct at end"][rep].append(c["correct"])
                row["commission at end"][rep].append(c["commission"])
                row["multi-piece glances"][rep].append(sum(1 for x in c["glances"] if len(x) > 1))
                hows["copy"].update(c["hows"])
                code_seq(p, c["glances"], *rel_copy[g])
                for x in c["glances"]:
                    if len(x) > 1:
                        idea_n["copy"][g] += 1
                        idea_cnt["copy"][g].update(ideas(x))
    gc.unfreeze()
    out = {"positions learned": n_learned, "records": len(L.records), "stored chunks": len(store),
           "memory": {f"K={K}": {t: {k: agg(v) for k, v in mem[K][t].items()} for t in TYPES} for K in KS},
           "table 8 (K=7)": {t: [rnd(mean(x), 2) for x in t8[t]] for t in TYPES},
           "copy": {t: {k: agg(v) for k, v in cp[t].items()} for t in TYPES},
           "decoding": {k: dict(v) for k, v in hows.items()}, "examples": examples}
    rel_ = {"copy within random": probs(rel_copy["random"][0]), "copy within game": probs(rel_copy["game"][0]),
            "memory within game": probs(rel_mem["game"][0]), "copy between random": probs(rel_copy["random"][1]),
            "copy between game": probs(rel_copy["game"][1]), "memory between game": probs(rel_mem["game"][1]),
            "a priori game": prior_p["game"], "a priori random": prior_p["random"],
            "memory within random": probs(rel_mem["random"][0]), "memory between random": probs(rel_mem["random"][1])}
    cols = list(rel_)[:7]
    out["relations"] = rel_
    out["relations correlations (Table 7)"] = {f"{a} | {b}": corr(rel_[a], rel_[b])
                                              for i, a in enumerate(cols) for b in cols[i + 1:]}
    ide = {}
    for task in idea_cnt:
        for g in idea_cnt[task]:
            n = idea_n[task][g]
            ide[f"{task} {g}"] = {"chunks": n, **{k: rnd(idea_cnt[task][g][k] / n if n else None) for k in IDEAS}}
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
    L._quiet = False
    for p in train[done:c]:
        learn(p)
    done = c
    t_learn = time.time() - t0
    pt = evaluate(c)
    L._quiet = False
    pt["seconds"] = round(time.time() - t0)
    res["points"][str(c)] = pt
    m = pt["memory"]["K=7"]
    print(json.dumps({"learned": c, "s": pt["seconds"], "learn s": round(t_learn), "stored chunks": len(store),
                      "trial1": {t: m[t]["trial 1 correct"]["mean"] for t in TYPES},
                      "chunks": {t: m[t]["chunks trial 1"]["mean"] for t in ("middle", "end")},
                      "trials": {t: m[t]["trials (to criterion or cap)"]["mean"] for t in ("middle", "end")},
                      "reached": {t: m[t]["reached criterion"]["mean"] for t in ("middle", "end")},
                      "ppg middle": pt["copy"]["middle"]["pieces per glance"]["mean"],
                      "C&S types memory game": pt["ideas"]["memory game"][CS_TYPES],
                      "decoding": pt["decoding"]["memory"]}), flush=True)
    with open(os.path.join(OUT_DIR, f"{tag}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


# -------------------------------------------------------------- human vs model
HUMAN = {"trial 1 middle": (16, 8, 4), "trial 1 end": (8, 7, 4), "trial 1 random": (2.0, 2.5, 3.0),
         "chunks/trial middle": (7.7, 5.7, 5.3), "chunks/trial end": (7.6, 6.4, 4.2),
         "first chunk middle": (3.8, 2.6, 2.4), "pieces/glance middle": (2.0, 2.8, 2.0),
         "trials to perfect middle": ("3-4", "4-6", "<=14"), "trials to perfect end": ("2-3", "3-4", "4-7")}


def model_row(pt):
    m, cpy = pt["memory"]["K=7"], pt["copy"]
    rnd_t1 = mean([m["random middle"]["trial 1 correct"]["mean"], m["random end"]["trial 1 correct"]["mean"]])
    return {"trial 1 middle": m["middle"]["trial 1 correct"]["mean"], "trial 1 end": m["end"]["trial 1 correct"]["mean"],
            "trial 1 random": rnd(rnd_t1, 2), "chunks/trial middle": m["middle"]["chunks trial 1"]["mean"],
            "chunks/trial end": m["end"]["chunks trial 1"]["mean"],
            "first chunk middle": m["middle"]["first chunk size"]["mean"],
            "pieces/glance middle": cpy["middle"]["pieces per glance"]["mean"],
            "trials to perfect middle": m["middle"]["trials (to criterion or cap)"]["mean"],
            "trials to perfect end": m["end"]["trials (to criterion or cap)"]["mean"]}


def nearest(n):
    return str(min(curve, key=lambda c: abs(c - n)))


rows = {name: model_row(res["points"][nearest(n)]) for name, n in HEADLINE.items()}
print("\nHuman (C&S 1973; random = GS96a estimate from graph) vs model (K=7, theta %g, %d subjects)" % (THETA, REPS))
print("%-26s %18s %22s" % ("", "human M / A / B", "model M / A / B"))
for k, h in HUMAN.items():
    mv = [rows[x][k] for x in ("M", "A", "B")]
    print("%-26s %18s %22s" % (k, " / ".join(str(x) for x in h), " / ".join("-" if x is None else f"{x:.2f}" for x in mv)))
print("headline players: " + ", ".join(f"{k}={nearest(n)}" for k, n in HEADLINE.items()) +
      "; trials to perfect = mean of trials run (cap 15 when not reached); no latencies (no timing)")
print("\nLearning curve, K=7 trial-1 correct (middle / end / random middle / random end), K=4 middle, "
      "chunks middle, trials middle, pieces/glance middle, C&S types share (memory, game)")
for c in curve:
    pt = res["points"][str(c)]
    m, m4 = pt["memory"]["K=7"], pt["memory"]["K=4"]
    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    print("%6d  %s  K4 %s  ch %s  tr %s  ppg %s  cs %s" % (
        c, " / ".join(f(m[t]["trial 1 correct"]["mean"]) for t in TYPES), f(m4["middle"]["trial 1 correct"]["mean"]),
        f(m["middle"]["chunks trial 1"]["mean"]), f(m["middle"]["trials (to criterion or cap)"]["mean"]),
        f(pt["copy"]["middle"]["pieces per glance"]["mean"]), f(pt["ideas"]["memory game"][CS_TYPES])))
print("wrote", os.path.join(OUT_DIR, f"{tag}_s{seed}.json"))
