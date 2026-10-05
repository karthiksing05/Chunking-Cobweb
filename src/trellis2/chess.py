"""Chess positions as experiences: parts joined by typed relations on a board.

A position is a set of pieces on squares; a primitive element is one piece,
whose token is its colour and kind (``wK``, ``bP``, ...).

**The star.** From a square, the context window reaches along the eight
queen rays to the edge of the board (the first piece on each ray, at any
distance) and to the eight squares a knight's jump away. It is an element's
representation context: what the representation hierarchy sees of an
element is its star, its anchor piece (whole, and by colour and kind), its
kind and its square, and its chunk context, as for sentences. The star is
written as two bags, one of the eight rays and one of the eight jumps, each
value naming its direction (``N:wP``, ``NNE:bN``). As separate attributes its
sixteen directions would outweigh what the element is, and the categories
would mix kinds of piece; as two bags the star weighs what a sentence
element's two neighbours weigh.

**Typed relations.** A composite joins two elements whose anchors see each
other: the second anchor is the first piece along a ray from the first
(relation: direction and distance, ``N1``, ``E3``) or a knight's jump away
(relation: the jump, ``NNE``). Its anchor is its first part's anchor. The
composition hierarchy describes a composite by its two parts' categories and
their relation.

**The top level.** A board is read square by square, a1 b1 ... h1 a2 ... h8,
as a sentence is read word by word. Each square that no earlier chunk covers
is coded as empty or as the anchor of a top-level element, from one
Dirichlet row per square and context. So that a chunk is decoded at its
first square, relations point forward in the scan: the N, NE, NW and E rays
and the four upward knight jumps (32 relations). A chunk therefore pays for
itself when it predicts its pieces better than their squares do.

**What has been read so far.** A square is read in the light of the pieces
on the squares before it: the context of the read is a set of features "at
least m pieces of kind k stand on earlier squares", each kept only if it
shortens the code of the training positions (``select_context``). Read
without it, every square is drawn on its own and a generated board has one
king of each colour only a third of the time; the features that pay for
themselves include "a white king is already on the board".

The positions are middlegame positions from the Lichess database (CC0); see
``experiments/v2/run_chess.py`` for how they are extracted into
``data/chess``.
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .grammar import Grammar, dm_code
from .memory import Memory, chunk_attrs
from .model import Trellis2

Square = Tuple[int, int]            # (file 0-7, rank 0-7)
Position = Dict[Square, str]        # square -> piece token
FILES = "abcdefgh"
EMPTY, EDGE = "·", "|"
RAYS = (("N", (0, 1)), ("NE", (1, 1)), ("E", (1, 0)), ("SE", (1, -1)),
        ("S", (0, -1)), ("SW", (-1, -1)), ("W", (-1, 0)), ("NW", (-1, 1)))
JUMPS = (("NNE", (1, 2)), ("ENE", (2, 1)), ("ESE", (2, -1)), ("SSE", (1, -2)),
         ("SSW", (-1, -2)), ("WSW", (-2, -1)), ("WNW", (-2, 1)), ("NNW", (-1, 2)))
FORWARD_RAYS = ("N", "NE", "NW", "E")
FORWARD_JUMPS = ("NNE", "ENE", "WNW", "NNW")
RELATIONS = [f"{d}{k}" for d in FORWARD_RAYS for k in range(1, 8)] + list(FORWARD_JUMPS)
OFFSET = {f"{d}{k}": (dx * k, dy * k) for d, (dx, dy) in RAYS if d in FORWARD_RAYS for k in range(1, 8)}
OFFSET.update({d: v for d, v in JUMPS if d in FORWARD_JUMPS})
Node = tuple    # (label, square) for a piece, (label, (node, relation, node)) for a chunk
TOKENS = {c + p for c in "wb" for p in "KQRBNP"}
KINDS = sorted(TOKENS)
Feature = Tuple[str, int]   # at least m pieces of this kind on earlier squares


def square_name(sq: Square) -> str:
    return FILES[sq[0]] + str(sq[1] + 1)


def on_board(sq: Square) -> bool:
    return 0 <= sq[0] < 8 and 0 <= sq[1] < 8


def parse_fen(board_fen: str) -> Position:
    """The board part of a FEN string as {square: token}."""
    position: Position = {}
    for r, row in enumerate(board_fen.split()[0].split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                position[(f, 7 - r)] = ("w" if ch.isupper() else "b") + ch.upper()
                f += 1
    return position


def default_positions_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "data", "chess", "positions_1800_ply30.fen"))


def load_positions(path: Optional[str] = None, n: Optional[int] = None) -> List[Position]:
    with open(path or default_positions_path()) as f:
        fens = [line.strip() for line in f if line.strip()]
    return [parse_fen(fen) for fen in fens[:n]]


def star(position: Position, sq: Square, own: frozenset = frozenset()) -> Dict[str, str]:
    """The context window of an element anchored at ``sq``: along each ray the
    first piece outside the element (EDGE if none), on each knight square the
    piece there (EMPTY if none or the element's own, EDGE off the board)."""
    x: Dict[str, str] = {}
    for d, (dx, dy) in RAYS:
        x[d] = EDGE
        f, r = sq[0] + dx, sq[1] + dy
        while 0 <= f < 8 and 0 <= r < 8:
            if (f, r) in position and (f, r) not in own:
                x[d] = position[(f, r)]
                break
            f, r = f + dx, r + dy
    for d, (dx, dy) in JUMPS:
        t = (sq[0] + dx, sq[1] + dy)
        x[d] = EDGE if not on_board(t) else (position[t] if t in position and t not in own else EMPTY)
    return x


def forward_neighbours(position: Position, sq: Square,
                       own: frozenset = frozenset()) -> Iterator[Tuple[str, Square]]:
    """The pieces seen from ``sq`` forward in the scan, with their relation:
    along each forward ray the first piece at any distance, and on each
    forward knight square the piece there. An element's own pieces (``own``)
    do not block its view, as in its star: a chunk sees past its members."""
    for d, (dx, dy) in RAYS:
        if d not in FORWARD_RAYS:
            continue
        for k in range(1, 8):
            t = (sq[0] + dx * k, sq[1] + dy * k)
            if not on_board(t):
                break
            if t in position and t not in own:
                yield f"{d}{k}", t
                break
    for d in FORWARD_JUMPS:
        dx, dy = OFFSET[d]
        t = (sq[0] + dx, sq[1] + dy)
        if t in position and t not in own:
            yield d, t


def is_chunk(node: Node) -> bool:
    return len(node[1]) == 3


def anchor_of(node: Node) -> Square:
    while is_chunk(node):
        node = node[1][0]
    return node[1]


def render(node: Node, position: Position) -> str:
    """A chunk written as nested [first relation second]."""
    if not is_chunk(node):
        return position[node[1]]
    x, rel, y = node[1]
    return f"[{render(x, position)} {rel} {render(y, position)}]"


# --------------------------------------------------------------------------- #
# The context of the square-by-square read
# --------------------------------------------------------------------------- #
def scan_rows(position: Position, features: Sequence[Feature]) -> np.ndarray:
    """The row of every square's read: the square, and for each feature
    whether at least m pieces of its kind stand on earlier squares."""
    rows = np.zeros(64, dtype=np.int64)
    placed: Counter = Counter()
    for i in range(64):
        r = i
        for kind, m in features:
            r = r * 2 + (placed[kind] >= m)
        rows[i] = r
        t = position.get((i % 8, i // 8))
        if t is not None:
            placed[t] += 1
    return rows


def select_context(positions: Sequence[Position], alpha: float = 0.001,
                   max_count: int = 8) -> List[Feature]:
    """Greedily add the feature "at least m pieces of kind k on earlier
    squares" that most shortens the code of the positions read square by
    square, while one does."""
    outcomes = KINDS + [EMPTY]
    index = {o: i for i, o in enumerate(outcomes)}
    out, square, before = [], [], []
    for position in positions:
        placed = np.zeros(len(KINDS), dtype=np.int64)
        for i in range(64):
            t = position.get((i % 8, i // 8), EMPTY)
            out.append(index[t])
            square.append(i)
            before.append(placed.copy())
            if t != EMPTY:
                placed[index[t]] += 1
    out, square, before = np.array(out), np.array(square), np.array(before)
    ones = np.ones(len(out))

    def code(features):
        rows = square.copy()
        for k, m in features:
            rows = rows * 2 + (before[:, k] >= m)
        return dm_code(rows, out, ones, len(outcomes), alpha)

    candidates = [(k, m) for k in range(len(KINDS)) for m in range(1, max_count + 1)
                  if (before[:, k] >= m).any() and (before[:, k] < m).any()]
    chosen: List[Tuple[int, int]] = []
    current = code(chosen)
    while True:
        scored = [(code(chosen + [c]), c) for c in candidates if c not in chosen]
        if not scored:
            break
        value, best = min(scored)
        if value >= current - 1e-6:
            break
        chosen.append(best)
        current = value
    return [(KINDS[k], m) for k, m in chosen]



# --------------------------------------------------------------------------- #
# The board's memory: star context, relations, the scan code
# --------------------------------------------------------------------------- #
class BoardMemory(Memory):
    """Element records for positions. Elements are trees of pieces joined by
    relations; each has an anchor square and the squares of its pieces."""

    relations = RELATIONS

    def __init__(self, spine_depth: int = 2, granularities: int = 2,
                 features: Sequence[Feature] = ()):
        super().__init__(spine_depth=spine_depth, granularities=granularities)
        self.attrs = ["rays", "jumps", "t", "colour", "piece", "k", "rank", "file"] + chunk_attrs(spine_depth, granularities)
        self.features = list(features)
        self.anchor: List[Square] = []
        self.members: List[frozenset] = []
        self.label: List[Hashable] = []      # the analysis' own label of each element
        self._scan = None

    def add_board(self, position: Position, tops: Sequence[Node], weight: float = 1.0) -> None:
        """Record every element of an analysed position, bottom up."""
        pid = len(self.sentences)
        self.sentences.append(position)
        self._scan = None

        def record(node: Node) -> int:
            lab, body = node
            if is_chunk(node):
                x, rel, y = body
                a, b = record(x), record(y)
                e = len(self.kind)
                self.kind.append(self.COMPOSITE)
                self.token.append(None)
                self.left.append(a)
                self.right.append(b)
                self.relation.append(rel)
                self.anchor.append(self.anchor[a])
                self.members.append(self.members[a] | self.members[b])
                self.parent[a] = self.parent[b] = e
            else:
                e = len(self.kind)
                self.kind.append(self.PRIMITIVE)
                self.token.append(position[body])
                self.left.append(-1)
                self.right.append(-1)
                self.relation.append(None)
                self.anchor.append(body)
                self.members.append(frozenset([body]))
            self.label.append(lab)
            self.parent.append(-1)
            self.top_left.append(-1)
            self.top_right.append(-1)
            self.is_root.append(False)
            self.weight.append(weight)
            self.sentence_of.append(pid)
            self.span.append(None)
            return e

        for node in tops:
            self.is_root[record(node)] = True

    def surface(self, e: int):
        position = self.sentences[self.sentence_of[e]]
        sq = self.anchor[e]
        context = star(position, sq, self.members[e])
        x = {"rays": {f"{d}:{context[d]}": 1 / 8 for d, _ in RAYS},
             "jumps": {f"{d}:{context[d]}": 1 / 8 for d, _ in JUMPS}}
        # The anchor piece, whole and by colour and kind (as a sentence's
        # element is described by its first and its last token).
        x["t"] = position[sq]
        x["colour"], x["piece"] = position[sq][0], position[sq][1]
        x["k"] = "P" if self.kind[e] == self.PRIMITIVE else "C"
        x["rank"] = str(sq[1] + 1)
        x["file"] = FILES[sq[0]]
        return x

    def describe(self, e: int) -> str:
        if self.kind[e] == self.PRIMITIVE:
            return self.token[e]
        return f"[{self.describe(self.left[e])} {self.relation[e]} {self.describe(self.right[e])}]"

    def _scan_arrays(self):
        """Per square read in each position: (its row: the square and its
        context, the top-level element anchored there or -1 if the square is
        empty, weight). Squares covered by an earlier chunk are not read."""
        if self._scan is None:
            at = {}
            for e, root in enumerate(self.is_root):
                if root:
                    at[(self.sentence_of[e], self.anchor[e])] = e
            weight = {self.sentence_of[e]: self.weight[e] for e in at.values()}
            q, el, w = [], [], []
            for pid, position in enumerate(self.sentences):
                rows = scan_rows(position, self.features)
                for i in range(64):
                    sq = (i % 8, i // 8)
                    e = at.get((pid, sq))
                    if e is not None or sq not in position:
                        q.append(rows[i])
                        el.append(-1 if e is None else e)
                        w.append(weight.get(pid, 1.0))
            self._scan = (np.array(q, dtype=np.int64), np.array(el, dtype=np.int64), np.array(w))
        return self._scan

    def top_level_nats(self, s: np.ndarray, K: int, alpha: float) -> float:
        q, el, w = self._scan_arrays()
        outcome = np.where(el >= 0, s[np.maximum(el, 0)], K)
        return dm_code(q, outcome, w, K + 1, alpha)

    def scan_counts(self, s: np.ndarray, K: int) -> np.ndarray:
        q, el, w = self._scan_arrays()
        outcome = np.where(el >= 0, s[np.maximum(el, 0)], K)
        counts = np.zeros((64 * 2 ** len(self.features), K + 1))
        np.add.at(counts, (q, outcome), w)
        return counts


# --------------------------------------------------------------------------- #
# The structure search: chunk moves under the scan code
# --------------------------------------------------------------------------- #
def _phi(c: float, a: float) -> float:
    return math.lgamma(c + a) - math.lgamma(a) if c else 0.0


class BoardSearch:
    """Greedy search over chunk moves, scored exactly by the plain code:
    category rows (outcomes: a piece, or (B, relation, C)) and the scan rows.
    A chunk move (B, relation, C) joins, in every position, the top-level
    elements labelled B and C whose anchors stand in that relation (each
    element at most once, in scan order)."""

    def __init__(self, positions: Sequence[Position], alpha: float = 0.001,
                 features: Sequence[Feature] = ()):
        self.a = alpha
        self.features = list(features)
        self.V = len({t for p in positions for t in p.values()}) + 1
        self.NR = len(RELATIONS)
        self.rows: Dict[Hashable, Counter] = defaultdict(Counter)
        self.sq: Dict[int, Counter] = defaultdict(Counter)     # scan row -> outcomes
        self.pos = []
        self.fresh = 0
        self.moves: List[tuple] = []      # (B, relation, C, Y) in the order applied
        for position in positions:
            tops = {}
            rows = scan_rows(position, self.features)
            for i in range(64):
                sq = (i % 8, i // 8)
                if sq in position:
                    t = position[sq]
                    tops[sq] = (t, sq)
                    self.rows[t][("w", t)] += 1
                    self.sq[rows[i]][t] += 1
                else:
                    self.sq[rows[i]][EMPTY] += 1
            self.pos.append({"position": position, "tops": tops, "owner": {sq: sq for sq in position},
                             "rows": rows})
        self._row_stats()

    def _row_stats(self):
        """Each scan row's total and the sum of its outcomes' log-gamma terms,
        so that a move is scored from the rows it changes."""
        self.row_tot = {r: sum(c.values()) for r, c in self.sq.items()}
        self.row_phi = {r: sum(_phi(v, self.a) for v in c.values()) for r, c in self.sq.items()}

    def _row_nats(self, K: int) -> float:
        A = (self.V + K * K * self.NR) * self.a
        return sum(math.lgamma(sum(r.values()) + A) - math.lgamma(A) - sum(_phi(c, self.a) for c in r.values())
                   for r in self.rows.values())

    def bits(self) -> float:
        K = len(self.rows)
        nats = self._row_nats(K)
        A = (K + 1) * self.a
        for c in self.sq.values():
            nats += math.lgamma(sum(c.values()) + A) - math.lgamma(A) - sum(_phi(v, self.a) for v in c.values())
        return nats / math.log(2)

    @staticmethod
    def _seen(p: dict, sq: Square) -> Iterator[Tuple[str, Square]]:
        """(relation, anchor) of the top-level elements that the element
        anchored at ``sq`` sees forward, looking past its own pieces."""
        owner, tops = p["owner"], p["tops"]
        own = frozenset(m for m, o in owner.items() if o == sq)
        for rel, t in forward_neighbours(p["position"], sq, own):
            if owner[t] == t and t in tops:
                yield rel, t

    def candidates(self) -> Dict[tuple, List[tuple]]:
        out: Dict[tuple, List[tuple]] = defaultdict(list)
        for pi, p in enumerate(self.pos):
            for sq, x in p["tops"].items():
                for rel, t in self._seen(p, sq):
                    out[(x[0], rel, p["tops"][t][0])].append((pi, sq, t))
        return out

    def replay(self, moves: Sequence[tuple]) -> None:
        """Apply chunk moves (B, relation, C, Y) learned elsewhere, in order."""
        for B, rel, C, Y in moves:
            pairs = [(pi, sq, t) for pi, p in enumerate(self.pos) for sq, x in p["tops"].items()
                     if x[0] == B for r, t in self._seen(p, sq) if r == rel and p["tops"][t][0] == C]
            chosen = self.select(pairs)
            if chosen:
                self.fresh = Y[1]
                self.apply((B, rel, C), chosen)

    @staticmethod
    def select(pairs: List[tuple]) -> List[tuple]:
        used, chosen = set(), []
        for pi, s, t in sorted(pairs, key=lambda x: (x[0], x[1][1], x[1][0], x[2][1], x[2][0])):
            if (pi, s) in used or (pi, t) in used:
                continue
            used.add((pi, s))
            used.add((pi, t))
            chosen.append((pi, s, t))
        return chosen

    def score(self, key: tuple, chosen: List[tuple], base_rows: Dict[int, float]) -> float:
        """Bits after the chunk move (exact)."""
        B, _, C = key
        a, K = self.a, len(self.rows) + 1
        n = len(chosen)
        A = (self.V + K * K * self.NR) * a
        if K not in base_rows:
            base_rows[K] = self._row_nats(K)
        nats = base_rows[K] + math.lgamma(n + A) - math.lgamma(A) - _phi(n, a)
        change: Dict[int, Counter] = defaultdict(Counter)
        for pi, s, t in chosen:
            rows = self.pos[pi]["rows"]
            i, j = rows[s[1] * 8 + s[0]], rows[t[1] * 8 + t[0]]
            change[i][B] -= 1
            change[i]["Y"] += 1
            change[j][C] -= 1
        A = (K + 1) * a
        if ("scan", K) not in base_rows:
            base_rows[("scan", K)] = sum(math.lgamma(t + A) - math.lgamma(A) - self.row_phi[r]
                                         for r, t in self.row_tot.items())
        nats += base_rows[("scan", K)]
        for r, d in change.items():
            new = Counter(self.sq[r])
            for o, dv in d.items():
                new[o] += dv
            vals = [v for v in new.values() if v]
            nats -= math.lgamma(self.row_tot[r] + A) - math.lgamma(A) - self.row_phi[r]
            nats += math.lgamma(sum(vals) + A) - math.lgamma(A) - sum(_phi(v, a) for v in vals)
        return nats / math.log(2)

    def apply(self, key: tuple, chosen: List[tuple]) -> Hashable:
        B, rel, C = key
        Y = ("chunk", self.fresh)
        self.fresh += 1
        self.moves.append((B, rel, C, Y))
        self.rows[Y][("p", B, rel, C)] += len(chosen)
        for pi, s, t in chosen:
            p = self.pos[pi]
            x, y = p["tops"][s], p["tops"].pop(t)
            p["tops"][s] = (Y, (x, rel, y))
            for m, owner in list(p["owner"].items()):
                if owner == t:
                    p["owner"][m] = s
            i, j = p["rows"][s[1] * 8 + s[0]], p["rows"][t[1] * 8 + t[0]]
            self.sq[i][B] -= 1
            self.sq[i][Y] += 1
            self.sq[j][C] -= 1
        self.sq = defaultdict(Counter, {r: Counter({o: v for o, v in c.items() if v})
                                        for r, c in self.sq.items()})
        self._row_stats()
        return Y

    def run(self, max_steps: int = 500, log=None) -> float:
        cur = self.bits()
        for step in range(max_steps):
            base_rows: Dict[int, float] = {}
            best = None
            for key, pairs in self.candidates().items():
                chosen = self.select(pairs)
                if len(chosen) < 2:
                    continue
                b = self.score(key, chosen, base_rows)
                if best is None or b < best[0] - 1e-9:
                    best = (b, key, chosen)
            if best is None or best[0] >= cur - 1e-6:
                break
            self.apply(best[1], best[2])
            cur = best[0]
            if log:
                log(step, best[1], len(best[2]), cur)
        return cur

    def analyses(self) -> List[List[Node]]:
        return [[p["tops"][sq] for sq in sorted(p["tops"], key=lambda s: (s[1], s[0]))] for p in self.pos]

    def held_out_bits(self, positions: Sequence[Position]) -> Tuple[float, float]:
        """Bits per position for new positions: analysed by replaying the
        learned chunk moves in order and coded with this search's counts
        (posterior predictive), and the same with no chunks (each square on
        its own, from the training positions' flat counts)."""
        analysed = BoardSearch(positions, self.a, self.features)
        analysed.replay(self.moves)
        flat = BoardSearch([p["position"] for p in self.pos], self.a, self.features)
        return (self.code_of(analysed) / len(positions),
                flat.code_of(BoardSearch(positions, self.a, self.features)) / len(positions))

    def code_of(self, other: "BoardSearch") -> float:
        """Bits of another set of analyses under this search's counts."""
        a, K = self.a, len(self.rows)
        A = (self.V + K * K * self.NR) * a
        bits = 0.0
        for r, theirs in other.sq.items():
            mine = self.sq.get(r, Counter())
            n = sum(mine.values())
            for o, m in theirs.items():
                bits -= m * math.log2((mine.get(o, 0) + a) / (n + (K + 1) * a))
        for lab, row in other.rows.items():
            mine = self.rows.get(lab, Counter())
            n = sum(mine.values())
            for o, m in row.items():
                bits -= m * math.log2((mine.get(o, 0) + a) / (n + A))
        return bits


# --------------------------------------------------------------------------- #
# Learning, and sampling positions from the grammar
# --------------------------------------------------------------------------- #
class ChessLearner:
    """TRELLIS v2 on positions alone: the structure search proposes chunks,
    consolidation into the two hierarchies forms the categories and rule
    classes, and the grammar read off them generates positions."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, max_steps: int = 500,
                 context: bool = True):
        self.seed, self.alpha, self.max_steps = seed, alpha, max_steps
        # Whether the square-by-square read learns its context (it does by
        # default; without it every square is read on its own).
        self.context = context
        self.features: List[Feature] = []
        self.model: Optional[Trellis2] = None
        self.search: Optional[BoardSearch] = None
        self.analyses: List[List[Node]] = []
        self.history: List[Dict] = []

    def sleep(self, positions: Sequence[Position]) -> Grammar:
        t0 = time.time()
        self.features = select_context(positions, self.alpha) if self.context else []
        search = BoardSearch(positions, self.alpha, self.features)
        flat = search.bits()
        self.history.append({"stage": "flat", "bits": flat, "seconds": 0.0})

        def log(step, key, n, bits):
            self.history.append({"stage": "chunk", "move": f"[{key[0]} {key[1]} {key[2]}] x{n}",
                                 "bits": bits, "seconds": time.time() - t0})
        search.run(self.max_steps, log)
        self.search = search
        self.analyses = search.analyses()
        mem = BoardMemory(features=self.features)
        model = Trellis2(seed=self.seed, alpha=self.alpha, memory=mem)
        for position, tops in zip(positions, self.analyses):
            mem.add_board(position, tops)
        ids: Dict[Hashable, int] = {}
        labels = np.array([ids.setdefault(lab, len(ids)) for lab in mem.label])
        g = model.consolidate(init_labels=[labels] * mem.granularities)
        self.history.append({"stage": "consolidate", "bits": g.info["total bits"],
                             "seconds": time.time() - t0})
        self.model = model
        return g

    @property
    def grammar(self) -> Grammar:
        return self.model.grammar

    def sample(self, rng: np.random.Generator, max_depth: int = 12):
        """A position read off the grammar square by square, or None if a
        chunk would put a piece off the board or on an occupied square."""
        g = self.grammar
        position: Position = {}
        tops: List[Node] = []

        def expand(sym: int, sq: Square, depth: int):
            c = int(rng.choice(g.M, p=g.U[sym]))
            if rng.random() < g.pk[c] or depth >= max_depth:
                tok = g.vocab[int(rng.choice(len(g.vocab), p=g.E[c]))]
                if not on_board(sq) or sq in position or tok not in TOKENS:
                    return None
                position[sq] = tok
                return (sym, sq)
            b, d = int(rng.choice(g.K, p=g.Lt[c])), int(rng.choice(g.K, p=g.Rt[c]))
            rel = g.relations[int(rng.choice(len(g.relations), p=g.Rel[c]))]
            x = expand(b, sq, depth + 1)
            if x is None:
                return None
            dx, dy = OFFSET[rel]
            y = expand(d, (sq[0] + dx, sq[1] + dy), depth + 1)
            return None if y is None else (sym, (x, rel, y))

        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in position:
                continue
            earlier = {s: t for s, t in position.items() if s[1] * 8 + s[0] < i}
            o = int(rng.choice(g.K + 1, p=g.Q[scan_rows(earlier, self.features)[i]]))
            if o == g.K:
                continue
            node = expand(o, sq, 0)
            if node is None:
                return None
            tops.append(node)
        return position, tops



START = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}


def plausibility(position: Position) -> Dict[str, bool]:
    """Simple checks of a generated position's chess sense. The last is the
    strictest: no side has more of any kind of piece than it starts with
    (promotions are rare by move 15)."""
    count = Counter(position.values())
    return {"one king each": count["wK"] == 1 and count["bK"] == 1,
            "no pawn on a back rank": all(not (t[1] == "P" and sq[1] in (0, 7)) for sq, t in position.items()),
            "at most 8 pawns each": count["wP"] <= 8 and count["bP"] <= 8,
            "at most 16 pieces each": sum(v for k, v in count.items() if k[0] == "w") <= 16
                                       and sum(v for k, v in count.items() if k[0] == "b") <= 16,
            "no more of any kind than at the start": all(count[c + k] <= n for c in "wb" for k, n in START.items())}
