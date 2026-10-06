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
as a sentence is read word by word. At each square that no earlier chunk
covers, the read asks of each kind of piece in turn (``KINDS``) whether a
top-level element is anchored here on a piece of that kind, until one is or
none is (the square is empty). Each question is answered in the light of
the square and of how many pieces of that kind stand on earlier squares, so
the read counts material as it goes: after one white king, another is all
but impossible. The element's symbol is then drawn given the kind of its
anchor (``T``), and the element from its symbol, its anchor's kind given.
So that a chunk is decoded at its first square, relations point forward in
the scan: the N, NE, NW and E rays and the four upward knight jumps (32
relations). A chunk pays for itself when it predicts its pieces better than
the read of their squares does.

The positions are middlegame positions from the Lichess database (CC0); see
``experiments/v2/run_chess.py`` for how they are extracted into
``data/chess``.
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .grammar import Grammar, inside_of_analysis
from .mdl import dm_code, rows_nats
from .memory import Memory, chunk_attrs
from .model import Learner

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
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
COUNTS = 11      # the read counts 0..10 pieces of a kind on earlier squares (10: ten or more)


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
# What the read knows: the pieces on earlier squares
# --------------------------------------------------------------------------- #
def counts_before(position: Position) -> np.ndarray:
    """(64, kinds): how many pieces of each kind stand on the squares before
    each square, in the order of the read."""
    out = np.zeros((64, len(KINDS)), dtype=np.int64)
    placed = np.zeros(len(KINDS), dtype=np.int64)
    for i in range(64):
        out[i] = placed
        t = position.get((i % 8, i // 8))
        if t is not None:
            placed[KIND_INDEX[t]] += 1
    return out


def read_row(i: int, before: np.ndarray) -> np.ndarray:
    """Per kind, the read's row at square i given the counts ``before``."""
    return i * COUNTS + np.minimum(before, COUNTS - 1)


def left_corner(g: Grammar) -> np.ndarray:
    """LC[A, k]: the probability that a derivation from symbol A is anchored
    on a piece of kind k (its first piece is of that kind)."""
    E = np.zeros((g.M, len(KINDS)))
    for k, t in enumerate(KINDS):
        if t in g.tok_index:
            E[:, k] = g.E[:, g.tok_index[t]]
    term = g.U @ (g.pk[:, None] * E)
    step = g.U @ (g.qk[:, None] * g.Lt)
    return np.linalg.solve(np.eye(g.K) - step, term)


# --------------------------------------------------------------------------- #
# The board's memory: star context, relations, the read
# --------------------------------------------------------------------------- #
class BoardMemory(Memory):
    """Element records for positions. Elements are trees of pieces joined by
    relations; each has an anchor square and the squares of its pieces."""

    relations = RELATIONS

    def __init__(self, spine_depth: int = 2, granularities: int = 2, counts: bool = True):
        super().__init__(spine_depth=spine_depth, granularities=granularities)
        self.attrs = ["rays", "jumps", "t", "colour", "piece", "k", "rank", "file"] + chunk_attrs(spine_depth, granularities)
        # Whether the read counts the pieces on earlier squares (without, each
        # square is read on its own).
        self.counts = counts
        self.anchor: List[Square] = []
        self.anchor_kind: List[int] = []     # the kind of piece on the anchor square
        self.members: List[frozenset] = []
        self._read = self._answer_counts = self._roots = None
        self._answer_nats: Dict[float, float] = {}

    # A position is recorded with its analysis (its top-level elements),
    # drawn from the grammar square by square, and coded the same way.
    def add(self, experience: Position, analysis: Sequence[Node], weight: float = 1.0) -> None:
        """Record every element of an analysed position, bottom up."""
        position, tops = experience, analysis
        pid = len(self.experiences)
        self.experiences.append(position)
        self._read = self._answer_counts = self._roots = None
        self._answer_nats = {}

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
                self.anchor_kind.append(self.anchor_kind[a])
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
                self.anchor_kind.append(KIND_INDEX[position[body]])
                self.members.append(frozenset([body]))
            self.label.append(lab)
            self.parent.append(-1)
            self.top_left.append(-1)
            self.top_right.append(-1)
            self.is_root.append(False)
            self.weight.append(weight)
            self.experience_of.append(pid)
            self.span.append(None)
            return e

        for node in tops:
            self.is_root[record(node)] = True

    def surface(self, e: int):
        position = self.experiences[self.experience_of[e]]
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

    def contexts(self):
        """The read's context is at the top level (the counts); inside an
        element every rule choice is made the same way."""
        return None

    # The read ----------------------------------------------------------- #
    def _read_arrays(self):
        """Per square read in each position: (the square, the top-level
        element anchored there or -1 if it is empty, how many pieces of each
        kind stand on earlier squares, weight). Squares that an earlier
        chunk covers are not read."""
        if self._read is None:
            at = {}
            for e, root in enumerate(self.is_root):
                if root:
                    at[(self.experience_of[e], self.anchor[e])] = e
            weight = {self.experience_of[e]: self.weight[e] for e in at.values()}
            sq, el, cnt, w = [], [], [], []
            for pid, position in enumerate(self.experiences):
                before = counts_before(position) if self.counts else np.zeros((64, len(KINDS)), dtype=np.int64)
                for i in range(64):
                    e = at.get((pid, (i % 8, i // 8)))
                    if e is not None or (i % 8, i // 8) not in position:
                        sq.append(i)
                        el.append(-1 if e is None else e)
                        cnt.append(before[i])
                        w.append(weight.get(pid, 1.0))
            self._read = (np.array(sq, dtype=np.int64), np.array(el, dtype=np.int64),
                          np.array(cnt, dtype=np.int64).reshape(-1, len(KINDS)), np.array(w))
        return self._read

    def _answers(self) -> np.ndarray:
        """The read's answers, (kind, row, no/yes): asked of every read square
        for each kind in turn, until one is the kind of the piece anchoring an
        element there."""
        if self._answer_counts is None:
            sq, el, cnt, w = self._read_arrays()
            kind = np.array(self.anchor_kind + [len(KINDS)])[el]     # el = -1: empty
            n = np.zeros((len(KINDS), 64 * COUNTS * 2))
            for k in range(len(KINDS)):
                m = kind >= k
                row = sq[m] * COUNTS + np.minimum(cnt[m, k], COUNTS - 1)
                n[k] = np.bincount(row * 2 + (kind[m] == k), weights=w[m], minlength=64 * COUNTS * 2)
            self._answer_counts = n.reshape(len(KINDS), -1, 2)
        return self._answer_counts

    def _symbols_at(self, s: np.ndarray, K: int) -> np.ndarray:
        """(kind and square, symbol): the symbols of top-level elements, by
        the kind and the square of their anchor."""
        if self._roots is None:        # kept until an element is added
            root = np.flatnonzero(self.is_root)
            where = np.array([self.anchor[e][1] * 8 + self.anchor[e][0] for e in root], dtype=np.int64)
            self._roots = (root, np.array(self.anchor_kind)[root] * 64 + where, np.array(self.weight)[root])
        root, row, w = self._roots
        n = np.zeros((len(KINDS) * 64, K))
        np.add.at(n, (row, s[root]), w)
        return n

    def top_level_nats(self, s: np.ndarray, K: int, alpha: float) -> float:
        """The read's answers, then each top-level element's symbol given the
        kind and the square of its anchor. The answers do not depend on the
        symbols, so their code is kept until an element is added."""
        if alpha not in self._answer_nats:
            self._answer_nats[alpha] = rows_nats(self._answers().reshape(-1, 2), alpha)
        if self._roots is None:
            self._symbols_at(s, K)
        root, row, w = self._roots
        # One row per kind and square, coded over its seen symbols only (the
        # table is sparse: most symbols never anchor a top-level element).
        return self._answer_nats[alpha] + dm_code(row, s[root], w, K, alpha)

    def layout_nats(self, alpha: float) -> float:
        """A board's read codes every square, so nothing is left to code."""
        return 0.0

    def top_level_tables(self, s: np.ndarray, K: int, alpha: float):
        """The read's tables Q[kind, square, count] = P(yes) and T[kind,
        square, symbol], and the symbols of top-level elements (S)."""
        root = np.array(self.is_root, dtype=bool)
        n_top = np.bincount(s[root], weights=np.array(self.weight)[root], minlength=K)
        answers, n_T = self._answers(), self._symbols_at(s, K)
        Q = ((answers[..., 1] + alpha) / (answers.sum(axis=-1) + 2 * alpha)).reshape(len(KINDS), 64, COUNTS)
        T = ((n_T + alpha) / (n_T + alpha).sum(axis=1, keepdims=True)).reshape(len(KINDS), 64, K)
        return ({"S": (n_top + alpha) / (n_top + alpha).sum(), "Q": Q, "T": T},
                [(n, 2) for n in answers] + [(n_T, K)])

    def _asked(self, g: Grammar, i: int, before: np.ndarray) -> np.ndarray:
        """P(yes) of each kind's question at square i."""
        return g.Q[np.arange(len(KINDS)), i, np.minimum(before, COUNTS - 1) if self.counts else 0]

    def sample(self, grammar, rng: np.random.Generator, max_depth: int = 12, **kw):
        """A position read off the grammar square by square: (position,
        top-level elements), or None if a chunk would put a piece off the
        board or on an occupied square."""
        g = grammar
        lc = left_corner(g)
        position: Position = {}
        tops: List[Node] = []

        def expand(sym: int, sq: Square, depth: int, kind: Optional[int] = None):
            # ``kind``: the kind of piece this element must be anchored on.
            if kind is None:
                c = int(rng.choice(g.M, p=g.U[sym]))
                prim = rng.random() < g.pk[c]
            else:
                t = g.tok_index.get(KINDS[kind])
                if t is None:            # a kind never seen in training
                    return None
                as_piece, as_chunk = g.pk * g.E[:, t], g.qk * (g.Lt @ lc[:, kind])
                p = g.U[sym] * (as_piece + as_chunk)
                c = int(rng.choice(g.M, p=p / p.sum()))
                prim = rng.random() < as_piece[c] / (as_piece[c] + as_chunk[c])
            if prim or depth >= max_depth:
                tok = (g.vocab[int(rng.choice(len(g.vocab), p=g.E[c]))] if kind is None
                       else KINDS[kind])
                if not on_board(sq) or sq in position or tok not in TOKENS:
                    return None
                position[sq] = tok
                return (sym, sq)
            if kind is None:
                b = int(rng.choice(g.K, p=g.Lt[c]))
            else:
                p = g.Lt[c] * lc[:, kind]
                b = int(rng.choice(g.K, p=p / p.sum()))
            d = int(rng.choice(g.K, p=g.Rt[c]))
            rel = g.relations[int(rng.choice(len(g.relations), p=g.Rel[c]))]
            x = expand(b, sq, depth + 1, kind)
            if x is None:
                return None
            dx, dy = OFFSET[rel]
            y = expand(d, (sq[0] + dx, sq[1] + dy), depth + 1)
            return None if y is None else (sym, (x, rel, y))

        before = np.zeros(len(KINDS), dtype=np.int64)    # pieces on the squares read so far
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq not in position:
                for k, yes in enumerate(self._asked(g, i, before)):
                    if rng.random() < yes:
                        node = expand(int(rng.choice(g.K, p=g.T[k, i])), sq, 0, k)
                        if node is None:
                            return None
                        tops.append(node)
                        break
            if sq in position and self.counts:
                before[KIND_INDEX[position[sq]]] += 1
        return position, tops

    def log_prob(self, grammar, experience, analysis) -> float:
        """ln P(position) under the grammar, given its analysis (its top-level
        elements): the read square by square; at an element's anchor, its
        symbol given the anchor's kind and the inside pass of the element
        given its symbol and that kind; squares that an earlier chunk covers
        are not read."""
        g, position = grammar, experience
        lc = left_corner(g)
        anchored = {anchor_of(n): n for n in analysis}
        before = counts_before(position) if self.counts else np.zeros((64, len(KINDS)), dtype=np.int64)
        lp = 0.0
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in anchored or sq not in position:
                yes = self._asked(g, i, before[i])
                if sq in anchored:
                    k = KIND_INDEX[position[sq]]
                    lp += np.log1p(-yes[:k]).sum() + np.log(yes[k])
                    scale, v = inside_of_analysis(
                        g, anchored[sq], parts=lambda n: (n[1][0], n[1][2]) if is_chunk(n) else None,
                        token=lambda n: position[n[1]], relation=lambda n: n[1][1])
                    lp += scale + np.log(g.T[k, i] @ (v / lc[:, k]))
                else:
                    lp += np.log1p(-yes).sum()
        return float(lp)


# --------------------------------------------------------------------------- #
# The structure search: chunk moves under the read's code
# --------------------------------------------------------------------------- #
def _phi(c: float, a: float) -> float:
    return math.lgamma(c + a) - math.lgamma(a) if c else 0.0


class BoardSearch:
    """Greedy search over chunk moves, scored exactly by the plain code: the
    read's answers (at each square, is an element anchored here on a piece
    of this kind? given the square and the count of that kind on earlier
    squares), the label of each top-level element given its anchor's kind,
    and each label's definition (a piece, or (B, relation, C)). A chunk move
    (B, relation, C) joins, in every position, the top-level elements
    labelled B and C whose anchors stand in that relation (each element at
    most once, in scan order); its second part's square is then no longer
    read."""

    def __init__(self, positions: Sequence[Position], alpha: float = 0.001, counts: bool = True):
        self.a = alpha
        self.counts = counts
        self.V = len({t for p in positions for t in p.values()}) + 1
        self.NR = len(RELATIONS)
        self.rows: Dict[Hashable, Counter] = defaultdict(Counter)    # label -> definition
        self.kind_of: Dict[Hashable, int] = {t: KIND_INDEX[t] for t in KINDS}
        self.top: Dict[tuple, Counter] = defaultdict(Counter)       # (anchor kind, square) -> labels
        self.read = np.zeros((len(KINDS), 64 * COUNTS, 2))           # the read's answers
        self.pos = []
        self.fresh = 0
        self.moves: List[tuple] = []      # (B, relation, C, Y) in the order applied
        for position in positions:
            before = counts_before(position) if counts else np.zeros((64, len(KINDS)), dtype=np.int64)
            tops = {}
            for i in range(64):
                sq = (i % 8, i // 8)
                if sq in position:
                    t = position[sq]
                    tops[sq] = (t, sq)
                    self.rows[t][("w", t)] += 1
                    self.top[(KIND_INDEX[t], i)][t] += 1
                self._ask(self.read, i, before[i], KIND_INDEX.get(position.get(sq), len(KINDS)), 1)
            self.pos.append({"position": position, "tops": tops, "owner": {sq: sq for sq in position},
                             "before": before})

    @staticmethod
    def _ask(read: np.ndarray, i: int, before: np.ndarray, stop: int, sign: float) -> None:
        """Add (sign 1) or remove (sign -1) the read's answers at square i,
        where the element anchored is on a piece of kind ``stop`` (or none:
        len(KINDS))."""
        rows = read_row(i, before)
        for k in range(min(stop + 1, len(KINDS))):
            read[k, rows[k], int(k == stop)] += sign

    def _row_nats(self, K: int) -> float:
        A = (self.V + K * K * self.NR) * self.a
        return sum(math.lgamma(sum(r.values()) + A) - math.lgamma(A) - sum(_phi(c, self.a) for c in r.values())
                   for r in self.rows.values())

    def _top_nats(self, rows: Iterable[Counter], K: int) -> float:
        A = K * self.a
        return sum(math.lgamma(sum(r.values()) + A) - math.lgamma(A) - sum(_phi(c, self.a) for c in r.values())
                   for r in rows if r)

    def bits(self) -> float:
        K = len(self.rows)
        return (rows_nats(self.read.reshape(-1, 2), self.a) + self._top_nats(self.top.values(), K)
                + self._row_nats(K)) / math.log(2)

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

    def _changes(self, key: tuple, chosen: List[tuple], read: np.ndarray, top) -> None:
        """Apply a chunk move's changes to the read's answers and the labels
        of top-level elements (in place; ``top(kind, square)`` gives the
        labels' row to change)."""
        B, _, C = key
        Y = ("chunk", self.fresh)
        kB, kC = self.kind_of[B], self.kind_of[C]
        for pi, s, t in chosen:
            p = self.pos[pi]
            i, j = s[1] * 8 + s[0], t[1] * 8 + t[0]
            self._ask(read, j, p["before"][j], kC, -1)       # the second part's square is no longer read
            top(kB, i)[B] -= 1
            top(kB, i)[Y] += 1
            top(kC, j)[C] -= 1

    def score(self, key: tuple, chosen: List[tuple], base_rows: Dict[int, float]) -> float:
        """Bits after the chunk move (exact)."""
        a, K = self.a, len(self.rows) + 1
        n = len(chosen)
        A = (self.V + K * K * self.NR) * a
        if K not in base_rows:
            base_rows[K] = self._row_nats(K)
        nats = base_rows[K] + math.lgamma(n + A) - math.lgamma(A) - _phi(n, a)
        if ("top", K) not in base_rows:
            base_rows[("top", K)] = self._top_nats(self.top.values(), K)
        read = self.read.copy()
        changed: Dict[tuple, Counter] = {}
        self._changes(key, chosen, read,
                      lambda k, i: changed.setdefault((k, i), Counter(self.top.get((k, i), {}))))
        nats += (rows_nats(read.reshape(-1, 2), a) + base_rows[("top", K)]
                 - self._top_nats([self.top[r] for r in changed if r in self.top], K)
                 + self._top_nats([Counter({lab: v for lab, v in r.items() if v}) for r in changed.values()], K))
        return nats / math.log(2)

    def apply(self, key: tuple, chosen: List[tuple]) -> Hashable:
        B, rel, C = key
        Y = ("chunk", self.fresh)
        self._changes(key, chosen, self.read, lambda k, i: self.top[(k, i)])
        self.top = defaultdict(Counter, {r: Counter({lab: v for lab, v in c.items() if v})
                                         for r, c in self.top.items()})
        self.kind_of[Y] = self.kind_of[B]
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
        (posterior predictive), and the same with no chunks (each square read
        alone, from the training positions' counts)."""
        analysed = BoardSearch(positions, self.a, self.counts)
        analysed.replay(self.moves)
        flat = BoardSearch([p["position"] for p in self.pos], self.a, self.counts)
        return (self.code_of(analysed) / len(positions),
                flat.code_of(BoardSearch(positions, self.a, self.counts)) / len(positions))

    def code_of(self, other: "BoardSearch") -> float:
        """Bits of another set of analyses under this search's counts."""
        a, K = self.a, len(self.rows)
        mine = self.read
        p_yes = (mine[..., 1] + a) / (mine.sum(axis=-1) + 2 * a)
        bits = -float(np.sum(other.read[..., 1] * np.log2(p_yes) + other.read[..., 0] * np.log2(1 - p_yes)))
        for r, theirs in other.top.items():
            row = self.top.get(r, Counter())
            n = sum(row.values())
            for lab, m in theirs.items():
                bits -= m * math.log2((row.get(lab, 0) + a) / (n + K * a))
        A = (self.V + K * K * self.NR) * a
        for lab, row in other.rows.items():
            mine_row = self.rows.get(lab, Counter())
            n = sum(mine_row.values())
            for o, m in row.items():
                bits -= m * math.log2((mine_row.get(o, 0) + a) / (n + A))
        return bits


# --------------------------------------------------------------------------- #
# Learning, and sampling positions from the grammar
# --------------------------------------------------------------------------- #
class ChessLearner(Learner):
    """Positions by day and by night: by night the structure search proposes
    chunks, and the positions so analysed are consolidated into the two
    hierarchies; a new position is analysed by replaying the chunk moves."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, max_steps: int = 500,
                 counts: bool = True):
        super().__init__(seed=seed, alpha=alpha)
        self.max_steps = max_steps
        # Whether the read counts the pieces on earlier squares (without, each
        # square is read on its own).
        self.counts = counts
        self.search: Optional[BoardSearch] = None

    def sleep(self) -> Grammar:
        """Learn from every position observed so far: the structure search,
        then consolidation into the two hierarchies."""
        t0 = time.time()
        search = BoardSearch(self.experiences, self.alpha, self.counts)
        self.history.append({"stage": "flat", "bits": search.bits(), "seconds": 0.0})

        def log(step, key, n, bits):
            self.history.append({"stage": "chunk", "move": f"[{key[0]} {key[1]} {key[2]}] x{n}",
                                 "bits": bits, "seconds": time.time() - t0})
        search.run(self.max_steps, log)
        self.search = search
        self.analyses = search.analyses()
        self.model = self.fit(self.analyses, BoardMemory(counts=self.counts))
        g = self.model.grammar
        self.history.append({"stage": "consolidate", "bits": g.info["total bits"],
                             "seconds": time.time() - t0})
        return g

    def analyse(self, position: Position) -> List[Node]:
        """A position's top-level elements: the learned chunk moves replayed."""
        search = BoardSearch([position], self.alpha, self.counts)
        search.replay(self.search.moves)
        return search.analyses()[0]


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
