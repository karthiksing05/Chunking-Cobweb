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

**The read's context** (``context=True``, the default; ``context=False``
codes each symbol given anchor kind and square alone, as before 2026-10-11). The one choice the read leaves to
the grammar, the symbol of each top-level element, is coded in the light of
what the read has already placed around its anchor, as a sentence's rule
choices are coded given the words read before them: the backward half of the
star (the first piece along the S, W, SW and SE rays and on the four backward
knight squares) as a chain of refinements (``CHAIN``), with the back-off
code of ``mdl.backoff_chain_coder``; description length chooses the depth and
weight (``CONTEXT_OPTIONS``). The search scores its moves under the same code
(``ContextBoardSearch``).

**The search** (``BoardSearch``) has chunk moves and, with ``merges=True``,
the merge moves of the shared sequence search, every move scored exactly.

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

from .grammar import CONTEXT_WEIGHTS, Grammar, inside_of_analysis
from .mdl import _before, backoff_chain_coder, dm_code, rows_nats
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
LN2 = math.log(2.0)


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


# --------------------------------------------------------------------------- #
# Typed relations: attack and defence
# --------------------------------------------------------------------------- #
# Beyond direction and distance, the domain supplies two relation types (as
# characters supply their operators): two pieces on a line the board's
# relations see (the first piece along a ray, or a knight's jump) *defend*
# each other if either could capture onto the other's square and they are of
# one colour, and *attack* if they are of different colours. A typed chunk
# move joins every such pair whatever its direction and distance; each
# instance keeps its geometric relation, so a typed chunk is a category of
# geometric ones (Chase and Simon's chunks: attack, defence, same colour).
TACTICAL = ("attacks", "defends")
REL_OF = {v: k for k, v in OFFSET.items()}


def hits(token: str, dx: int, dy: int) -> bool:
    """Whether a piece could capture onto the square (dx, dy) away, the line
    between them being clear."""
    k, ax, ay = token[1], abs(dx), abs(dy)
    if k == "N":
        return (ax, ay) in ((1, 2), (2, 1))
    if k == "K":
        return max(ax, ay) == 1
    if k == "R":
        return dx == 0 or dy == 0
    if k == "B":
        return ax == ay
    if k == "Q":
        return dx == 0 or dy == 0 or ax == ay
    return ax == 1 and dy == (1 if token[0] == "w" else -1)       # a pawn


def interaction(a: str, c: str, rel: str) -> Optional[str]:
    """The tactical type of two pieces, c standing in relation ``rel`` from a."""
    dx, dy = OFFSET[rel]
    if hits(a, dx, dy) or hits(c, -dx, -dy):
        return "defends" if a[0] == c[0] else "attacks"
    return None


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

    def __init__(self, spine_depth: int = 2, granularities: int = 2, counts: bool = True,
                 context: bool = True, read_context: bool = False, context_of: str = "pieces"):
        super().__init__(spine_depth=spine_depth, granularities=granularities)
        # What a context value is: the piece in that direction ("pieces"), or
        # the symbol of the top-level element holding it ("elements": a chunk
        # is one value; the top-level symbols are then coded explicitly).
        if context_of not in ("pieces", "elements"):
            raise ValueError(context_of)
        self.context_of = context_of
        # Whether the read's yes-or-no answers are also coded in that context
        # (backing off to the counted row).
        self.read_context = read_context
        self._rctx = None
        self._read_fitted: Dict[float, tuple] = {}
        # Whether each top-level symbol is coded in the read's context
        # (``CHAIN``, as a sentence's rule choices are coded given the words
        # before them), or given its anchor's kind and square alone.
        self.context = context
        self._ctx = self._coders = None
        self._fitted: Dict[int, tuple] = {}
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
        self._ctx = self._coders = None
        self._fitted = {}
        self._rctx = None
        self._read_fitted = {}

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
            sq, el, cnt, w, pids = [], [], [], [], []
            for pid, position in enumerate(self.experiences):
                before = counts_before(position) if self.counts else np.zeros((64, len(KINDS)), dtype=np.int64)
                for i in range(64):
                    e = at.get((pid, (i % 8, i // 8)))
                    if e is not None or (i % 8, i // 8) not in position:
                        sq.append(i)
                        el.append(-1 if e is None else e)
                        cnt.append(before[i])
                        w.append(weight.get(pid, 1.0))
                        pids.append(pid)
            self._read_pids = np.array(pids, dtype=np.int64)
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
            self._answer_nats[alpha] = (self.answers_nats(alpha)[0] + CHOICE_NATS if self.read_context
                                        else rows_nats(self._answers().reshape(-1, 2), alpha))
        if self.context:
            nats, _ = self.symbols_nats(s[np.flatnonzero(self.is_root)], K, alpha, s_all=s)
            return self._answer_nats[alpha] + nats + CHOICE_NATS
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

    # The top level in context ------------------------------------------- #
    def _context_arrays(self):
        """Per top-level element in reading order: its group (anchor kind and
        square), its context ids level by level, its weight; and a back-off
        coder per depth."""
        if self._ctx is None:
            root = np.flatnonzero(self.is_root)
            self.ctx_ids = ContextIds()
            vals = [self.ctx_ids.ids(chain_values(self.experiences[self.experience_of[e]], self.anchor[e]))
                    for e in root]
            g = np.array([self.anchor_kind[e] * 64 + self.anchor[e][1] * 8 + self.anchor[e][0] for e in root],
                         dtype=np.int64)
            c = np.array(vals, dtype=np.int64).reshape(-1, len(CHAIN))
            w = np.array(self.weight)[root]
            self._ctx = (root, g, [c[:, l] for l in range(len(CHAIN))], w)
            self._coders = {d: backoff_chain_coder(g, [c[:, l] for l in range(d)], w)
                            for d in range(len(CHAIN) + 1)}
            if self.context_of == "elements":
                # Per top-level element and backward direction: the top-level
                # element holding the piece seen there, or EDGE (-1) / EMPTY (-2).
                holder = {}
                for e in root:
                    for m in self.members[e]:
                        holder[(self.experience_of[e], m)] = e
                O = np.empty((len(root), len(BACK_DIRS)), dtype=np.int64)
                for n, e in enumerate(root):
                    pid = self.experience_of[e]
                    b = backward_squares(self.experiences[pid], self.anchor[e])
                    for j, d in enumerate(BACK_DIRS):
                        v = b[d]
                        O[n, j] = holder[(pid, v)] if isinstance(v, tuple) else (-1 if v == EDGE else -2)
                self._holder = O
        return self._ctx

    def _element_contexts(self, s_all: np.ndarray) -> List[np.ndarray]:
        """Context ids level by level when the context is made of the
        symbols of neighbouring top-level elements (under symbols ``s_all``)."""
        O = self._holder
        V = np.where(O >= 0, np.asarray(s_all)[np.maximum(O, 0)], O)
        out = []
        for dirs in CHAIN:
            cols = [BACK_DIRS.index(d) for d in dirs]
            _, inv = np.unique(V[:, cols], axis=0, return_inverse=True)
            out.append(inv.reshape(-1).astype(np.int64))
        return out

    def symbols_nats(self, s_root: np.ndarray, K: int, alpha: float,
                     s_all: Optional[np.ndarray] = None) -> Tuple[float, tuple]:
        """The top-level symbols' code in context (nats), the best of
        ``CONTEXT_OPTIONS``, and that option. With a context made of
        elements, ``s_all`` (every element's symbol) gives the neighbours'."""
        root, g, _, w = self._context_arrays()
        if self.context_of == "elements":
            c = self._element_contexts(s_all)
            coders = {d: backoff_chain_coder(g, c[:d], w) for d in range(len(CHAIN) + 1)}
        else:
            coders = self._coders
        best = None
        for d, b in CONTEXT_OPTIONS:
            n = coders[d](s_root, K, alpha, [b] * d)
            if best is None or n < best[0] - 1e-12:
                best = (n, (d, b))
        return best

    def _predictor(self, g: Grammar) -> tuple:
        """The training symbols' counts for the back-off predictive, under
        the option description length chose for this grammar."""
        if id(g) not in self._fitted:
            root, grp, c, w = self._context_arrays()
            s_all = np.asarray(g.elem_symbol)
            s = s_all[root]
            _, (d, beta) = self.symbols_nats(s, g.K, g.alpha, s_all=s_all)
            n_g = np.zeros((len(KINDS) * 64, g.K))
            np.add.at(n_g, (grp, s), w)
            if self.context_of == "elements":
                # Keyed by the context values themselves (neighbours' symbols).
                O = self._holder
                V = np.where(O >= 0, s_all[np.maximum(O, 0)], O)
                c = [[tuple(int(V[n, BACK_DIRS.index(dd)]) for dd in dirs) for n in range(len(root))]
                     for dirs in CHAIN]
            n_gx = []
            for l in range(d):
                t: Dict[tuple, np.ndarray] = {}
                for gg, cc, ss, ww in zip(grp, c[l], s, w):
                    key = (int(gg), cc if isinstance(cc, tuple) else int(cc))
                    t.setdefault(key, np.zeros(g.K))[ss] += ww
                n_gx.append(t)
            self._fitted[id(g)] = (d, beta, n_g, n_gx)
        return self._fitted[id(g)]

    def top_row(self, g: Grammar, k: int, i: int, position: Position, held: Optional[dict] = None) -> np.ndarray:
        """P(symbol | anchor kind k, square i, and, in context, what the read
        has placed around it in ``position``; with a context made of
        elements, ``held[square]`` is the symbol of the top-level element
        holding the piece there)."""
        if not self.context:
            return g.T[k, i]
        d, beta, n_g, n_gx = self._predictor(g)
        grp = k * 64 + i
        p = (n_g[grp] + g.alpha) / (n_g[grp].sum() + g.K * g.alpha)
        if d:
            sq = (i % 8, i // 8)
            if self.context_of == "elements":
                vals = chain_values(position, sq, value_of=lambda q: int(held[q]))
                vals = [tuple(-1 if v == EDGE else -2 if v == EMPTY else v for v in lev) for lev in vals]
                ids = vals
            else:
                ids = self.ctx_ids.ids(chain_values(position, sq), grow=False)
            for l in range(d):
                r = n_gx[l].get((grp, ids[l])) if (isinstance(ids[l], tuple) or ids[l] >= 0) else None
                if r is not None:
                    p = (r + beta * p) / (r.sum() + beta)
        return p

    # The read's answers in context -------------------------------------- #
    def _read_events(self):
        """Every question the read asks, in reading order (position, square,
        kind): its row (kind, square, count of that kind on earlier squares),
        its answer, the context ids of its square level by level, its
        weight; and a back-off coder per depth."""
        if self._rctx is None:
            sq, el, cnt, w = self._read_arrays()
            pid = self._read_pids
            kind = np.array(self.anchor_kind + [len(KINDS)])[el]
            self.read_ids = ContextIds()
            ctx = [self.read_ids.ids(chain_values(self.experiences[p], (i % 8, i // 8))) for p, i in zip(pid, sq)]
            ctx = np.array(ctx, dtype=np.int64).reshape(-1, len(CHAIN))
            n_q = np.minimum(kind + 1, len(KINDS))            # questions asked at each read square
            at = np.repeat(np.arange(len(sq)), n_q)
            k = np.concatenate([np.arange(n) for n in n_q]) if len(n_q) else np.zeros(0, np.int64)
            cnt_k = np.minimum(cnt[at, k], COUNTS - 1) if self.counts else np.zeros(len(at), np.int64)
            groups = k * (64 * COUNTS) + sq[at] * COUNTS + cnt_k
            keys = (k == kind[at]).astype(np.int64)
            contexts = [ctx[at, l] for l in range(len(CHAIN))]
            weights = w[at]
            coders = {d: backoff_chain_coder(groups, contexts[:d], weights) for d in range(len(CHAIN) + 1)}
            self._rctx = (groups, keys, contexts, weights, coders)
        return self._rctx

    def answers_nats(self, alpha: float) -> Tuple[float, tuple]:
        """The read's answers coded in context (nats; the best of
        ``CONTEXT_OPTIONS``) and that option. Depth 0 is the counted read."""
        groups, keys, contexts, weights, coders = self._read_events()
        best = None
        for d, b in CONTEXT_OPTIONS:
            n = coders[d](keys, 2, alpha, [b] * d)
            if best is None or n < best[0] - 1e-12:
                best = (n, (d, b))
        return best

    def _read_predictor(self, alpha: float) -> tuple:
        if alpha not in self._read_fitted:
            groups, keys, contexts, weights, _ = self._read_events()
            _, (d, beta) = self.answers_nats(alpha)
            n = np.bincount(groups, weights=weights, minlength=len(KINDS) * 64 * COUNTS)
            yes = np.bincount(groups, weights=weights * keys, minlength=len(KINDS) * 64 * COUNTS)
            levels = []
            for l in range(d):
                t: Dict[tuple, list] = {}
                for gg, xx, kk, ww in zip(groups, contexts[l], keys, weights):
                    c = t.setdefault((int(gg), int(xx)), [0.0, 0.0])
                    c[0] += ww
                    c[1] += ww * kk
                levels.append(t)
            self._read_fitted[alpha] = (d, beta, n, yes, levels)
        return self._read_fitted[alpha]

    def _asked(self, g: Grammar, i: int, before: np.ndarray, position: Optional[Position] = None) -> np.ndarray:
        """P(yes) of each kind's question at square i (in context: given what
        the read has placed around it in ``position``)."""
        if not self.read_context:
            return g.Q[np.arange(len(KINDS)), i, np.minimum(before, COUNTS - 1) if self.counts else 0]
        d, beta, n, yes, levels = self._read_predictor(g.alpha)
        rows = np.arange(len(KINDS)) * (64 * COUNTS) + i * COUNTS + (np.minimum(before, COUNTS - 1)
                                                                      if self.counts else 0)
        p = (yes[rows] + g.alpha) / (n[rows] + 2 * g.alpha)
        if d:
            self._read_events()
            ids = self.read_ids.ids(chain_values(position, (i % 8, i // 8)), grow=False)
            for l in range(d):
                if ids[l] < 0:
                    continue
                for k, r in enumerate(rows):
                    c = levels[l].get((int(r), ids[l]))
                    if c is not None:
                        p[k] = (c[1] + beta * p[k]) / (c[0] + beta)
        return p

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
        held: Dict[Square, int] = {}                     # square -> symbol of the element holding it
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq not in position:
                for k, yes in enumerate(self._asked(g, i, before, position)):
                    if rng.random() < yes:
                        T = self.top_row(g, k, i, position, held)
                        node = expand(int(rng.choice(g.K, p=T / T.sum())), sq, 0, k)
                        if node is None:
                            return None
                        tops.append(node)
                        for m in _squares_of(node):
                            held[m] = node[0]
                        break
            if sq in position and self.counts:
                before[KIND_INDEX[position[sq]]] += 1
        return position, tops

    def log_prob(self, grammar, experience, analysis, chosen: Optional[list] = None) -> float:
        """ln P(position) under the grammar, given its analysis (its top-level
        elements): the read square by square; at an element's anchor, its
        symbol given the anchor's kind and the inside pass of the element
        given its symbol and that kind; squares that an earlier chunk covers
        are not read. With a context made of elements, ``chosen`` (a list)
        receives the top-level symbols coded, in reading order."""
        g, position = grammar, experience
        lc = left_corner(g)
        anchored = {anchor_of(n): n for n in analysis}
        before = counts_before(position) if self.counts else np.zeros((64, len(KINDS)), dtype=np.int64)
        held: Dict[Square, int] = {}         # square -> symbol of the top-level element holding it
        if self.context and self.context_of == "elements":
            self._predictor(g)
        lp = 0.0
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in anchored or sq not in position:
                yes = self._asked(g, i, before[i], position)
                if sq in anchored:
                    k = KIND_INDEX[position[sq]]
                    lp += np.log1p(-yes[:k]).sum() + np.log(yes[k])
                    scale, v = inside_of_analysis(
                        g, anchored[sq], parts=lambda n: (n[1][0], n[1][2]) if is_chunk(n) else None,
                        token=lambda n: position[n[1]], relation=lambda n: n[1][1])
                    if self.context and self.context_of == "elements":
                        # The top-level symbol is coded explicitly (the
                        # neighbours' symbols are the context of later ones).
                        # An analysis that names grammar symbols keeps them;
                        # one that names the search's labels (a new position)
                        # takes, element by element in reading order, the
                        # symbol that codes it shortest: an encoder's choice
                        # that the decoder reads, so the code stays exact.
                        T = self.top_row(g, k, i, position, held)
                        joint = T * v / lc[:, k]
                        lab = anchored[sq][0]
                        sym = int(lab) if isinstance(lab, (int, np.integer)) else int(np.argmax(joint))
                        lp += scale + np.log(joint[sym])
                        if chosen is not None:
                            chosen.append(sym)
                        for m in _squares_of(anchored[sq]):
                            held[m] = sym
                    else:
                        lp += scale + np.log(self.top_row(g, k, i, position) @ (v / lc[:, k]))
                else:
                    lp += np.log1p(-yes).sum()
        return float(lp)


# --------------------------------------------------------------------------- #
# The read's context: what the read has already placed around a square
# --------------------------------------------------------------------------- #
# The backward half of the star, as a chain of refinements (as a sentence's
# context is the word before, then the two words before): the first piece
# along the S ray; S and W; S, W, SW and SE; and the four backward knight
# squares. Every square these see comes earlier in the read.
BACK_RAYS = ("S", "W", "SW", "SE")
BACK_JUMPS = ("SSW", "SSE", "WSW", "ESE")
CHAIN = (("S",), ("S", "W"), ("S", "W", "SW", "SE"), BACK_RAYS + BACK_JUMPS)
# Description length chooses the depth (0: no context) and one weight beta.
CONTEXT_OPTIONS = [(0, 0.0)] + [(d, b) for d in range(1, len(CHAIN) + 1) for b in CONTEXT_WEIGHTS]
CHOICE_NATS = math.log(len(CONTEXT_OPTIONS))
_DIRECTION = dict(RAYS)
_DIRECTION.update(JUMPS)


BACK_DIRS = BACK_RAYS + BACK_JUMPS


def _squares_of(node: Node) -> List[Square]:
    if is_chunk(node):
        return _squares_of(node[1][0]) + _squares_of(node[1][2])
    return [node[1]]


def backward_squares(position: Position, sq: Square) -> Dict[str, object]:
    """Per backward direction, the square of the first piece along the ray
    (or on the knight square), else EDGE or EMPTY. Every such square comes
    earlier in the read."""
    out: Dict[str, object] = {}
    for d in BACK_RAYS:
        dx, dy = _DIRECTION[d]
        out[d] = EDGE
        f, r = sq[0] + dx, sq[1] + dy
        while 0 <= f < 8 and 0 <= r < 8:
            if (f, r) in position:
                out[d] = (f, r)
                break
            f, r = f + dx, r + dy
    for d in BACK_JUMPS:
        dx, dy = _DIRECTION[d]
        t = (sq[0] + dx, sq[1] + dy)
        out[d] = EDGE if not on_board(t) else (t if t in position else EMPTY)
    return out


def backward(position: Position, sq: Square) -> Dict[str, str]:
    """The first piece along each backward ray from ``sq`` and the piece on
    each backward knight square: what the read has placed around it."""
    return {d: (position[v] if isinstance(v, tuple) else v) for d, v in backward_squares(position, sq).items()}


def chain_values(position: Position, sq: Square, value_of=None) -> Tuple[tuple, ...]:
    """The context of a top-level element anchored at ``sq``, level by level:
    what stands in each backward direction. By default the piece there;
    ``value_of(square)`` gives instead the category of the top-level element
    that holds that piece (context made of elements: a chunk is one value)."""
    b = backward_squares(position, sq)
    val = (lambda v: position[v]) if value_of is None else value_of
    b = {d: (val(v) if isinstance(v, tuple) else v) for d, v in b.items()}
    return tuple(tuple(b[d] for d in dirs) for dirs in CHAIN)


class ContextIds:
    """Integer ids of context values, one table per level."""

    def __init__(self):
        self.tables: List[dict] = [dict() for _ in CHAIN]

    def ids(self, values: Sequence[tuple], grow: bool = True) -> List[int]:
        out = []
        for t, v in zip(self.tables, values):
            if v not in t:
                if not grow:
                    out.append(-1)
                    continue
                t[v] = len(t)
            out.append(t[v])
        return out


def context_nats(groups: np.ndarray, contexts: Sequence[np.ndarray], keys: np.ndarray, K: int,
                 alpha: float, option: tuple) -> float:
    """The prequential back-off code (nats; ``mdl.backoff_chain_coder``) of
    ``keys`` in their groups and contexts under one option (depth, beta)."""
    if len(keys) == 0:
        return 0.0
    d, beta = option
    return backoff_chain_coder(groups, list(contexts[:d]), np.ones(len(keys)))(keys, K, alpha, [beta] * d)


def backoff_terms(groups: np.ndarray, contexts: Sequence[np.ndarray], keys: np.ndarray, weights: np.ndarray,
                  alphabet: int, alpha: float, beta: float) -> np.ndarray:
    """Each event's share (nats) of ``mdl.backoff_chain_coder``'s code, at
    every depth 0..len(contexts) with weight ``beta`` at every level:
    row d summed is that coder's code at depth d (the same predictive, kept
    per event so that a search can recompute the groups a move touches)."""
    g, k, span = groups.astype(np.int64), keys.astype(np.int64), np.int64(alphabet)
    p = (_before(g * span + k, weights) + alpha) / (_before(g, weights) + alphabet * alpha)
    out = [-weights * np.log(p)]
    for x in contexts:
        x = x.astype(np.int64)
        gx = g * np.int64(x.max() + 1) + x
        p = (_before(gx * span + k, weights) + beta * p) / (_before(gx, weights) + beta)
        out.append(-weights * np.log(p))
    return np.array(out)


# --------------------------------------------------------------------------- #
# The structure search: chunk and merge moves under the read's code
# --------------------------------------------------------------------------- #
def _phi(c: float, a: float) -> float:
    return math.lgamma(c + a) - math.lgamma(a) if c else 0.0


def _row_term(row: Counter, A: float, a: float) -> float:
    n = sum(row.values())
    if not n:
        return 0.0
    return math.lgamma(n + A) - math.lgamma(A) - sum(_phi(c, a) for c in row.values())


class BoardSearch:
    """Best-first search over chunk and merge moves, scored exactly by the
    plain code: the read's answers (at each square, is an element anchored
    here on a piece of this kind? given the square and the count of that
    kind on earlier squares), the label of each top-level element given its
    anchor's kind and square, and each label's definition (a piece, or
    (B, relation, C)).

    * *Chunk* (B, relation, C) joins, in every position, the top-level
      elements labelled B and C whose anchors stand in that relation (each
      element at most once, in scan order); its second part's square is
      then no longer read.
    * *Merge* (A, A') (``merges=True``), with the semantics of the shared
      sequence search (``mdl_search._State``): A' is renamed A everywhere.
      Rows A and A' pool, every row holding A' as a part is renamed (its
      outcomes may collide), every top-level row (anchor kind, square)
      pools its counts of A and A', and the number of labels drops by one.

    Each move changes a few rows, plus the number of labels on which every
    row's normaliser depends, and is scored exactly from cached sums. The
    kind of a top-level element's anchor is read from its square, so a
    category may hold labels anchored on different kinds."""

    def __init__(self, positions: Sequence[Position], alpha: float = 0.001, counts: bool = True,
                 merges: bool = False, merge_kinds: str = "any", typed: bool = False):
        self.a = alpha
        # Whether chunk moves may also join pairs by a tactical type
        # (``TACTICAL``), whatever their direction and distance.
        self.typed = typed
        self.counts = counts
        self.merges = merges
        # "any": any two labels may merge; "same": only labels anchored on
        # one and the same kind.
        self.merge_kinds = merge_kinds
        self.V = len({t for p in positions for t in p.values()}) + 1
        self.NR = len(RELATIONS)
        self.rows: Dict[Hashable, Counter] = defaultdict(Counter)    # label -> definition
        self.kind_of: Dict[Hashable, int] = {t: KIND_INDEX[t] for t in KINDS}
        self.top: Dict[tuple, Counter] = defaultdict(Counter)       # (anchor kind, square) -> labels
        self.read = np.zeros((len(KINDS), 64 * COUNTS, 2))           # the read's answers
        self.pos = []
        self.fresh = 0
        self.moves: List[tuple] = []      # (B, relation, C, Y) and ("merge", A, A'), in order
        self.alias: Dict[Hashable, Hashable] = {}
        self.parents: Dict[Hashable, set] = defaultdict(set)        # label -> rows holding it as a part
        self.top_of: Dict[Hashable, Counter] = defaultdict(Counter)  # label -> top-level rows
        self.anchor_kinds: Dict[Hashable, set] = {t: {KIND_INDEX[t]} for t in KINDS}
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
                    self.top_of[t][(KIND_INDEX[t], i)] += 1
                self._ask(self.read, i, before[i], KIND_INDEX.get(position.get(sq), len(KINDS)), 1)
            self.pos.append({"position": position, "tops": tops, "owner": {sq: sq for sq in position},
                             "before": before})

    def _like(self, positions: Sequence[Position]) -> "BoardSearch":
        """A fresh search of the same kind and settings over ``positions``."""
        return BoardSearch(positions, self.a, self.counts, self.merges, self.merge_kinds, self.typed)

    @staticmethod
    def _ask(read: np.ndarray, i: int, before: np.ndarray, stop: int, sign: float) -> None:
        """Add (sign 1) or remove (sign -1) the read's answers at square i,
        where the element anchored is on a piece of kind ``stop`` (or none:
        len(KINDS))."""
        rows = read_row(i, before)
        for k in range(min(stop + 1, len(KINDS))):
            read[k, rows[k], int(k == stop)] += sign

    # The code ----------------------------------------------------------- #
    def _row_nats(self, K: int) -> float:
        A = (self.V + K * K * self.NR) * self.a
        return sum(math.lgamma(sum(r.values()) + A) - math.lgamma(A) - sum(_phi(c, self.a) for c in r.values())
                   for r in self.rows.values())

    def _top_nats(self, rows: Iterable[Counter], K: int) -> float:
        A = K * self.a
        return sum(math.lgamma(sum(r.values()) + A) - math.lgamma(A) - sum(_phi(c, self.a) for c in r.values())
                   for r in rows if r)

    def top_level_nats(self) -> float:
        """The labels of the top-level elements given anchor kind and square."""
        return self._top_nats(self.top.values(), len(self.rows))

    def _read_nats(self) -> float:
        """The read's answers (nats): one Dirichlet-multinomial row per kind,
        square and count."""
        return rows_nats(self.read.reshape(-1, 2), self.a)

    def parts(self) -> Dict[str, float]:
        """Bits by part: the read's answers, the top-level labels, the rows."""
        return {"read": self._read_nats() / LN2,
                "top": self.top_level_nats() / LN2, "rows": self._row_nats(len(self.rows)) / LN2}

    def bits(self) -> float:
        return sum(self.parts().values())

    # Chunk moves -------------------------------------------------------- #
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
            position = p["position"]
            for sq, x in p["tops"].items():
                for rel, t in self._seen(p, sq):
                    out[(x[0], rel, p["tops"][t][0])].append((pi, sq, t))
                    if self.typed:
                        kind = interaction(position[sq], position[t], rel)
                        if kind is not None:
                            out[(x[0], kind, p["tops"][t][0])].append((pi, sq, t))
        return out

    @staticmethod
    def _relations(key: tuple, chosen: List[tuple]) -> Counter:
        """The geometric relations of a move's instances (one for a
        geometric chunk move; any for a typed one)."""
        if key[1] not in TACTICAL:
            return Counter({key[1]: len(chosen)})
        return Counter(REL_OF[(t[0] - s[0], t[1] - s[1])] for _, s, t in chosen)

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
        for pi, s, t in chosen:
            p = self.pos[pi]
            kB, kC = KIND_INDEX[p["position"][s]], KIND_INDEX[p["position"][t]]
            i, j = s[1] * 8 + s[0], t[1] * 8 + t[0]
            self._ask(read, j, p["before"][j], kC, -1)       # the second part's square is no longer read
            top(kB, i)[B] -= 1
            top(kB, i)[Y] += 1
            top(kC, j)[C] -= 1

    def _chunk_rows_nats(self, relations: Counter, base: Dict) -> float:
        """The rows' code after a chunk move (a new label whose row holds one
        outcome per geometric relation of its instances)."""
        a, K = self.a, len(self.rows) + 1
        A = (self.V + K * K * self.NR) * a
        if K not in base:
            base[K] = self._row_nats(K)
        n = sum(relations.values())
        return base[K] + math.lgamma(n + A) - math.lgamma(A) - sum(_phi(c, a) for c in relations.values())

    def score(self, key: tuple, chosen: List[tuple], base_rows: Dict) -> float:
        """Bits after the chunk move (exact)."""
        a, K = self.a, len(self.rows) + 1
        nats = self._chunk_rows_nats(self._relations(key, chosen), base_rows)
        if ("top", K) not in base_rows:
            base_rows[("top", K)] = self._top_nats(self.top.values(), K)
        read = self.read.copy()
        changed: Dict[tuple, Counter] = {}
        self._changes(key, chosen, read,
                      lambda k, i: changed.setdefault((k, i), Counter(self.top.get((k, i), {}))))
        nats += (rows_nats(read.reshape(-1, 2), a) + base_rows[("top", K)]
                 - self._top_nats([self.top[r] for r in changed if r in self.top], K)
                 + self._top_nats([Counter({lab: v for lab, v in r.items() if v}) for r in changed.values()], K))
        return nats / LN2

    def apply(self, key: tuple, chosen: List[tuple]) -> Hashable:
        B, rel, C = key
        Y = ("chunk", self.fresh)
        kinds = set()
        for pi, s, t in chosen:
            p = self.pos[pi]
            kB, kC = KIND_INDEX[p["position"][s]], KIND_INDEX[p["position"][t]]
            i, j = s[1] * 8 + s[0], t[1] * 8 + t[0]
            kinds.add(kB)
            for lab, r, d in ((B, (kB, i), -1), (Y, (kB, i), 1), (C, (kC, j), -1)):
                self.top_of[lab][r] += d
                if not self.top_of[lab][r]:
                    del self.top_of[lab][r]
        self._changes(key, chosen, self.read, lambda k, i: self.top[(k, i)])
        self.top = defaultdict(Counter, {r: Counter({lab: v for lab, v in c.items() if v})
                                         for r, c in self.top.items()})
        self.kind_of[Y] = self.kind_of.get(B)
        self.anchor_kinds[Y] = kinds
        self.parents[B].add(Y)
        self.parents[C].add(Y)
        self.fresh += 1
        self.moves.append((B, rel, C, Y))
        for pi, s, t in chosen:
            r = rel if rel not in TACTICAL else REL_OF[(t[0] - s[0], t[1] - s[1])]
            self.rows[Y][("p", B, r, C)] += 1
            p = self.pos[pi]
            x, y = p["tops"][s], p["tops"].pop(t)
            p["tops"][s] = (Y, (x, r, y))
            for m, owner in list(p["owner"].items()):
                if owner == t:
                    p["owner"][m] = s
        return Y

    # Merge moves -------------------------------------------------------- #
    def _merged_rows(self, A, B):
        """Rows A and A' (= B) pooled, and every other row holding A' as a
        part, with A' renamed A."""
        def sub(o):
            return o if o[0] == "w" else ("p", A if o[1] == B else o[1], o[2], A if o[3] == B else o[3])
        merged: Counter = Counter()
        for r in (A, B):
            for o, c in self.rows[r].items():
                merged[sub(o)] += c
        moved = {}
        for r in self.parents.get(B, ()):
            if r != A and r != B and r in self.rows:
                row: Counter = Counter()
                for o, c in self.rows[r].items():
                    row[sub(o)] += c
                moved[r] = row
        return merged, moved

    def _merge_base(self, cache: dict) -> tuple:
        """Per-step terms at K - 1 labels: every row's and every top row's
        code, and their sums."""
        if "merge" not in cache:
            K1, a = len(self.rows) - 1, self.a
            A = (self.V + K1 * K1 * self.NR) * a
            terms = {lab: _row_term(r, A, a) for lab, r in self.rows.items()}
            tops = {r: _row_term(c, K1 * a, a) for r, c in self.top.items() if sum(c.values())}
            cache["merge"] = (A, terms, sum(terms.values()), tops, sum(tops.values()))
        return cache["merge"]

    def merge_allowed(self, A, B) -> bool:
        if self.merge_kinds == "same":
            return len(self.anchor_kinds.get(A, set()) | self.anchor_kinds.get(B, set())) == 1
        return True

    def _merge_rows_nats(self, A, B, cache: dict) -> float:
        """The rows' code after merging A' = B into A."""
        a = self.a
        AA, terms, row_sum, _, _ = self._merge_base(cache)
        merged, moved = self._merged_rows(A, B)
        nats = row_sum - terms[A] - terms[B] + _row_term(merged, AA, a)
        for r, row in moved.items():
            nats += _row_term(row, AA, a) - terms[r]
        if "read" not in cache:
            cache["read"] = self._read_nats()
        return nats + cache["read"]

    def merge_score(self, A, B, cache: dict) -> float:
        """Bits after merging A' = B into A (exact)."""
        a, K1 = self.a, len(self.rows) - 1
        _, _, _, tops, top = self._merge_base(cache)
        ta, tb = self.top_of.get(A, {}), self.top_of.get(B, {})
        for r in (ta.keys() & tb.keys()):
            new = Counter(self.top[r])
            new[A] += new.pop(B)
            top += _row_term(new, K1 * a, a) - tops[r]
        return (self._merge_rows_nats(A, B, cache) + top) / LN2

    def apply_merge(self, A, B) -> None:
        merged, moved = self._merged_rows(A, B)
        children = [z for o in self.rows[B] if o[0] == "p" for z in (o[1], o[3])]
        self.rows[A] = merged
        del self.rows[B]
        for r, row in moved.items():
            self.rows[r] = row
        holders = set(self.parents.get(A, ())) | set(self.parents.get(B, ()))
        self.parents.pop(B, None)
        self.parents[A] = {A if r == B else r for r in holders}
        for z in children:
            z = A if z == B else z
            self.parents[z] = {A if r == B else r for r in self.parents.get(z, ())} | {A}
        for z in list(self.parents):
            if B in self.parents[z]:
                self.parents[z] = {A if r == B else r for r in self.parents[z]}
        for r, c in self.top_of.pop(B, Counter()).items():
            self.top[r][A] += c
            del self.top[r][B]
            self.top_of[A][r] += c
        self.anchor_kinds[A] = self.anchor_kinds.get(A, set()) | self.anchor_kinds.pop(B, set())
        self.kind_of.setdefault(A, self.kind_of.get(B))
        for p in self.pos:
            for sq, x in p["tops"].items():
                if x[0] == B:
                    p["tops"][sq] = (A, x[1])
        self.alias = {k: (A if v == B else v) for k, v in self.alias.items()}
        self.alias[B] = A
        self.moves.append(("merge", A, B))

    # The search --------------------------------------------------------- #
    def scored_moves(self):
        """(bits after the move, move) for every chunk move, and every merge
        move if merges are on."""
        base_rows: Dict = {}
        for key, pairs in self.candidates().items():
            chosen = self.select(pairs)
            if len(chosen) < 2:
                continue
            yield self.score(key, chosen, base_rows), ("chunk", key, chosen)
        if self.merges:
            cache: dict = {}
            labels = sorted(self.rows, key=str)
            for i, A in enumerate(labels):
                for B in labels[i + 1:]:
                    if self.merge_allowed(A, B):
                        yield self.merge_score(A, B, cache), ("merge", A, B)

    def do(self, move) -> None:
        if move[0] == "chunk":
            self.apply(move[1], move[2])
        else:
            self.apply_merge(move[1], move[2])

    def run(self, max_steps: int = 500, log=None, patience: int = 0) -> float:
        """Best-first search; with ``patience``, continue past steps without
        a new shortest code (Stolcke's lookahead) and return to the shortest
        found. ``log(step, move, bits)``."""
        cur = best = self.bits()
        best_moves = list(self.moves)
        stale = 0
        for step in range(max_steps):
            top = None
            for b, move in self.scored_moves():
                if top is None or b < top[0] - 1e-9:
                    top = (b, move)
            if top is None or (patience == 0 and top[0] >= cur - 1e-6):
                break
            self.do(top[1])
            cur = top[0]
            if log:
                log(step, top[1], cur)
            if cur < best - 1e-6:
                best, best_moves, stale = cur, list(self.moves), 0
            else:
                stale += 1
                if stale > patience:
                    break
        if len(self.moves) != len(best_moves):
            fresh = self._like([p["position"] for p in self.pos])
            fresh.replay(best_moves)
            self.__dict__.update(fresh.__dict__)
        return best

    def replay(self, moves: Sequence[tuple]) -> None:
        """Apply chunk moves (B, relation, C, Y) and merges ("merge", A, A')
        learned elsewhere, in order."""
        for move in moves:
            if len(move) == 3 and move[0] == "merge":
                A, B = move[1], move[2]
                if A in self.rows and B in self.rows:
                    self.apply_merge(A, B)
                elif B in self.rows:             # A unseen here: A' takes A's name
                    self.rows[A] = Counter()
                    self.anchor_kinds.setdefault(A, set())
                    self.apply_merge(A, B)
                continue
            B, rel, C, Y = move
            if rel in TACTICAL:
                pairs = [(pi, sq, t) for pi, p in enumerate(self.pos) for sq, x in p["tops"].items()
                         if x[0] == B for r, t in self._seen(p, sq)
                         if p["tops"][t][0] == C and interaction(p["position"][sq], p["position"][t], r) == rel]
            else:
                pairs = [(pi, sq, t) for pi, p in enumerate(self.pos) for sq, x in p["tops"].items()
                         if x[0] == B for r, t in self._seen(p, sq) if r == rel and p["tops"][t][0] == C]
            chosen = self.select(pairs)
            if chosen:
                self.fresh = Y[1]
                self.apply((B, rel, C), chosen)

    # Analyses and codes ------------------------------------------------- #
    def resolve(self, lab):
        return self.alias.get(lab, lab)

    def analyses(self) -> List[List[Node]]:
        def rename(node):
            lab, body = node
            if is_chunk(node):
                x, rel, y = body
                return (self.resolve(lab), (rename(x), rel, rename(y)))
            return (self.resolve(lab), body)
        return [[rename(p["tops"][sq]) for sq in sorted(p["tops"], key=lambda s: (s[1], s[0]))] for p in self.pos]

    def held_out_bits(self, positions: Sequence[Position]) -> Tuple[float, float]:
        """Bits per position for new positions: analysed by replaying the
        learned moves in order and coded with this search's counts
        (posterior predictive), and the same with no chunks (each square read
        alone, from the training positions' counts)."""
        analysed = self._like(positions)
        analysed.replay(self.moves)
        flat = self._like([p["position"] for p in self.pos])
        return self.code_of(analysed) / len(positions), flat.code_of(flat._like(positions)) / len(positions)

    def _read_and_rows_bits_of(self, other: "BoardSearch") -> float:
        a, K = self.a, len(self.rows)
        mine = self.read
        p_yes = (mine[..., 1] + a) / (mine.sum(axis=-1) + 2 * a)
        bits = -float(np.sum(other.read[..., 1] * np.log2(p_yes) + other.read[..., 0] * np.log2(1 - p_yes)))
        A = (self.V + K * K * self.NR) * a
        for lab, row in other.rows.items():
            mine_row = self.rows.get(lab, Counter())
            n = sum(mine_row.values())
            for o, m in row.items():
                bits -= m * math.log2((mine_row.get(o, 0) + a) / (n + A))
        return bits

    def code_of(self, other: "BoardSearch") -> float:
        """Bits of another set of analyses under this search's counts."""
        a, K = self.a, len(self.rows)
        bits = self._read_and_rows_bits_of(other)
        for r, theirs in other.top.items():
            row = self.top.get(r, Counter())
            n = sum(row.values())
            for lab, m in theirs.items():
                bits -= m * math.log2((row.get(lab, 0) + a) / (n + K * a))
        return bits


class ContextBoardSearch(BoardSearch):
    """BoardSearch whose top-level labels are coded in the read's context
    (``CHAIN``): the prequential back-off code of ``mdl.backoff_chain_coder``,
    groups (anchor kind, square), in reading order, the best of
    ``CONTEXT_OPTIONS`` (its choice paid). Chunk and merge moves are scored
    exactly: the code of the groups a move touches is recomputed, the rest
    cached per number of labels. The context is made of pieces, never of
    the analysis, so moves never change it."""

    def __init__(self, positions: Sequence[Position], alpha: float = 0.001, counts: bool = True,
                 merges: bool = False, merge_kinds: str = "any", ids: Optional[ContextIds] = None,
                 read_context: bool = False, context_of: str = "pieces", typed: bool = False):
        super().__init__(positions, alpha, counts, merges, merge_kinds, typed)
        self.ids = ids or ContextIds()
        self.read_context = read_context
        # "pieces": the piece in each backward direction; "elements": the
        # label of the top-level element holding it (a chunk is one value,
        # and moves change the context of the elements that see it).
        self.context_of = context_of
        grow = ids is None
        self._grow = grow
        # One event per piece, in reading order; it is coded while it
        # anchors a top-level element.
        ev_g, ev_c, self.ev_at = [], [], {}
        for pi, p in enumerate(self.pos):
            for i in range(64):
                sq = (i % 8, i // 8)
                if sq in p["position"]:
                    self.ev_at[(pi, sq)] = len(ev_g)
                    ev_g.append(KIND_INDEX[p["position"][sq]] * 64 + i)
                    ev_c.append(self.ids.ids(chain_values(p["position"], sq), grow))
        self.ev_g = np.array(ev_g, dtype=np.int64)
        c = np.array(ev_c, dtype=np.int64).reshape(-1, len(CHAIN))
        # A context unseen in training (ids not grown) gets one fresh id per level.
        self.ev_c = [np.where(c[:, l] < 0, len(self.ids.tables[l]), c[:, l]) for l in range(len(CHAIN))]
        self.by_group: Dict[int, np.ndarray] = {}
        order = np.argsort(self.ev_g, kind="stable")
        gs = self.ev_g[order]
        cuts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            self.by_group[int(gs[a])] = np.sort(order[a:b])
        self.option = CONTEXT_OPTIONS[0]
        self._lid: Dict[Hashable, int] = {}
        if context_of == "elements":
            # What each event sees, and who sees each piece.
            self.ev_where = [None] * len(ev_g)
            self.ev_seen: List[Dict[str, object]] = [None] * len(ev_g)
            self.watchers: Dict[tuple, List[int]] = defaultdict(list)
            for (pi, sq), e in self.ev_at.items():
                b = backward_squares(self.pos[pi]["position"], sq)
                self.ev_where[e], self.ev_seen[e] = (pi, sq), b
                for v in set(v for v in b.values() if isinstance(v, tuple)):
                    self.watchers[(pi, v)].append(e)
        if read_context:
            self._init_read(grow)

    def _like(self, positions: Sequence[Position]) -> "ContextBoardSearch":
        return ContextBoardSearch(positions, self.a, self.counts, self.merges, self.merge_kinds,
                                  read_context=self.read_context, context_of=self.context_of, typed=self.typed)

    # A context made of elements ------------------------------------------ #
    def _holder_label(self, pi: int, sq: Square, override: Optional[dict] = None):
        if override is not None and (pi, sq) in override:
            return override[(pi, sq)]
        p = self.pos[pi]
        return p["tops"][p["owner"][sq]][0]

    def _element_ids(self, e: int, override: Optional[dict] = None) -> List[int]:
        pi, _ = self.ev_where[e]
        b = self.ev_seen[e]
        vals = tuple(tuple((self._holder_label(pi, b[d], override) if isinstance(b[d], tuple) else b[d])
                           for d in dirs) for dirs in CHAIN)
        ids = self.ids.ids(vals, self._grow)
        return [len(self.ids.tables[l]) if v < 0 else v for l, v in enumerate(ids)]

    def _refresh(self, events) -> None:
        for e in events:
            for l, v in enumerate(self._element_ids(e)):
                self.ev_c[l][e] = v

    def _owned(self, pi: int, anchor: Square) -> List[Square]:
        return [m for m, o in self.pos[pi]["owner"].items() if o == anchor]

    def _seen_change(self, pieces: Dict[tuple, Hashable], label_of: Dict[int, Hashable]):
        """The coded events whose context changes when the pieces ``pieces``
        come to be held by elements of the given labels: their new context
        ids, level by level."""
        out = {}
        for pq in pieces:
            for e in self.watchers.get(pq, ()):
                if e in label_of and e not in out:
                    out[e] = self._element_ids(e, pieces)
        return out

    # The read's answers in context -------------------------------------- #
    def _init_read(self, grow: bool) -> None:
        """Every question the read asks, in reading order (position, square,
        kind), with its row (kind, square, count), answer and the context
        ids of its square; each square's questions are dropped when a chunk
        covers it. The code is kept per row and option."""
        g, key, ctx, self.r_square = [], [], [], {}
        for pi, p in enumerate(self.pos):
            position, before = p["position"], p["before"]
            for i in range(64):
                sq = (i % 8, i // 8)
                stop = KIND_INDEX.get(position.get(sq), len(KINDS))
                ids = self.ids.ids(chain_values(position, sq), grow)
                start = len(g)
                for k in range(min(stop + 1, len(KINDS))):
                    g.append(k * (64 * COUNTS) + i * COUNTS + min(int(before[i][k]), COUNTS - 1))
                    key.append(int(k == stop))
                    ctx.append(ids)
                self.r_square[(pi, sq)] = np.arange(start, len(g))
        self.r_g = np.array(g, dtype=np.int64)
        self.r_key = np.array(key, dtype=np.int64)
        c = np.array(ctx, dtype=np.int64).reshape(-1, len(CHAIN))
        self.r_ctx = [np.where(c[:, l] < 0, len(self.ids.tables[l]), c[:, l]) for l in range(len(CHAIN))]
        self.r_active = np.ones(len(g), dtype=bool)
        _, self.r_gid = np.unique(self.r_g, return_inverse=True)
        order = np.argsort(self.r_gid, kind="stable")
        cuts = np.r_[0, np.flatnonzero(np.diff(self.r_gid[order])) + 1, len(order)]
        self.r_members = [np.sort(order[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
        self.r_code = self._read_codes(np.arange(len(g)))

    def _read_codes(self, idx: np.ndarray) -> Dict[tuple, np.ndarray]:
        """Per option, the code (nats) of each row among the active events
        of ``idx`` (whole rows, in reading order)."""
        idx = idx[self.r_active[idx]]
        G = len(self.r_members)
        out = {}
        if len(idx) == 0:
            return {opt: np.zeros(G) for opt in CONTEXT_OPTIONS}
        gid, w = self.r_gid[idx], np.ones(len(idx))
        for b in CONTEXT_WEIGHTS:
            t = backoff_terms(self.r_g[idx], [c[idx] for c in self.r_ctx], self.r_key[idx], w, 2, self.a, b)
            if (0, 0.0) not in out:
                out[(0, 0.0)] = np.bincount(gid, weights=t[0], minlength=G)
            for d in range(1, len(CHAIN) + 1):
                out[(d, b)] = np.bincount(gid, weights=t[d], minlength=G)
        return out

    def _read_nats(self, option=None) -> float:
        if not self.read_context:
            return super()._read_nats()
        totals = {opt: float(v.sum()) for opt, v in self.r_code.items()}
        if option is not None:
            return totals[option]
        self.read_option = min(CONTEXT_OPTIONS, key=lambda o: totals[o])
        return totals[self.read_option] + CHOICE_NATS

    def _read_after(self, squares: List[tuple], cache: dict) -> float:
        """The read's code (nats, best option, choice paid) once the given
        (position, square)s are no longer read."""
        if not self.read_context:
            read = self.read.copy()
            for pi, t in squares:
                p = self.pos[pi]
                j = t[1] * 8 + t[0]
                self._ask(read, j, p["before"][j], KIND_INDEX[p["position"][t]], -1)
            return rows_nats(read.reshape(-1, 2), self.a)
        gone = np.concatenate([self.r_square[x] for x in squares])
        gids = np.unique(self.r_gid[gone])
        idx = np.concatenate([self.r_members[i] for i in gids])
        keep = self.r_active.copy()
        keep[gone] = False
        idx = idx[keep[idx]]
        if "read totals" not in cache:
            cache["read totals"] = {opt: float(v.sum()) for opt, v in self.r_code.items()}
        totals = cache["read totals"]
        G = len(self.r_members)
        new = {}
        if len(idx):
            w = np.ones(len(idx))
            for b in CONTEXT_WEIGHTS:
                t = backoff_terms(self.r_g[idx], [c[idx] for c in self.r_ctx], self.r_key[idx], w, 2, self.a, b)
                new.setdefault((0, 0.0), t[0].sum())
                for d in range(1, len(CHAIN) + 1):
                    new[(d, b)] = t[d].sum()
        return min(totals[o] - float(self.r_code[o][gids].sum()) + new.get(o, 0.0)
                   for o in CONTEXT_OPTIONS) + CHOICE_NATS

    def apply(self, key: tuple, chosen: List[tuple]) -> Hashable:
        if self.read_context:
            gone = np.concatenate([self.r_square[(pi, t)] for pi, _, t in chosen])
            self.r_active[gone] = False
            gids = np.unique(self.r_gid[gone])
            fresh = self._read_codes(np.concatenate([self.r_members[i] for i in gids]))
            for opt in CONTEXT_OPTIONS:
                self.r_code[opt][gids] = fresh[opt][gids]
        Y = super().apply(key, chosen)
        if self.context_of == "elements":
            self._refresh({e for pi, s, _ in chosen for m in self._owned(pi, s) for e in self.watchers.get((pi, m), ())})
        return Y

    def apply_merge(self, A, B) -> None:
        super().apply_merge(A, B)
        if self.context_of == "elements":
            self._refresh(range(len(self.ev_g)))

    def _event_labels(self) -> Dict[int, Hashable]:
        out = {}
        for pi, p in enumerate(self.pos):
            for sq, x in p["tops"].items():
                out[self.ev_at[(pi, sq)]] = x[0]
        return out

    def _prep(self, idx: np.ndarray, label_of: Dict[int, Hashable]):
        """The coded events among ``idx`` (reading order) and their keys.
        The code depends on the labels only through which are equal within a
        group, so each group's labels are numbered 0, 1, ... within it, and
        group ids are spread (2g) so that no two groups' keys meet whatever
        the alphabet."""
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        if len(idx):
            idx = idx[np.fromiter((j in label_of for j in idx), bool, len(idx))]
        if len(idx) == 0:
            return idx, idx
        lid = self._lid
        labs = np.fromiter((lid.setdefault(label_of[j], len(lid)) for j in idx), np.int64, len(idx))
        uniq, inv = np.unique(self.ev_g[idx] * np.int64(1 << 32) + labs, return_inverse=True)
        ug = uniq >> 32
        first = np.r_[0, np.flatnonzero(ug[1:] != ug[:-1]) + 1]
        start = np.repeat(first, np.diff(np.r_[first, len(uniq)]))
        return idx, (np.arange(len(uniq)) - start)[inv]

    def _code(self, idx, keys, K: int, option, new_ctx: Optional[dict] = None) -> float:
        if len(idx) == 0:
            return 0.0
        ctx = [c[idx] for c in self.ev_c]
        if new_ctx:
            for n, j in enumerate(idx):
                if j in new_ctx:
                    for l in range(len(CHAIN)):
                        ctx[l][n] = new_ctx[j][l]
        return context_nats(2 * self.ev_g[idx], ctx, keys, K, self.a, option)

    def top_nats(self, option=None) -> Tuple[float, tuple]:
        """The top-level labels' code (nats) and its option: the best of
        ``CONTEXT_OPTIONS``, or the one given (the choice not included)."""
        label_of = self._event_labels()
        idx, keys = self._prep(np.array(sorted(label_of), dtype=np.int64), label_of)
        best = None
        for opt in (CONTEXT_OPTIONS if option is None else [option]):
            n = self._code(idx, keys, len(self.rows), opt)
            if best is None or n < best[0] - 1e-12:
                best = (n, opt)
        return best

    def top_level_nats(self) -> float:
        nats, self.option = self.top_nats()
        return nats + CHOICE_NATS

    def _replaced(self, cache: dict, K: int, groups, new_label_of, new_ctx: Optional[dict] = None) -> float:
        """The best over the options of the top-level code at K labels, with
        the events of ``groups`` relabelled by ``new_label_of`` (None: no
        longer top level) and given the contexts ``new_ctx``; the other
        groups' code is cached per K."""
        label_of = cache.setdefault("label_of", self._event_labels())
        if ("base", K) not in cache:
            i, k = self._prep(np.array(sorted(label_of), dtype=np.int64), label_of)
            cache[("base", K)] = {opt: self._code(i, k, K, opt) for opt in CONTEXT_OPTIONS}
        base = cache[("base", K)]
        idx = np.concatenate([self.by_group[g] for g in groups]) if groups else np.zeros(0, np.int64)
        new = {j: label_of[j] for j in idx if j in label_of}
        for j, lab in new_label_of.items():
            if lab is None:
                new.pop(j, None)
            else:
                new[j] = lab
        oi, ok = self._prep(idx, label_of)
        ni, nk = self._prep(idx, new)
        return min(base[opt] - self._code(oi, ok, K, opt) + self._code(ni, nk, K, opt, new_ctx)
                   for opt in CONTEXT_OPTIONS)

    def score(self, key: tuple, chosen: List[tuple], base_rows: Dict) -> float:
        """Bits after the chunk move (exact, the best of the options)."""
        K = len(self.rows) + 1
        nats = self._chunk_rows_nats(self._relations(key, chosen), base_rows)
        Y = ("chunk", self.fresh)
        new_label_of, groups = {}, set()
        for pi, s, t in chosen:
            es, et = self.ev_at[(pi, s)], self.ev_at[(pi, t)]
            new_label_of[es], new_label_of[et] = Y, None
            groups.update((int(self.ev_g[es]), int(self.ev_g[et])))
        new_ctx = None
        if self.context_of == "elements":
            label_of = base_rows.setdefault("label_of", self._event_labels())
            pieces = {(pi, m): Y for pi, s, t in chosen for a in (s, t) for m in self._owned(pi, a)}
            after = {j: lab for j, lab in label_of.items() if new_label_of.get(j, lab) is not None}
            new_ctx = self._seen_change(pieces, after)
            groups.update(int(self.ev_g[e]) for e in new_ctx)
        top = self._replaced(base_rows, K, sorted(groups), new_label_of, new_ctx)
        read = self._read_after([(pi, t) for pi, _, t in chosen], base_rows)
        return (nats + read + top + CHOICE_NATS) / LN2

    def merge_score(self, A, B, cache: dict) -> float:
        """Bits after merging A' = B into A (exact). Only groups holding
        both labels change their pattern of equal labels."""
        K1 = len(self.rows) - 1
        ta, tb = self.top_of.get(A, {}), self.top_of.get(B, {})
        label_of = cache.setdefault("label_of", self._event_labels())
        if self.context_of == "elements":
            # Every element labelled A' is renamed, and so is every context
            # that sees one of its pieces.
            renamed = {j: A for j, lab in label_of.items() if lab == B}
            pieces = {}
            for j in renamed:
                pi, sq = self.ev_where[j]
                for m in self._owned(pi, sq):
                    pieces[(pi, m)] = A
            new_ctx = self._seen_change(pieces, label_of)
            groups = sorted({int(self.ev_g[j]) for j in renamed} | {int(self.ev_g[e]) for e in new_ctx})
            top = self._replaced(cache, K1, groups, renamed, new_ctx)
            return (self._merge_rows_nats(A, B, cache) + top + CHOICE_NATS) / LN2
        groups = sorted(r[0] * 64 + r[1] for r in (ta.keys() & tb.keys()))
        idx = np.concatenate([self.by_group[g] for g in groups]) if groups else np.zeros(0, np.int64)
        top = self._replaced(cache, K1, groups, {j: A for j in idx if label_of.get(j) == B})
        return (self._merge_rows_nats(A, B, cache) + top + CHOICE_NATS) / LN2

    def held_out_bits(self, positions: Sequence[Position]) -> Tuple[float, float]:
        analysed = ContextBoardSearch(positions, self.a, self.counts, self.merges, self.merge_kinds, ids=self.ids,
                                      read_context=self.read_context, context_of=self.context_of, typed=self.typed)
        analysed.replay(self.moves)
        flat = self._like([p["position"] for p in self.pos])
        flat_held = ContextBoardSearch(positions, self.a, self.counts, ids=flat.ids, read_context=self.read_context)
        return self.code_of(analysed) / len(positions), flat.code_of(flat_held) / len(positions)

    def code_of(self, other: "ContextBoardSearch") -> float:
        """Bits of another set of analyses under this search's counts: the
        read and the rows as BoardSearch, the top-level labels by the
        back-off predictive of this search's option."""
        _, (d, beta) = self.top_nats()
        K = len(self.rows)
        n_g: Dict[int, Counter] = defaultdict(Counter)
        n_gx: List[Dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(d)]
        for j, lab in self._event_labels().items():
            g = int(self.ev_g[j])
            n_g[g][lab] += 1
            for l in range(d):
                n_gx[l][(g, int(self.ev_c[l][j]))][lab] += 1
        bits = self._read_and_rows_bits_of(other)
        if self.read_context:
            bits += self._read_bits_of(other) - self._counted_read_bits_of(other)
        for j, lab in other._event_labels().items():
            g = int(other.ev_g[j])
            row = n_g.get(g, Counter())
            p = (row.get(lab, 0) + self.a) / (sum(row.values()) + K * self.a)
            for l in range(d):
                r = n_gx[l].get((g, int(other.ev_c[l][j])), Counter())
                p = (r.get(lab, 0) + beta * p) / (sum(r.values()) + beta)
            bits -= math.log2(p)
        return bits


    def _counted_read_bits_of(self, other: "BoardSearch") -> float:
        a = self.a
        p_yes = (self.read[..., 1] + a) / (self.read.sum(axis=-1) + 2 * a)
        return -float(np.sum(other.read[..., 1] * np.log2(p_yes) + other.read[..., 0] * np.log2(1 - p_yes)))

    def _read_bits_of(self, other: "ContextBoardSearch") -> float:
        """The read's answers of another set of analyses (bits) by the
        back-off predictive of this search's answers and read option."""
        self._read_nats()
        d, beta = self.read_option
        on = self.r_active
        g, key = self.r_g[on], self.r_key[on]
        n = Counter(g.tolist())
        yes = Counter(g[key == 1].tolist())
        levels = []
        for l in range(d):
            x = self.r_ctx[l][on]
            levels.append((Counter(zip(g.tolist(), x.tolist())), Counter(zip(g[key == 1].tolist(), x[key == 1].tolist()))))
        on = other.r_active
        bits = 0.0
        for gg, kk, *xs in zip(other.r_g[on].tolist(), other.r_key[on].tolist(), *[c[on].tolist() for c in other.r_ctx]):
            p = (yes.get(gg, 0) + self.a) / (n.get(gg, 0) + 2 * self.a)
            for l in range(d):
                nn, yy = levels[l]
                p = (yy.get((gg, xs[l]), 0) + beta * p) / (nn.get((gg, xs[l]), 0) + beta)
            bits -= math.log2(p if kk else 1 - p)
        return bits


def code_of_analyses(positions: Sequence[Position], analyses: Sequence[List[Node]], alpha: float,
                     counts: bool = True) -> Dict[str, float]:
    """The plain code (bits) of analysed positions, recomputed from scratch:
    the reference against which the search's scores are checked (the
    top-level labels without context)."""
    a = alpha
    V = len({t for p in positions for t in p.values()}) + 1
    rows: Dict[Hashable, Counter] = defaultdict(Counter)
    top: Dict[tuple, Counter] = defaultdict(Counter)
    read = np.zeros((len(KINDS), 64 * COUNTS, 2))
    for position, tops in zip(positions, analyses):
        before = counts_before(position) if counts else np.zeros((64, len(KINDS)), dtype=np.int64)
        anchored = {anchor_of(n): n for n in tops}
        covered = set()
        for n in tops:
            stack = [n]
            while stack:
                m = stack.pop()
                lab, body = m
                if is_chunk(m):
                    x, rel, y = body
                    rows[lab][("p", x[0], rel, y[0])] += 1
                    stack.extend([x, y])
                else:
                    rows[lab][("w", position[body])] += 1
                    covered.add(body)
        assert covered == set(position), "every piece in exactly one element"
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in anchored:
                k = KIND_INDEX[position[sq]]
                top[(k, i)][anchored[sq][0]] += 1
                BoardSearch._ask(read, i, before[i], k, 1)
            elif sq not in position:
                BoardSearch._ask(read, i, before[i], len(KINDS), 1)
    K = len(rows)
    A = (V + K * K * len(RELATIONS)) * a
    row_nats = sum(_row_term(r, A, a) for r in rows.values())
    top_nats = sum(_row_term(r, K * a, a) for r in top.values())
    read_nats = rows_nats(read.reshape(-1, 2), a)
    return {"read": read_nats / LN2, "top": top_nats / LN2, "rows": row_nats / LN2,
            "total": (read_nats + top_nats + row_nats) / LN2}


# --------------------------------------------------------------------------- #
# Learning, and sampling positions from the grammar
# --------------------------------------------------------------------------- #
class ChessLearner(Learner):
    """Positions by day and by night: by night the structure search proposes
    chunks (and, with ``merges``, categories), and the positions so analysed
    are consolidated into the two hierarchies; a new position is analysed by
    replaying the learned moves. With ``context`` (the default) the
    top-level labels are coded in the read's context, in the search and in
    the grammar; ``context=False`` is the learner before 2026-10-11."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, max_steps: int = 500,
                 counts: bool = True, merges: bool = False, patience: int = 0, context: bool = True,
                 read_context: bool = False, context_of: str = "pieces", typed: bool = False):
        super().__init__(seed=seed, alpha=alpha)
        self.typed = typed
        # What the context is made of: "pieces" or "elements" (the categories
        # of the neighbouring top-level elements; a chunk is one value).
        self.context_of = context_of
        self.max_steps = max_steps
        # Whether the read counts the pieces on earlier squares (without, each
        # square is read on its own).
        self.counts = counts
        self.merges, self.patience, self.context = merges, patience, context
        # Whether the read's answers are coded in context too (needs context).
        self.read_context = read_context and context
        self.search: Optional[BoardSearch] = None

    def sleep(self) -> Grammar:
        """Learn from every position observed so far: the structure search,
        then consolidation into the two hierarchies."""
        t0 = time.time()
        if self.context:
            search = ContextBoardSearch(self.experiences, self.alpha, self.counts, self.merges,
                                        read_context=self.read_context, context_of=self.context_of, typed=self.typed)
        else:
            search = BoardSearch(self.experiences, self.alpha, self.counts, self.merges, typed=self.typed)
        self.history.append({"stage": "flat", "bits": search.bits(), "seconds": 0.0})

        def log(step, move, bits):
            if move[0] == "chunk":
                key = move[1]
                self.history.append({"stage": "chunk", "move": f"[{key[0]} {key[1]} {key[2]}] x{len(move[2])}",
                                     "bits": bits, "seconds": time.time() - t0})
            else:
                self.history.append({"stage": "merge", "move": f"merge({move[1]}, {move[2]})",
                                     "bits": bits, "seconds": time.time() - t0})
        search.run(self.max_steps, log, self.patience)
        self.search = search
        self.analyses = search.analyses()
        self.model = self.fit(self.analyses, BoardMemory(counts=self.counts, context=self.context,
                                                                read_context=self.read_context,
                                                                context_of=self.context_of))
        g = self.model.grammar
        self.history.append({"stage": "consolidate", "bits": g.info["total bits"],
                             "seconds": time.time() - t0})
        return g

    def analyse(self, position: Position) -> List[Node]:
        """A position's top-level elements: the learned moves replayed."""
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
