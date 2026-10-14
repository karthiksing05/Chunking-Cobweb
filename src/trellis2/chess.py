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
