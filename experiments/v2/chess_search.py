"""The board's structure search with merge moves, as in the shared sequence search.

``BoardSearch`` (src/trellis2/chess.py) has one move: chunk (B, relation, C).
The shared search (``mdl_search._State``) alternates chunk and merge moves,
so chunks form over categories and pool evidence. ``HierBoardSearch`` adds
the merge move with ``mdl_search``'s semantics:

* *Merge* (A, A'): A' is renamed A everywhere. Rows A and A' pool (their
  outcomes renamed), every row that holds A' as a part sees its outcomes
  renamed (and may see outcomes collide), every top-level row (anchor kind,
  square) pools its counts of A and A', and the number of labels K drops by
  one, which changes every row's normaliser.

The code is BoardSearch's plain code unchanged (the read's answers, the
label of each top-level element given its anchor's kind and square, one
Dirichlet-multinomial row per label over V + K^2 * relations outcomes), and
both moves are scored exactly from the rows they change. The only other
change: the kind of a top-level element's anchor is taken from the piece
on its anchor square (BoardSearch took it from the label, which is the same
while labels never mix kinds).

Nothing here is about chess beyond what BoardSearch already codes: a merge
is a renaming of labels, scored by the same code.

Search: best-first over both moves, with Stolcke's lookahead (``patience``:
continue up to that many steps without a new shortest code, then return to
the shortest found, by replaying its moves).
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from trellis2.chess import (KIND_INDEX, KINDS, RELATIONS, BoardSearch, Position,  # noqa: E402
                            anchor_of, counts_before, is_chunk)
from trellis2.mdl import rows_nats  # noqa: E402

LN2 = math.log(2.0)


def _phi(c: float, a: float) -> float:
    return math.lgamma(c + a) - math.lgamma(a) if c else 0.0


def _row_term(row: Counter, A: float, a: float) -> float:
    n = sum(row.values())
    if not n:
        return 0.0
    return math.lgamma(n + A) - math.lgamma(A) - sum(_phi(c, a) for c in row.values())


class HierBoardSearch(BoardSearch):
    """BoardSearch with merge moves (mdl_search semantics), exact scores."""

    def __init__(self, positions: Sequence[Position], alpha: float = 0.001, counts: bool = True,
                 merges: bool = True, merge_kinds: str = "any"):
        super().__init__(positions, alpha, counts)
        self.merges = merges
        # "any": any two labels; "same": only labels whose elements are all
        # anchored on one and the same kind of piece.
        self.merge_kinds = merge_kinds
        self.alias: Dict[Hashable, Hashable] = {}
        self.parents: Dict[Hashable, set] = defaultdict(set)
        self.top_of: Dict[Hashable, Counter] = defaultdict(Counter)
        self.anchor_kinds: Dict[Hashable, set] = {t: {KIND_INDEX[t]} for t in KINDS}
        for r, c in self.top.items():
            for lab, v in c.items():
                self.top_of[lab][r] += v

    # ------------------------------------------------------------------ #
    # Code
    # ------------------------------------------------------------------ #
    def parts(self) -> Dict[str, float]:
        """Bits by part: the read's answers, the top-level labels, the rows."""
        K = len(self.rows)
        return {"read": rows_nats(self.read.reshape(-1, 2), self.a) / LN2,
                "top": self._top_nats(self.top.values(), K) / LN2,
                "rows": self._row_nats(K) / LN2}

    # ------------------------------------------------------------------ #
    # Chunk move (as BoardSearch, with each anchor's kind from its square)
    # ------------------------------------------------------------------ #
    def _changes(self, key: tuple, chosen: List[tuple], read: np.ndarray, top) -> None:
        B, _, C = key
        Y = ("chunk", self.fresh)
        for pi, s, t in chosen:
            p = self.pos[pi]
            kB, kC = KIND_INDEX[p["position"][s]], KIND_INDEX[p["position"][t]]
            i, j = s[1] * 8 + s[0], t[1] * 8 + t[0]
            self._ask(read, j, p["before"][j], kC, -1)
            top(kB, i)[B] -= 1
            top(kB, i)[Y] += 1
            top(kC, j)[C] -= 1

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
        out = super().apply(key, chosen)
        assert out == Y
        self.anchor_kinds[Y] = kinds
        self.parents[B].add(Y)
        self.parents[C].add(Y)
        return Y

    # ------------------------------------------------------------------ #
    # Merge move (mdl_search._State semantics)
    # ------------------------------------------------------------------ #
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
            ka, kb = self.anchor_kinds.get(A, set()), self.anchor_kinds.get(B, set())
            return len(ka | kb) == 1
        return True

    def merge_score(self, A, B, cache: dict) -> float:
        """Bits after merging A' = B into A (exact)."""
        a, K1 = self.a, len(self.rows) - 1
        AA, terms, row_sum, tops, top_sum = self._merge_base(cache)
        merged, moved = self._merged_rows(A, B)
        nats = row_sum - terms[A] - terms[B] + _row_term(merged, AA, a)
        for r, row in moved.items():
            nats += _row_term(row, AA, a) - terms[r]
        top = top_sum
        ta, tb = self.top_of.get(A, {}), self.top_of.get(B, {})
        for r in (ta.keys() & tb.keys()):
            row = self.top[r]
            new = Counter(row)
            new[A] += new.pop(B)
            top += _row_term(new, K1 * a, a) - tops[r]
        if "read" not in cache:
            cache["read"] = rows_nats(self.read.reshape(-1, 2), a)
        return (nats + top + cache["read"]) / LN2

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

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def scored_moves(self):
        """(bits after the move, move) for every chunk and merge move."""
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
        """Best-first search over chunk and merge moves; with ``patience``,
        continue past a step without a new shortest code (Stolcke's
        lookahead) and return to the shortest found."""
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
            self._restore(best_moves)
        return best

    def _restore(self, moves: List[tuple]) -> None:
        fresh = HierBoardSearch([p["position"] for p in self.pos], self.a, self.counts,
                                self.merges, self.merge_kinds)
        fresh.replay(moves)
        self.__dict__.update(fresh.__dict__)

    def replay(self, moves: Sequence[tuple]) -> None:
        """Apply chunk moves (B, relation, C, Y) and merges ("merge", A, A')
        learned elsewhere, in order."""
        for move in moves:
            if len(move) == 3 and move[0] == "merge":
                if move[1] in self.rows and move[2] in self.rows:
                    self.apply_merge(move[1], move[2])
                elif move[2] in self.rows:          # A unseen here: A' takes A's name
                    self._rename(move[2], move[1])
                continue
            B, rel, C, Y = move
            pairs = [(pi, sq, t) for pi, p in enumerate(self.pos) for sq, x in p["tops"].items()
                     if x[0] == B for r, t in self._seen(p, sq) if r == rel and p["tops"][t][0] == C]
            chosen = self.select(pairs)
            if chosen:
                self.fresh = Y[1]
                self.apply((B, rel, C), chosen)

    def _rename(self, B, A) -> None:
        """Rename label B to A (A not in use): a merge with an empty A."""
        self.rows[A] = Counter()
        self.anchor_kinds.setdefault(A, set())
        self.apply_merge(A, B)

    # ------------------------------------------------------------------ #
    # Analyses and checks
    # ------------------------------------------------------------------ #
    def resolve(self, lab):
        return self.alias.get(lab, lab)

    def analyses(self) -> List[list]:
        def rename(node):
            lab, body = node
            if is_chunk(node):
                x, rel, y = body
                return (self.resolve(lab), (rename(x), rel, rename(y)))
            return (self.resolve(lab), body)
        return [[rename(p["tops"][sq]) for sq in sorted(p["tops"], key=lambda s: (s[1], s[0]))]
                for p in self.pos]

    def levels(self) -> Dict[Hashable, int]:
        """Each chunk label's depth (a chunk of pieces: 1)."""
        memo: Dict[Hashable, int] = {}

        def depth(lab, seen=()):
            if lab in memo:
                return memo[lab]
            if lab in seen:
                return 1
            d = 0
            for o in self.rows.get(lab, {}):
                if o[0] == "p":
                    d = max(d, 1 + max(depth(o[1], seen + (lab,)), depth(o[3], seen + (lab,))))
            memo[lab] = d
            return d
        return {lab: depth(lab) for lab in self.rows}

    def held_out_bits(self, positions: Sequence[Position]) -> Tuple[float, float]:
        analysed = HierBoardSearch(positions, self.a, self.counts, self.merges, self.merge_kinds)
        analysed.replay(self.moves)
        flat = BoardSearch([p["position"] for p in self.pos], self.a, self.counts)
        return (self.code_of(analysed) / len(positions),
                flat.code_of(BoardSearch(positions, self.a, self.counts)) / len(positions))


def code_of_analyses(positions: Sequence[Position], analyses: Sequence[list], alpha: float,
                     counts: bool = True) -> Dict[str, float]:
    """The plain code (bits) of analysed positions, recomputed from scratch:
    the reference against which the moves' scores are checked."""
    a = alpha
    V = len({t for p in positions for t in p.values()}) + 1
    NR = len(RELATIONS)
    rows: Dict[Hashable, Counter] = defaultdict(Counter)
    top: Dict[tuple, Counter] = defaultdict(Counter)
    read = np.zeros((len(KINDS), 64 * 11, 2))
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
    A = (V + K * K * NR) * a
    row_nats = sum(_row_term(r, A, a) for r in rows.values())
    top_nats = sum(_row_term(r, K * a, a) for r in top.values())
    read_nats = rows_nats(read.reshape(-1, 2), a)
    return {"read": read_nats / LN2, "top": top_nats / LN2, "rows": row_nats / LN2,
            "total": (read_nats + top_nats + row_nats) / LN2}
