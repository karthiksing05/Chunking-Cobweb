"""TRELLIS v2 chess without nights: a continuously incremental board learner.

The night learner (``chess.ChessLearner``) stores positions by day and, by
night, runs the structure search (``ContextBoardSearch``) over every stored
position from scratch and consolidates the analyses into fresh hierarchies.
This learner has no night. Its state is persistent and every unit of work is
per position or a bounded step on it:

* the stored analyses, one per position (its top-level elements, which
  piece each element owns), and the chunk types learned so far, in the order
  they were learned (``moves``);
* the code's count tables, exactly those the night's search keeps: each
  label's definition row, the read's answers (kind, square, count of that
  kind on earlier squares), and the top-level labels coded in the read's
  context (one event per piece, in reading order; the prequential back-off
  code of ``mdl.backoff_chain_coder``, the best of ``CONTEXT_OPTIONS``, its
  choice paid). The top-level code is kept per group (anchor kind and
  square) and option, for the current number of labels K and for K + 1
  (what a chunk move needs);
* an index of every candidate chunk instance (B, relation, C) -> the
  top-level pairs in every stored position, updated when a position's
  analysis changes.

Each position:

1. is perceived with what has been learned: the chunk types are replayed on
   it in order (``BoardSearch.replay``), and its counts are added to the
   tables (its events are appended at the end of the reading order, so only
   the groups it touches are recoded);
2. the chunk moves it touches (the keys of its own top-level pairs, and after
   a move the keys made with the new label) are scored exactly from the rows
   and groups they change, and the best is applied while the total code
   shrinks (a few steps per position). A chunk move applies at every
   instance in memory (through the index), so every stored analysis stays
   the replay of the learned moves: the state equals the night's search
   state for the same moves, and is re-analysed exactly where a move
   touched it.

Screening: a key is rescored when its number of instances has grown since it
was last scored (by one instance when it was close to paying, by a fraction
otherwise); acceptance is always by the exact code.

Nothing is rebuilt from the whole memory while learning, except the
top-level code's dependence on the number of labels: each accepted move
(K -> K + 1) recodes every group at the new K and K + 1 (a vectorised pass
over the events; a few times per run). The grammar read off the state for
evaluation (``read_grammar``) takes the labels as symbols and one rule class
per composition key.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .chess import (CHAIN, CHOICE_NATS, CONTEXT_OPTIONS, COUNTS, KIND_INDEX, KINDS, LN2, RELATIONS,
                    BoardMemory, BoardSearch, ContextBoardSearch, ContextIds, Node, Position, _phi, backoff_terms,
                    chain_values, read_row)
from .grammar import CONTEXT_WEIGHTS, UNK, Grammar
from .mdl import dm_code, elias_delta_bits, rows_nats
from .model import Trellis2

N_OPT = len(CONTEXT_OPTIONS)
OPT_INDEX = {o: i for i, o in enumerate(CONTEXT_OPTIONS)}
N_GROUPS = len(KINDS) * 64


def _before_many(keys: Sequence[np.ndarray]) -> List[np.ndarray]:
    """For each array of keys (in event order), how many earlier events had
    the same key (``mdl._before`` with unit weights), all in one sort
    (keys below 2**44)."""
    n = len(keys[0])
    if n == 0:
        return [np.zeros(0) for _ in keys]
    flat = np.concatenate([k.astype(np.int64) + np.int64(i) * np.int64(1 << 44) for i, k in enumerate(keys)])
    order = np.argsort(flat, kind="stable")
    f = flat[order]
    idx = np.arange(len(f))
    start = np.empty(len(f), dtype=bool)
    start[0] = True
    start[1:] = f[1:] != f[:-1]
    first = np.maximum.accumulate(np.where(start, idx, 0))
    out = np.empty(len(f))
    out[order] = idx - first
    return [out[i * n:(i + 1) * n] for i in range(len(keys))]


def option_counts(groups: np.ndarray, ctx: np.ndarray, keys: np.ndarray) -> tuple:
    """What the back-off code counts before each event (reading order
    within each group), whatever K and beta: n(g, k), n(g), and per level
    n(g, x_l, k), n(g, x_l)."""
    g = groups.astype(np.int64)
    k = keys.astype(np.int64)
    span = np.int64(int(k.max()) + 1 if len(k) else 1)
    arrays = [g * span + k, g]
    for l in range(ctx.shape[1]):
        x = ctx[:, l].astype(np.int64)
        gx = g * np.int64(int(x.max()) + 1) + x
        arrays += [gx * span + k, gx]
    b = _before_many(arrays)
    return b[0], b[1], [(b[2 + 2 * l], b[3 + 2 * l]) for l in range(ctx.shape[1])]


def option_terms_from(counts: tuple, K: int, alpha: float) -> np.ndarray:
    """(options, events): each event's share (nats) of the prequential
    back-off code (``mdl.backoff_chain_coder``) under every option of
    ``CONTEXT_OPTIONS``, at K labels."""
    n_gk, n_g, levels = counts
    out = np.empty((N_OPT, len(n_g)))
    p0 = (n_gk + alpha) / (n_g + K * alpha)
    out[0] = -np.log(p0)
    for b in CONTEXT_WEIGHTS:
        p = p0
        for d, (n_xk, n_x) in enumerate(levels, start=1):
            p = (n_xk + b * p) / (n_x + b)
            out[OPT_INDEX[(d, b)]] = -np.log(p)
    return out


def option_terms(groups: np.ndarray, ctx: np.ndarray, keys: np.ndarray, K: int, alpha: float) -> np.ndarray:
    """(options, events): each event's share (nats) of the prequential
    back-off code under every option of ``CONTEXT_OPTIONS`` (events in
    reading order within each group; ``keys`` distinct within a group)."""
    if len(keys) == 0:
        return np.empty((N_OPT, 0))
    return option_terms_from(option_counts(groups, ctx, keys), K, alpha)


def _local_keys(groups: np.ndarray, labs: np.ndarray) -> np.ndarray:
    """Label ids renumbered 0, 1, ... within each group (the code depends on
    the labels only through which are equal within a group)."""
    if len(labs) == 0:
        return labs
    uniq, inv = np.unique(groups.astype(np.int64) * np.int64(1 << 32) + labs, return_inverse=True)
    ug = uniq >> 32
    first = np.r_[0, np.flatnonzero(ug[1:] != ug[:-1]) + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(uniq)]))
    return (np.arange(len(uniq)) - start)[inv.reshape(-1)]


def _dm_row(n0: float, n1: float, a: float) -> float:
    """One read row's code (nats): a Dirichlet-multinomial over yes/no."""
    if n0 + n1 <= 0:
        return 0.0
    return (math.lgamma(n0 + n1 + 2 * a) - math.lgamma(2 * a)
            - (math.lgamma(n0 + a) - math.lgamma(a)) - (math.lgamma(n1 + a) - math.lgamma(a)))


def node_rows(node: Node, position: Position, out: Dict[Hashable, Counter], sign: int = 1) -> None:
    """Add (sign 1) or remove (sign -1) an element's definitions to label rows."""
    stack = [node]
    while stack:
        lab, body = stack.pop()
        if len(body) == 3:
            x, rel, y = body
            out[lab][("p", x[0], rel, y[0])] += sign
            stack.extend([x, y])
        else:
            out[lab][("w", position[body])] += sign


class IncrementalBoardState:
    """The night search's state (``ContextBoardSearch``, read context off),
    kept persistently over a growing memory: positions are added one at a
    time; chunk moves and dissolve moves are scored and applied against the
    stored counts."""

    def __init__(self, alpha: float = 0.001, counts: bool = True):
        self.a = alpha
        self.counts = counts
        self.NR = len(RELATIONS)
        self.rows: Dict[Hashable, Counter] = defaultdict(Counter)
        self.top: Dict[tuple, Counter] = defaultdict(Counter)
        self.read = np.zeros((len(KINDS), 64 * COUNTS, 2))
        self.pos: List[dict] = []
        self.tokens: set = set()
        self.moves: List[tuple] = []
        self.fresh = 0
        self.ids = ContextIds()
        # Events: one per piece, in reading order (positions in order).
        self.n_ev = 0
        self._g = np.zeros(1024, dtype=np.int64)
        self._c = np.zeros((1024, len(CHAIN)), dtype=np.int64)
        self._lab = np.full(1024, -1, dtype=np.int64)      # label id, -1: not top level
        self.ev_at: Dict[tuple, int] = {}
        self.group_events: Dict[int, List[int]] = defaultdict(list)
        self._garr: Dict[int, np.ndarray] = {}
        self.lid: Dict[Hashable, int] = {}
        # Candidate index: key -> {(position, anchor, second anchor)}; label -> keys holding it.
        self.index: Dict[tuple, set] = defaultdict(set)
        self.label_keys: Dict[Hashable, set] = defaultdict(set)
        self.pos_pairs: List[List[tuple]] = []
        # label -> positions holding an element of it.
        self.label_pos: Dict[Hashable, set] = defaultdict(set)
        # Top-level code per K: (groups, options) and its column sums.
        self.gcode: Dict[int, np.ndarray] = {}
        self.gtot: Dict[int, np.ndarray] = {}
        self._read_cache = None
        self.stats = Counter()

    # Sizes and the code --------------------------------------------------- #
    @property
    def K(self) -> int:
        return len(self.rows)

    @property
    def V(self) -> int:
        return len(self.tokens) + 1

    def _rows_nats(self, rows, K: int) -> float:
        A = (self.V + K * K * self.NR) * self.a
        a = self.a
        out = 0.0
        for r in rows.values():
            n = sum(r.values())
            if n:
                out += math.lgamma(n + A) - math.lgamma(A) - sum(_phi(c, a) for c in r.values())
        return out

    def _row_nats(self, K: int) -> float:
        return self._rows_nats(self.rows, K)

    def read_nats(self) -> float:
        if self._read_cache is None:
            self._read_cache = rows_nats(self.read.reshape(-1, 2), self.a)
        return self._read_cache

    def top_nats(self, K: Optional[int] = None) -> Tuple[float, tuple]:
        K = self.K if K is None else K
        tot = self.gtot[K]
        i = int(np.argmin(tot))
        return float(tot[i]), CONTEXT_OPTIONS[i]

    def nats(self) -> float:
        return self._row_nats(self.K) + self.read_nats() + self.top_nats()[0] + CHOICE_NATS

    def bits(self) -> float:
        return self.nats() / LN2

    def parts(self) -> Dict[str, float]:
        return {"read": self.read_nats() / LN2, "top": (self.top_nats()[0] + CHOICE_NATS) / LN2,
                "rows": self._row_nats(self.K) / LN2}

    # Events and group codes ------------------------------------------------ #
    def _lab_id(self, lab) -> int:
        if lab is None:
            return -1
        return self.lid.setdefault(lab, len(self.lid))

    def _add_event(self, g: int, ctx: tuple, lab) -> int:
        e = self.n_ev
        if e >= len(self._g):
            n = 2 * len(self._g)
            self._g = np.resize(self._g, n)
            self._c = np.resize(self._c, (n, len(CHAIN)))
            lab_ = np.full(n, -1, dtype=np.int64)
            lab_[:e] = self._lab[:e]
            self._lab = lab_
        self._g[e], self._c[e], self._lab[e] = g, ctx, self._lab_id(lab)
        self.n_ev += 1
        self.group_events[g].append(e)
        self._garr.pop(g, None)
        return e

    def _group_arr(self, g: int) -> np.ndarray:
        arr = self._garr.get(g)
        if arr is None:
            arr = self._garr[g] = np.array(self.group_events.get(g, ()), dtype=np.int64)
        return arr

    def _group_code(self, g: int, K, override: Optional[Dict[int, int]] = None):
        """One group's code (nats) under every option at K labels (or a dict
        K -> code for a sequence of K), its events labelled as stored except
        for ``override`` (event -> label id, -1: not top level)."""
        Ks = [K] if isinstance(K, int) else list(K)
        idx = self._group_arr(g)
        labs = self._lab[idx]
        if override:
            labs = labs.copy()
            for e, l in override.items():
                labs[np.searchsorted(idx, e)] = l
        on = labs >= 0
        if not on.any():
            out = {k: np.zeros(N_OPT) for k in Ks}
        else:
            labs = labs[on]
            _, keys = np.unique(labs, return_inverse=True)
            counts = option_counts(np.zeros(len(labs), dtype=np.int64), self._c[idx[on]], keys.reshape(-1))
            out = {k: option_terms_from(counts, k, self.a).sum(axis=1) for k in Ks}
        return out[K] if isinstance(K, int) else out

    def _code_all(self, K):
        """Every group's code (groups, options) at K labels (or a dict K ->
        code for a sequence of K): one vectorised pass over the events."""
        Ks = [K] if isinstance(K, int) else list(K)
        n = self.n_ev
        on = np.flatnonzero(self._lab[:n] >= 0)
        out = {k: np.zeros((N_GROUPS, N_OPT)) for k in Ks}
        if len(on):
            g = self._g[on]
            keys = _local_keys(g, self._lab[on])
            counts = option_counts(g, self._c[on], keys)
            for k in Ks:
                t = option_terms_from(counts, k, self.a)
                m = out[k]
                for o in range(N_OPT):
                    m[:, o] = np.bincount(g, weights=t[o], minlength=N_GROUPS)
        return out[K] if isinstance(K, int) else out

    def _recode_all(self) -> None:
        """Every group's code at K and K + 1 (after the number of labels or
        many labels changed)."""
        self.stats["full recodes"] += 1
        self.gcode = self._code_all((self.K, self.K + 1))
        self.gtot = {K: m.sum(axis=0) for K, m in self.gcode.items()}

    def _ensure_K(self, K: int) -> None:
        if K not in self.gcode:
            self.gcode[K] = self._code_all(K)
            self.gtot[K] = self.gcode[K].sum(axis=0)

    def _recode_groups(self, groups) -> None:
        Ks = list(self.gcode)
        for g in groups:
            new = self._group_code(g, Ks)
            for K in Ks:
                self.gtot[K] += new[K] - self.gcode[K][g]
                self.gcode[K][g] = new[K]

    def _top_after(self, K: int, override: Dict[int, int]) -> float:
        """The top-level code (nats, best option) at K labels with the events
        of ``override`` relabelled."""
        self._ensure_K(K)
        by_group: Dict[int, Dict[int, int]] = defaultdict(dict)
        for e, l in override.items():
            by_group[int(self._g[e])][e] = l
        tot = self.gtot[K].copy()
        m = self.gcode[K]
        for g, ov in by_group.items():
            tot += self._group_code(g, K, ov) - m[g]
        return float(tot.min())

    # The candidate index --------------------------------------------------- #
    def _pairs_of(self, pi: int) -> List[tuple]:
        p = self.pos[pi]
        out = []
        for sq, x in p["tops"].items():
            for rel, t in BoardSearch._seen(p, sq):
                out.append(((x[0], rel, p["tops"][t][0]), (pi, sq, t)))
        return out

    def _unindex(self, pi: int) -> None:
        for key, inst in self.pos_pairs[pi]:
            s = self.index.get(key)
            if s is not None:
                s.discard(inst)
                if not s:
                    del self.index[key]
                    for lab in (key[0], key[2]):
                        self.label_keys[lab].discard(key)
        self.pos_pairs[pi] = []

    def _reindex(self, pi: int) -> set:
        """Recompute position pi's candidate pairs; returns its keys."""
        self._unindex(pi)
        pairs = self._pairs_of(pi)
        self.pos_pairs[pi] = pairs
        keys = set()
        for key, inst in pairs:
            self.index[key].add(inst)
            self.label_keys[key[0]].add(key)
            self.label_keys[key[2]].add(key)
            keys.add(key)
        return keys

    def _labels_in(self, tops) -> set:
        out = set()
        stack = list(tops.values())
        while stack:
            lab, body = stack.pop()
            out.add(lab)
            if len(body) == 3:
                stack.extend([body[0], body[2]])
        return out

    # Perception: a new position ------------------------------------------- #
    def perceive(self, position: Position, moves: Optional[Sequence[tuple]] = None) -> BoardSearch:
        """The position analysed by replaying the learned chunk types in order."""
        one = BoardSearch([position], self.a, self.counts)
        one.replay(self.moves if moves is None else moves)
        return one

    def add_position(self, position: Position) -> set:
        """Perceive a position with the learned chunk types, add its counts;
        returns the candidate keys it touches."""
        one = self.perceive(position)
        p = one.pos[0]
        pi = len(self.pos)
        self.pos.append(p)
        K0 = self.K
        self.read += one.read
        self._read_cache = None
        for lab, row in one.rows.items():
            self.rows[lab].update(row)
        for r, row in one.top.items():
            self.top[r].update({k: v for k, v in row.items() if v})
        self.tokens.update(position.values())
        for lab in self._labels_in(p["tops"]):
            self.label_pos[lab].add(pi)
        touched = set()
        for i in range(64):
            sq = (i % 8, i // 8)
            if sq in position:
                g = KIND_INDEX[position[sq]] * 64 + i
                lab = p["tops"][sq][0] if sq in p["tops"] else None
                self.ev_at[(pi, sq)] = self._add_event(g, tuple(self.ids.ids(chain_values(position, sq))), lab)
                if lab is not None:
                    touched.add(g)
        if self.K != K0 or not self.gcode:
            self._recode_all()
        else:
            self._recode_groups(touched)
        self.pos_pairs.append([])
        return self._reindex(pi)

    # Chunk moves ----------------------------------------------------------- #
    def chosen(self, key: tuple) -> List[tuple]:
        return BoardSearch.select(list(self.index.get(key, ())))

    def _read_cells_delta(self, cells: Dict[tuple, List[float]]) -> float:
        a, out = self.a, 0.0
        for (k, r), (d0, d1) in cells.items():
            if d0 or d1:
                n0, n1 = self.read[k, r]
                out += _dm_row(n0 + d0, n1 + d1, a) - _dm_row(n0, n1, a)
        return out

    def _ask_cells(self, cells: Dict[tuple, List[float]], pi: int, sq, sign: float) -> None:
        """The read's answers at a square of position pi that holds a piece
        (as BoardSearch._ask) into ``cells``."""
        p = self.pos[pi]
        j = sq[1] * 8 + sq[0]
        kC = KIND_INDEX[p["position"][sq]]
        rows = read_row(j, p["before"][j])
        for k in range(min(kC + 1, len(KINDS))):
            c = cells.setdefault((k, int(rows[k])), [0.0, 0.0])
            c[int(k == kC)] += sign

    def score(self, key: tuple, chosen: List[tuple]) -> float:
        """Bits after the chunk move (exact)."""
        a, K1 = self.a, self.K + 1
        self.stats["scores"] += 1
        rel = BoardSearch._relations(key, chosen)
        A = (self.V + K1 * K1 * self.NR) * a
        n = sum(rel.values())
        nats = self._row_nats(K1) + math.lgamma(n + A) - math.lgamma(A) - sum(_phi(c, a) for c in rel.values())
        cells: Dict[tuple, List[float]] = {}
        for pi, s, t in chosen:
            self._ask_cells(cells, pi, t, -1)
        nats += self.read_nats() + self._read_cells_delta(cells)
        y = self._lab_id(("chunk", self.fresh))
        override = {}
        for pi, s, t in chosen:
            override[self.ev_at[(pi, s)]] = y
            override[self.ev_at[(pi, t)]] = -1
        nats += self._top_after(K1, override) + CHOICE_NATS
        return nats / LN2

    def apply(self, key: tuple, chosen: List[tuple]) -> Tuple[Hashable, set]:
        """Apply a chunk move at the chosen instances; returns the new label
        and the candidate keys made with it."""
        B, rel, C = key
        Y = ("chunk", self.fresh)
        y = self._lab_id(Y)
        touched_pos = set()
        for pi, s, t in chosen:
            p = self.pos[pi]
            kB, kC = KIND_INDEX[p["position"][s]], KIND_INDEX[p["position"][t]]
            i, j = s[1] * 8 + s[0], t[1] * 8 + t[0]
            BoardSearch._ask(self.read, j, p["before"][j], kC, -1)
            for r, lab, d in (((kB, i), B, -1), ((kB, i), Y, 1), ((kC, j), C, -1)):
                self.top[r][lab] += d
                if not self.top[r][lab]:
                    del self.top[r][lab]
            self.rows[Y][("p", B, rel, C)] += 1
            x, y_ = p["tops"][s], p["tops"].pop(t)
            p["tops"][s] = (Y, (x, rel, y_))
            for m_, owner in list(p["owner"].items()):
                if owner == t:
                    p["owner"][m_] = s
            self._lab[self.ev_at[(pi, s)]] = y
            self._lab[self.ev_at[(pi, t)]] = -1
            self.label_pos[Y].add(pi)
            touched_pos.add(pi)
        self._read_cache = None
        self.fresh += 1
        self.moves.append((B, rel, C, Y))
        self._recode_all()
        for pi in touched_pos:
            self._reindex(pi)
        return Y, set(self.label_keys.get(Y, ()))

    # Dissolve moves -------------------------------------------------------- #
    def chunk_labels(self) -> List[Hashable]:
        return [m[3] for m in self.moves]

    def dependents(self, Y) -> set:
        """Y and every chunk type built on it (transitively)."""
        out = {Y}
        for B, _, C, Z in self.moves:
            if B in out or C in out:
                out.add(Z)
        return out

    def _dissolve_plan(self, Y):
        """The moves without Y and its dependents, and per affected position
        its new analysis (the replay of those moves)."""
        gone = self.dependents(Y)
        moves = [m for m in self.moves if m[3] not in gone]
        pis = sorted(set().union(*(self.label_pos.get(l, set()) for l in gone)))
        new = {pi: self.perceive(self.pos[pi]["position"], moves).pos[0] for pi in pis}
        return gone, moves, new

    def _replace_delta(self, new: Dict[int, dict]):
        """Rows after, read cells and event overrides when the given
        positions take the given analyses."""
        rows = {lab: Counter(r) for lab, r in self.rows.items()}
        rows = defaultdict(Counter, rows)
        cells: Dict[tuple, List[float]] = {}
        override: Dict[int, int] = {}
        for pi, q in new.items():
            p = self.pos[pi]
            position = p["position"]
            for node in p["tops"].values():
                node_rows(node, position, rows, -1)
            for node in q["tops"].values():
                node_rows(node, position, rows, 1)
            for sq in position:
                was, now = sq in p["tops"], sq in q["tops"]
                if was != now:
                    self._ask_cells(cells, pi, sq, 1 if now else -1)
                lab = q["tops"][sq][0] if now else None
                old = p["tops"][sq][0] if was else None
                if lab != old:
                    override[self.ev_at[(pi, sq)]] = self._lab_id(lab)
        rows = {lab: Counter({o: c for o, c in r.items() if c}) for lab, r in rows.items()}
        rows = {lab: r for lab, r in rows.items() if r}
        return rows, cells, override

    def score_dissolve(self, Y) -> Tuple[float, tuple]:
        """Bits after dissolving chunk type Y (and its dependents) wherever
        it stands, the positions re-analysed by replay (exact)."""
        self.stats["dissolve scores"] += 1
        gone, moves, new = self._dissolve_plan(Y)
        rows, cells, override = self._replace_delta(new)
        K = len(rows)
        nats = (self._rows_nats(rows, K) + self.read_nats() + self._read_cells_delta(cells)
                + self._top_after(K, override) + CHOICE_NATS)
        return nats / LN2, (gone, moves, new)

    def apply_replace(self, moves: List[tuple], new: Dict[int, dict]) -> set:
        """The given positions take the given analyses and the move list
        becomes ``moves``; returns the candidate keys of those positions."""
        rows, cells, override = self._replace_delta(new)
        self.rows = defaultdict(Counter, rows)
        for (k, r), (d0, d1) in cells.items():
            self.read[k, r, 0] += d0
            self.read[k, r, 1] += d1
        self._read_cache = None
        for e, l in override.items():
            self._lab[e] = l
        keys = set()
        for pi, q in new.items():
            p = self.pos[pi]
            for lab in self._labels_in(p["tops"]):
                self.label_pos[lab].discard(pi)
            for sq, x in p["tops"].items():
                r = (KIND_INDEX[p["position"][sq]], sq[1] * 8 + sq[0])
                self.top[r][x[0]] -= 1
                if not self.top[r][x[0]]:
                    del self.top[r][x[0]]
            p["tops"], p["owner"] = q["tops"], q["owner"]
            for sq, x in p["tops"].items():
                self.top[(KIND_INDEX[p["position"][sq]], sq[1] * 8 + sq[0])][x[0]] += 1
            for lab in self._labels_in(p["tops"]):
                self.label_pos[lab].add(pi)
        for lab in [l for l, s in self.label_pos.items() if not s]:
            del self.label_pos[lab]
        self.moves = list(moves)
        self._recode_all()
        for pi in new:
            keys |= self._reindex(pi)
        return keys

    # Reading out ------------------------------------------------------------ #
    def analyses(self) -> List[List[Node]]:
        return [[p["tops"][sq] for sq in sorted(p["tops"], key=lambda s: (s[1], s[0]))] for p in self.pos]

    def reference(self) -> ContextBoardSearch:
        """The night's search state for the same positions and moves, built
        from scratch (for tests and evaluation only)."""
        ref = ContextBoardSearch([p["position"] for p in self.pos], self.a, self.counts)
        ref.replay(self.moves)
        return ref


class IncrementalChessLearner:
    """Positions one at a time, no night (see the module docstring).

    ``config``: ``steps`` (moves tried per position), ``margin`` (bits: a key
    whose last score was within this of paying is rescored at every new
    instance), ``growth`` (otherwise rescored when its instances grew by
    this fraction)."""

    DEFAULTS = {"steps": 4, "margin": 30.0, "growth": 0.1, "min_instances": 2,
                "dissolve": True, "dissolve_growth": 0.1}

    def __init__(self, seed: int = 0, alpha: float = 0.001, counts: bool = True, config: Optional[dict] = None):
        self.seed, self.alpha, self.counts = seed, alpha, counts
        self.config = dict(self.DEFAULTS, **(config or {}))
        self.state = IncrementalBoardState(alpha, counts)
        self.last: Dict[tuple, Tuple[int, float]] = {}
        self.last_dissolve: Dict[Hashable, int] = {}
        self.history: List[dict] = []
        self.seconds: List[float] = []
        self.model: Optional[Trellis2] = None
        self._grammar_at = None

    # Learning --------------------------------------------------------------- #
    def _due(self, key: tuple, n: int) -> bool:
        last = self.last.get(key)
        if last is None:
            return True
        n0, delta = last
        if n <= n0:
            return False
        if delta < self.config["margin"]:
            return True
        return n >= n0 * (1.0 + self.config["growth"])

    def _chunk_steps(self, keys: set, t0: float) -> None:
        st = self.state
        fresh: set = set()
        for step in range(self.config["steps"]):
            cur = st.bits()
            best = None
            for key in sorted(keys | fresh, key=str):
                n = len(st.index.get(key, ()))
                if n < self.config["min_instances"]:
                    continue
                if key not in fresh and not self._due(key, n):
                    continue
                chosen = st.chosen(key)
                if len(chosen) < 2:
                    self.last[key] = (n, float("inf"))
                    continue
                after = st.score(key, chosen)
                self.last[key] = (n, after - cur)
                if after < cur - 1e-6 and (best is None or after < best[0] - 1e-9):
                    best = (after, key, chosen)
            if best is None:
                break
            after, key, chosen = best
            Y, fresh = st.apply(key, chosen)
            self.last_dissolve[Y] = len(st.pos)
            # Keys that paid this step stay due (K changed under them).
            for k in keys:
                if k in self.last and self.last[k][1] < 0:
                    del self.last[k]
            self.history.append({"position": len(st.pos), "move": f"[{key[0]} {key[1]} {key[2]}] x{len(chosen)}",
                                 "label": str(Y), "bits": after, "seconds": time.time() - t0})

    def _dissolve_step(self, t0: float) -> Optional[set]:
        st = self.state
        n = len(st.pos)
        cur = None
        best = None
        for Y in st.chunk_labels():
            last = self.last_dissolve.get(Y, 0)
            if n < max(last + 1, last * (1.0 + self.config["dissolve_growth"])):
                continue
            self.last_dissolve[Y] = n
            if cur is None:
                cur = st.bits()
            after, plan = st.score_dissolve(Y)
            if after < cur - 1e-6 and (best is None or after < best[0] - 1e-9):
                best = (after, Y, plan)
        if best is None:
            return None
        after, Y, (gone, moves, new) = best
        keys = st.apply_replace(moves, new)
        self.history.append({"position": n, "move": f"dissolve {Y} ({len(gone)} types, {len(new)} positions)",
                             "label": str(Y), "bits": after, "seconds": time.time() - t0})
        return keys

    def observe(self, position: Position) -> None:
        t0 = time.time()
        keys = self.state.add_position(position)
        self._chunk_steps(keys, t0)
        if self.config["dissolve"]:
            keys = self._dissolve_step(t0)
            if keys is not None:
                self._chunk_steps(keys, t0)
        self.seconds.append(time.time() - t0)

    def sleep(self):
        """No night: the grammar is read off the current state."""
        return self.grammar

    # The grammar read off the state ---------------------------------------- #
    def read_grammar(self) -> Tuple[Trellis2, Grammar]:
        """The grammar of the stored analyses: the labels as symbols, one
        rule class per composition key, the read and the contextual top
        level as ``BoardMemory`` codes them (evaluation)."""
        st = self.state
        mem = BoardMemory(counts=self.counts, context=True)
        for p, tops in zip(st.pos, st.analyses()):
            mem.add(p["position"], tops)
        labels: Dict[Hashable, int] = {}
        for lab in sorted(st.rows, key=lambda l: (isinstance(l, tuple), str(l))):
            labels[lab] = len(labels)
        s = np.array([labels[l] for l in mem.label], dtype=np.int64)
        K = len(labels)
        vocab = mem.vocabulary() + [UNK]
        tok_index = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)
        rel_index = {r: i for i, r in enumerate(mem.relations)}
        prim = np.array([k == mem.PRIMITIVE for k in mem.kind])
        keys = []
        for e in range(len(s)):
            if prim[e]:
                keys.append(("P", mem.token[e]))
            else:
                keys.append(("C", int(s[mem.left[e]]), int(s[mem.right[e]]), mem.relation[e]))
        cls: Dict[tuple, int] = {}
        for k in sorted(set(keys), key=lambda k: (k[0], str(k[1:]))):
            cls[k] = len(cls)
        c = np.array([cls[k] for k in keys], dtype=np.int64)
        M = len(cls)
        w = np.array(mem.weight)
        comp = ~prim
        tok = np.array([tok_index[t] if t is not None else -1 for t in mem.token], dtype=np.int64)
        left = np.array(mem.left, dtype=np.int64)
        right = np.array(mem.right, dtype=np.int64)
        rel = np.array([rel_index[r] if r is not None else -1 for r in mem.relation], dtype=np.int64)
        a = self.alpha
        n_U = np.zeros((K, M))
        np.add.at(n_U, (s, c), w)
        n_prim = np.bincount(c[prim], weights=w[prim], minlength=M)
        n_comp = np.bincount(c[comp], weights=w[comp], minlength=M)
        n_E = np.zeros((M, V))
        np.add.at(n_E, (c[prim], tok[prim]), w[prim])
        n_L, n_R = np.zeros((M, K)), np.zeros((M, K))
        np.add.at(n_L, (c[comp], s[left[comp]]), w[comp])
        np.add.at(n_R, (c[comp], s[right[comp]]), w[comp])
        n_Rel = np.zeros((M, len(mem.relations)))
        np.add.at(n_Rel, (c[comp], rel[comp]), w[comp])
        kind = comp.astype(np.int64)
        rules = (dm_code(s, c, w, M, a) + dm_code(c, kind, w, 2, a)
                 + dm_code(c[prim], tok[prim], w[prim], V, a))
        if comp.any():
            rules += (dm_code(c[comp], s[left[comp]], w[comp], K, a) + dm_code(c[comp], s[right[comp]], w[comp], K, a)
                      + dm_code(c[comp], rel[comp], w[comp], len(mem.relations), a))
        top = mem.top_level_nats(s, K, a)
        info = {"bits (factored grammar)": (top + rules) / LN2,
                "structure bits": elias_delta_bits(K) + elias_delta_bits(M)}
        info["total bits"] = info["bits (factored grammar)"] + info["structure bits"]
        info["symbols"], info["rule classes"] = K, M
        info["chunk types"] = len({(int(s[e]), int(s[left[e]]), int(rel[e]), int(s[right[e]]))
                                   for e in np.flatnonzero(comp)})

        def normalize(n):
            n = n + a
            return n / n.sum(axis=-1, keepdims=True)
        top_fields, _ = mem.top_level_tables(s, K, a)
        g = Grammar(vocab=vocab, U=normalize(n_U), pk=(n_prim + a) / (n_prim + n_comp + 2 * a), **top_fields,
                    Lt=normalize(n_L), Rt=normalize(n_R), E=normalize(n_E), alpha=a, info=info, elem_symbol=s,
                    elem_rule=c, relations=list(mem.relations), Rel=normalize(n_Rel))
        model = Trellis2(seed=self.seed, alpha=a, memory=mem)
        model._grammar = g
        return model, g

    @property
    def grammar(self) -> Grammar:
        stamp = (len(self.state.pos), tuple(self.state.moves))
        if self.model is None or self._grammar_at != stamp:
            self.model, _ = self.read_grammar()
            self._grammar_at = stamp
        return self.model.grammar

    def analyse(self, position: Position) -> List[Node]:
        search = BoardSearch([position], self.alpha, self.counts)
        search.replay(self.state.moves)
        return search.analyses()[0]

    def log_prob(self, position: Position) -> float:
        self.grammar
        return self.model.log_prob(position, self.analyse(position))

    def generate(self, n: int, rng: Optional[np.random.Generator] = None, **kw):
        self.grammar
        return self.model.generate(n, rng, **kw)
