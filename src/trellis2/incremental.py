"""TRELLIS v2 without nights: a continuously incremental learner.

The night learner (``unsupervised.UnsupervisedLearner``) stores experiences by
day and, by night, searches the structure again from word classes, replays
every analysis into fresh Cobweb hierarchies over several rounds, re-analyses
every stored experience and chooses among whole-memory candidates. This
learner has no night. Its state is persistent and every unit of work is per
experience or a small bounded step on it:

* the stored analyses, one per experience (elements in a ``Memory``; an
  analysis that is replaced is tombstoned, never replayed);
* the grammar's count tables (``CodeTables``): the Dirichlet-multinomial
  counts of the factored grammar of ``grammar.py`` (rule class given symbol,
  primitive or composite given rule class, token, left and right part, the
  top level and the layout), and with them the exact code of the stored
  analyses, the same ``total bits`` that ``compile_grammar`` reports for the
  same symbols and rule classes;
* the representation hierarchy, a Cobweb tree over every live element's
  description (surface and chunk context in the current categories), kept
  with insertion and removal (``incremental_cobweb``): an element enters
  when its analysis is stored and is re-described (removed and re-inserted)
  when a move changes the categories it is described with;
* the composition hierarchy, a Cobweb tree over the composition keys
  (a primitive's token, a composite's two part symbols) weighted by their
  counts, kept the same way.

Symbols are labels of elements and rule classes are labels of composition
keys, both read from and refined along the hierarchies. Each experience:

1. is perceived with the current grammar (Viterbi analysis; a new word takes
   the category its context implies) and its analysis is added to the counts;
2. local moves touching what it touched are scored exactly from the count
   rows they change, and the best is applied while the code shrinks: chunk
   (B, C) at all its top-level occurrences; dissolve a top-level chunk
   category; merge two symbols; move a unit (a symbol's elements with one
   composition key) to another or a new symbol; split a symbol along the
   representation hierarchy; merge two rule classes; move a key to another or
   a new rule class; split a rule class along the composition hierarchy;
3. a bounded number of stored experiences are re-analysed under the current
   grammar (those the accepted moves touched first, then a few rehearsed at
   random) and their new analyses replace the old ones if the code shrinks.

Every move and re-analysis is accepted only if the total code (``CodeTables``,
order-free: the Dirichlet-multinomial code of the stored analyses plus the
grammar's size) shrinks. Nothing is ever rebuilt from the whole memory.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .chart import Chart
from .data import Tree
from .grammar import UNK, Grammar
from .incremental_cobweb import IncrementalCobwebTree
from .mdl import backoff_chain_coder, backoff_coder, elias_delta_bits
from .memory import BOS, Memory

LN2 = math.log(2.0)
EPS = 1e-9
TABLES = ("U", "KIND", "E", "L", "R", "S", "SP", "MODE", "STOP")
# The read's context options the night's grammar offers (none; one word at
# three weights; two words at nine): naming the one taken costs log2(13) bits
# in ``compile_grammar``'s structure bits, so the same constant is added here.
CONTEXT_OPTION_BITS = math.log2(1 + 3 + 9)
CONTEXT_WEIGHTS = (1.0, 4.0, 16.0)
# None: no context; b1: the word read before, trusted with weight b1 against
# the symbol's own row; (b1, b2): the two words read before, backing off to it.
CONTEXT_OPTIONS = ([None] + list(CONTEXT_WEIGHTS)
                   + [(b1, b2) for b1 in CONTEXT_WEIGHTS for b2 in CONTEXT_WEIGHTS])


def _phi(c: float, a: float) -> float:
    return math.lgamma(c + a) - math.lgamma(a) if c > 0.0 else 0.0


def _norm(n: float, A: float) -> float:
    """lgamma(n + A) - lgamma(A): a row's normalizer (A = alphabet * alpha)."""
    return math.lgamma(n + A) - math.lgamma(A) if n > 0.0 else 0.0


def _elias(n: int) -> float:
    return elias_delta_bits(n) if n >= 1 else 0.0


# --------------------------------------------------------------------------- #
# The count tables and their exact code
# --------------------------------------------------------------------------- #
class Patch:
    """A change of the count tables: cells (table, row, outcome) -> change,
    and the reference counts of symbols, rule classes and tokens (which fix
    the alphabets K, M and V)."""

    __slots__ = ("cells", "refs")

    def __init__(self):
        self.cells: Dict[tuple, float] = defaultdict(float)
        self.refs: Dict[tuple, float] = defaultdict(float)

    def record(self, sym, cls, key, w: float) -> None:
        """An element of symbol ``sym`` whose composition key ``key`` is in
        rule class ``cls`` (weight ``w``; negative to remove)."""
        c = self.cells
        c[("U", sym, cls)] += w
        if key[0] == "P":
            c[("KIND", cls, 0)] += w
            c[("E", cls, key[1])] += w
            self.refs[("tok", key[1])] += w
        else:
            c[("KIND", cls, 1)] += w
            c[("L", cls, key[1])] += w
            c[("R", cls, key[2])] += w
        self.refs[("sym", sym)] += w
        self.refs[("cls", cls)] += w

    def top(self, roots: Sequence, w: float) -> None:
        """An experience's top level (the symbols of its roots, in order):
        one tree, or a forest of two or more pieces."""
        c = self.cells
        if len(roots) == 1:
            c[("S", 0, roots[0])] += w
            c[("MODE", 0, 0)] += w
        elif len(roots) > 1:
            for r in roots:
                c[("SP", 0, r)] += w
            c[("MODE", 0, 1)] += w
            c[("STOP", 0, 0)] += w
            if len(roots) > 2:
                c[("STOP", 0, 1)] += w * (len(roots) - 2)

    def add(self, other: "Patch", sign: float = 1.0) -> "Patch":
        for k, v in other.cells.items():
            self.cells[k] += sign * v
        for k, v in other.refs.items():
            self.refs[k] += sign * v
        return self


class CodeTables:
    """The factored grammar's count tables (as ``grammar._stage_b`` counts
    them) and their code: per table row a Dirichlet-multinomial over the
    table's alphabet (U: the M rule classes; KIND: primitive or composite;
    E: the V tokens with <unk>; L, R, S, SP: the K symbols; MODE, STOP: two),
    plus Elias codes of K and M and the context option. ``delta`` scores any
    ``Patch`` exactly; ``commit`` applies it."""

    def __init__(self, alpha: float, extra_bits: float = CONTEXT_OPTION_BITS):
        self.alpha = alpha
        self.extra_bits = extra_bits
        self.cells: Dict[str, Dict[Hashable, Dict[Hashable, float]]] = {t: {} for t in TABLES}
        self.tot: Dict[str, Dict[Hashable, float]] = {t: {} for t in TABLES}
        self.phis: Dict[str, Dict[Hashable, float]] = {t: {} for t in TABLES}
        self.ref: Dict[str, Dict[Hashable, float]] = {"sym": {}, "cls": {}, "tok": {}}
        self.norm = {t: 0.0 for t in TABLES}
        self.phisum = {t: 0.0 for t in TABLES}
        self.version = 0

    # Sizes ---------------------------------------------------------------- #
    @property
    def K(self) -> int:
        return len(self.ref["sym"])

    @property
    def M(self) -> int:
        return len(self.ref["cls"])

    @property
    def V(self) -> int:
        return len(self.ref["tok"]) + 1

    @staticmethod
    def _alphabets(K: int, M: int, V: int) -> Dict[str, int]:
        return {"U": M, "KIND": 2, "E": V, "L": K, "R": K, "S": K, "SP": K, "MODE": 2, "STOP": 2}

    def structure_bits(self, K: Optional[int] = None, M: Optional[int] = None) -> float:
        K = self.K if K is None else K
        M = self.M if M is None else M
        return _elias(K) + _elias(M) + (self.extra_bits if K else 0.0)

    def nats(self) -> float:
        return sum(self.norm[t] - self.phisum[t] for t in TABLES)

    def bits(self) -> float:
        """The total code (bits): ``info['total bits']`` of the grammar."""
        return self.nats() / LN2 + self.structure_bits()

    # Scoring ------------------------------------------------------------- #
    def _sizes_after(self, refs: Dict[tuple, float]) -> Tuple[int, int, int]:
        d = {"sym": 0, "cls": 0, "tok": 0}
        for (kind, key), v in refs.items():
            if v == 0.0:
                continue
            old = self.ref[kind].get(key, 0.0)
            new = old + v
            was, now = old > EPS, new > EPS
            if was != now:
                d[kind] += 1 if now else -1
        return self.K + d["sym"], self.M + d["cls"], self.V + d["tok"]

    def _row_changes(self, cells: Dict[tuple, float]):
        a = self.alpha
        rows: Dict[str, Dict[Hashable, List[float]]] = defaultdict(dict)
        for (t, row, o), v in cells.items():
            if v == 0.0:
                continue
            c = self.cells[t].get(row, {}).get(o, 0.0)
            c2 = c + v
            if c2 <= EPS:
                c2 = 0.0
            e = rows[t].get(row)
            if e is None:
                e = rows[t][row] = [0.0, 0.0]
            e[0] += v
            e[1] += _phi(c2, a) - _phi(c, a)
        return rows

    def delta(self, patch: Patch) -> float:
        """Change of the total code (bits) if ``patch`` were committed."""
        a = self.alpha
        K2, M2, V2 = self._sizes_after(patch.refs)
        A1 = self._alphabets(self.K, self.M, self.V)
        A2 = self._alphabets(K2, M2, V2)
        rows = self._row_changes(patch.cells)
        d = 0.0
        for t in TABLES:
            changed = rows.get(t, {})
            tot = self.tot[t]
            if A1[t] == A2[t]:
                A = A1[t] * a
                for row, (dt, dp) in changed.items():
                    n = tot.get(row, 0.0)
                    n2 = n + dt
                    if n2 <= EPS:
                        n2 = 0.0
                    d += _norm(n2, A) - _norm(n, A) - dp
            else:
                Aold, Anew = A1[t] * a, A2[t] * a
                new = 0.0
                for row, n in tot.items():
                    if row in changed:
                        continue
                    new += _norm(n, Anew)
                for row, (dt, dp) in changed.items():
                    n2 = tot.get(row, 0.0) + dt
                    new += _norm(n2 if n2 > EPS else 0.0, Anew)
                    d -= dp
                d += new - self.norm[t]
        return d / LN2 + self.structure_bits(K2, M2) - self.structure_bits()

    def commit(self, patch: Patch) -> None:
        self.version += 1
        a = self.alpha
        K1, M1, V1 = self.K, self.M, self.V
        for (kind, key), v in patch.refs.items():
            if v == 0.0:
                continue
            r = self.ref[kind]
            n = r.get(key, 0.0) + v
            if n > EPS:
                r[key] = n
            else:
                r.pop(key, None)
        A1 = self._alphabets(K1, M1, V1)
        A2 = self._alphabets(self.K, self.M, self.V)
        touched: Dict[str, set] = defaultdict(set)
        for (t, row, o), v in patch.cells.items():
            if v == 0.0:
                continue
            cells = self.cells[t].setdefault(row, {})
            c = cells.get(o, 0.0)
            c2 = c + v
            if c2 <= EPS:
                c2 = 0.0
                cells.pop(o, None)
            else:
                cells[o] = c2
            self.phis[t][row] = self.phis[t].get(row, 0.0) + _phi(c2, a) - _phi(c, a)
            self.phisum[t] += _phi(c2, a) - _phi(c, a)
            touched[t].add(row)
        for t in TABLES:
            tot = self.tot[t]
            if A1[t] != A2[t]:
                for row in touched.get(t, ()):
                    self._settle(t, row)
                A = A2[t] * a
                self.norm[t] = math.fsum(_norm(n, A) for n in tot.values())
            else:
                A = A1[t] * a
                for row in touched.get(t, ()):
                    old = _norm(tot.get(row, 0.0), A)
                    self._settle(t, row)
                    self.norm[t] += _norm(tot.get(row, 0.0), A) - old

    def _settle(self, t: str, row) -> None:
        """Recompute a touched row's total (and drop it if empty)."""
        cells = self.cells[t].get(row, {})
        if not cells:
            self.cells[t].pop(row, None)
            self.tot[t].pop(row, None)
            self.phisum[t] -= self.phis[t].pop(row, 0.0)
            return
        self.tot[t][row] = math.fsum(cells.values())

    def snapshot(self):
        """The tables' state (its size depends on the grammar, not on the
        memory), for ``restore`` after a trial."""
        return ({t: {r: dict(c) for r, c in self.cells[t].items()} for t in TABLES},
                {t: dict(self.tot[t]) for t in TABLES}, {t: dict(self.phis[t]) for t in TABLES},
                {k: dict(v) for k, v in self.ref.items()}, dict(self.norm), dict(self.phisum))

    def restore(self, snap) -> None:
        self.version += 1
        cells, tot, phis, ref, norm, phisum = snap
        self.cells, self.tot, self.phis, self.ref = cells, tot, phis, ref
        self.norm, self.phisum = dict(norm), dict(phisum)

    def recompute_bits(self) -> float:
        """The code recomputed from the cells (for tests)."""
        a = self.alpha
        A = self._alphabets(self.K, self.M, self.V)
        nats = 0.0
        for t in TABLES:
            for row, cells in self.cells[t].items():
                n = sum(cells.values())
                nats += _norm(n, A[t] * a) - sum(_phi(c, a) for c in cells.values())
        return nats / LN2 + self.structure_bits()

    def resync(self) -> None:
        """Recompute the cached sums from the cells (removes rounding drift)."""
        a = self.alpha
        A = self._alphabets(self.K, self.M, self.V)
        for t in TABLES:
            self.tot[t] = {row: math.fsum(c.values()) for row, c in self.cells[t].items()}
            self.phis[t] = {row: math.fsum(_phi(v, a) for v in c.values()) for row, c in self.cells[t].items()}
            self.norm[t] = math.fsum(_norm(n, A[t] * a) for n in self.tot[t].values())
            self.phisum[t] = math.fsum(self.phis[t].values())

    def data_bits(self) -> float:
        """The code under the best-fitting parameters (as ``mdl.split_bits``)."""
        nats = 0.0
        for t in TABLES:
            for row, cells in self.cells[t].items():
                n = sum(cells.values())
                nats -= sum(c * math.log(c / n) for c in cells.values() if c > 0)
        return nats / LN2


# --------------------------------------------------------------------------- #
# Word classes from the class-bigram code, kept online
# --------------------------------------------------------------------------- #
class OnlineWordClasses:
    """The night's word classes (``mdl_search.word_classes``: word types
    merged while a class-bigram code shrinks) from counts kept per sentence:
    the bigram table of word types (with sentence start and end) is the
    sufficient statistic, so the merge path is recomputed from it, never from
    the stored sentences, and equals the night's path on the same sentences."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.bigrams: Counter = Counter()        # (previous or <s>, word or </s>) -> count
        self.vocab: set = set()
        self._path = None
        self._version = 0
        self._path_version = -1

    def add(self, tokens: Sequence[str]) -> None:
        prev = None
        for w in tokens:
            self.bigrams[(prev, w)] += 1
            self.vocab.add(w)
            prev = w
        self.bigrams[(prev, None)] += 1
        self._version += 1

    def path(self) -> List[Dict[str, str]]:
        if self._path_version != self._version:
            self._path = self._compute()
            self._path_version = self._version
        return self._path

    def _compute(self) -> List[Dict[str, str]]:
        from scipy.special import gammaln
        alpha = self.alpha
        vocab = sorted(self.vocab)
        K, V, lg = len(vocab), len(vocab) + 1, gammaln
        index = {w: i for i, w in enumerate(vocab)}
        T = np.zeros((K + 1, K + 1))
        for (a, b), n in self.bigrams.items():
            T[K if a is None else index[a], K if b is None else index[b]] += n

        # Counts are whole numbers: phi from a table (the same values gammaln
        # gives for the same arguments).
        table = lg(np.arange(int(T.sum()) * 4 + 2) + alpha) - lg(alpha)

        def phi(x):
            return table[np.asarray(x, dtype=np.int64)]
        emitted = T[:, :K].sum(axis=0)
        members = [[w] for w in vocab]
        alive = list(range(K))
        path = [{w: w for w in vocab}]
        while len(alive) > 1:
            k = len(alive)
            sub = T[np.ix_(alive + [K], alive + [K])]
            X, P = sub[:, :k], phi(sub[:, :k])
            cols = np.zeros((k, k))
            for r in range(k + 1):
                cols += phi(X[r][:, None] + X[r][None, :]) - P[r][:, None] - P[r][None, :]
            rows = np.zeros((k, k))
            for c in range(k):
                rows += phi(X[:k, c][:, None] + X[:k, c][None, :])
            d = np.diag(X[:k])
            Xab = X[:k]
            col_change = (cols - (phi(d[:, None] + Xab) - phi(d)[:, None] - phi(Xab))
                          - (phi(Xab.T + d[None, :]) - phi(Xab.T) - phi(d)[None, :]))
            end = sub[:k, k]
            row_change = (rows - phi(d[:, None] + Xab.T) - phi(Xab + d[None, :])
                          + phi(d[:, None] + Xab + Xab.T + d[None, :])
                          + phi(end[:, None] + end[None, :])
                          - phi(sub[:k]).sum(axis=1)[:, None] - phi(sub[:k]).sum(axis=1)[None, :])
            N = sub.sum(axis=1)

            def norm(n, a):
                return np.where(n > 0, lg(n + a) - lg(a), 0.0)
            after = norm(N, k * alpha)
            norm_change = (after.sum() - after[:k][:, None] - after[:k][None, :]
                           + norm(N[:k][:, None] + N[:k][None, :], k * alpha)
                           - norm(N, (k + 1) * alpha).sum())
            m = emitted[alive]
            emit_change = (norm(m[:, None] + m[None, :], V * alpha)
                           - norm(m, V * alpha)[:, None] - norm(m, V * alpha)[None, :])
            change = (norm_change + emit_change - col_change - row_change) / LN2
            iu = np.triu_indices(k, 1)
            best = int(np.argmin(change[iu]))
            if change[iu][best] >= -1e-9:
                break
            a, b = alive[iu[0][best]], alive[iu[1][best]]
            T[a, :] += T[b, :]
            T[:, a] += T[:, b]
            T[b, :] = 0
            T[:, b] = 0
            emitted[a] += emitted[b]
            members[a] += members[b]
            alive.remove(b)
            path.append({w: vocab[c] for c in alive for w in members[c]})
        return path


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "moves_per_step": 6,        # greedy moves applied per experience at most
    "rehearse": 2,              # stored experiences re-analysed at random per step
    "rehearse_moves": 0,        # stored experiences drawn at random whose moves are searched per step
    "touched": 4,               # re-analyses of experiences touched by moves, per step
    "tree_splits": 6,           # representation-hierarchy split proposals per symbol
    "class_splits": 4,          # composition-hierarchy split proposals per rule class
    "min_pair": 2,              # occurrences for a chunk proposal
    "exchange_targets": 0,      # 0: every symbol; else the nearest few by code
    "redescribe": True,         # keep the representation hierarchy's descriptions current
    "chunk_window": 4,          # joint chunks of 3..this many top-level categories (2: none)
    "seed_merges": True,        # a new word's own symbol is merged where the code says
    "rotations": True,
    "inner_chunks": True,       # chunk (B, C) wherever B and C are adjacent inside trees (one rotation)
    "joint_depth": 2,           # levels a split is carried up to the parents (0: none)          # rotations of local patterns ((A B) C) <-> (A (B C))
    "wc_levels": 12,            # word-class partitions proposed as splits (0: none)
    "wc_every": 1,              # recompute the word-class merge path every this many experiences
    "context": True,            # read the context option chosen by the online code into the grammar
    "compound": 8,
    "lookahead": 6,             # greedy continuation moves scored after each first move (0: none)
    "lookahead_width": 4,       # first moves (structure moves, split compounds) scored by rollout
    "lookahead_region": 8,      # experiences a rollout's continuation looks at
    "lookahead_splits": 4,
    "neighbour_splits": 6,
    "neighbour_hops": 3,        # steps through the analyses from a split to the symbols whose splits may follow it      # symbols next to a rolled-out split whose splits continue it      # best splits also scored by rollout
    "rollout_full": 0,          # experiences of the region whose every move a rollout may take
    "lookahead_dissolve": False,  # rollouts may reopen top-level chunks (dissolve, then rebuild)
    "profile_depth": 3,         # levels of structure profiling a symbol's units for context splits (0: none)
    "graph_joint": 0,           # best splits carried through the composition graph per step (0: none)
    "graph_rounds": 3,          # rounds of that propagation
    "samples": 1,               # analyses stored per experience, drawn from the posterior (1: Viterbi)
    "sample_temperature": 1.0,
    "resample": "always",       # hedged re-analysis: a fresh sample replaces ("always") or only if shorter
    "context_code": False,      # accept moves and re-analyses by the contextual full code
    "context_checks": 4,        # candidates tried in order under it per move
    "surprise_after": 20,       # stalled experiences before a wider search step (0: never)
    "surprise_factor": 4,       # its budget factor
    "rollout_windows": 8,       # joint-chunk windows a rollout continuation scores (most frequent)
    "first_choice": 0,          # take the k-th best first structure decision (a beam's hypotheses)              # best splits (of top-level elements) also scored with their best follow-up chunk
}


class IncrementalLearner:
    """Sentences, one at a time, with no night (see the module docstring).

    ``observe(tokens)`` perceives and stores a sentence and runs the step's
    bounded work; ``grammar`` is read off the count tables at any time.
    ``sleep()`` exists for the experiment scripts' interface and does
    nothing but return the current grammar."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, config: Optional[dict] = None,
                 memory: Optional[Memory] = None, check: bool = False, **memory_kwargs):
        self.seed, self.alpha = seed, alpha
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.memory = memory if memory is not None else Memory(granularities=1, **memory_kwargs)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.check = check
        self.tables = CodeTables(alpha)
        # Elements (memory ids): symbol, alive.
        self.sym: List[int] = []
        self.alive: List[bool] = []
        # Experiences: tokens, element ids of the current analysis, roots in order.
        self.experiences: List[List[str]] = []     # per stored analysis (slot)
        self.real_of: List[int] = []               # slot -> experience
        self.slots: List[List[int]] = []           # experience -> its slots
        self.elems: List[List[int]] = []
        self.roots: List[List[int]] = []
        self.weight: List[float] = []
        # Indexes over live elements.
        self.members: Dict[int, set] = defaultdict(set)          # symbol -> elements
        self.units: Dict[tuple, set] = defaultdict(set)          # (symbol, key) -> elements
        self.rec: Dict[tuple, float] = {}                        # (symbol, key) -> weight
        self.rec_by_sym: Dict[int, set] = defaultdict(set)       # symbol -> keys
        self.rec_by_child: Dict[int, set] = defaultdict(set)     # child symbol -> (symbol, key)
        self.syms_of_key: Dict[tuple, set] = defaultdict(set)    # key -> symbols
        self.class_of: Dict[tuple, int] = {}
        self.keys_of_class: Dict[int, set] = defaultdict(set)
        # The read's context of each rule choice (the word, and the two
        # words, read before the element): counts per (symbol, context) of
        # rule classes, and per record the contexts of its elements.
        self.ctx_rec: Dict[tuple, Counter] = defaultdict(Counter)
        self.uc: Dict[tuple, Counter] = defaultdict(Counter)
        self.uc2: Dict[tuple, Counter] = defaultdict(Counter)
        # Prequential code (bits) of every new experience's rule choices under
        # each context option, before its own counts: the online choice.
        self.preq: Dict[object, float] = {opt: 0.0 for opt in CONTEXT_OPTIONS}
        # Top level: experiences holding each adjacent pair / each symbol.
        self.pair_exps: Dict[tuple, set] = defaultdict(set)
        self.top_exps: Dict[int, set] = defaultdict(set)
        # Hierarchies.
        self.rtree = IncrementalCobwebTree(self.memory.attrs, seed=seed)
        self.placed: Dict[int, Tuple[object, dict]] = {}
        self.ctree = IncrementalCobwebTree(("kind", "tok", "L", "R"), seed=seed)
        self.cplaced: Dict[tuple, List] = {}                     # key -> [(leaf, instance, weight)]
        self.wc = OnlineWordClasses(alpha)
        self._wc_path: List[Dict[str, str]] = []
        self._wc_at = -10 ** 9
        self.next_sym = 0
        self.next_cls = 0
        self._grammar: Optional[Grammar] = None
        self._index = None
        self.queue: List[int] = []
        # Trials (scoring a move by what follows it) journal every change of
        # the counts and indexes and leave the hierarchies alone.
        self._journal: Optional[list] = None
        self._rolling = False
        self._decided = False           # whether a structure move has been taken
        self._stall = 0                 # experiences since a move was last accepted
        self._budget = 1                # lookahead budget factor (surprise steps)
        self._branches: Optional[list] = None   # set by a HedgedBeam: where runner-ups go
        self._branched = False
        self.history: List[dict] = []
        self.stats = Counter()
        self.analyses = self.elems     # interface of the night learner (one analysis per experience)

    # ------------------------------------------------------------------ #
    # Keys and records
    # ------------------------------------------------------------------ #
    def _key(self, e: int, sym: Optional[Dict[int, int]] = None) -> tuple:
        m = self.memory
        if m.kind[e] == Memory.PRIMITIVE:
            return ("P", m.token[e])
        s = self.sym if sym is None else sym
        return ("C", s[m.left[e]], s[m.right[e]])

    def _new_sym(self) -> int:
        self.next_sym += 1
        return self.next_sym - 1

    def _new_cls(self) -> int:
        self.next_cls += 1
        return self.next_cls - 1

    def _top_syms(self, i: int, sym=None) -> List[int]:
        s = self.sym if sym is None else sym
        return [s[r] for r in self.roots[i]]

    # ------------------------------------------------------------------ #
    # Indexes
    # ------------------------------------------------------------------ #
    def _jr(self, fn, *args) -> None:
        """Journal the inverse of a change made during a trial."""
        if self._journal is not None and not self._rolling:
            self._journal.append((fn, args))

    @property
    def _light(self) -> bool:
        return self._journal is not None

    def _ctx(self, e: int) -> Tuple[Hashable, tuple]:
        """The word and the two words read before element e."""
        m = self.memory
        toks, i = m.experiences[m.experience_of[e]], m.span[e][0]
        w1 = toks[i - 1] if i > 0 else BOS
        return w1, (toks[i - 2] if i > 1 else BOS, w1)

    def _ctx_add(self, s_, key, x, w) -> None:
        c = self.class_of[key]
        x1, x2 = x
        for table, k in ((self.uc, (s_, x1)), (self.uc2, (s_, x2))):
            row = table[k]
            n = row[c] + w
            if n > EPS:
                row[c] = n
            else:
                del row[c]
                if not row:
                    del table[k]

    def _index_add(self, e: int) -> None:
        self._jr(self._index_remove, e)
        s, key, w = self.sym[e], self._key(e), self.memory.weight[e]
        x = self._ctx(e)
        self.ctx_rec[(s, key)][x] += w
        self._ctx_add(s, key, x, w)
        self.members[s].add(e)
        self.units[(s, key)].add(e)
        self.rec[(s, key)] = self.rec.get((s, key), 0.0) + w
        self.rec_by_sym[s].add(key)
        self.syms_of_key[key].add(s)
        if key[0] == "C":
            for c in key[1:]:
                self.rec_by_child[c].add((s, key))

    def _index_remove(self, e: int) -> None:
        self._jr(self._index_add, e)
        s, key, w = self.sym[e], self._key(e), self.memory.weight[e]
        x = self._ctx(e)
        cr = self.ctx_rec[(s, key)]
        n = cr[x] - w
        if n > EPS:
            cr[x] = n
        else:
            del cr[x]
            if not cr:
                del self.ctx_rec[(s, key)]
        self._ctx_add(s, key, x, -w)
        self.members[s].discard(e)
        if not self.members[s]:
            del self.members[s]
        u = self.units[(s, key)]
        u.discard(e)
        if not u:
            del self.units[(s, key)]
        n = self.rec[(s, key)] - w
        if n > EPS:
            self.rec[(s, key)] = n
            return
        del self.rec[(s, key)]
        self.rec_by_sym[s].discard(key)
        if not self.rec_by_sym[s]:
            del self.rec_by_sym[s]
        self.syms_of_key[key].discard(s)
        if not self.syms_of_key[key]:
            del self.syms_of_key[key]
        if key[0] == "C":
            for c in key[1:]:
                rb = self.rec_by_child.get(c)
                if rb is not None:
                    rb.discard((s, key))
                    if not rb:
                        del self.rec_by_child[c]

    def _key_weight(self, key) -> float:
        return sum(self.rec.get((s, key), 0.0) for s in self.syms_of_key.get(key, ()))

    def _set_class(self, key, cls) -> None:
        old = self.class_of.get(key)
        if old == cls:
            return
        self._jr(self._restore_class, key, old)
        # The key's elements now choose another class: move their context counts.
        if old is not None:
            for s_ in self.syms_of_key.get(key, ()):
                for x, w in self.ctx_rec.get((s_, key), {}).items():
                    self._ctx_add(s_, key, x, -w)
        self._set_class_raw(key, old, cls)
        if old is not None:
            for s_ in self.syms_of_key.get(key, ()):
                for x, w in self.ctx_rec.get((s_, key), {}).items():
                    self._ctx_add(s_, key, x, w)

    def _set_class_raw(self, key, old, cls) -> None:
        if old is not None:
            self.keys_of_class[old].discard(key)
            if not self.keys_of_class[old]:
                del self.keys_of_class[old]
        self.class_of[key] = cls
        self.keys_of_class[cls].add(key)

    def _restore_class(self, key, cls) -> None:
        if cls is not None:
            self._set_class(key, cls)
            return
        old = self.class_of.pop(key)
        self.keys_of_class[old].discard(key)
        if not self.keys_of_class[old]:
            del self.keys_of_class[old]

    def _set_sym(self, e: int, s_: int) -> None:
        self.sym[e] = s_

    def _set_alive(self, e: int, v: bool) -> None:
        self.alive[e] = v

    def _set_analysis(self, i: int, value) -> None:
        if value is None:
            self.elems.pop()
            self.roots.pop()
            self.weight.pop()
        else:
            self.elems[i], self.roots[i], self.weight[i] = value

    def _top_index(self, i: int, add: bool) -> None:
        self._jr(self._top_index, i, not add)
        labs = self._top_syms(i)
        for a in labs:
            (self.top_exps[a].add(i) if add else self.top_exps[a].discard(i))
        for p in zip(labs, labs[1:]):
            (self.pair_exps[p].add(i) if add else self.pair_exps[p].discard(i))
        if not add:
            for a in labs:
                if not self.top_exps.get(a, True):
                    del self.top_exps[a]
            for p in zip(labs, labs[1:]):
                if not self.pair_exps.get(p, True):
                    del self.pair_exps[p]

    # ------------------------------------------------------------------ #
    # The representation hierarchy (descriptions kept current)
    # ------------------------------------------------------------------ #
    def _describe(self, e: int) -> dict:
        return self.memory.instance(e, [self.sym])

    def _place(self, e: int) -> None:
        if self._light:
            return
        x = self._describe(e)
        leaf = self.rtree.ifit(x, self.memory.weight[e])
        self.placed[e] = (leaf, x)
        self.stats["rep inserts"] += 1

    def _unplace(self, e: int) -> None:
        if self._light:
            return
        leaf, x = self.placed.pop(e)
        self.rtree.unfit(leaf, x, self.memory.weight[e])
        self.stats["rep removals"] += 1

    def _redescribe(self, exps: Iterable[int]) -> None:
        """Re-describe the elements of these experiences whose description
        changed (categories of their neighbours): remove and re-insert."""
        if not self.config["redescribe"] or self._light:
            return
        for i in exps:
            for e in self.elems[i]:
                x = self._describe(e)
                leaf, old = self.placed[e]
                if x != old:
                    self.rtree.unfit(leaf, old, self.memory.weight[e])
                    self.placed[e] = (self.rtree.ifit(x, self.memory.weight[e]), x)
                    self.stats["rep re-descriptions"] += 1

    # The composition hierarchy over keys -------------------------------- #
    @staticmethod
    def _cinstance(key) -> dict:
        if key[0] == "P":
            return {"kind": "P", "tok": key[1], "L": "-", "R": "-"}
        return {"kind": "C", "tok": "-", "L": f"S{key[1]}", "R": f"S{key[2]}"}

    def _csync(self, keys: Iterable[tuple]) -> None:
        """Bring the composition hierarchy's weight of each key to its count."""
        if self._light:
            return
        for key in keys:
            have = sum(w for _, _, w in self.cplaced.get(key, ()))
            want = self._key_weight(key)
            if abs(have - want) <= EPS:
                continue
            if want > have:
                x = self._cinstance(key)
                leaf = self.ctree.ifit(x, want - have)
                self.cplaced.setdefault(key, []).append((leaf, x, want - have))
            else:
                excess = have - want
                kept = []
                for leaf, x, w in reversed(self.cplaced.get(key, [])):
                    if excess > EPS:
                        take = min(w, excess)
                        self.ctree.unfit(leaf, x, take)
                        excess -= take
                        if w - take > EPS:
                            kept.append((leaf, x, w - take))
                    else:
                        kept.append((leaf, x, w))
                kept.reverse()
                if kept:
                    self.cplaced[key] = kept
                else:
                    self.cplaced.pop(key, None)

    # ------------------------------------------------------------------ #
    # Patches
    # ------------------------------------------------------------------ #
    def _class_for(self, key, fallback) -> int:
        """The rule class of a key: its own if it is in use, else ``fallback``."""
        if key in self.class_of and self._key_weight(key) > EPS:
            return self.class_of[key]
        return fallback

    def _tree_patch(self, tokens, tree: Tree, labels: Dict, classes: Dict, w: float, sign: float,
                    patch: Patch) -> None:
        """Add (sign +1) or remove (-1) the records of an analysis given as a
        Tree with a symbol per span; ``classes`` maps each key to its class."""
        for span in tree.bottom_up():
            i, j = span
            if j - i == 1:
                key = ("P", tokens[i])
            else:
                k = tree.split[span]
                key = ("C", labels[(i, k)], labels[(k, j)])
            patch.record(labels[span], classes[key], key, sign * w)
        patch.top([labels[r] for r in tree.roots], sign * w)

    def _experience_patch(self, i: int, sign: float, patch: Patch) -> None:
        """Add or remove the records of experience i's stored analysis."""
        w = self.weight[i]
        for e in self.elems[i]:
            key = self._key(e)
            patch.record(self.sym[e], self.class_of[key], key, sign * self.memory.weight[e])
        patch.top(self._top_syms(i), sign * w)

    def own_classes(self, mapping: Dict[int, int]) -> Dict[tuple, int]:
        """The coupled class split of a symbol split: the keys that only the
        relabelled elements use (all of their occurrences move) leave their
        rule class for a new one, one new class per (old class, new symbol),
        so that the new symbol's tokens and parts are coded apart."""
        weight: Dict[tuple, float] = defaultdict(float)
        target: Dict[tuple, set] = defaultdict(set)
        for e, s in mapping.items():
            key = self._key(e)
            weight[key] += self.memory.weight[e]
            target[key].add(s)
        remap, fresh = {}, {}
        for key in sorted(weight, key=str):
            if len(target[key]) != 1 or abs(weight[key] - self._key_weight(key)) > EPS:
                continue
            c = self.class_of[key]
            slot = (c, next(iter(target[key])))
            if slot not in fresh:
                fresh[slot] = self.next_cls + len(fresh)
            remap[key] = fresh[slot]
        # A class all of whose live keys move keeps its label (no new class).
        for (c, s), c2 in list(fresh.items()):
            live = {k for k in self.keys_of_class[c] if self._key_weight(k) > EPS}
            moved = {k for k, v in remap.items() if v == c2}
            if live == moved:
                for k in moved:
                    remap[k] = c
        return remap

    def _relabel_patch(self, mapping: Dict[int, int], class_remap: Optional[Dict[tuple, int]] = None,
                       fresh_renamed: bool = False) -> Tuple[Patch, Dict[tuple, int]]:
        """Elements relabelled (element -> new symbol): their records, their
        parents' keys, and the top level of the experiences they are roots in.
        A key that is new takes the class of the heaviest old key that
        becomes it; ``class_remap`` moves keys of the relabelled elements to
        other classes (``own_classes``); with ``fresh_renamed`` a new key of
        a relabelled element takes a new class (one per old class and new
        symbol) instead (a joint split: the parents' compositions are coded
        apart too). Returns the patch and the classes of the keys whose
        class is new or changed."""
        m = self.memory
        class_remap = class_remap or {}
        fresh_votes: Dict[tuple, Counter] = defaultdict(Counter)
        affected = set(mapping)
        for e in mapping:
            p = m.parent[e]
            if p >= 0:
                affected.add(p)
        new_sym = lambda e: mapping.get(e, self.sym[e])
        patch = Patch()
        votes: Dict[tuple, Counter] = defaultdict(Counter)
        moves = []
        for e in affected:
            key = self._key(e)
            if m.kind[e] == Memory.PRIMITIVE:
                key2 = key
            else:
                key2 = ("C", new_sym(m.left[e]), new_sym(m.right[e]))
            w = m.weight[e]
            moves.append((e, key, key2, w))
            if key2 != key:
                votes[key2][self.class_of[key]] += w
                if fresh_renamed and e in mapping:
                    fresh_votes[key2][(self.class_of[key], mapping[e])] += w
        new_classes = dict(class_remap)
        nxt = max([self.next_cls] + [c + 1 for c in class_remap.values()])
        slots: Dict[tuple, int] = {}
        for key2, v in sorted(votes.items(), key=lambda kv: str(kv[0])):
            if key2 in self.class_of and self._key_weight(key2) > EPS:
                new_classes[key2] = self.class_of[key2]
            elif key2 in fresh_votes:
                slot = max(fresh_votes[key2].items(), key=lambda kv: (kv[1], str(kv[0])))[0]
                if slot not in slots:
                    slots[slot] = nxt
                    nxt += 1
                new_classes[key2] = slots[slot]
            else:
                new_classes[key2] = max(v.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        for e, key, key2, w in moves:
            patch.record(self.sym[e], self.class_of[key], key, -w)
            c2 = new_classes.get(key2, self.class_of.get(key2)) if key2 != key else new_classes.get(key, self.class_of[key])
            patch.record(new_sym(e), c2, key2, w)
        exps = {m.experience_of[e] for e in mapping if m.is_root[e]}
        for mid in exps:
            i = self._exp_of_mid[mid]
            patch.top(self._top_syms(i), -self.weight[i])
            patch.top([new_sym(r) for r in self.roots[i]], self.weight[i])
        return patch, new_classes

    def _profile(self, els: Iterable[int]) -> Tuple[Counter, Counter]:
        """What relabelling the elements ``els`` to one symbol touches,
        aggregated: each affected record (the elements and their parents) as
        (symbol, key, itself moved, left part moved, right part moved) ->
        weight, and the moved roots' top-level cells (table, symbol) ->
        weight. Computed once, it scores the move to every target symbol."""
        m = self.memory
        els = els if isinstance(els, (set, frozenset)) else set(els)
        aff: Counter = Counter()
        tops: Counter = Counter()
        seen = set()
        for e in els:
            for x in (e, m.parent[e]):
                if x < 0 or x in seen:
                    continue
                seen.add(x)
                lm = rm = False
                if m.kind[x] == Memory.COMPOSITE:
                    lm, rm = m.left[x] in els, m.right[x] in els
                aff[(self.sym[x], self._key(x), x in els, lm, rm)] += m.weight[x]
            if m.is_root[e]:
                whole = m.top_left[e] < 0 and m.top_right[e] < 0
                tops[("S" if whole else "SP", self.sym[e])] += m.weight[e]
        return aff, tops

    def _profile_patch(self, profile, T: int, class_remap: Optional[Dict[tuple, int]] = None
                       ) -> Tuple[Patch, Dict[tuple, int]]:
        """``_relabel_patch`` for a move of a profiled set to symbol T."""
        aff, tops = profile
        patch = Patch()
        votes: Dict[tuple, Counter] = defaultdict(Counter)
        moves = []
        for (s_, key, me, lm, rm), w in aff.items():
            key2 = key if key[0] == "P" else ("C", T if lm else key[1], T if rm else key[2])
            moves.append((s_, key, T if me else s_, key2, w))
            if key2 != key:
                votes[key2][self.class_of[key]] += w
        new_classes = dict(class_remap or {})
        for key2, v in votes.items():
            if key2 in self.class_of and self._key_weight(key2) > EPS:
                new_classes[key2] = self.class_of[key2]
            else:
                new_classes[key2] = max(v.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        for s_, key, s2, key2, w in moves:
            patch.record(s_, self.class_of[key], key, -w)
            c2 = new_classes.get(key2) if key2 != key else new_classes.get(key, self.class_of[key])
            patch.record(s2, c2, key2, w)
        for (t, s_), w in tops.items():
            patch.cells[(t, 0, s_)] -= w
            patch.cells[(t, 0, T)] += w
        return patch, new_classes

    def _merge_patch(self, A: int, B: int, class_map: Optional[Dict[int, int]] = None
                     ) -> Tuple[Patch, Dict[tuple, int]]:
        """Symbol B merged into A, scored from the aggregated records (the same
        patch as relabelling every element of B, without visiting them);
        ``class_map`` merges classes used only by A and B as well."""
        ren = lambda s: A if s == B else s
        recs = {(B, k) for k in self.rec_by_sym.get(B, ())} | set(self.rec_by_child.get(B, ()))
        if class_map:
            recs |= {(A, k) for k in self.rec_by_sym.get(A, ())}
        patch = Patch()
        votes: Dict[tuple, Counter] = defaultdict(Counter)
        moves = []
        for (s, key) in recs:
            w = self.rec[(s, key)]
            key2 = key if key[0] == "P" else ("C", ren(key[1]), ren(key[2]))
            moves.append((s, key, key2, w))
            if key2 != key:
                votes[key2][self.class_of[key]] += w
        new_classes = {}
        for key2, v in votes.items():
            if key2 in self.class_of and self._key_weight(key2) > EPS:
                new_classes[key2] = self.class_of[key2]
            else:
                new_classes[key2] = max(v.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if class_map:
            for _, key, key2, _ in moves:
                c = new_classes[key2] if key2 != key else self.class_of[key]
                if c in class_map:
                    new_classes[key2] = class_map[c]
        for s, key, key2, w in moves:
            patch.record(s, self.class_of[key], key, -w)
            c2 = new_classes[key2] if key2 in new_classes else self.class_of[key]
            patch.record(ren(s), c2, key2, w)
        for t in ("S", "SP"):
            n = self.tables.cells[t].get(0, {}).get(B, 0.0)
            if n:
                patch.cells[(t, 0, B)] -= n
                patch.cells[(t, 0, A)] += n
        return patch, new_classes

    def _class_patch(self, keys: Iterable[tuple], cls: int) -> Patch:
        """Keys moved to rule class ``cls``."""
        patch = Patch()
        for key in keys:
            old = self.class_of[key]
            if old == cls:
                continue
            for s in self.syms_of_key.get(key, ()):
                w = self.rec[(s, key)]
                patch.record(s, old, key, -w)
                patch.record(s, cls, key, w)
        return patch

    # ------------------------------------------------------------------ #
    # Committing
    # ------------------------------------------------------------------ #
    @property
    def _exp_of_mid(self) -> Dict[int, int]:
        return self.__dict__.setdefault("_mid_map", {})

    def _install(self, i: int, tokens, tree: Tree, labels: Dict, classes: Dict, w: float) -> None:
        """Store an analysis of experience i (its records are committed by
        the caller's patch): memory elements, indexes, hierarchies."""
        m = self.memory
        start = len(m)
        mid = len(m.experiences)
        m.add(tokens, tree, w)
        self._exp_of_mid[mid] = i
        self._jr(self._exp_of_mid.pop, mid)
        ids = list(range(start, len(m)))
        for e in ids:
            self.sym.append(labels[m.span[e]])
            self.alive.append(True)
        roots = sorted((e for e in ids if m.is_root[e]), key=lambda e: m.span[e])
        if i == len(self.elems):
            self._jr(self._set_analysis, i, None)
            self.elems.append(ids)
            self.roots.append(roots)
            self.weight.append(w)
        else:
            self._jr(self._set_analysis, i, (self.elems[i], self.roots[i], self.weight[i]))
            self.elems[i], self.roots[i], self.weight[i] = ids, roots, w
        for key, c in classes.items():
            if key not in self.class_of or self._key_weight(key) <= EPS:
                self._set_class(key, c)
        for e in ids:
            self._index_add(e)
        self._top_index(i, True)
        for e in ids:
            self._place(e)
        self._csync({self._key(e) for e in ids})
        self._grammar = None

    def _uninstall(self, i: int) -> None:
        keys = {self._key(e) for e in self.elems[i]}
        self._top_index(i, False)
        for e in self.elems[i]:
            self._unplace(e)
            self._index_remove(e)
            self._jr(self._set_alive, e, True)
            self.alive[e] = False
        self._csync(keys)
        self._grammar = None

    def _replace(self, i: int, tree: Tree, labels: Dict, classes: Dict, patch: Patch) -> None:
        self.tables.commit(patch)
        self._uninstall(i)
        self._install(i, self.experiences[i], tree, labels, classes, self.weight[i])
        self._after_commit()

    # Trials -------------------------------------------------------------- #
    _MEMORY_LISTS = ("kind", "token", "left", "right", "relation", "parent", "top_left", "top_right",
                     "is_root", "weight", "label", "experience_of", "span", "experiences")

    def _txn_begin(self) -> dict:
        """Start a trial: from here every change of the counts and indexes is
        journaled (``_txn_rollback`` undoes it exactly) and the hierarchies
        are not touched. Trials nest."""
        m = self.memory
        tok = {"outer": self._journal, "tables": self.tables.snapshot(),
               "next": (self.next_sym, self.next_cls), "queue": list(self.queue),
               "mem": {a: len(getattr(m, a)) for a in self._MEMORY_LISTS},
               "n_sym": len(self.sym), "n_alive": len(self.alive)}
        self._journal = []
        tok["journal"] = self._journal
        return tok

    def _txn_rollback(self, tok: dict) -> None:
        self._rolling = True
        try:
            for fn, args in reversed(tok["journal"]):
                fn(*args)
        finally:
            self._rolling = False
        self._journal = tok["outer"]
        self.tables.restore(tok["tables"])
        m = self.memory
        for a, n in tok["mem"].items():
            del getattr(m, a)[n:]
        m.__dict__.pop("_top_level_kept", None)
        del self.sym[tok["n_sym"]:]
        del self.alive[tok["n_alive"]:]
        self.next_sym, self.next_cls = tok["next"]
        self.queue = tok["queue"]
        self._grammar = None

    def _apply_relabel(self, mapping: Dict[int, int], new_classes: Dict[tuple, int], patch: Patch,
                       light: bool = False) -> None:
        """Commit a relabelling: counts, indexes and, unless ``light`` (a
        trial, see ``_trial``), the hierarchies."""
        m = self.memory
        affected = set(mapping)
        for e in mapping:
            if m.parent[e] >= 0:
                affected.add(m.parent[e])
        old_keys = {self._key(e) for e in affected}
        exps = {self._exp_of_mid[m.experience_of[e]] for e in mapping}
        for i in exps:
            self._top_index(i, False)
        for e in affected:
            self._index_remove(e)
        for e, s_ in mapping.items():
            self._jr(self._set_sym, e, self.sym[e])
            self.sym[e] = s_
        for key, c in new_classes.items():
            self._set_class(key, c)
        for e in affected:
            self._index_add(e)
        for i in exps:
            self._top_index(i, True)
        self.tables.commit(patch)
        self._grammar = None
        if light or self._light:
            return
        self._redescribe(exps | {self._exp_of_mid[m.experience_of[e]] for e in affected})
        self._csync(old_keys | {self._key(e) for e in affected})
        self._grammar = None
        self._after_commit()

    def _apply_classes(self, keys: Iterable[tuple], cls: int, patch: Patch) -> None:
        for key in keys:
            self._set_class(key, cls)
        self.tables.commit(patch)
        self._grammar = None
        self._after_commit()

    def _after_commit(self) -> None:
        if self._light:
            return
        self.stats["commits"] += 1
        if self.check:
            self.verify()

    # ------------------------------------------------------------------ #
    # The grammar read off the count tables
    # ------------------------------------------------------------------ #
    @property
    def grammar(self) -> Grammar:
        if self._grammar is None:
            self._grammar = self._read_grammar()
        return self._grammar

    def _read_grammar(self) -> Grammar:
        t, a = self.tables, self.alpha
        syms = sorted(t.ref["sym"])
        clss = sorted(t.ref["cls"])
        vocab = sorted(t.ref["tok"]) + [UNK]
        si = {s: k for k, s in enumerate(syms)}
        ci = {c: k for k, c in enumerate(clss)}
        vi = {w: k for k, w in enumerate(vocab)}
        K, M, V = len(syms), len(clss), len(vocab)
        nU = np.zeros((K, M))
        for s, row in t.cells["U"].items():
            for c, n in row.items():
                nU[si[s], ci[c]] = n
        kind = np.zeros((M, 2))
        for c, row in t.cells["KIND"].items():
            for o, n in row.items():
                kind[ci[c], o] = n
        nE = np.zeros((M, V))
        for c, row in t.cells["E"].items():
            for w, n in row.items():
                nE[ci[c], vi[w]] = n
        nL, nR = np.zeros((M, K)), np.zeros((M, K))
        for tab, arr in (("L", nL), ("R", nR)):
            for c, row in t.cells[tab].items():
                for s, n in row.items():
                    arr[ci[c], si[s]] = n
        nS, nP = np.zeros(K), np.zeros(K)
        for tab, arr in (("S", nS), ("SP", nP)):
            for s, n in t.cells[tab].get(0, {}).items():
                arr[si[s]] = n
        mode = t.cells["MODE"].get(0, {})
        stop = t.cells["STOP"].get(0, {})

        def normalize(n):
            n = n + a
            return n / n.sum(axis=-1, keepdims=True)
        total = t.bits()
        data = t.data_bits()
        info = {"total bits": total, "bits (factored grammar)": t.nats() / LN2,
                "structure bits": t.structure_bits(), "data bits": data,
                "model bits": total - data, "symbols": K, "rule classes": M,
                "composite rule classes": int(np.sum(kind[:, 1] > 0)),
                "chunk types": sum(1 for (s, k) in self.rec if k[0] == "C"),
                "bits per sentence": total / max(len(self.slots), 1)}
        g = Grammar(vocab=vocab, S=normalize(nS), U=normalize(nU),
                    pk=(kind[:, 0] + a) / (kind.sum(axis=1) + 2 * a),
                    Lt=normalize(nL), Rt=normalize(nR), E=normalize(nE), alpha=a,
                    p_stop=float((stop.get(0, 0.0) + a) / (sum(stop.values()) + 2 * a)),
                    p_whole=float((mode.get(0, 0.0) + a) / (sum(mode.values()) + 2 * a)),
                    S_piece=normalize(nP), info=info)
        if not self.config["context"]:
            opt = None
        elif self.config["context_code"]:
            opt = self.full_code()[1]
            info["total bits (order-free)"] = info["total bits"]
            info["total bits"] = self.full_code()[0]
        else:
            opt = self.context_option
        info["read context (online choice)"] = str(opt)
        if opt is not None:
            b1 = opt[0] if isinstance(opt, tuple) else opt
            Un = normalize(nU)
            contexts = sorted({x for (_, x) in self.uc} | {BOS}, key=str)
            xi = {x: k for k, x in enumerate(contexts)}
            nUc = np.zeros((K, len(contexts), M))
            for (A, x), row in self.uc.items():
                for c, n in row.items():
                    nUc[si[A], xi[x], ci[c]] = n
            Uc = (nUc + b1 * Un[:, None, :]) / (nUc.sum(axis=2, keepdims=True) + b1)
            Uc2 = None
            if isinstance(opt, tuple):
                rows: Dict[tuple, Tuple[list, list]] = defaultdict(lambda: ([], []))
                for (A, x2), row in self.uc2.items():
                    r = np.zeros(M)
                    for c, n in row.items():
                        r[ci[c]] = n
                    base = Uc[si[A], xi[x2[-1]]]
                    rows[x2][0].append(si[A])
                    rows[x2][1].append((r + opt[1] * base) / (r.sum() + opt[1]))
                Uc2 = {x2: (np.array(As, dtype=np.int64), np.array(rs)) for x2, (As, rs) in rows.items()}
            g.contexts = contexts
            g.Uc, g.Uc2 = Uc, Uc2
            g.context_index = {x: k for k, x in enumerate(contexts)}
        g.inc_symbols, g.inc_classes = syms, clss
        return g

    def _parse(self, tokens: Sequence[str]) -> Tuple[Tree, Dict, Dict]:
        """The Viterbi analysis under the current grammar: the tree, its
        symbol labels and the classes of its keys (a key in use keeps its
        class; a new key takes the class that best derives it)."""
        g = self.grammar
        return self._analysis(tokens, Chart(g, tokens).viterbi_tree(), g)

    def _sample(self, tokens: Sequence[str]) -> Tuple[Tree, Dict, Dict]:
        """An analysis drawn from the posterior under the current grammar (at
        ``sample_temperature``), with the classes of its keys as ``_parse``."""
        g = self.grammar
        T = self.config["sample_temperature"]
        gt = g if T == 1.0 else g.tempered(T)
        return self._analysis(tokens, Chart(gt, tokens).sample_tree(self.np_rng), g)

    def _analysis(self, tokens, tree: Tree, g: Grammar) -> Tuple[Tree, Dict, Dict]:
        syms, clss = g.inc_symbols, g.inc_classes
        labels = {span: syms[int(a)] for span, a in tree.label.items()}
        classes = {}
        ids = g.token_ids(tokens)
        for span in tree.bottom_up():
            i, j = span
            A = int(tree.label[span])
            Urow = g.rules(g.read(tokens, i))[A]
            if j - i == 1:
                key = ("P", tokens[i])
                score = Urow * g.pk * g.E[:, ids[i]]
            else:
                k = tree.split[span]
                key = ("C", labels[(i, k)], labels[(k, j)])
                score = Urow * g.qk * g.Lt[:, int(tree.label[(i, k)])] * g.Rt[:, int(tree.label[(k, j)])]
            if key not in classes:
                classes[key] = self._class_for(key, clss[int(np.argmax(score))])
        return tree, labels, classes

    # ------------------------------------------------------------------ #
    # Perception and the step
    # ------------------------------------------------------------------ #
    def observe(self, tokens: Sequence[str]) -> Tree:
        t0 = time.time()
        tokens = list(tokens)
        i = len(self.experiences)
        self.wc.add(tokens)
        # Hedged incorporation: k analyses drawn from the posterior, each of
        # weight 1/k (k = 1: the Viterbi analysis), all from the grammar
        # before this experience.
        k = self.config["samples"]
        analyses = []
        own: Dict[str, tuple] = {}
        for t in range(k):
            if self.tables.K == 0:
                a = self._flat(tokens, own)
            else:
                a = self._parse(tokens) if k == 1 else self._sample(tokens)
                # A new word: its own symbol and rule class, merged below where
                # the code says (so its first category is the code's).
                if self.config["seed_merges"]:
                    a = self._own_new_words(tokens, *a, own=own)
            analyses.append(a)
        r = len(self.slots)
        self.slots.append([])
        for t, (tree, labels, classes) in enumerate(analyses):
            # A key an earlier sample put in use keeps that sample's class.
            classes = {key: self._class_for(key, c) for key, c in classes.items()}
            slot = len(self.experiences)
            self.experiences.append(tokens)
            self.real_of.append(r)
            self.slots[r].append(slot)
            patch = Patch()
            self._tree_patch(tokens, tree, labels, classes, 1.0 / k, +1.0, patch)
            if t == 0:
                self._prequential(tokens, tree, labels, classes)
            self.tables.commit(patch)
            self._install(slot, tokens, tree, labels, classes, 1.0 / k)
            self._after_commit()
        tree = analyses[0][0]
        self.stats["observed"] += 1
        self._step(i)
        self.history.append({"experience": i, "bits": self.tables.bits(), "seconds": time.time() - t0,
                             "symbols": self.tables.K, "rule classes": self.tables.M})
        return tree

    def _flat(self, tokens, own=None):
        """Before any grammar: every word a piece, each new word type its own
        symbol and rule class."""
        n = len(tokens)
        tree = Tree(n, {}, {}, [(k, k + 1) for k in range(n)])
        labels, classes = {}, {}
        seen = own if own is not None else {}
        for k, w in enumerate(tokens):
            if w not in seen:
                seen[w] = (self._new_sym(), self._new_cls())
            labels[(k, k + 1)] = seen[w][0]
            classes[("P", w)] = seen[w][1]
        tree.label = dict(labels)
        return tree, labels, classes

    def _own_new_words(self, tokens, tree, labels, classes, own=None):
        known = self.tables.ref["tok"]
        new = {w for w in tokens if w not in known}
        if not new:
            return tree, labels, classes
        own = own if own is not None else {}
        for w in sorted(new):
            if w not in own:
                own[w] = (self._new_sym(), self._new_cls())
        # Relabel the primitives of new words and the keys of their parents.
        labels = dict(labels)
        for k, w in enumerate(tokens):
            if w in own:
                labels[(k, k + 1)] = own[w][0]
        classes = {}
        for span in tree.bottom_up():
            i, j = span
            if j - i == 1:
                key = ("P", tokens[i])
                if tokens[i] in own:
                    classes[key] = own[tokens[i]][1]
                else:
                    classes[key] = self.class_of[key] if key in self.class_of else self._new_cls()
            else:
                k = tree.split[span]
                key = ("C", labels[(i, k)], labels[(k, j)])
                if key not in classes:
                    classes[key] = self._class_for(key, self._new_cls())
        return tree, labels, classes

    def _prequential(self, tokens, tree: Tree, labels: Dict, classes: Dict) -> None:
        """Code a new experience's rule choices (each element's class given
        its symbol) under every context option, from the counts before it
        (and before its own earlier elements): the online code by which the
        grammar's context option is chosen (``context_option``)."""
        a = self.alpha
        M = max(self.tables.M, 1)
        U = self.tables.cells["U"]
        local_u: Counter = Counter()
        local_1: Counter = Counter()
        local_2: Counter = Counter()
        for span in tree.bottom_up():
            i, j = span
            A = labels[span]
            if j - i == 1:
                key = ("P", tokens[i])
            else:
                k = tree.split[span]
                key = ("C", labels[(i, k)], labels[(k, j)])
            c = classes[key]
            w1 = tokens[i - 1] if i > 0 else BOS
            x2 = (tokens[i - 2] if i > 1 else BOS, w1)
            row = U.get(A, {})
            nA = sum(row.values()) + sum(v for (s_, _), v in local_u.items() if s_ == A)
            p0 = (row.get(c, 0.0) + local_u[(A, c)] + a) / (nA + M * a)
            r1 = self.uc.get((A, w1), {})
            n1 = sum(r1.values()) + sum(v for (s_, x, _), v in local_1.items() if s_ == A and x == w1)
            c1 = r1.get(c, 0.0) + local_1[(A, w1, c)]
            r2 = self.uc2.get((A, x2), {})
            n2 = sum(r2.values()) + sum(v for (s_, x, _), v in local_2.items() if s_ == A and x == x2)
            c2 = r2.get(c, 0.0) + local_2[(A, x2, c)]
            for opt in CONTEXT_OPTIONS:
                if opt is None:
                    p = p0
                elif isinstance(opt, tuple):
                    p1 = (c1 + opt[0] * p0) / (n1 + opt[0])
                    p = (c2 + opt[1] * p1) / (n2 + opt[1])
                else:
                    p = (c1 + opt * p0) / (n1 + opt)
                self.preq[opt] -= math.log2(p)
            local_u[(A, c)] += 1.0
            local_1[(A, w1, c)] += 1.0
            local_2[(A, x2, c)] += 1.0

    @property
    def context_option(self):
        """The context option with the shortest online code so far (ties: the
        simpler)."""
        return min(CONTEXT_OPTIONS, key=lambda o: (round(self.preq[o], 6), CONTEXT_OPTIONS.index(o)))

    def sleep(self) -> Grammar:
        """No night: the current grammar."""
        return self.grammar

    def _step(self, i: int) -> None:
        cfg = self.config
        # Surprise: after ``surprise_after`` experiences without an accepted
        # move, one step searches with a wider budget (more first moves and
        # splits rolled out); bounded, and on average cheap.
        wide = cfg["surprise_after"] > 0 and self._stall >= cfg["surprise_after"]
        if wide:
            self._budget = cfg["surprise_factor"]
            self._stall = 0
            self.stats["surprise steps"] += 1
        moved = False
        try:
            for _ in range(cfg["moves_per_step"]):
                if not self._best_move(i):
                    break
                moved = True
        finally:
            self._budget = 1
        self._stall = 0 if moved else self._stall + 1
        # Re-analysis: the new experience under the grammar it changed, the
        # experiences touched by moves, and a few rehearsed at random.
        todo = [i]
        while self.queue and len(todo) < 1 + cfg["touched"]:
            j = self.queue.pop(0)
            if j not in todo:
                todo.append(j)
        n = len(self.experiences)
        for _ in range(cfg["rehearse"]):
            if n > 1:
                todo.append(self.rng.randrange(n))
        for j in todo:
            self.reanalyse(j)
        # Rehearsal of moves: the moves touching a few stored experiences
        # drawn at random are searched as for the new one (the night's search,
        # spread over the stream a few experiences at a time).
        for _ in range(cfg["rehearse_moves"]):
            if n > 1:
                j = self.rng.randrange(n - 1)
                for _ in range(cfg["moves_per_step"]):
                    if not self._best_move(j):
                        break

    def reanalyse(self, i: int) -> bool:
        """Re-parse a stored experience under the current grammar; keep the
        new analysis if the total code shrinks."""
        tokens = self.experiences[i]
        hedged = self.config["samples"] > 1
        tree, labels, classes = self._sample(tokens) if hedged else self._parse(tokens)
        if self._same_analysis(i, tree, labels):
            return False
        patch = Patch()
        self._experience_patch(i, -1.0, patch)
        self._tree_patch(tokens, tree, labels, classes, self.weight[i], +1.0, patch)
        d = self.tables.delta(patch)
        self.stats["reanalyses scored"] += 1
        # Hedged: a fresh posterior sample replaces the stored one (the stored
        # samples follow the posterior); otherwise kept only if shorter.
        if self.config["context_code"] and not (hedged and self.config["resample"] == "always"):
            d = self._full_delta(lambda: self._replace(i, tree, labels, classes, patch))
        if d >= -1e-9 and not (hedged and self.config["resample"] == "always"):
            return False
        self._replace(i, tree, labels, classes, patch)
        self.stats["reanalyses accepted"] += 1
        return True

    def _same_analysis(self, i: int, tree: Tree, labels: Dict) -> bool:
        m = self.memory
        old = {m.span[e]: self.sym[e] for e in self.elems[i]}
        if set(old) != set(labels) or any(old[s] != labels[s] for s in old):
            return False
        old_roots = [m.span[r] for r in self.roots[i]]
        if old_roots != list(tree.roots):
            return False
        for e in self.elems[i]:
            if m.kind[e] == Memory.COMPOSITE:
                sp = m.span[e]
                if tree.split.get(sp) != m.span[m.left[e]][1]:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Moves
    # ------------------------------------------------------------------ #
    STRUCTURE = ("chunk", "joint", "inner chunk", "rotate", "dissolve", "compound")

    def _best_move(self, i: int) -> bool:
        """Score the moves touching experience i and apply the best if it
        shortens the code. A split of a category pays only once structure
        uses it, so the best few splits are also scored with their best
        follow-up chunk (compounds). With ``lookahead`` > 0, the best few
        structure moves are scored by what they lead to: their own change
        plus that of a bounded greedy continuation of structure moves in the
        experiences they touch (a rollout, undone exactly), as the night's
        beam search judges a move by the codes reachable after it."""
        cfg = self.config
        pool, splits, every = [], [], []
        for d, move in self._candidates(i):
            self.stats["moves scored"] += 1
            pool.append((d, move))
            if move[0] == "relabel" and len(move) > 4:
                every.append((d, move))
            if move[0] == "relabel" and len(move) > 4 and any(self.memory.is_root[e] for e in move[1]):
                splits.append((d, move))
        if cfg["graph_joint"] > 0 and every:
            # Joint splits through the composition graph, for the best few splits.
            for d, move in sorted(every, key=lambda x: x[0])[:cfg["graph_joint"]]:
                mp = move[1]
                first = min(mp)
                seedmap = {e: v for e, v in mp.items() if self.sym[e] == self.sym[first]}
                joint = self._propagate_graph(seedmap, cfg["graph_rounds"])
                self.stats["graph joint splits scored"] += 1
                if len(joint) > len(seedmap):
                    patch, nc = self._relabel_patch(joint, self.own_classes(joint))
                    jm = ("relabel", joint, nc, patch, "split")
                    dj = self.tables.delta(patch)
                    pool.append((dj, jm))
                    if any(self.memory.is_root[e] for e in joint):
                        splits.append((dj, jm))
        if cfg["compound"] > 0 and splits:
            splits.sort(key=lambda x: x[0])
            for d, move in splits[:cfg["compound"]]:
                undo = self._trial(move)
                follows = sorted(self._followups(move[1]), key=lambda x: x[0])
                self._undo(undo)
                self.stats["compounds scored"] += 1
                for f in follows[:max(1, cfg["lookahead_width"] if cfg["lookahead"] else 1)]:
                    pool.append((d + f[0], ("compound", move, f[1])))
        if not pool:
            return False
        pool.sort(key=lambda x: x[0])
        best = pool[0] if pool[0][0] < -1e-9 else None
        if cfg["lookahead"] > 0:
            budget = getattr(self, "_budget", 1)
            firsts = [x for x in pool if x[1][0] in self.STRUCTURE][:cfg["lookahead_width"] * budget]
            if cfg["lookahead_splits"]:
                # Splits whose gain shows only after the moves that use them
                # (the substantive groupings: word classes, structural context).
                seen_sets = set()
                for d, move in sorted(every, key=lambda x: x[0]):
                    if move[4] not in ("word classes", "context"):
                        continue
                    sig = frozenset(move[1])
                    if sig in seen_sets:
                        continue
                    seen_sets.add(sig)
                    firsts.append((d, move))
                    if len(seen_sets) >= cfg["lookahead_splits"] * budget:
                        break
            if cfg["lookahead_dissolve"]:
                # Reopening: dissolving the experience's top-level chunks never
                # pays alone; it is judged by what the continuation rebuilds.
                firsts += [x for x in pool if x[1][0] == "dissolve" and x not in firsts]
            if best is not None and best[1][0] not in self.STRUCTURE:
                firsts.append(best)
            scored = []
            for d, move in firsts:
                value = d + self._rollout(i, move, cfg["lookahead"])
                self.stats["rollouts"] += 1
                scored.append((value, d, move))
            if scored:
                scored.sort(key=lambda x: x[0])
                value, d, move = scored[0]
                # A hypothesis of a beam may take another first structure
                # decision (``first_choice``: the k-th best by rollout).
                k = cfg["first_choice"]
                if k and not self._decided and move[0] in self.STRUCTURE:
                    ok = [x for x in scored if x[0] < -1e-9 and x[2][0] in self.STRUCTURE]
                    if len(ok) > k:
                        value, d, move = ok[k]
                        best = None
                if value < -1e-9 and (best is None or value < best[0] - 1e-9):
                    best = (d, move)
                    self.stats["lookahead choices"] += 1
                # Hedging: a beam (``HedgedBeam``) keeps the runner-up structure
                # decision alive as another hypothesis.
                if self._branches is not None and not self._branched:
                    alts = [x for x in scored if x[0] < -1e-9 and x[2][0] in self.STRUCTURE
                            and (best is None or x[2] is not best[1])]
                    if alts and best is not None and best[1][0] in self.STRUCTURE:
                        self._branched = True
                        branches, self._branches = self._branches, None
                        child = __import__("copy").deepcopy(self)
                        self._branches = branches
                        child._apply(alts[0][2])
                        child.stats["branched"] += 1
                        branches.append(child)
        if cfg["context_code"]:
            # Accept under the code the grammar uses: the best candidates in
            # order, the first that shortens the contextual full code.
            order = ([best] if best is not None else []) + [x for x in pool if best is None or x[1] is not best[1]]
            best = None
            for d, move in order[:cfg["context_checks"]]:
                if self._full_delta(lambda mv=move: self._apply(mv)) < -1e-9:
                    best = (d, move)
                    break
        if best is not None and best[1][0] in self.STRUCTURE:
            self._decided = True
        if best is None:
            return False
        self._apply(best[1])
        self.stats["moves accepted"] += 1
        self.stats[f"accepted {best[1][0]}"] += 1
        return True

    def _rollout(self, i: int, move, depth: int) -> float:
        """The change of a greedy continuation after ``move`` (applied as a
        trial): up to ``depth`` best structure moves (chunks, joint and inner
        chunks, and merges of the symbols the continuation created) among the
        experiences touched so far; everything undone."""
        tok = self._txn_begin()
        try:
            q0, s0 = len(self.queue), self.next_sym
            near = set()
            if move[0] == "relabel" and self.config["neighbour_splits"]:
                # After a split, the splits of the symbols next to it (its
                # elements' parents, siblings and parts) are its follow-ups.
                ring = set(list(move[1])[:2000])
                seen_e = set(ring)
                for _ in range(self.config["neighbour_hops"]):
                    nxt_ring = set()
                    for e in ring:
                        for _, n in self._neighbours(e):
                            if n not in seen_e:
                                seen_e.add(n)
                                nxt_ring.add(n)
                                near.add(self.sym[n])
                    ring = nxt_ring
            self._apply(move)
            total = 0.0
            cap = self.config["lookahead_region"]
            for _ in range(depth):
                region = [i] + [j for j in self.queue[q0:] if j != i][:cap - 1]
                fresh = [A for A in range(s0, self.next_sym) if A in self.members]
                best = None
                full = self.config["rollout_full"]
                cands = (c for j in region[:full] for c in self._candidates(j)) if full else ()
                if near:
                    cands = itertools.chain(cands, self._neighbour_split_candidates(near))
                for d, mv in itertools.chain(self._rollout_candidates(region, fresh), cands):
                    if best is None or d < best[0]:
                        best = (d, mv)
                if best is None or best[0] >= -1e-9:
                    break
                self._apply(best[1])
                total += best[0]
            return total
        finally:
            self._txn_rollback(tok)

    def _neighbour_split_candidates(self, syms):
        cap = self.config["neighbour_splits"]
        for B in sorted((b for b in syms if b in self.members), key=lambda b: -len(self.members[b]))[:cap]:
            for groups in self._context_splits(B) + self._word_class_splits(B):
                yield from self._split_candidates(groups)

    def _rollout_candidates(self, region: Sequence[int], fresh: Sequence[int]):
        win = self.config["chunk_window"]
        pairs, windows = set(), set()
        for j in region:
            labs = self._top_syms(j)
            pairs |= set(zip(labs, labs[1:]))
            if win >= 3:
                for size in range(3, min(win, len(labs)) + 1):
                    for t in range(len(labs) - size + 1):
                        windows.add(tuple(labs[t:t + size]))
        for pair in sorted(pairs):
            yield from self._chunk_moves(pair)
        if windows:
            from .mdl_search import shapes
            # Proposals bounded: the most frequent windows (occurrences), as
            # the night keeps its ``top_windows``.
            occs = [(w, self._window_occ(w)) for w in sorted(windows)]
            occs = [(w, o, sum(len(p) for p in o.values())) for w, o in occs]
            occs = sorted((x for x in occs if x[2] >= self.config["min_pair"]), key=lambda x: -x[2])
            for window, occ, _ in occs[:self.config["rollout_windows"]]:
                for shape in shapes(0, len(window)):
                    yield from self._joint_move(window, shape, occ)
        if self.config["inner_chunks"]:
            inner = set()
            for j in region:
                inner |= self._adjacent_pairs(j)
            for pair in sorted(inner):
                yield from self._inner_chunk(pair)
        if self.config["lookahead_dissolve"]:
            m = self.memory
            for Y in sorted({self.sym[r] for j in region for r in self.roots[j] if m.kind[r] == Memory.COMPOSITE}):
                d, move = self._dissolve_move(Y)
                if move is not None:
                    yield d, move
        for A in fresh:
            for B in sorted(self.members):
                if B != A:
                    a, b = (A, B) if A < B else (B, A)
                    patch, nc = self._merge_patch(a, b)
                    yield self.tables.delta(patch), ("merge", a, b, patch, nc)

    def _trial(self, move) -> dict:
        """Apply a move as a trial (``_txn_begin``); ``_undo`` restores."""
        tok = self._txn_begin()
        self._apply(move)
        return tok

    def _undo(self, tok: dict) -> None:
        self._txn_rollback(tok)

    def _followups(self, mapping: Dict[int, int]):
        """Chunk and joint chunk moves over top-level pairs and windows that
        hold a symbol just created by ``mapping`` (scored in the trial state):
        (code change, what to redo)."""
        m = self.memory
        new = set(mapping.values())
        exps = {self._exp_of_mid[m.experience_of[e]] for e in mapping if m.is_root[e]}
        pairs, windows = set(), set()
        win = self.config["chunk_window"]
        for i in exps:
            labs = self._top_syms(i)
            for a, b in zip(labs, labs[1:]):
                if a in new or b in new:
                    pairs.add((a, b))
            for size in range(3, min(win, len(labs)) + 1):
                for t in range(len(labs) - size + 1):
                    w = tuple(labs[t:t + size])
                    if new & set(w):
                        windows.add(w)
        for pair in sorted(pairs):
            for d, _ in self._chunk_moves(pair):
                yield d, ("chunk", pair)
        for window in sorted(windows):
            occ = self._window_occ(window)
            if sum(len(p) for p in occ.values()) < self.config["min_pair"]:
                continue
            from .mdl_search import shapes
            for shape in shapes(0, len(window)):
                for d, _ in self._joint_move(window, shape, occ):
                    yield d, ("joint", window, shape)

    def _window_occ(self, window) -> Dict[int, List[int]]:
        from .mdl_search import match_positions
        occ = {}
        for i in sorted(self.pair_exps.get(window[:2], ())):
            pos = match_positions(self._top_syms(i), window)
            if pos:
                occ[i] = pos
        return occ

    def _candidates(self, i: int):
        m = self.memory
        syms = sorted({self.sym[e] for e in self.elems[i]})
        keys = sorted({self._key(e) for e in self.elems[i]}, key=str)
        all_syms = sorted(self.members)
        # Merges of the experience's symbols with every other symbol, alone
        # and with their own rule classes merged too.
        seen = set()
        for A in syms:
            for B in all_syms:
                if A == B or (min(A, B), max(A, B)) in seen:
                    continue
                seen.add((min(A, B), max(A, B)))
                a, b = (A, B) if A < B else (B, A)
                for class_map in self._merge_class_maps(a, b):
                    patch, nc = self._merge_patch(a, b, class_map)
                    yield self.tables.delta(patch), ("merge", a, b, patch, nc)
        # Units of the experience moved to another symbol or a new one.
        units = sorted({(self.sym[e], self._key(e)) for e in self.elems[i]}, key=str)
        for (A, key) in units:
            els = frozenset(self.units.get((A, key), ()))
            if not els:
                continue
            prof = self._profile(els)
            own = self.own_classes({e: A for e in els})
            for B in all_syms:
                if B == A:
                    continue
                for remap in self._join_classes(own, B):
                    patch, nc = self._profile_patch(prof, B, remap)
                    yield self.tables.delta(patch), ("move", els, B, nc, patch)
            if len(self.members[A]) > len(els):
                yield from self._split_candidates([set(els)])
        # Splits along the representation hierarchy, and of a symbol's
        # primitives by the word classes.
        for A in syms:
            for groups in self._tree_splits(A):
                yield from self._split_candidates(groups)
            for groups in self._word_class_splits(A):
                yield from self._split_candidates(groups, "word classes")
            for groups in self._context_splits(A):
                yield from self._split_candidates(groups, "context")
        # Chunks of adjacent top-level categories, and dissolving chunks.
        labs = self._top_syms(i)
        for pair in sorted(set(zip(labs, labs[1:]))):
            yield from self._chunk_moves(pair)
        win = self.config["chunk_window"]
        if win >= 3 and len(labs) >= 3:
            yield from self._joint_chunk_moves(labs, win)
        if self.config["rotations"]:
            yield from self._rotation_moves(i)
        if self.config["inner_chunks"]:
            yield from self._inner_chunk_moves(i)
        for Y in sorted({self.sym[r] for r in self.roots[i] if m.kind[r] == Memory.COMPOSITE}):
            d, move = self._dissolve_move(Y)
            if move is not None:
                yield d, move
        # Rule classes: merges, key moves, splits along the composition hierarchy.
        classes = sorted({self.class_of[k] for k in keys})
        all_cls = sorted(c for c in self.keys_of_class if self._live_keys(c))
        seen = set()
        for c in classes:
            for c2 in all_cls:
                if c == c2 or (min(c, c2), max(c, c2)) in seen:
                    continue
                seen.add((min(c, c2), max(c, c2)))
                a, b = (c, c2) if c < c2 else (c2, c)
                live = self._live_keys(b)
                patch = self._class_patch(live, a)
                yield self.tables.delta(patch), ("classes", live, a, patch)
        for key in keys:
            c = self.class_of[key]
            targets = [c2 for c2 in all_cls if c2 != c]
            if len(self._live_keys(c)) > 1:
                targets.append(self.next_cls)
            for c2 in targets:
                patch = self._class_patch([key], c2)
                yield self.tables.delta(patch), ("classes", [key], c2, patch)
        for c in classes:
            for subset in self._class_splits(c):
                patch = self._class_patch(subset, self.next_cls)
                yield self.tables.delta(patch), ("classes", subset, self.next_cls, patch)

    def _live_keys(self, c: int) -> List[tuple]:
        return sorted((k for k in self.keys_of_class.get(c, ()) if self._key_weight(k) > EPS), key=str)

    def _split_candidates(self, groups: List[set], source: str = "other"):
        """Groups of a symbol's elements, each relabelled to a new symbol:
        alone, and with their own rule classes (``own_classes``)."""
        mapping = {}
        for k, g in enumerate(groups):
            for e in g:
                mapping[e] = self.next_sym + k
        for remap in (None, self.own_classes(mapping)):
            if remap is not None and not remap:
                continue
            patch, nc = self._relabel_patch(mapping, remap)
            yield self.tables.delta(patch), ("relabel", mapping, nc, patch, source)
        # Joint splits: the split carried to the parents holding the new parts.
        last = len(mapping)
        for depth in range(1, self.config["joint_depth"] + 1):
            joint = self._propagate(mapping, depth)
            if len(joint) == last:
                break
            last = len(joint)
            patch, nc = self._relabel_patch(joint, self.own_classes(joint), fresh_renamed=True)
            yield self.tables.delta(patch), ("relabel", joint, nc, patch, source)

    def _neighbours(self, e: int):
        m = self.memory
        p = m.parent[e]
        if p >= 0:
            yield "parent", p
            yield "sibling", (m.right[p] if m.left[p] == e else m.left[p])
        if m.kind[e] == Memory.COMPOSITE:
            yield "left", m.left[e]
            yield "right", m.right[e]

    def _propagate_graph(self, mapping: Dict[int, int], rounds: int) -> Dict[int, int]:
        """A split carried through the composition graph (as the night's joint
        split, ``splits.propagate``, but locally): each round, the elements
        next to moved ones (parent, sibling, parts) are grouped by which new
        symbols they neighbour, and every symbol whose elements so divide is
        split by group (the relations by which they touch moved elements, so
        each symbol splits into a few parts at most); then each propagated
        part is merged back while that shortens the exact code (pruning the
        parts that only cost)."""
        mapping = dict(mapping)
        nxt = max(mapping.values()) + 1
        frontier = set(mapping)
        parts: Dict[int, set] = {}
        for _ in range(rounds):
            sig: Dict[int, Dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
            for e in frontier:
                for _, n in self._neighbours(e):
                    if n in mapping:
                        continue
                    key = ()                     # one part per symbol: its elements next to moved ones
                    sig[self.sym[n]][key].add(n)
            frontier = set()
            for B in sorted(sig):
                groups = sig[B]
                if len(groups) == 1 and sum(len(g) for g in groups.values()) == len(self.members.get(B, ())):
                    continue
                for k in sorted(groups, key=str):
                    for n in groups[k]:
                        mapping[n] = nxt
                    parts[nxt] = set(groups[k])
                    nxt += 1
                    frontier |= groups[k]
            if not frontier:
                break
        if not parts:
            return mapping
        score = lambda mp: self.tables.delta(self._relabel_patch(mp, self.own_classes(mp))[0])
        current = score(mapping)
        for part in sorted(parts, key=lambda q: len(parts[q])):
            trial = {e: v for e, v in mapping.items() if v != part}
            value = score(trial)
            if value < current - 1e-9:
                mapping, current = trial, value
        return mapping

    def _propagate(self, mapping: Dict[int, int], depth: int) -> Dict[int, int]:
        """A split carried up ``depth`` levels: a symbol's elements whose
        parts moved to new symbols are split by which new parts they hold
        (each combination a new symbol), unless all of its elements hold the
        same (then its compositions change but it stays one symbol)."""
        m = self.memory
        mapping = dict(mapping)
        frontier = set(mapping)
        nxt = max(mapping.values()) + 1
        for _ in range(depth):
            groups: Dict[int, Dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
            for e in frontier:
                p = m.parent[e]
                if p < 0 or p in mapping:
                    continue
                groups[self.sym[p]][(mapping.get(m.left[p]), mapping.get(m.right[p]))].add(p)
            frontier = set()
            for P in sorted(groups):
                g = groups[P]
                if len(g) == 1 and sum(len(x) for x in g.values()) == len(self.members[P]):
                    continue
                for sig in sorted(g, key=str):
                    for p in g[sig]:
                        mapping[p] = nxt
                    nxt += 1
                    frontier |= g[sig]
            if not frontier:
                break
        return mapping

    def _join_classes(self, own: Dict[tuple, int], B: int):
        """Class options for elements moved into symbol B: their keys keep
        their classes, or the keys only they use (``own``: their
        ``own_classes``) join B's heaviest class of the same kind."""
        yield None
        if not own:
            return
        best: Dict[str, Tuple[float, int]] = {}
        for key in self.rec_by_sym.get(B, ()):
            c = self.class_of[key]
            kind = key[0]
            w = self.rec[(B, key)]
            if kind not in best or w > best[kind][0]:
                best[kind] = (w, c)
        remap = {}
        for key in own:
            if key[0] in best:
                remap[key] = best[key[0]][1]
        if remap:
            yield remap

    def _merge_class_maps(self, A: int, B: int):
        """Class options of a symbol merge: none, or the classes used only
        by A and B merged by kind (primitive and composite classes apart)."""
        yield None
        mine = set()
        for s in (A, B):
            for key in self.rec_by_sym.get(s, ()):
                mine.add(self.class_of[key])
        groups: Dict[str, List[int]] = defaultdict(list)
        for c in sorted(mine):
            live = self._live_keys(c)
            if not live or any(not self.syms_of_key[k] <= {A, B} for k in live):
                continue
            kinds = {k[0] for k in live}
            if len(kinds) == 1:
                groups[kinds.pop()].append(c)
        out = {}
        for kind in ("P", "C"):
            cs = groups.get(kind, [])
            if len(cs) > 1:
                out.update({c: cs[0] for c in cs[1:]})
                yield dict(out)

    def _apply(self, move) -> None:
        kind = move[0]
        if kind == "compound":
            _, split, spec = move
            self._apply(split)
            if spec[0] == "chunk":
                follow = list(self._chunk_moves(spec[1]))
            else:
                follow = list(self._joint_move(spec[1], spec[2], self._window_occ(spec[1])))
            self._apply(follow[0][1])
            return
        if kind == "move":
            _, els, B, nc, patch = move
            move = ("relabel", {e: B for e in els}, nc, patch)
            kind = "relabel"
        if kind == "merge":
            _, A, B, patch, nc = move
            mapping = {e: A for e in self.members[B]}
            self._apply_relabel(mapping, nc, patch)
            self._touch(mapping)
        if kind == "relabel":
            _, mapping, nc, patch = move[:4]
            for s in set(mapping.values()):
                self.next_sym = max(self.next_sym, s + 1)
            for c in set(nc.values()):
                self.next_cls = max(self.next_cls, c + 1)
            self._apply_relabel(mapping, nc, patch)
            self._touch(mapping)
        elif kind == "classes":
            _, keys, c, patch = move
            if c >= self.next_cls:
                self.next_cls = c + 1
            self._apply_classes(keys, c, patch)
        elif kind in ("chunk", "dissolve", "joint", "rotate", "inner chunk"):
            _, rewrites, patch = move[:3]
            if callable(rewrites):
                rewrites = rewrites()
            self.tables.commit(patch)
            fresh = move[3] if len(move) > 3 else {}
            for s in fresh.get("syms", ()):
                self.next_sym = max(self.next_sym, s + 1)
            for c in fresh.get("classes", ()):
                self.next_cls = max(self.next_cls, c + 1)
            for i, (tree, labels, classes) in rewrites.items():
                self._uninstall(i)
                self._install(i, self.experiences[i], tree, labels, classes, self.weight[i])
            self._after_commit()
            for i in rewrites:
                self._enqueue(i)

    def _touch(self, mapping) -> None:
        m = self.memory
        for e in mapping:
            self._enqueue(self._exp_of_mid[m.experience_of[e]])

    def _enqueue(self, i: int) -> None:
        if i not in self.queue:
            self.queue.append(i)

    # Structure moves ------------------------------------------------------ #
    def _rewrite(self, i: int, groups: Dict[Tuple[int, int], List]) -> Tuple[Tree, Dict, Dict]:
        """Experience i's analysis with top-level runs grouped: ``groups``
        maps the index of a run's first root to (number of roots, list of
        (span, symbol, left span, right span) bottom up) for the new
        composites."""
        m = self.memory
        n = len(self.experiences[i])
        split, labels, classes = {}, {}, {}
        for e in self.elems[i]:
            labels[m.span[e]] = self.sym[e]
            classes[self._key(e)] = self.class_of[self._key(e)]
            if m.kind[e] == Memory.COMPOSITE:
                split[m.span[e]] = m.span[m.left[e]][1]
        roots, r = [], 0
        spans = [m.span[x] for x in self.roots[i]]
        while r < len(spans):
            if r in groups:
                size, nodes = groups[r]
                for span, s, ls, rs, cls in nodes:
                    split[span] = ls[1]
                    labels[span] = s
                    classes[("C", labels[ls], labels[rs])] = cls
                roots.append((spans[r][0], spans[r + size - 1][1]))
                r += size
            else:
                roots.append(spans[r])
                r += 1
        return Tree(n, split, labels, roots), labels, classes

    def _chunk_moves(self, pair):
        """Chunk (B, C): every non-overlapping top-level occurrence of the
        pair becomes a composite of a new symbol Y. Its key's rule class is
        its own if the key is in use, else a new class or any existing one,
        whichever codes shortest."""
        B, C = pair
        key = ("C", B, C)
        Y = self.next_sym
        occ = {}
        n = 0
        for i in sorted(self.pair_exps.get(pair, ())):
            labs = self._top_syms(i)
            pos, t = [], 0
            while t + 1 < len(labs):
                if labs[t] == B and labs[t + 1] == C:
                    pos.append(t)
                    t += 2
                else:
                    t += 1
            if pos:
                occ[i] = pos
                n += len(pos)
        if n < self.config["min_pair"]:
            return
        if key in self.class_of and self._key_weight(key) > EPS:
            options = [self.class_of[key]]
        else:
            options = [self.next_cls] + sorted(self.keys_of_class)
        base = Patch()
        for i, pos in occ.items():
            labs = self._top_syms(i)
            new, t = [], 0
            starts = set(pos)
            while t < len(labs):
                if t in starts:
                    new.append(Y)
                    t += 2
                else:
                    new.append(labs[t])
                    t += 1
            base.top(labs, -self.weight[i])
            base.top(new, self.weight[i])
        w_tot = sum(self.weight[i] * len(pos) for i, pos in occ.items())
        best = None
        for c in options:
            patch = Patch().add(base)
            patch.record(Y, c, key, w_tot)
            d = self.tables.delta(patch)
            if best is None or d < best[0]:
                best = (d, c, patch)
        d, c, patch = best
        m = self.memory

        def rewrites():                 # built only if the move is applied
            out = {}
            for i, pos in occ.items():
                spans = [m.span[x] for x in self.roots[i]]
                groups = {t: (2, [((spans[t][0], spans[t + 1][1]), Y, spans[t], spans[t + 1], c)]) for t in pos}
                out[i] = self._rewrite(i, groups)
            return out
        yield d, ("chunk", rewrites, patch, {"syms": [Y], "classes": [c]})

    def _joint_chunk_moves(self, labs, win):
        """Joint chunks: a window of 3..``win`` adjacent top-level symbols of
        the experience, at all its non-overlapping top-level occurrences, as a
        nested composite (every bracketing), each internal node a new symbol
        unless a symbol's composition already holds its pair (as
        ``mdl_search.CompositeState``)."""
        from .mdl_search import match_positions, shape_spans, shapes
        seen = set()
        for size in range(3, min(win, len(labs)) + 1):
            for t in range(len(labs) - size + 1):
                window = tuple(labs[t:t + size])
                if window in seen:
                    continue
                seen.add(window)
                # Experiences holding the window: those holding its first pair.
                occ = {}
                for i in sorted(self.pair_exps.get(window[:2], ())):
                    pos = match_positions(self._top_syms(i), window)
                    if pos:
                        occ[i] = pos
                n = sum(len(p) for p in occ.values())
                if n < self.config["min_pair"]:
                    continue
                for shape in shapes(0, size):
                    yield from self._joint_move(window, shape, occ)

    def _joint_move(self, window, shape, occ):
        from .mdl_search import shape_spans
        spans_ = shape_spans(shape)               # internal nodes bottom up (leaf-index spans)
        holders = {}
        fresh_syms, nodes = [], []
        lab_of: Dict[Tuple[int, int], int] = {(k, k + 1): window[k] for k in range(len(window))}
        new_pairs = {}
        for (a, b) in spans_:
            # Children of this internal node: find the split.
            for k in range(a + 1, b):
                if (a, k) in lab_of and (k, b) in lab_of:
                    break
            l, r = lab_of[(a, k)], lab_of[(k, b)]
            key = ("C", l, r)
            if (l, r) in new_pairs:
                s = new_pairs[(l, r)]
            else:
                holder = max(((self.rec[(s2, key)], -s2) for s2 in self.syms_of_key.get(key, ())), default=None)
                if holder is not None:
                    s = -holder[1]
                else:
                    s = self.next_sym + len(fresh_syms)
                    fresh_syms.append(s)
                new_pairs[(l, r)] = s
            lab_of[(a, b)] = s
            nodes.append(((a, b), s, (a, k), (k, b), key))
        root = lab_of[(0, len(window))]
        new_cls, classes = [], {}
        for (_, s, _, _, key) in nodes:
            if key in classes:
                continue
            if key in self.class_of and self._key_weight(key) > EPS:
                classes[key] = self.class_of[key]
            else:
                classes[key] = self.next_cls + len(new_cls)
                new_cls.append(classes[key])
        patch = Patch()
        m = self.memory
        rewrites = {}
        size = len(window)
        for i, pos in occ.items():
            labs = self._top_syms(i)
            new, t, starts = [], 0, set(pos)
            while t < len(labs):
                if t in starts:
                    new.append(root)
                    t += size
                else:
                    new.append(labs[t])
                    t += 1
            patch.top(labs, -self.weight[i])
            patch.top(new, self.weight[i])
            for _ in pos:
                for (_, s, _, _, key) in nodes:
                    patch.record(s, classes[key], key, self.weight[i])

        def rewrites():                 # built only if the move is applied
            out = {}
            for i, pos in occ.items():
                spans = [m.span[x] for x in self.roots[i]]
                groups = {}
                for t0 in pos:
                    built = []
                    for (a, b), s, (a1, k1), (k2, b2), key in nodes:
                        built.append(((spans[t0 + a][0], spans[t0 + b - 1][1]), s,
                                      (spans[t0 + a1][0], spans[t0 + k1 - 1][1]),
                                      (spans[t0 + k2][0], spans[t0 + b2 - 1][1]), classes[key]))
                    groups[t0] = (size, built)
                out[i] = self._rewrite(i, groups)
            return out
        yield self.tables.delta(patch), ("joint", rewrites, patch, {"syms": fresh_syms, "classes": new_cls})

    def _rotation_moves(self, i: int):
        """Rotations of the experience's local patterns: every occurrence of
        a parent P = (Y, C) with Y = (A, B) (P, Y of given symbols and keys)
        rewritten as P = (A, Z), Z = (B, C), or the mirror image; Z is the
        symbol that already composes (B, C) or a new one."""
        m = self.memory
        seen = set()
        for e in self.elems[i]:
            if m.kind[e] != Memory.COMPOSITE:
                continue
            for side, inner in (("right", m.left[e]), ("left", m.right[e])):
                if m.kind[inner] != Memory.COMPOSITE:
                    continue
                pat = (side, self.sym[e], self._key(e), self.sym[inner], self._key(inner))
                if pat not in seen:
                    seen.add(pat)
                    yield from self._rotate(pat)

    def _rotate(self, pat):
        side, p, keyP, y, keyY = pat
        m = self.memory
        occ, used = [], set()
        for e in sorted(self.units.get((p, keyP), ()), key=lambda x: -(m.span[x][1] - m.span[x][0])):
            inner = m.left[e] if side == "right" else m.right[e]
            if self.sym[inner] != y or self._key(inner) != keyY or e in used or inner in used:
                continue
            occ.append((e, inner))
            used.update((e, inner))
        if len(occ) < self.config["min_pair"]:
            return
        n = sum(m.weight[e] for e, _ in occ)
        if side == "right":                  # (Y=(a b)) c -> a (Z=(b c))
            a, b, c = keyY[1], keyY[2], keyP[2]
            zkey = ("C", b, c)
        else:                                # a (Y=(b c)) -> (Z=(a b)) c
            a, b, c = keyP[1], keyY[1], keyY[2]
            zkey = ("C", a, b)
        z_opts = []
        holder = max(((self.rec[(s2, zkey)], -s2) for s2 in self.syms_of_key.get(zkey, ())), default=None)
        if holder is not None:
            z_opts.append(-holder[1])
        z_opts.append(self.next_sym)
        best = None
        for z in z_opts:
            pkey = ("C", a, z) if side == "right" else ("C", z, c)
            fresh = []
            if zkey in self.class_of and self._key_weight(zkey) > EPS:
                zc_opts = [self.class_of[zkey]]
            else:
                zc_opts = [self.next_cls, self.class_of[keyY]]
            if pkey in self.class_of and self._key_weight(pkey) > EPS:
                pc_opts = [self.class_of[pkey]]
            else:
                pc_opts = [self.class_of[keyP], None]       # None: a new class
            for zc in zc_opts:
                for pc in pc_opts:
                    if pc is None:
                        pc = self.next_cls + (1 if zc == self.next_cls else 0)
                    patch = Patch()
                    patch.record(p, self.class_of[keyP], keyP, -n)
                    patch.record(y, self.class_of[keyY], keyY, -n)
                    patch.record(p, pc, pkey, n)
                    patch.record(z, zc, zkey, n)
                    d = self.tables.delta(patch)
                    if best is None or d < best[0]:
                        best = (d, z, zc, pc, pkey, patch)
        d, z, zc, pc, pkey, patch = best
        rewrites: Dict[int, list] = defaultdict(list)
        for e, inner in occ:
            rewrites[self._exp_of_mid[m.experience_of[e]]].append((e, inner))
        out = {}
        for i, pairs in rewrites.items():
            split, labels, classes = self._tree_parts(i)
            for e, inner in pairs:
                (i0, j0), (i1, j1) = m.span[e], m.span[inner]
                k_inner = m.span[m.left[inner]][1]
                del split[(i1, j1)]
                del labels[(i1, j1)]
                if side == "right":           # P=(i0,j0) split k_inner; Z=(k_inner, j0) split j1
                    split[(i0, j0)] = k_inner
                    split[(k_inner, j0)] = j1
                    labels[(k_inner, j0)] = z
                else:                         # P split k_inner; Z=(i0, k_inner) split i1
                    split[(i0, j0)] = k_inner
                    split[(i0, k_inner)] = i1
                    labels[(i0, k_inner)] = z
            classes[zkey] = zc
            classes[pkey] = pc
            n_tok = len(self.experiences[i])
            out[i] = (Tree(n_tok, split, labels, [m.span[r] for r in self.roots[i]]), labels, classes)
        yield d, ("rotate", out, patch, {"syms": [z], "classes": [zc, pc]})

    def _adjacent_pairs(self, i: int):
        """Symbol pairs (B, C) adjacent inside experience i's trees where one
        rotation makes them siblings: B the right part of P's left part and C
        P's right part, or B P's left part and C the left part of P's right
        part."""
        m = self.memory
        out = set()
        for e in self.elems[i]:
            if m.kind[e] != Memory.COMPOSITE:
                continue
            l, r = m.left[e], m.right[e]
            if m.kind[l] == Memory.COMPOSITE:
                out.add((self.sym[m.right[l]], self.sym[r]))
            if m.kind[r] == Memory.COMPOSITE:
                out.add((self.sym[l], self.sym[m.left[r]]))
        return out

    def _inner_chunk_moves(self, i: int):
        """Chunk (B, C) inside trees: every occurrence of B and C adjacent
        in a rotatable position is grouped under one symbol Z (the symbol
        that already composes (B, C), or a new one), as the chunk move groups
        top-level pairs: P = ((A B) C) -> (A (B C)), P = (B (C D)) -> ((B C) D).
        Occurrences are taken left to right without sharing a node."""
        for pair in sorted(self._adjacent_pairs(i)):
            yield from self._inner_chunk(pair)

    def _inner_chunk(self, pair):
        B, C = pair
        m = self.memory
        zkey = ("C", B, C)
        occ, used = [], set()
        for b in sorted(self.members.get(B, ())):
            p1 = m.parent[b]
            if p1 < 0:
                continue
            # Case R: b is the right part of x, x the left part of p, c = p's right part.
            if m.right[p1] == b:
                x, p = p1, m.parent[p1]
                if p >= 0 and m.left[p] == x and self.sym[m.right[p]] == C:
                    c = m.right[p]
                    if not used & {x, p, b, c}:
                        occ.append(("R", p, x, b, c))
                        used.update((x, p, b, c))
                        continue
            # Case L: b is p's left part, c the left part of y = p's right part.
            if m.left[p1] == b:
                p, y = p1, m.right[p1]
                if m.kind[y] == Memory.COMPOSITE and self.sym[m.left[y]] == C:
                    c = m.left[y]
                    if not used & {y, p, b, c}:
                        occ.append(("L", p, y, b, c))
                        used.update((y, p, b, c))
        if len(occ) < self.config["min_pair"]:
            return
        holder = max(((self.rec[(s2, zkey)], -s2) for s2 in self.syms_of_key.get(zkey, ())), default=None)
        z_opts = ([-holder[1]] if holder is not None else []) + [self.next_sym]
        best = None
        for z in z_opts:
            if zkey in self.class_of and self._key_weight(zkey) > EPS:
                zc_opts = [self.class_of[zkey]]
            else:
                zc_opts = [self.next_cls]
            for zc in zc_opts:
                patch, pclasses = Patch(), {}
                for case, p, x, b, c in occ:
                    w = m.weight[p]
                    keyP, keyX = self._key(p), self._key(x)
                    if case == "R":
                        pkey = ("C", self.sym[m.left[x]], z)
                    else:
                        pkey = ("C", z, self.sym[m.right[x]])
                    if pkey in self.class_of and self._key_weight(pkey) > EPS:
                        pc = self.class_of[pkey]
                    else:
                        pc = pclasses.setdefault(pkey, self.class_of[keyP])
                    patch.record(self.sym[p], self.class_of[keyP], keyP, -w)
                    patch.record(self.sym[x], self.class_of[keyX], keyX, -w)
                    patch.record(self.sym[p], pc, pkey, w)
                    patch.record(z, zc, zkey, w)
                d = self.tables.delta(patch)
                if best is None or d < best[0]:
                    best = (d, z, zc, patch, pclasses)
        d, z, zc, patch, pclasses = best
        by_exp: Dict[int, list] = defaultdict(list)
        for o in occ:
            by_exp[self._exp_of_mid[m.experience_of[o[1]]]].append(o)
        out = {}
        for i, items in by_exp.items():
            split, labels, classes = self._tree_parts(i)
            for case, p, x, b, c in items:
                (i0, j0), (ix, jx) = m.span[p], m.span[x]
                del split[(ix, jx)]
                del labels[(ix, jx)]
                if case == "R":            # p=(i0,j0) = (x=(i0, kb) + c); new p = (a, z=(kb.. j0))
                    kb = m.span[b][0]
                    split[(i0, j0)] = kb
                    split[(kb, j0)] = m.span[c][0]
                    labels[(kb, j0)] = z
                    classes[("C", self.sym[m.left[x]], z)] = pclasses.get(("C", self.sym[m.left[x]], z),
                                                                         self.class_of.get(("C", self.sym[m.left[x]], z)))
                else:                      # p = (b, x=(c, d)); new p = (z=(b c), d)
                    kc = m.span[c][1]
                    split[(i0, j0)] = kc
                    split[(i0, kc)] = m.span[b][1]
                    labels[(i0, kc)] = z
                    classes[("C", z, self.sym[m.right[x]])] = pclasses.get(("C", z, self.sym[m.right[x]]),
                                                                          self.class_of.get(("C", z, self.sym[m.right[x]])))
            classes[zkey] = zc
            out[i] = (Tree(len(self.experiences[i]), split, labels, [m.span[r] for r in self.roots[i]]),
                      labels, classes)
        yield d, ("inner chunk", out, patch, {"syms": [z], "classes": [zc] + list(pclasses.values())})

    def _tree_parts(self, i: int):
        """Experience i's analysis as (split, labels, classes of its keys)."""
        m = self.memory
        split, labels, classes = {}, {}, {}
        for e in self.elems[i]:
            labels[m.span[e]] = self.sym[e]
            key = self._key(e)
            classes[key] = self.class_of[key]
            if m.kind[e] == Memory.COMPOSITE:
                split[m.span[e]] = m.span[m.left[e]][1]
        return split, labels, classes

    def _dissolve_move(self, Y):
        """Dissolve chunk category Y at the top level: each top-level Y
        composite is replaced by its two parts."""
        m = self.memory
        patch = Patch()
        rewrites = {}
        for i in sorted(self.top_exps.get(Y, ())):
            roots = self.roots[i]
            if not any(self.sym[r] == Y and m.kind[r] == Memory.COMPOSITE for r in roots):
                continue
            labs = self._top_syms(i)
            new = []
            for r in roots:
                if self.sym[r] == Y and m.kind[r] == Memory.COMPOSITE:
                    key = self._key(r)
                    patch.record(Y, self.class_of[key], key, -m.weight[r])
                    new += [self.sym[m.left[r]], self.sym[m.right[r]]]
                else:
                    new.append(self.sym[r])
            patch.top(labs, -self.weight[i])
            patch.top(new, self.weight[i])
            # The new tree: the dissolved roots' spans removed.
            n = len(self.experiences[i])
            split, labels = {}, {}
            for e in self.elems[i]:
                labels[m.span[e]] = self.sym[e]
                if m.kind[e] == Memory.COMPOSITE:
                    split[m.span[e]] = m.span[m.left[e]][1]
            tops = []
            for r in roots:
                if self.sym[r] == Y and m.kind[r] == Memory.COMPOSITE:
                    split.pop(m.span[r])
                    labels.pop(m.span[r])
                    tops += [m.span[m.left[r]], m.span[m.right[r]]]
                else:
                    tops.append(m.span[r])
            classes = {self._key(e): self.class_of[self._key(e)] for e in self.elems[i]}
            rewrites[i] = (Tree(n, split, labels, tops), labels, classes)
        if not rewrites:
            return 0.0, None
        return self.tables.delta(patch), ("dissolve", rewrites, patch)

    # Proposals from the hierarchies --------------------------------------- #
    def _tree_splits(self, A: int) -> List[List[set]]:
        """Split proposals for symbol A along the representation hierarchy.
        Below the lowest concept holding all of A's elements, at each
        concept on the way down to the heaviest child: A's elements under
        each child concept as a new symbol (a refinement, as the night's cut
        search refines a cut node into its children; the heaviest child keeps
        A), and A's elements under one child against the rest; at most
        ``tree_splits`` proposals. Each proposal is a list of groups."""
        cap = self.config["tree_splits"]
        els = self.members.get(A, ())
        if cap <= 0 or len(els) < 2:
            return []
        mass: Dict[int, float] = defaultdict(float)
        under: Dict[int, List[int]] = defaultdict(list)
        for e in els:
            if e not in self.placed:          # (during a trial new elements are not placed)
                continue
            node = self.placed[e][0]
            while node is not None:
                mass[node.id] += self.memory.weight[e]
                under[node.id].append(e)
                node = node.parent
        out: List[List[set]] = []
        node = self.rtree.root
        while node is not None and len(out) < cap:
            kids = sorted((c for c in node.children if mass.get(c.id, 0.0) > 0),
                          key=lambda c: (-mass[c.id], c.id))
            if len(kids) == 1:
                node = kids[0]
                continue
            if len(kids) < 2:
                break
            if len(kids) > 2:
                out.append([set(under[c.id]) for c in kids[1:]])
            for c in kids[1:]:
                if len(out) >= cap:
                    break
                out.append([set(under[c.id])])
            node = kids[0]
        return out[:cap]

    def _word_class_splits(self, A: int) -> List[List[set]]:
        """Split proposals for symbol A from the word classes: its primitive
        elements grouped by their word's class, at each of the last
        ``wc_levels`` partitions of the merge path (as the night starts its
        search from them); the largest group keeps A."""
        levels = self.config["wc_levels"]
        if levels <= 0:
            return []
        n = len(self.experiences)
        if n - self._wc_at >= self.config["wc_every"]:
            self._wc_path, self._wc_at = self.wc.path(), n
        m = self.memory
        prims = [e for e in self.members.get(A, ()) if m.kind[e] == Memory.PRIMITIVE]
        if len(prims) < 2:
            return []
        out, seen = [], set()
        for cls in self._wc_path[-levels:]:
            groups: Dict[str, set] = defaultdict(set)
            for e in prims:
                groups[cls.get(m.token[e], m.token[e])].add(e)
            if len(groups) < 2:
                continue
            parts = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
            sig = tuple(sorted(tuple(sorted(g)) for g in parts))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(parts[1:])
        return out

    def _context_splits(self, A: int) -> List[List[set]]:
        """Split proposals for symbol A from the structure around it: A's
        units (its elements grouped by composition key) are profiled by the
        symbols of their ancestors up to ``profile_depth`` levels and of the
        siblings beside that path (with sides), and parted in two by 2-means
        (L1, weighted); the smaller part is proposed as a new symbol."""
        depth = self.config["profile_depth"]
        if depth <= 0:
            return []
        m = self.memory
        units: Dict[tuple, Counter] = {}
        weight: Dict[tuple, float] = {}
        for key in self.rec_by_sym.get(A, ()):
            prof: Counter = Counter()
            for e in self.units.get((A, key), ()):
                node = e
                for d in range(1, depth + 1):
                    p = m.parent[node]
                    if p < 0:
                        prof[(d, "top", len(self.roots[self._exp_of_mid[m.experience_of[e]]]) == 1)] += 1
                        break
                    left = m.left[p] == node
                    sib = m.right[p] if left else m.left[p]
                    prof[(d, "a", self.sym[p], left)] += 1
                    prof[(d, "s", self.sym[sib], left)] += 1
                    node = p
            tot = sum(prof.values())
            if tot:
                units[key] = Counter({f: v / tot for f, v in prof.items()})
                weight[key] = self.rec[(A, key)]
        if len(units) < 2:
            return []
        keys = sorted(units, key=lambda k: (-weight[k], str(k)))
        dist = lambda a, b: sum(abs(a.get(f, 0.0) - b.get(f, 0.0)) for f in set(a) | set(b))
        top = keys[:20]
        a, b = max(((x, y) for i, x in enumerate(top) for y in top[i + 1:]),
                   key=lambda xy: (dist(units[xy[0]], units[xy[1]]), str(xy)))
        if dist(units[a], units[b]) < 1e-9:
            return []
        cents = [Counter(units[a]), Counter(units[b])]
        assign = {}
        for _ in range(10):
            new = {k: int(dist(units[k], cents[1]) < dist(units[k], cents[0])) for k in keys}
            if new == assign:
                break
            assign = new
            for g in (0, 1):
                members = [k for k in keys if assign[k] == g]
                tot = sum(weight[k] for k in members) or 1.0
                c: Counter = Counter()
                for k in members:
                    for f, v in units[k].items():
                        c[f] += v * weight[k] / tot
                cents[g] = c
        groups = [[k for k in keys if assign[k] == g] for g in (0, 1)]
        if not groups[0] or not groups[1]:
            return []
        small = min(groups, key=lambda g: (sum(weight[k] for k in g), str(g)))
        return [[{e for k in small for e in self.units[(A, k)]}]]

    def _class_splits(self, c: int) -> List[List[tuple]]:
        """Split proposals for rule class c along the composition hierarchy:
        its keys under one child concept against the rest."""
        cap = self.config["class_splits"]
        keys = [k for k in self.keys_of_class.get(c, ()) if self._key_weight(k) > EPS]
        if cap <= 0 or len(keys) < 2:
            return []
        under: Dict[int, set] = defaultdict(set)
        mass: Dict[int, float] = defaultdict(float)
        for k in keys:
            for leaf, _, w in self.cplaced.get(k, ()):
                node = leaf
                while node is not None:
                    under[node.id].add(k)
                    mass[node.id] += w
                    node = node.parent
        out, frontier = [], [self.ctree.root]
        while frontier and len(out) < cap:
            node = frontier.pop(0)
            kids = sorted((ch for ch in node.children if under.get(ch.id)), key=lambda ch: -mass[ch.id])
            if len(kids) == 1:
                frontier.insert(0, kids[0])
                continue
            if len(kids) < 2:
                continue
            for ch in kids:
                if len(out) >= cap:
                    break
                sub = sorted(under[ch.id], key=str)
                if 0 < len(sub) < len(keys):
                    out.append(sub)
            frontier.append(kids[0])
        return out

    # ------------------------------------------------------------------ #
    # Checks and the night learner's interface
    # ------------------------------------------------------------------ #
    def verify(self) -> None:
        """Assert that the count tables equal the stored analyses' counts and
        their code its recomputation (for tests; ``check=True`` runs it after
        every commit)."""
        patch = Patch()
        for i in range(len(self.elems)):
            self._experience_patch(i, +1.0, patch)
        fresh = CodeTables(self.alpha, self.tables.extra_bits)
        fresh.commit(patch)
        for t in TABLES:
            a = {r: {o: v for o, v in c.items()} for r, c in self.tables.cells[t].items() if c}
            b = {r: {o: v for o, v in c.items()} for r, c in fresh.cells[t].items() if c}
            assert set(a) == set(b), f"table {t}: rows differ"
            for r in a:
                assert set(a[r]) == set(b[r]), f"table {t} row {r}: outcomes differ"
                for o in a[r]:
                    assert abs(a[r][o] - b[r][o]) < 1e-6, f"table {t} row {r} {o}: {a[r][o]} != {b[r][o]}"
        assert abs(self.tables.bits() - fresh.bits()) < 1e-6, (self.tables.bits(), fresh.bits())
        assert abs(self.tables.bits() - self.tables.recompute_bits()) < 1e-6
        uc, uc2 = defaultdict(Counter), defaultdict(Counter)
        for x in self.elems:
            for e in x:
                c = self.class_of[self._key(e)]
                x1, x2 = self._ctx(e)
                uc[(self.sym[e], x1)][c] += self.memory.weight[e]
                uc2[(self.sym[e], x2)][c] += self.memory.weight[e]
        for mine, ref in ((self.uc, uc), (self.uc2, uc2)):
            assert set(k for k, v in mine.items() if v) == set(ref), "context rows differ"
            for k in ref:
                assert set(mine[k]) == set(ref[k]) and all(abs(mine[k][c] - ref[k][c]) < 1e-6 for c in ref[k])
        live = sum(len(x) for x in self.elems)
        assert len(self.placed) == live, (len(self.placed), live)
        assert abs(self.rtree.root.count - sum(self.memory.weight[e] for x in self.elems for e in x)) < 1e-6

    def full_code(self) -> Tuple[float, object]:
        """The total code with the rule choices coded as the night's grammar
        codes them (``grammar._factored_code``): under each context option
        (none; the word before at weights 1, 4, 16; the two words before at
        the nine weight pairs) the rule-class choices are a prequential
        backoff code in learning order (experiences in order, each analysis'
        elements as ``Memory.add`` records them), and the shortest option
        counts. Returns (bits, option). The rest of the code is the
        order-free tables'. Exact; O(elements) per call (cached until the
        next commit)."""
        key = (self.tables.version, self._journal is None and len(self.memory))
        hit = self.__dict__.get("_full_cache")
        if hit is not None and hit[0] == key and self._journal is None:
            return hit[1]
        a = self.alpha
        order = [e for x in self.slots for slot in x for e in self.elems[slot]]
        sym = np.array([self.sym[e] for e in order], dtype=np.int64)
        cls_ids: Dict[int, int] = {}
        keys = np.array([cls_ids.setdefault(self.class_of[self._key(e)], len(cls_ids)) for e in order], dtype=np.int64)
        x1_ids: Dict[Hashable, int] = {}
        x2_ids: Dict[Hashable, int] = {}
        x1, x2 = [], []
        for e in order:
            c1, c2 = self._ctx(e)
            x1.append(x1_ids.setdefault(c1, len(x1_ids)))
            x2.append(x2_ids.setdefault(c2, len(x2_ids)))
        x1, x2 = np.array(x1, dtype=np.int64), np.array(x2, dtype=np.int64)
        w = np.array([self.memory.weight[e] for e in order], dtype=float)
        M = self.tables.M
        u_nats = self.tables.norm["U"] - self.tables.phisum["U"]
        best = (u_nats, None)
        one = backoff_coder(sym, x1, w)
        two = backoff_chain_coder(sym, [x1, x2], w)
        for opt in CONTEXT_OPTIONS[1:]:
            nats = two(keys, M, a, opt) if isinstance(opt, tuple) else one(keys, M, a, opt)
            if nats < best[0] - 1e-9:
                best = (nats, opt)
        bits = (self.tables.nats() - u_nats + best[0]) / LN2 + self.tables.structure_bits()
        out = (bits, best[1])
        if self._journal is None:
            self._full_cache = (key, out)
        return out

    def _full_delta(self, apply) -> float:
        """Change of the contextual full code if ``apply()`` were done (a trial)."""
        before = self.full_code()[0]
        tok = self._txn_begin()
        try:
            apply()
            after = self.full_code()[0]
        finally:
            self._txn_rollback(tok)
        self.stats["contextual checks"] += 1
        return after - before

    def hard_code(self) -> float:
        """Evaluation only (never used for learning): the total code of one
        analysis per experience, its Viterbi analysis under the current
        grammar, with the learner's rule classes (a key never seen takes the
        class that best derives it). For a hedged learner this is the code
        comparable to a learner that stores single analyses."""
        tables = CodeTables(self.alpha, self.tables.extra_bits)
        fixed: Dict[tuple, int] = {}
        patch = Patch()
        for x in self.slots:
            tokens = self.experiences[x[0]]
            tree, labels, classes = self._parse(tokens)
            for key, c in classes.items():
                fixed.setdefault(key, self.class_of.get(key, c))
            self._tree_patch(tokens, tree, labels, fixed, 1.0, +1.0, patch)
        tables.commit(patch)
        return tables.bits()

    def chart(self, tokens: Sequence[str]) -> Chart:
        return Chart(self.grammar, tokens)

    def parse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).mbr_tree()

    def analyse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).viterbi_tree()

    def log_prob(self, tokens: Sequence[str]) -> float:
        return Chart(self.grammar, tokens).log_prob

    def generate(self, n: int, rng: Optional[np.random.Generator] = None, max_len: int = 30,
                 max_tries: int = 100, whole_only: bool = False):
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        g = self.grammar
        out, rejected = [], 0
        while len(out) < n:
            for _ in range(max_tries):
                s = g.sample(rng, max_len=max_len, whole_only=whole_only)
                if s is not None:
                    out.append(s)
                    break
                rejected += 1
            else:
                raise RuntimeError("grammar keeps producing over-long derivations")
        return out, rejected

    def complete(self, prompt, n: int = 1, rng=None, temperature: float = 1.0, max_len: int = 30):
        from .prompt import complete
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        return complete(self.grammar, prompt, n, rng, temperature, max_len)

    def stored_trees(self, every: bool = False) -> List[Tree]:
        """The stored analyses as Trees labelled with their symbols: one per
        experience (its first sample), or every stored sample."""
        m = self.memory
        out = []
        for i in (range(len(self.elems)) if every else [x[0] for x in self.slots]):
            split, labels = {}, {}
            for e in self.elems[i]:
                labels[m.span[e]] = self.sym[e]
                if m.kind[e] == Memory.COMPOSITE:
                    split[m.span[e]] = m.span[m.left[e]][1]
            out.append(Tree(len(self.experiences[i]), split, labels, [m.span[r] for r in self.roots[i]]))
        return out


class IncrementalBeam:
    """A few hypotheses kept side by side, each an ``IncrementalLearner``
    with its own moves (e.g. with and without joint chunks, as the night
    searches with and without them and lets the full code choose). Each
    perceives every experience once; the grammar at any time is that of the
    hypothesis with the shortest total code. Nothing is replayed: the
    hypotheses are independent persistent learners."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, configs: Sequence[dict] = ({},), **kw):
        self.learners = [IncrementalLearner(seed=seed, alpha=alpha, config=c, **kw) for c in configs]
        self.seed = seed

    def observe(self, tokens):
        for L in self.learners:
            L.observe(tokens)

    @property
    def best(self) -> IncrementalLearner:
        return min(self.learners, key=lambda L: L.tables.bits())

    def sleep(self):
        return self.best.grammar

    def __getattr__(self, name):
        # Everything else (grammar, chart, generate, stored_trees, ...) is the best hypothesis'.
        if name in ("learners", "seed"):
            raise AttributeError(name)
        return getattr(self.best, name)

    @property
    def stats(self):
        out = Counter()
        for k, L in enumerate(self.learners):
            out[f"hypothesis {k} bits"] = L.tables.bits()
            for key, v in L.stats.items():
                out[key] += v
        return out

    @property
    def history(self):
        return [{"seconds": sum(L.history[t]["seconds"] for L in self.learners)}
                for t in range(len(self.learners[0].history))]


class HedgedBeam:
    """Hedged commitments: up to ``width`` hypotheses (``IncrementalLearner``s).
    When a hypothesis takes a structure decision with a runner-up whose
    rollout also shortens the code, a copy of it takes the runner-up (at
    most one branch per hypothesis and experience); after each experience
    the ``width`` hypotheses with the shortest total code are kept. The
    grammar is that of the best. Copying a state replays nothing; each
    hypothesis remains a persistent incremental learner."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, width: int = 4, config: Optional[dict] = None):
        self.width = width
        self.learners = [IncrementalLearner(seed=seed, alpha=alpha, config=config)]
        self.seed = seed
        self._history: List[dict] = []
        self.branch_events = 0

    def observe(self, tokens):
        t0 = time.time()
        pool = []
        for L in self.learners:
            L._branches, L._branched = [], False
            L.observe(tokens)
            pool.append(L)
            pool += L._branches
            self.branch_events += len(L._branches)
            L._branches = None
        for L in pool:
            L._branches = None
        pool.sort(key=lambda L: L.tables.bits())
        kept = []
        for L in pool:                       # equal codes: the same hypothesis, kept once
            if all(abs(L.tables.bits() - K.tables.bits()) > 1e-6 for K in kept):
                kept.append(L)
        self.learners = kept[:self.width]
        self._history.append({"seconds": time.time() - t0, "hypotheses": len(pool)})

    @property
    def best(self) -> IncrementalLearner:
        return min(self.learners, key=lambda L: L.tables.bits())

    def sleep(self):
        return self.best.grammar

    def __getattr__(self, name):
        if name in ("learners", "seed", "width", "_history", "branch_events"):
            raise AttributeError(name)
        return getattr(self.best, name)

    @property
    def stats(self):
        out = Counter(self.best.stats)
        out["branch events"] = self.branch_events
        out["hypotheses"] = len(self.learners)
        return out

    @property
    def history(self):
        return self._history
