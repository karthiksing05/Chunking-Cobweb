"""Search for the analyses that describe a corpus in the fewest bits.

Analyses are kept symbolically: a sentence is a list of top-level nodes, and a
node is ``(label, token)`` for a word or ``(label, (left, right))`` for a
chunk. The description length is the plain-PCFG code of ``mdl.py``: one
Dirichlet-multinomial row per category (outcomes: a token, or a pair of child
categories), a row for the category of each top-level chunk, and a
continue/stop row. Under this code a chunk pays for its definition by making
the top level shorter, which is the pressure towards structure.

Two searches lower the code, in the tradition of Brown et al. (1992), SNPR
(Wolff 1982), GRIDS (Langley & Stromsten 2000) and Bayesian model merging
(Stolcke & Omohundro 1994):

* ``word_classes`` greedily merges word types into classes while the code of a
  class-bigram model shrinks, and returns the whole merge path.
* ``chunk_and_merge`` is a beam search over two moves. *Chunk* (B, C) replaces
  every non-overlapping adjacent pair of top-level categories B C with a new
  chunk. *Merge* (A, A') makes two categories one. Every move is global: it
  changes all sentences consistently, which is what lets a chunk type pay for
  itself. A move changes only a few rows of the code, so every candidate is
  scored exactly without rebuilding the analyses.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from .data import Tree

Node = Tuple[Hashable, object]   # (label, token) or (label, (left, right))
LN2 = math.log(2.0)


def _dm(counts, alphabet: int, alpha: float) -> float:
    c = np.array([v for v in counts if v > 0], dtype=float)
    if c.size == 0:
        return 0.0
    a = alphabet * alpha
    return float(gammaln(c.sum() + a) - gammaln(a) - np.sum(gammaln(c + alpha) - gammaln(alpha)))


def _nodes(node: Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n[1], str):
            stack.extend(n[1])


def code_bits(analyses: Sequence[List[Node]], n_tokens: int, alpha: float) -> float:
    """Description length (bits) of the analyses under the plain-PCFG code."""
    rows: Dict[Hashable, Counter] = defaultdict(Counter)
    start: Counter = Counter()
    cont = 0
    labels = set()
    for tops in analyses:
        cont += len(tops) - 1
        for top in tops:
            start[top[0]] += 1
            for lab, body in _nodes(top):
                labels.add(lab)
                if isinstance(body, str):
                    rows[lab][("w", body)] += 1
                else:
                    rows[lab][("p", body[0][0], body[1][0])] += 1
    K = len(labels)
    nats = sum(_dm(r.values(), n_tokens + K * K, alpha) for r in rows.values())
    nats += _dm(start.values(), K, alpha) + _dm([len(analyses), cont], 2, alpha)
    return nats / LN2


def class_bigram_bits(sentences: Sequence[Sequence[str]], cls: Dict[str, Hashable],
                      alpha: float) -> float:
    """Code length (bits) of the sentences under a class-bigram model: a
    transition row per previous class (outcomes: a class or the end of the
    sentence) and an emission row per class (outcomes: words)."""
    V = len({w for s in sentences for w in s}) + 1
    trans, emit = defaultdict(Counter), defaultdict(Counter)
    for s in sentences:
        prev = "<s>"
        for w in s:
            c = cls[w]
            trans[prev][c] += 1
            emit[c][w] += 1
            prev = c
        trans[prev]["</s>"] += 1
    K = len(set(cls.values()))
    return (sum(_dm(r.values(), K + 1, alpha) for r in trans.values())
            + sum(_dm(r.values(), V, alpha) for r in emit.values())) / LN2


def word_classes(sentences: Sequence[Sequence[str]], alpha: float
                 ) -> List[Dict[str, Hashable]]:
    """Merge word types into classes while the class-bigram code shrinks.

    Returns the merge path: every partition from word types (first) to the
    shortest-code partition (last). Each step scores every pair of classes
    exactly: a merge pools two transition rows, two transition columns and
    two emission rows, and shrinks every transition row's alphabet by one,
    so the change in code is computed from those counts alone
    (O(K^3) per step for K classes). A merged class keeps the name of its
    alphabetically first word, and ties go to the first pair in that order.
    """
    vocab = sorted({w for s in sentences for w in s})
    K, V, lg = len(vocab), len(vocab) + 1, gammaln
    index = {w: i for i, w in enumerate(vocab)}

    def phi(x):
        return lg(x + alpha) - lg(alpha)

    # T[r, c]: rows are classes and then <s>; columns are classes and then </s>.
    T = np.zeros((K + 1, K + 1))
    for s in sentences:
        prev = K
        for w in s:
            T[prev, index[w]] += 1
            prev = index[w]
        T[prev, K] += 1
    emitted = T[:, :K].sum(axis=0)          # tokens emitted by each class
    members = [[w] for w in vocab]
    alive = list(range(K))                  # classes in name order
    path = [{w: w for w in vocab}]
    while len(alive) > 1:
        k = len(alive)
        sub = T[np.ix_(alive + [K], alive + [K])]
        X, P = sub[:, :k], phi(sub[:, :k])
        # Column merge in every row: sum_r phi(X[r,a] + X[r,b]) - phi(X[r,a]) - phi(X[r,b]).
        cols = np.zeros((k, k))
        for r in range(k + 1):
            cols += phi(X[r][:, None] + X[r][None, :]) - P[r][:, None] - P[r][None, :]
        # Row merge over the class columns: sum_c phi(X[a,c] + X[b,c]).
        rows = np.zeros((k, k))
        for c in range(k):
            rows += phi(X[:k, c][:, None] + X[:k, c][None, :])
        d = np.diag(X[:k])
        Xab = X[:k]                                          # Xab[a, b] = transitions a -> b
        # Rows a and b are pooled (with their a/b columns merged), not column-merged.
        col_change = (cols - (phi(d[:, None] + Xab) - phi(d)[:, None] - phi(Xab))
                      - (phi(Xab.T + d[None, :]) - phi(Xab.T) - phi(d)[None, :]))
        end = sub[:k, k]
        row_change = (rows - phi(d[:, None] + Xab.T) - phi(Xab + d[None, :])
                      + phi(d[:, None] + Xab + Xab.T + d[None, :])
                      + phi(end[:, None] + end[None, :])
                      - phi(sub[:k]).sum(axis=1)[:, None] - phi(sub[:k]).sum(axis=1)[None, :])
        # Normalizers: every transition row's alphabet shrinks from k+1 to k.
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


def _phi(c: float, alpha: float) -> float:
    """log Gamma(c + alpha) - log Gamma(alpha): one outcome's share of a row's code."""
    return math.lgamma(c + alpha) - math.lgamma(alpha) if c else 0.0


def _outcome(body) -> tuple:
    return ("w", body) if isinstance(body, str) else ("p", body[0][0], body[1][0])


class _State:
    """Analyses with the counts their code depends on, so that the code after
    any chunk or merge move is computed exactly from a few changed rows.

    Moves are applied incrementally. A child state shares every unchanged
    sentence and row with its parent. A merge only renames: it records
    A' -> A in a table that is applied whenever labels are read, so the
    analyses themselves are rewritten only when they are needed
    (``analyses``)."""

    def __init__(self, analyses: List[List[Node]], n_tokens: int, alpha: float):
        self.tops, self.alias, self._analyses = analyses, {}, analyses
        self.n_tokens, self.alpha, self._sig = n_tokens, alpha, None
        rows: Dict[Hashable, Counter] = defaultdict(Counter)
        start: Counter = Counter()
        self.cont = 0
        for tops in analyses:
            self.cont += len(tops) - 1
            for top in tops:
                start[top[0]] += 1
                for lab, body in _nodes(top):
                    rows[lab][_outcome(body)] += 1
        self.rows, self.start, self.n_sent = dict(rows), start, len(analyses)
        self.parents: Dict[Hashable, set] = defaultdict(set)
        for lab, row in rows.items():
            for o in row:
                if o[0] == "p":
                    self.parents[o[1]].add(lab)
                    self.parents[o[2]].add(lab)
        self.parents = dict(self.parents)
        self.row_n = {lab: sum(r.values()) for lab, r in rows.items()}
        self.row_s = {lab: self._phis(r.values()) for lab, r in rows.items()}
        self.s_total = sum(self.row_s.values())
        self.start_n = sum(start.values())
        self.start_s = self._phis(start.values())
        self.fresh = 1 + max([lab[1] for lab in rows if isinstance(lab, tuple) and len(lab) == 2
                              and lab[0] == "chunk" and isinstance(lab[1], int)], default=-1)
        self._set_nats()

    def _set_nats(self):
        K = len(self.rows)
        self.nats = self._nats(K, self._alphabet_term(K), self.s_total, self.start_n,
                               self.start_s, self.cont)

    @property
    def bits(self) -> float:
        return self.nats / LN2

    @property
    def analyses(self) -> List[List[Node]]:
        """The analyses with every label renamed by the merges so far."""
        if self._analyses is None:
            alias = self.alias

            def rename(node):
                lab, body = node
                lab = alias.get(lab, lab)
                if isinstance(body, str):
                    return (lab, body)
                return (lab, (rename(body[0]), rename(body[1])))
            self._analyses = [[rename(n) for n in tops] for tops in self.tops]
        return self._analyses

    def _phis(self, counts) -> float:
        return sum(_phi(c, self.alpha) for c in counts)

    def _alphabet_term(self, K: int) -> float:
        aa = (self.n_tokens + K * K) * self.alpha
        return sum(math.lgamma(n + aa) for n in self.row_n.values())

    def _nats(self, K, alphabet_term, s_total, start_n, start_s, cont) -> float:
        a, lg = self.alpha, math.lgamma
        aa = (self.n_tokens + K * K) * a
        rows = alphabet_term - K * lg(aa) - s_total
        start = lg(start_n + K * a) - lg(K * a) - start_s
        stop = (lg(self.n_sent + cont + 2 * a) - lg(2 * a)
                - _phi(self.n_sent, a) - _phi(cont, a))
        return rows + start + stop

    def _labels(self, tops) -> list:
        alias = self.alias
        return [alias.get(t[0], t[0]) for t in tops]

    def pairs(self) -> Counter:
        """Adjacent top-level category pairs, counted as non-overlapping
        left-to-right replacements (a run of L equal categories holds L//2)."""
        pairs: Counter = Counter()
        for tops in self.tops:
            labs = self._labels(tops)
            run = 1
            for x, y in zip(labs, labs[1:]):
                if x != y:
                    pairs[(x, y)] += 1
                    if run > 1:
                        pairs[(x, x)] += run // 2
                    run = 1
                else:
                    run += 1
            if run > 1:
                pairs[(labs[-1], labs[-1])] += run // 2
        return pairs

    def _start_after_chunk(self, B, C, n) -> float:
        phi, sB, sC = self._phi_one, self.start[B], self.start[C]
        if B != C:
            return self.start_s - phi(sB) - phi(sC) + phi(sB - n) + phi(sC - n) + phi(n)
        return self.start_s - phi(sB) + phi(sB - 2 * n) + phi(n)

    def _merged_rows(self, A, B):
        """Rows A and A' pooled, and every other row holding A' as a child,
        with A' renamed A."""
        def sub(o):
            return o if o[0] == "w" else ("p", A if o[1] == B else o[1], A if o[2] == B else o[2])
        merged: Counter = Counter()
        for r in (A, B):
            for o, c in self.rows[r].items():
                merged[sub(o)] += c
        moved = {}
        for r in self.parents.get(B, ()):
            if r != A and r != B:
                row: Counter = Counter()
                for o, c in self.rows[r].items():
                    row[sub(o)] += c
                moved[r] = row
        return merged, moved

    def scored_moves(self):
        """Yield (nats after the move, move) for every chunk and merge move."""
        a, phi, lg = self.alpha, self._phi_one, math.lgamma
        K = len(self.rows)
        # Chunk (B, C) -> Y: a new row Y holding n pairs; n fewer top-level nodes.
        up = self._alphabet_term(K + 1)
        aa = (self.n_tokens + (K + 1) ** 2) * a
        for (B, C), n in self.pairs().items():
            if n < 2:
                continue
            nats = self._nats(K + 1, up + lg(n + aa), self.s_total + phi(n),
                              self.start_n - n, self._start_after_chunk(B, C, n), self.cont - n)
            yield nats, ("chunk", B, C)
        # Merge (A, A'): A' is renamed A everywhere, so rows A and A' pool and
        # every row holding A' as a child may see outcomes collide.
        down = self._alphabet_term(K - 1)
        aa = (self.n_tokens + (K - 1) ** 2) * a
        labels = sorted(self.rows, key=str)
        for A, B in itertools.combinations(labels, 2):
            merged, moved = self._merged_rows(A, B)
            s_total = self.s_total - self.row_s[A] - self.row_s[B] + self._phis(merged.values())
            for r, row in moved.items():
                s_total += self._phis(row.values()) - self.row_s[r]
            nA, nB = self.row_n[A], self.row_n[B]
            sA, sB = self.start[A], self.start[B]
            nats = self._nats(K - 1, down - lg(nA + aa) - lg(nB + aa) + lg(nA + nB + aa),
                              s_total, self.start_n,
                              self.start_s - phi(sA) - phi(sB) + phi(sA + sB), self.cont)
            yield nats, ("merge", A, B)

    def _phi_one(self, c) -> float:
        return _phi(c, self.alpha)

    def _child(self) -> "_State":
        c = object.__new__(_State)
        c.n_tokens, c.alpha, c.n_sent = self.n_tokens, self.alpha, self.n_sent
        c.tops, c.alias, c._analyses, c._sig = self.tops, self.alias, None, None
        c.rows, c.row_n, c.row_s = dict(self.rows), dict(self.row_n), dict(self.row_s)
        c.parents, c.start = dict(self.parents), Counter(self.start)
        c.s_total, c.start_n, c.start_s = self.s_total, self.start_n, self.start_s
        c.cont, c.fresh = self.cont, self.fresh
        return c

    def apply(self, move) -> "_State":
        kind, x, y = move
        c = self._child()
        if kind == "chunk":
            B, C, Y = x, y, ("chunk", self.fresh)
            c.fresh += 1
            tops, n = [], 0
            for sentence in self.tops:
                labs = self._labels(sentence)
                if not any(p == B and q == C for p, q in zip(labs, labs[1:])):
                    tops.append(sentence)
                    continue
                out, t = [], 0
                while t < len(sentence):
                    if t + 1 < len(sentence) and labs[t] == B and labs[t + 1] == C:
                        out.append((Y, (sentence[t], sentence[t + 1])))
                        t += 2
                        n += 1
                    else:
                        out.append(sentence[t])
                        t += 1
                tops.append(out)
            c.tops = tops
            c.rows[Y] = Counter({("p", B, C): n})
            c.row_n[Y], c.row_s[Y] = n, self._phi_one(n)
            c.s_total += c.row_s[Y]
            c.start_s = self._start_after_chunk(B, C, n)
            c.start[B] -= n
            c.start[C] -= n
            c.start[Y] = n
            c.start_n -= n
            c.cont -= n
            for z in (B, C):
                c.parents[z] = set(self.parents.get(z, ())) | {Y}
        else:
            A, B = x, y
            merged, moved = self._merged_rows(A, B)
            c.rows[A] = merged
            c.row_n[A] = self.row_n[A] + self.row_n[B]
            c.row_s[A] = self._phis(merged.values())
            c.s_total += c.row_s[A] - self.row_s[A] - self.row_s[B]
            for r, row in moved.items():
                c.rows[r] = row
                c.row_s[r] = self._phis(row.values())
                c.s_total += c.row_s[r] - self.row_s[r]
            for table in (c.rows, c.row_n, c.row_s):
                del table[B]
            # Rows that held A or A' as a child now hold A; the children of
            # row A' now have row A as a parent.
            holders = set(self.parents.get(A, ())) | set(self.parents.get(B, ()))
            c.parents.pop(B, None)
            c.parents[A] = {A if r == B else r for r in holders}
            for o in self.rows[B]:
                if o[0] == "p":
                    for z in (o[1], o[2]):
                        z = A if z == B else z
                        c.parents[z] = {A if r == B else r for r in c.parents.get(z, ())} | {A}
            sA, sB = self.start[A], self.start[B]
            c.start_s = self.start_s - self._phi_one(sA) - self._phi_one(sB) + self._phi_one(sA + sB)
            c.start[A] = sA + sB
            del c.start[B]
            c.alias = {k: (A if v == B else v) for k, v in self.alias.items()}
            c.alias[B] = A
        c.start = Counter({k: v for k, v in c.start.items() if v})
        c._set_nats()
        return c

    def signature(self) -> tuple:
        """The analyses with categories renamed in order of first appearance:
        equal for states that differ only in the names of their categories."""
        if self._sig is None:
            canon: Dict[Hashable, int] = {}
            out = []
            for tops in self.analyses:
                seq = []
                for top in tops:
                    stack = [top]
                    while stack:
                        lab, body = stack.pop()
                        k = canon.setdefault(lab, len(canon))
                        if isinstance(body, str):
                            seq.append((k, body))
                        else:
                            seq.append(k)
                            stack.append(body[1])
                            stack.append(body[0])
                out.append(tuple(seq))
            self._sig = tuple(out)
        return self._sig


def _describe(move) -> str:
    return f"{move[0]}({move[1]}, {move[2]})"


def chunk_and_merge(analyses: List[List[Node]], n_tokens: int, alpha: float,
                    beam: int = 1, patience: int = 0, max_steps: int = 500,
                    log=None) -> Tuple[List[List[Node]], float]:
    """Beam search over chunk and merge moves for the shortest code.

    Each step expands every analysis set in the beam by all its chunk and
    merge moves, scores the successors exactly and keeps the ``beam`` best
    distinct ones, even if they are longer than their parent (GRIDS, Langley
    & Stromsten 2000). The search stops after ``patience`` consecutive steps
    without a new shortest code (Stolcke 1994) and returns the shortest found.
    ``beam=1, patience=0`` is greedy best-first search.
    """
    best = _State(analyses, n_tokens, alpha)
    frontier, stale = [best], 0
    for step in range(max_steps):
        # Codes equal to 1e-7 nats are ties, broken by the order of the moves.
        scored = sorted(((nats, i, move) for i, st in enumerate(frontier)
                         for nats, move in st.scored_moves()), key=lambda x: round(x[0], 7))
        successors = []
        for nats, i, move in scored:
            if len(successors) == beam:
                break
            child = frontier[i].apply(move)
            # Two states can only be the same analyses if their codes are equal.
            ties = [s for s, _ in successors if abs(s.nats - child.nats) < 1e-7]
            if any(s.signature() == child.signature() for s in ties):
                continue
            successors.append((child, move))
        if not successors:
            break
        frontier = [c for c, _ in successors]
        if frontier[0].nats < best.nats - 1e-9 * LN2:
            best, stale = frontier[0], 0
            if log:
                log(step, _describe(successors[0][1]), best.bits)
        else:
            stale += 1
            if stale > patience:
                break
    return best.analyses, best.bits


def to_tree(tops: List[Node]) -> Tree:
    """A symbolic analysis as a Tree (a forest if it has several top nodes)."""
    split, roots, label = {}, [], {}

    def rec(node, i):
        lab, body = node
        if isinstance(body, str):
            label[(i, i + 1)] = lab
            return i + 1
        k = rec(body[0], i)
        j = rec(body[1], k)
        split[(i, j)] = k
        label[(i, j)] = lab
        return j

    pos = 0
    for node in tops:
        j = rec(node, pos)
        roots.append((pos, j))
        pos = j
    return Tree(pos, split, label, roots)


def from_tree(tokens: Sequence[str], tree: Tree, label_of) -> List[Node]:
    """A Tree (or forest) as a symbolic analysis; ``label_of(span)`` names the
    category of each span."""
    def rec(i, j):
        if j - i == 1:
            return (label_of((i, j)), tokens[i])
        k = tree.split[(i, j)]
        return (label_of((i, j)), (rec(i, k), rec(k, j)))
    return [rec(i, j) for i, j in tree.roots]
