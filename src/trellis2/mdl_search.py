"""Search for the analyses that describe a corpus in the fewest bits.

Analyses are kept symbolically: a sentence is a list of top-level nodes, and a
node is ``(label, token)`` for a word or ``(label, (left, right))`` for a
chunk. The description length is the plain-PCFG code of ``mdl.py``: one
Dirichlet-multinomial row per category (outcomes: a token, or a pair of child
categories), a row for the category of each top-level chunk, and a
continue/stop row. Under this code a chunk pays for its definition by making
the top level shorter, which is the pressure towards structure.

Two greedy searches lower the code, in the tradition of Brown et al. (1992),
SNPR (Wolff 1982), GRIDS (Langley & Stromsten 2000) and Bayesian model merging
(Stolcke & Omohundro 1994):

* ``word_classes`` merges word types into classes while the code of a
  class-bigram model shrinks: classes that best predict their neighbours.
* ``chunk_and_merge`` repeatedly applies the best of two moves. *Chunk* (B, C)
  replaces every non-overlapping adjacent pair of top-level categories B C
  with a new chunk. *Merge* (A, A') makes two categories one. It stops when no
  move shortens the code. Every move is global: it changes all sentences
  consistently, which is what lets a chunk type pay for itself.
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


def word_classes(sentences: Sequence[Sequence[str]], alpha: float) -> Dict[str, Hashable]:
    """Merge word types into classes while the class-bigram code shrinks."""
    vocab = sorted({w for s in sentences for w in s})
    V = len(vocab) + 1
    cls = {w: w for w in vocab}

    def bits(cls):
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

    current = bits(cls)
    while True:
        best = None
        for a, b in itertools.combinations(sorted(set(cls.values())), 2):
            trial = {w: (a if c == b else c) for w, c in cls.items()}
            value = bits(trial)
            if value < current - 1e-9 and (best is None or value < best[0]):
                best = (value, trial)
        if best is None:
            return cls
        current, cls = best


def _chunk(analyses, B, C, Y):
    out = []
    for tops in analyses:
        new, t = [], 0
        while t < len(tops):
            if t + 1 < len(tops) and tops[t][0] == B and tops[t + 1][0] == C:
                new.append((Y, (tops[t], tops[t + 1])))
                t += 2
            else:
                new.append(tops[t])
                t += 1
        out.append(new)
    return out


def _relabel(node: Node, a, b) -> Node:
    lab, body = node
    lab = a if lab == b else lab
    if isinstance(body, str):
        return (lab, body)
    return (lab, (_relabel(body[0], a, b), _relabel(body[1], a, b)))


def chunk_and_merge(analyses: List[List[Node]], n_tokens: int, alpha: float,
                    max_steps: int = 500, log=None) -> Tuple[List[List[Node]], float]:
    """Greedy best-first chunk and merge moves while the code shrinks."""
    current = code_bits(analyses, n_tokens, alpha)
    fresh = 0
    for step in range(max_steps):
        pairs = Counter()
        for tops in analyses:
            for x, y in zip(tops, tops[1:]):
                pairs[(x[0], y[0])] += 1
        labels = sorted({n[0] for tops in analyses for top in tops for n in _nodes(top)}, key=str)
        best = None
        for (B, C), count in pairs.items():
            if count < 2:
                continue
            trial = _chunk(analyses, B, C, ("chunk", fresh))
            value = code_bits(trial, n_tokens, alpha)
            if value < current - 1e-9 and (best is None or value < best[0]):
                best = (value, trial, f"chunk({B}, {C})")
        for a, b in itertools.combinations(labels, 2):
            trial = [[_relabel(n, a, b) for n in tops] for tops in analyses]
            value = code_bits(trial, n_tokens, alpha)
            if value < current - 1e-9 and (best is None or value < best[0]):
                best = (value, trial, f"merge({a}, {b})")
        if best is None:
            break
        current, analyses, move = best
        if move.startswith("chunk"):
            fresh += 1
        if log:
            log(step, move, current)
    return analyses, current


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
