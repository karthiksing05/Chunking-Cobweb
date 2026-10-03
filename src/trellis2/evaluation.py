"""Evaluation against the generating grammar (never used for learning).

Omission and commission follow Langley & Stromsten (2000): on the parse side,
omission is the share of gold substructures (composite spans) the learner
misses and commission the share of its substructures that are not gold; on
the generation side, commission is the share of generated sentences the
target grammar rejects.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .data import Span, Tree


class CFG:
    """A recognizer for the (binary + lexical) synthetic target grammars."""

    def __init__(self, grammar: Dict[str, List[List[str]]], start: str = "S"):
        self.start = start
        self.lex: Dict[str, Set[str]] = defaultdict(set)
        self.binary: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.unary: Dict[str, Set[str]] = defaultdict(set)
        for lhs, prods in grammar.items():
            for p in prods:
                if len(p) == 1 and p[0] not in grammar:
                    self.lex[p[0]].add(lhs)
                elif len(p) == 1:
                    self.unary[p[0]].add(lhs)
                elif len(p) == 2:
                    self.binary[(p[0], p[1])].add(lhs)
                elif len(p) > 2:
                    raise ValueError(f"non-binary production {lhs} -> {p}")

    def _close(self, cell: Set[str]) -> Set[str]:
        frontier = list(cell)
        while frontier:
            b = frontier.pop()
            for a in self.unary.get(b, ()):
                if a not in cell:
                    cell.add(a)
                    frontier.append(a)
        return cell

    def chart(self, tokens: Sequence[str]) -> Dict[Span, Set[str]]:
        n = len(tokens)
        cells: Dict[Span, Set[str]] = {}
        for i, t in enumerate(tokens):
            cells[(i, i + 1)] = self._close(set(self.lex.get(t, ())))
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell: Set[str] = set()
                for k in range(i + 1, j):
                    for b in cells[(i, k)]:
                        for c in cells[(k, j)]:
                            cell |= self.binary.get((b, c), set())
                cells[(i, j)] = self._close(cell)
        return cells

    def recognizes(self, tokens: Sequence[str]) -> bool:
        if not tokens or any(t not in self.lex for t in tokens):
            return False
        return self.start in self.chart(tokens)[(0, len(tokens))]

    def gold_labels(self, tokens: Sequence[str], tree: Tree) -> Dict[Span, str]:
        """Category of each span of a gold tree (the label consistent with the
        bracketing; ties broken alphabetically). For diagnostics only."""
        labels: Dict[Span, Set[str]] = {}
        for (i, j) in tree.bottom_up():
            if j - i == 1:
                labels[(i, j)] = self._close(set(self.lex.get(tokens[i], ())))
            else:
                k = tree.split[(i, j)]
                cell: Set[str] = set()
                for b in labels[(i, k)]:
                    for c in labels[(k, j)]:
                        cell |= self.binary.get((b, c), set())
                labels[(i, j)] = self._close(cell)
        # Resolve top-down so every label is used by its parent's production.
        out: Dict[Span, str] = {}
        stack = [((0, tree.n), self.start if self.start in labels[(0, tree.n)]
                  else min(labels[(0, tree.n)] or {"?"}))]
        while stack:
            (i, j), lab = stack.pop()
            out[(i, j)] = lab
            if j - i < 2:
                continue
            k = tree.split[(i, j)]
            pick = None
            for b in sorted(labels[(i, k)]):
                for c in sorted(labels[(k, j)]):
                    if lab in self.binary.get((b, c), ()):
                        pick = (b, c)
                        break
                if pick:
                    break
            if pick is None:
                pick = (min(labels[(i, k)] or {"?"}), min(labels[(k, j)] or {"?"}))
            stack.append(((i, k), pick[0]))
            stack.append(((k, j), pick[1]))
        return out


@dataclass
class BracketTally:
    matched: int = 0
    gold: int = 0
    predicted: int = 0
    exact: int = 0
    sentences: int = 0

    def add(self, gold: Tree, pred: Tree) -> None:
        g, p = gold.brackets(), pred.brackets()
        self.matched += len(g & p)
        self.gold += len(g)
        self.predicted += len(p)
        self.exact += int(g == p)
        self.sentences += 1

    @property
    def omission(self) -> float:
        return 1.0 - self.matched / max(self.gold, 1)

    @property
    def commission(self) -> float:
        return 1.0 - self.matched / max(self.predicted, 1)

    @property
    def exact_match(self) -> float:
        return self.exact / max(self.sentences, 1)


def purity(assignments: Iterable[Tuple[int, str]]) -> float:
    """Share of items whose cluster's majority label equals their own label."""
    by_cluster: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    for cluster, label in assignments:
        by_cluster[cluster][label] += 1
        total += 1
    return sum(max(d.values()) for d in by_cluster.values()) / max(total, 1)
