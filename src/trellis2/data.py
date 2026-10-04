"""Sentences, binary analyses, and the paper's synthetic corpora.

A ``Tree`` over ``n`` tokens is a binary bracketing of the half-open span
``[0, n)``. Every composite span ``(i, j)`` (``j - i >= 2``) stores its split
point ``k`` (children ``(i, k)`` and ``(k, j)``). Token positions are the
primitive spans ``(i, i + 1)``. A partial analysis (a forest) lists its
top-level spans in ``roots``; they tile ``[0, n)`` left to right. By default
there is a single root, the whole sentence.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

Span = Tuple[int, int]


@dataclass
class Tree:
    n: int
    split: Dict[Span, int] = field(default_factory=dict)
    # Optional category label per span (primitive or composite), e.g. the
    # symbol index chosen by a parser; not used by evaluation.
    label: Dict[Span, object] = field(default_factory=dict)
    roots: Optional[List[Span]] = None

    def __post_init__(self):
        if self.roots is None:
            self.roots = [(0, self.n)] if self.n else []
            if self.n >= 2 and (0, self.n) not in self.split:
                raise ValueError("tree must contain the full span")
        else:
            self.roots = [tuple(r) for r in self.roots]
            pos = 0
            for (i, j) in self.roots:
                if i != pos or j <= i:
                    raise ValueError("roots must tile [0, n) left to right")
                pos = j
            if pos != self.n:
                raise ValueError("roots must tile [0, n) left to right")

    @property
    def is_forest(self) -> bool:
        return len(self.roots) > 1

    def copy(self, labels: bool = True) -> "Tree":
        return Tree(self.n, dict(self.split), dict(self.label) if labels else {},
                    list(self.roots))

    def composite_spans(self) -> List[Span]:
        """Composite spans, each root's subtree in pre-order, left to right."""
        out = []
        stack = [r for r in reversed(self.roots) if r[1] - r[0] >= 2]
        while stack:
            i, j = stack.pop()
            out.append((i, j))
            k = self.split[(i, j)]
            if not i < k < j:
                raise ValueError(f"split {k} outside span {(i, j)}")
            for c in ((k, j), (i, k)):
                if c[1] - c[0] >= 2:
                    stack.append(c)
        return out

    def children(self, span: Span) -> Tuple[Span, Span]:
        i, j = span
        k = self.split[span]
        return (i, k), (k, j)

    def brackets(self) -> Set[Span]:
        """All composite spans, as a set (the unit of omission/commission)."""
        return set(self.composite_spans())

    def bottom_up(self) -> List[Span]:
        """Primitive spans left to right, then composites by increasing length."""
        prims = [(i, i + 1) for i in range(self.n)]
        comps = sorted(self.composite_spans(), key=lambda s: (s[1] - s[0], s[0]))
        return prims + comps

    def is_valid(self) -> bool:
        seen = set()
        try:
            spans = self.composite_spans()
        except (KeyError, ValueError):  # missing or out-of-range split point
            return False
        for span in spans:
            if span in seen:
                return False
            seen.add(span)
        return len(seen) == sum(j - i - 1 for i, j in self.roots)

    def to_string(self, tokens: Sequence[str], labels: bool = False) -> str:
        def rec(i, j):
            if j - i == 1:
                lab = self.label.get((i, j)) if labels and (i, j) in self.roots else None
                return tokens[i] if lab is None else f"{tokens[i]}/{lab}"
            k = self.split[(i, j)]
            body = f"{rec(i, k)} {rec(k, j)}"
            lab = self.label.get((i, j)) if labels else None
            return f"[{lab} {body}]" if lab is not None else f"[{body}]"
        return " · ".join(rec(i, j) for i, j in self.roots)

    @staticmethod
    def from_brackets(n: int, spans: Set[Span]) -> "Tree":
        """Build a tree from a complete set of non-crossing composite spans."""
        spans = set(spans) | ({(0, n)} if n >= 2 else set())
        split = {}
        for (i, j) in spans:
            # The split point is the end of the longest proper sub-span starting at i.
            ks = [b for (a, b) in spans if a == i and b < j]
            if ks:
                split[(i, j)] = max(ks)
            else:
                # Left child is the token i; check the right side is covered.
                split[(i, j)] = i + 1
        t = Tree(n, split)
        if not t.is_valid():
            raise ValueError("spans do not form a binary tree")
        return t


@dataclass
class Example:
    tokens: List[str]
    tree: Tree
    source: str = ""

    @property
    def sentence(self) -> str:
        return " ".join(self.tokens)


def tree_from_merges(n: int, merges: Sequence[dict]) -> Tree:
    """Rebuild a binary tree from v1's merge list.

    Units are identified by their centre position; merging the units centred
    at ``left`` and ``right`` yields a unit centred at their mean.
    """
    units: Dict[float, Span] = {float(i): (i, i + 1) for i in range(n)}
    split: Dict[Span, int] = {}

    def find(c: float) -> float:
        for key in units:
            if abs(key - c) < 1e-9:
                return key
        raise KeyError(c)

    for m in merges:
        lk, rk = find(float(m["left"])), find(float(m["right"]))
        (i, k), (k2, j) = units.pop(lk), units.pop(rk)
        if k != k2:
            raise ValueError(f"non-adjacent merge {(i, k)} + {(k2, j)}")
        split[(i, j)] = k
        units[(float(m["left"]) + float(m["right"])) / 2.0] = (i, j)
    if n >= 2 and len(units) != 1:
        raise ValueError(f"merges leave {len(units)} units, expected 1")
    return Tree(n, split)


def load_corpus(directory: str) -> List[Example]:
    """Load every ``*.json`` sentence/merges file, sorted by path (as v1 did)."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if "sentence" not in d or "merges" not in d:
            continue
        tokens = d["sentence"].split()
        out.append(Example(tokens, tree_from_merges(len(tokens), d["merges"]),
                           os.path.basename(path)))
    return out


def v1_split(examples: Sequence[Example], seed: int,
             train_frac: float = 0.8, n_test_eval: int = 40
             ) -> Tuple[List[Example], List[Example]]:
    """Reproduce v1's split: seeded shuffle, 80/20, first 40 test sentences.

    ``experiments/learning_curves.py`` (trellis_v1) seeds Python's global RNG
    with ``seed``, shuffles the path-sorted corpus, takes the first 80% for
    training, and scores the first 40 multi-token test sentences.
    """
    data = list(examples)
    random.Random(seed).shuffle(data)
    cut = int(train_frac * len(data))
    train, test = data[:cut], data[cut:]
    test = [ex for ex in test if len(ex.tokens) >= 2][:n_test_eval]
    return train, test


def default_data_root() -> str:
    """The paper corpora live in the trellis_v1 snapshot next to this repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    snapshot = os.path.abspath(os.path.join(repo, "..", "trellis_v1", "data"))
    return snapshot if os.path.isdir(snapshot) else os.path.join(repo, "data")


CONDITIONS = {
    # name: (corpus directory, util.cfg grammar attribute or lexicon variant)
    "small": "cfg_grammar_experiment_small",
    "med": "cfg_grammar_experiment_med",
    "large": "cfg_grammar_experiment_large",
    "term_low": "cfg_terminal_low",
    "term_med": "cfg_terminal_med",
    "term_high": "cfg_terminal_high",
}


def target_grammar(condition: str) -> dict:
    """The generating CFG for a condition (used only for evaluation)."""
    from util.cfg import (LEXICON_VARIANTS, TEST_GRAMMAR_LARGE, TEST_GRAMMAR_MED,
                          TEST_GRAMMAR_SMALL, make_grammar_variant)
    if condition == "small":
        return TEST_GRAMMAR_SMALL
    if condition == "med":
        return TEST_GRAMMAR_MED
    if condition == "large":
        return TEST_GRAMMAR_LARGE
    if condition.startswith("term_"):
        grammar, _ = make_grammar_variant(LEXICON_VARIANTS[condition[len("term_"):]])
        return grammar
    raise KeyError(condition)
