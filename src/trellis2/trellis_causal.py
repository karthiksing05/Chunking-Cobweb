"""TRELLIS on sequences with a causal context (the sequence counterpart of chess_causal).

The learner's context tree holds each occurrence by what a left-to-right reader has decoded
when it arrives: the ``window`` tokens before it (l_window ... l1, <s> before the start), and
nothing to its right; the own-content facet (cc) is dropped, since a reader does not know a
unit's content before reading it (as chess_causal drops it). Everything else is Trellis3:
parsing, chunking, the content tree and the chart grammar read the same machinery, only the
context description is causal, so the context concepts organise by what a reader knows.

nest=True (r164) describes the left context by its suffixes instead of its positions: l1, then
l1 with l2 as one value ("l2 l1"), up to the window. A reader's context is a nested set (the
last token, the last two, ...), and r163 found the positional description scatters the
occurrences that follow one token over many subtrees (at the sorted leaf 93% share the previous
token but hold only 42% of its occurrences), so the tree's back-off is not nested the way a
reader's knowledge is. Suffix values let category utility group by the last token first.

right=True (r172, with nest) adds the right context as nested prefixes (r1, "r1 r2", ...) and keeps
the own-content facet: the learner is two-sided again, so the chart reads both sides (r171 found
the chart on a left-only learner ~3.5 bits/char worse), while a causal reader sorts only the
left suffixes into the same tree (a partial description). One learner for both codes.

Used by experiments/v3/chars_coders.py with the learner config "seq_causal": true (off by
default). Kept out of trellis3.py so the base learner is untouched.
"""
from __future__ import annotations

from .trellis3 import BOS, EOS, Trellis3


def suffixes(left, window):
    """{l1: t1, l12: "t2 t1", ...}: the nested suffixes of the left context (left[0] = l1)."""
    return {("l1" if k == 1 else f"l1{k}"): " ".join(reversed(left[:k])) for k in range(1, window + 1)}


def prefixes(right, window):
    """{r1: t1, r12: "t1 t2", ...}: the nested prefixes of the right context (right[0] = r1)."""
    return {("r1" if k == 1 else f"r1{k}"): " ".join(right[:k]) for k in range(1, window + 1)}


class CausalTrellis(Trellis3):
    def __init__(self, nest: bool = False, right: bool = False, **kw):
        self.nest, self.right = nest, right
        if right and not nest:
            raise ValueError("seq_causal right: needs nest")
        if not right:
            kw["donate_ctx"] = False             # the own content is not known before the unit
        super().__init__(**kw)
        if self.boundary != "marker" or self.window_names or self.neighbour_types or \
                self.neighbour_concepts or self.neighbour_chunks or self.relational:
            raise ValueError("seq_causal: the left window only (boundary marker, no neighbour facets)")

    def _surface_names(self, window: int) -> tuple:
        if self.nest:
            names = tuple("l1" if k == 1 else f"l1{k}" for k in range(1, window + 1))
            if self.right:
                names += tuple("r1" if k == 1 else f"r1{k}" for k in range(1, window + 1))
            return names
        return tuple(f"l{k}" for k in range(window, 0, -1))

    def _surface(self, tokens, i, j) -> dict:
        left = [tokens[i - k] if i - k >= 0 else BOS for k in range(1, self.window + 1)]
        if self.nest:
            x = suffixes(left, self.window)
            if self.right:
                x.update(prefixes([tokens[j + k] if j + k < len(tokens) else EOS for k in range(self.window)], self.window))
            return x
        return {f"l{k}": left[k - 1] for k in range(1, self.window + 1)}
