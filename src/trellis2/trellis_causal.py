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

Used by experiments/v3/chars_coders.py with the learner config "seq_causal": true (off by
default). Kept out of trellis3.py so the base learner is untouched.
"""
from __future__ import annotations

from .trellis3 import BOS, Trellis3


def suffixes(left, window):
    """{l1: t1, l12: "t2 t1", ...}: the nested suffixes of the left context (left[0] = l1)."""
    return {("l1" if k == 1 else f"l1{k}"): " ".join(reversed(left[:k])) for k in range(1, window + 1)}


class CausalTrellis(Trellis3):
    def __init__(self, nest: bool = False, **kw):
        self.nest = nest
        kw["donate_ctx"] = False                 # the own content is not known before the unit
        super().__init__(**kw)
        if self.boundary != "marker" or self.window_names or self.neighbour_types or \
                self.neighbour_concepts or self.neighbour_chunks or self.relational:
            raise ValueError("seq_causal: the left window only (boundary marker, no neighbour facets)")

    def _surface_names(self, window: int) -> tuple:
        if self.nest:
            return tuple("l1" if k == 1 else f"l1{k}" for k in range(1, window + 1))
        return tuple(f"l{k}" for k in range(window, 0, -1))

    def _surface(self, tokens, i, j) -> dict:
        left = [tokens[i - k] if i - k >= 0 else BOS for k in range(1, self.window + 1)]
        if self.nest:
            return suffixes(left, self.window)
        return {f"l{k}": left[k - 1] for k in range(1, self.window + 1)}
