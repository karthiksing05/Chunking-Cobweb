"""TRELLIS on sequences with a causal context (the sequence counterpart of chess_causal).

The learner's context tree holds each occurrence by what a left-to-right reader has decoded
when it arrives: the ``window`` tokens before it (l_window ... l1, <s> before the start), and
nothing to its right; the own-content facet (cc) is dropped, since a reader does not know a
unit's content before reading it (as chess_causal drops it). Everything else is Trellis3:
parsing, chunking, the content tree and the chart grammar read the same machinery, only the
context description is causal, so the context concepts organise by what a reader knows.

Used by experiments/v3/chars_coders.py with the learner config "seq_causal": true (off by
default). Kept out of trellis3.py so the base learner is untouched.
"""
from __future__ import annotations

from .trellis3 import BOS, Trellis3


class CausalTrellis(Trellis3):
    def __init__(self, **kw):
        kw["donate_ctx"] = False                 # the own content is not known before the unit
        super().__init__(**kw)
        if self.boundary != "marker" or self.window_names or self.neighbour_types or \
                self.neighbour_concepts or self.neighbour_chunks or self.relational:
            raise ValueError("seq_causal: the left window only (boundary marker, no neighbour facets)")

    @staticmethod
    def _surface_names(window: int) -> tuple:
        return tuple(f"l{k}" for k in range(window, 0, -1))

    def _surface(self, tokens, i, j) -> dict:
        return {f"l{k}": tokens[i - k] if i - k >= 0 else BOS for k in range(1, self.window + 1)}
