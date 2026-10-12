"""The board search with merge moves now lives in ``trellis2.chess``
(``BoardSearch(merges=True)``); this module keeps the names the experiment
scripts use."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from trellis2.chess import LN2, BoardSearch, _phi, _row_term, code_of_analyses  # noqa: E402,F401


class HierBoardSearch(BoardSearch):
    """BoardSearch with merge moves on by default."""

    def __init__(self, positions, alpha=0.001, counts=True, merges=True, merge_kinds="any"):
        super().__init__(positions, alpha, counts, merges, merge_kinds)

    def _like(self, positions):
        return HierBoardSearch(positions, self.a, self.counts, self.merges, self.merge_kinds)
