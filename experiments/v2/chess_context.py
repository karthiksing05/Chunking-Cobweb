"""The board's top level read in context now lives in ``trellis2.chess``
(``ContextBoardSearch``, ``BoardMemory(context=True)``,
``ChessLearner(context=True)``); this module keeps the names the experiment
scripts use."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from trellis2.chess import (CHAIN, CHOICE_NATS, CONTEXT_OPTIONS, BoardMemory, ChessLearner,  # noqa: E402,F401
                            ContextBoardSearch, ContextIds, backward, chain_values, context_nats)

OPTIONS = CONTEXT_OPTIONS


def ContextBoardMemory(*args, **kw):
    return BoardMemory(*args, context=True, **kw)


class ContextChessLearner(ChessLearner):
    def __init__(self, seed=0, alpha=0.001, max_steps=500, counts=True, merges=False, patience=0, log=None):
        super().__init__(seed=seed, alpha=alpha, max_steps=max_steps, counts=counts, merges=merges,
                         patience=patience, context=True)
