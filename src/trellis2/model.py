"""TRELLIS v2: learn, consolidate, parse, generate."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chart import Chart
from .data import Tree
from .grammar import Grammar, compile_grammar
from .memory import Memory


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


class Trellis2:
    """Two Cobweb hierarchies plus the grammar read out of them.

    ``learn`` records every element of an analysed sentence. ``consolidate``
    re-describes the recorded elements with the current categories, rebuilds
    the representation hierarchy by replay, chooses the cuts by description
    length and rebuilds the composition hierarchy; it repeats until the
    categories stop changing and keeps the round with the shortest code.
    ``parse`` and ``generate`` use that one grammar.
    """

    def __init__(self, context_width: int = 1, spine_depth: int = 2, granularities: int = 2,
                 composition_ref: bool = False, alpha: float = 0.001, seed: int = 0,
                 search: bool = True, merge: bool = True, max_rounds: int = 6):
        self.memory = Memory(context_width=context_width, spine_depth=spine_depth,
                             granularities=granularities, composition_ref=composition_ref)
        self.alpha = alpha
        self.seed = seed
        self.search = search
        self.merge = merge
        self.max_rounds = max_rounds
        self.history: List[Dict[str, float]] = []
        self._grammar: Optional[Grammar] = None

    # Learning ----------------------------------------------------------- #
    def learn(self, tokens: Sequence[str], tree: Tree, weight: float = 1.0) -> None:
        self.memory.add_tree(tokens, tree, weight)
        self._grammar = None

    def consolidate(self) -> Grammar:
        labels = rules = None
        best: Optional[Grammar] = None
        self.history = []
        for r in range(self.max_rounds):
            rtree, leaves = self.memory.build_hierarchy(labels, rules, seed=self.seed)
            g = compile_grammar(self.memory, rtree, leaves, alpha=self.alpha,
                                seed=self.seed, search=self.search, merge=self.merge)
            bits = g.info["bits (factored grammar)"]
            self.history.append({"round": r, "bits": bits, "symbols": g.K,
                                 "rules": g.M})
            if best is None or bits < best.info["bits (factored grammar)"]:
                best = g
            if labels is not None and same_partition(labels[0], g.elem_symbol):
                break
            labels = [g.elem_symbol, g.elem_fine][:self.memory.granularities]
            rules = [g.elem_rule, g.elem_rule_fine][:self.memory.granularities]
        best.info["consolidation rounds"] = len(self.history)
        self._grammar = best
        return best

    @property
    def grammar(self) -> Grammar:
        return self._grammar if self._grammar is not None else self.consolidate()

    # Performance -------------------------------------------------------- #
    def parse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).mbr_tree()

    def chart(self, tokens: Sequence[str]) -> Chart:
        return Chart(self.grammar, tokens)

    def log_prob(self, tokens: Sequence[str]) -> float:
        """Natural-log probability of a token sequence (summed over trees)."""
        return Chart(self.grammar, tokens).log_prob

    def generate(self, n: int, rng: Optional[np.random.Generator] = None,
                 max_len: int = 30, max_tries: int = 100
                 ) -> Tuple[List[Tuple[List[str], Tree]], int]:
        """Sample ``n`` sentences; returns (samples, number rejected as too long)."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        out, rejected = [], 0
        while len(out) < n:
            for _ in range(max_tries):
                s = self.grammar.sample(rng, max_len=max_len)
                if s is not None:
                    out.append(s)
                    break
                rejected += 1
            else:
                raise RuntimeError("grammar keeps producing over-long derivations")
        return out, rejected
