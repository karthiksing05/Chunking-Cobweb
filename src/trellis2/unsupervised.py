"""Unsupervised learning by minimum description length.

Only sentences are given. The learner looks for analyses and a grammar that
transmit the corpus in the fewest bits, so a chunk type exists only if it pays
for its definition. Superfluous chunks never form; reusable ones do.

1. Word classes: word types are merged while the code of a class-bigram
   model shrinks (``mdl_search.word_classes``).
2. Structure: from flat sentences, greedy chunk and merge moves while the
   plain-PCFG code of the corpus shrinks (``mdl_search.chunk_and_merge``).
   The moves are global, so analyses stay mutually consistent; a sentence may
   remain a forest of chunks if no larger chunk pays.
3. Concepts: the analyses are consolidated into the two hierarchies
   (``Trellis2``), which re-form the categories with chunk context, choose
   the cuts by description length and read off the grammar used for parsing
   and generation.
4. Re-analysis (hard EM): every sentence gets its Viterbi analysis under that
   grammar, kept only if the full description length shrinks.

Sentences, not analyses, are the memory: every analysis is revisable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chart import Chart
from .data import Tree
from .grammar import Grammar
from .mdl_search import chunk_and_merge, code_bits, to_tree, word_classes
from .model import Trellis2


class UnsupervisedLearner:
    def __init__(self, reanalysis_steps: int = 5, alpha: float = 0.001, seed: int = 0,
                 **trellis_kwargs):
        self.reanalysis_steps = reanalysis_steps
        self.alpha = alpha
        self.seed = seed
        self.trellis_kwargs = trellis_kwargs
        self.sentences: List[List[str]] = []
        self.model: Optional[Trellis2] = None
        self.trees: List[Tree] = []
        self.classes: Dict[str, object] = {}
        self.history: List[Dict] = []

    def observe(self, tokens: Sequence[str]) -> None:
        self.sentences.append(list(tokens))
        self.model = None

    def _fit(self, trees: Sequence[Tree]) -> Tuple[Trellis2, float]:
        model = Trellis2(seed=self.seed, alpha=self.alpha, **self.trellis_kwargs)
        for tokens, tree in zip(self.sentences, trees):
            model.learn(tokens, tree)
        g = model.consolidate()
        return model, g.info["total bits"]

    def consolidate(self) -> Grammar:
        self.history = []
        n_tokens = len({w for s in self.sentences for w in s}) + 1
        self.classes = word_classes(self.sentences, self.alpha)
        analyses = [[(("w", self.classes[w]), w) for w in s] for s in self.sentences]
        self.history.append({"stage": "flat", "move": "start",
                             "bits": code_bits(analyses, n_tokens, self.alpha)})

        def log(step, move, bits):
            self.history.append({"stage": "chunk and merge", "move": move, "bits": bits})

        analyses, _ = chunk_and_merge(analyses, n_tokens, self.alpha, log=log)
        trees = [Tree(t.n, t.split, {}, t.roots) for t in map(to_tree, analyses)]
        model, bits = self._fit(trees)
        self.history.append({"stage": "concepts", "move": "consolidate", "bits": bits})
        for _ in range(self.reanalysis_steps):
            new = [Chart(model.grammar, s).viterbi_tree() for s in self.sentences]
            if all(a.brackets() == b.brackets() and a.roots == b.roots
                   for a, b in zip(trees, new)):
                break
            m2, b2 = self._fit([Tree(t.n, t.split, {}, t.roots) for t in new])
            if b2 >= bits - 1e-6:
                break
            trees, model, bits = new, m2, b2
            self.history.append({"stage": "re-analysis", "move": "viterbi", "bits": bits})
        self.model, self.trees = model, trees
        return model.grammar

    @property
    def grammar(self) -> Grammar:
        if self.model is None:
            self.consolidate()
        return self.model.grammar

    def parse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).mbr_tree()

    def analyse(self, tokens: Sequence[str]) -> Tree:
        """The shortest-code analysis (possibly partial: a forest of chunks)."""
        return Chart(self.grammar, tokens).viterbi_tree()

    def chart(self, tokens: Sequence[str]) -> Chart:
        return Chart(self.grammar, tokens)

    def generate(self, n: int, rng: Optional[np.random.Generator] = None, **kw):
        self.grammar  # ensure consolidated
        return self.model.generate(n, rng, **kw)
