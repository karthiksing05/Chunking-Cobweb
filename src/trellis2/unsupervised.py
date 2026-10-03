"""Unsupervised learning: the analyses come from the learner's own parses.

Raw sentences are stored. Consolidation alternates two steps that lower the
same description length (grammar plus derivations):

1. learn a grammar from the current analyses (``Trellis2.consolidate``:
   hierarchies rebuilt, cuts chosen by MDL);
2. re-analyse every stored sentence: the Viterbi tree is the derivation with
   the shortest code under that grammar; a tree sampled from the tempered
   posterior explores alternatives (annealed stochastic EM).

The stored sentences, not the trees, are the memory, so an early bad analysis
is revisited at every consolidation rather than becoming ground truth. Every
iteration's code is recorded and the shortest is kept (an MDL choice made
without gold trees). With ``curriculum`` the learner first works on the
shortest sentences and admits longer ones stage by stage, parsing each
newcomer with the grammar learned so far ("baby steps").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chart import Chart
from .data import Tree
from .grammar import Grammar
from .model import Trellis2


def initial_tree(n: int, kind: str, rng: np.random.Generator) -> Tree:
    """A starting analysis: right-branching, left-branching, balanced or random."""
    split = {}
    stack = [(0, n)] if n >= 2 else []
    while stack:
        i, j = stack.pop()
        if kind == "right":
            k = i + 1
        elif kind == "left":
            k = j - 1
        elif kind == "balanced":
            k = (i + j) // 2
        elif kind == "random":
            k = int(rng.integers(i + 1, j))
        else:
            raise ValueError(kind)
        split[(i, j)] = k
        stack.extend(s for s in ((i, k), (k, j)) if s[1] - s[0] >= 2)
    return Tree(n, split)


class UnsupervisedLearner:
    """EM over analyses, scored by description length.

    ``temperatures`` gives, per iteration, the temperature at which new trees
    are drawn from the posterior; ``None`` (or running past the list) means
    the Viterbi tree.
    """

    def __init__(self, inits: Sequence[str] = ("random",), em_iterations: int = 10,
                 temperatures: Optional[Sequence[Optional[float]]] = None,
                 curriculum: bool = False, seed: int = 0, **trellis_kwargs):
        self.inits = list(inits)
        self.em_iterations = em_iterations
        self.temperatures = list(temperatures) if temperatures is not None else []
        self.curriculum = curriculum
        self.seed = seed
        self.trellis_kwargs = trellis_kwargs
        self.sentences: List[List[str]] = []
        self.model: Optional[Trellis2] = None
        self.trees: List[Tree] = []
        self.history: List[Dict[str, float]] = []

    def observe(self, tokens: Sequence[str]) -> None:
        self.sentences.append(list(tokens))
        self.model = None

    def _fit(self, sentences: Sequence[List[str]], trees: Sequence[Tree]) -> Trellis2:
        model = Trellis2(seed=self.seed, **self.trellis_kwargs)
        for tokens, tree in zip(sentences, trees):
            model.learn(tokens, tree)
        model.consolidate()
        return model

    def _em(self, sentences: List[List[str]], trees: List[Tree], tag: str,
            rng: np.random.Generator) -> Tuple[float, Trellis2, List[Tree]]:
        best = None
        for it in range(self.em_iterations):
            model = self._fit(sentences, trees)
            code = model.grammar.info["bits (factored grammar)"]
            record = {"stage": tag, "iteration": it, "sentences": len(sentences),
                      "bits": code, "symbols": model.grammar.K}
            self.history.append(record)
            if best is None or code < best[0]:
                best = (code, model, trees)
            temp = self.temperatures[it] if it < len(self.temperatures) else None
            if temp is None:
                new = [Chart(model.grammar, s).viterbi_tree() for s in sentences]
            else:
                hot = model.grammar.tempered(temp)
                new = [Chart(hot, s).sample_tree(rng) for s in sentences]
            changed = sum(a.brackets() != b.brackets() for a, b in zip(trees, new))
            record["changed"], record["temperature"] = changed, temp
            if changed == 0 and temp is None:
                break
            trees = new
        return best

    def consolidate(self) -> Grammar:
        rng = np.random.default_rng(self.seed)
        self.history = []
        best = None
        for init in self.inits:
            if not self.curriculum:
                trees = [initial_tree(len(s), init, rng) for s in self.sentences]
                result = self._em(self.sentences, trees, init, rng)
                order = list(range(len(self.sentences)))
            else:
                order = sorted(range(len(self.sentences)), key=lambda i: len(self.sentences[i]))
                lengths = sorted({len(self.sentences[i]) for i in order})
                trees_of: Dict[int, Tree] = {}
                model = None
                for L in lengths:
                    stage = [i for i in order if len(self.sentences[i]) <= L]
                    for i in stage:
                        if i not in trees_of:
                            s = self.sentences[i]
                            trees_of[i] = (Chart(model.grammar, s).viterbi_tree() if model
                                           else initial_tree(len(s), init, rng))
                    code, model, trees = self._em([self.sentences[i] for i in stage],
                                                  [trees_of[i] for i in stage],
                                                  f"{init}<= {L}", rng)
                    trees_of.update(zip(stage, trees))
                result = (code, model, trees)
                order = stage
            if best is None or result[0] < best[0]:
                best = (result[0], result[1], [result[2][order.index(i)] for i in range(len(order))])
        _, self.model, self.trees = best
        return self.model.grammar

    @property
    def grammar(self) -> Grammar:
        if self.model is None:
            self.consolidate()
        return self.model.grammar

    def parse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).mbr_tree()

    def chart(self, tokens: Sequence[str]) -> Chart:
        return Chart(self.grammar, tokens)

    def generate(self, n: int, rng: Optional[np.random.Generator] = None, **kw):
        self.grammar  # ensure consolidated
        return self.model.generate(n, rng, **kw)
