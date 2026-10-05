"""Unsupervised learning by minimum description length, by day and by night.

Only sentences are given. The learner looks for analyses and a grammar that
transmit the corpus in the fewest bits, so a chunk type exists only if it pays
for its definition. Superfluous chunks never form; reusable ones do.

By day (``observe``) each sentence is perceived with the current grammar: its
shortest-code analysis, a forest of chunks wherever the grammar has no larger
chunk that pays, with unknown words categorized by their context. The
sentence is stored with that analysis.

By night (``sleep``) the stored analyses are consolidated:

1. Structure: a beam search over chunk and merge moves lowers the plain-PCFG
   code of the corpus (``mdl_search.chunk_and_merge``). The moves are global,
   so analyses stay mutually consistent; a sentence may remain a forest of
   chunks if no larger chunk pays. The search runs from flat sentences, once
   from each partition on the word-class merge path
   (``mdl_search.word_classes``), and from the stored analyses in the
   categories of the previous night.
2. Concepts: the analyses are consolidated into the two hierarchies
   (``Trellis2``), which re-form the categories with chunk context, starting
   from the search's categories, choose the cuts by description length and
   read off the grammar used for parsing and generation.
3. Re-analysis (hard EM): every sentence gets its Viterbi analysis under that
   grammar, kept only if the full description length shrinks.

Steps 2 and 3 run on each of the best few distinct search results, and the
grammar with the shortest total code wins: the plain code guides the search,
the full code decides.

The stored analyses are then written in the new grammar's categories, which
the next day perceives with and the next night starts from. Sleeping once
after observing everything is batch learning.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from .chart import Chart
from .data import Tree
from .grammar import Grammar
from .mdl_search import Node, chunk_and_merge, code_bits, from_tree, to_tree, word_classes
from .model import Learner, Trellis2


class UnsupervisedLearner(Learner):
    """Sentences by day and by night (see the module docstring). Per
    sentence, ``analyses`` holds its current analysis (top-level nodes), or
    None if it arrived before there was a grammar to perceive it with."""

    def __init__(self, beam: int = 4, patience: int = 3, levels: int = 12,
                 consolidations: int = 3, reanalysis_steps: int = 5, alpha: float = 0.001,
                 seed: int = 0, **trellis_kwargs):
        super().__init__(seed=seed, alpha=alpha, **trellis_kwargs)
        self.beam = beam
        self.consolidations = consolidations
        self.patience = patience
        self.levels = levels
        self.reanalysis_steps = reanalysis_steps
        self.trees: List[Tree] = []
        self.nights = 0

    # Day ---------------------------------------------------------------- #
    def observe(self, tokens: Sequence[str]) -> Optional[Tree]:
        """Perceive a sentence with the current grammar and store it. Returns
        the analysis (None before the first night)."""
        tokens = list(tokens)
        self.experiences.append(tokens)
        if self.model is None:
            self.analyses.append(None)
            return None
        tree = Chart(self.model.grammar, tokens).viterbi_tree()
        self.analyses.append(from_tree(tokens, tree, lambda span: ("s", tree.label[span])))
        return tree

    # Night -------------------------------------------------------------- #
    def sleep(self) -> Grammar:
        t0 = time.time()
        night = self.nights
        n_tokens = len({w for s in self.experiences for w in s}) + 1

        def log(stage, move, bits, seconds=None):
            self.history.append({"night": night, "stage": stage, "move": move, "bits": bits,
                                 "sentences": len(self.experiences),
                                 "seconds": time.time() - t0 if seconds is None else seconds})

        # Starting points: flat sentences in each partition on the word-class
        # merge path, and (after the first night) the stored analyses, in the
        # categories of the last night. Categories can only merge during the
        # search, and more data may call for finer ones, so the night may
        # always start over; the shortest code decides.
        path = word_classes(self.experiences, self.alpha)
        starts = [(f"{len(set(cls.values()))} word classes",
                   [[(("w", cls[w]), w) for w in s] for s in self.experiences])
                  for cls in path[-self.levels:]]
        if self.model is not None:
            starts.append(("perceived analyses", self.analyses))
        flat = [[(("w", path[-1][w]), w) for w in s] for s in self.experiences]
        log("flat", "start", code_bits(flat, n_tokens, self.alpha))
        results = []
        for name, start in starts:
            analyses, bits = chunk_and_merge(start, n_tokens, self.alpha,
                                             beam=self.beam, patience=self.patience)
            results.append((bits, name, analyses))
        search_seconds = time.time() - t0
        # The plain code guides the search, but the night minimizes the full
        # code: each of the best few distinct search results is consolidated
        # and re-analysed, and the one whose grammar describes the corpus in
        # the fewest bits wins.
        results.sort(key=lambda r: r[0])
        candidates = []
        for r in results:
            if all(abs(r[0] - c[0]) > 1e-6 for c in candidates):
                candidates.append(r)
            if len(candidates) == self.consolidations:
                break
        best = None
        for bits_plain, name, analyses in candidates:
            outcome = self._consolidate([to_tree(a) for a in analyses])
            # Equal full codes (up to rounding): keep the shorter plain code.
            if best is None or outcome[1] < best[1] - 1e-6:
                best = outcome + (name, bits_plain)
        model, bits, trees, steps, name, bits_plain = best
        log("structure", f"chunk and merge from {name}", bits_plain, search_seconds)
        log("concepts", "consolidate", steps[0])
        for b in steps[1:]:
            log("re-analysis", "viterbi", b)
        self.model, self.trees = model, trees
        self.analyses = self._in_categories(model, trees)
        self.nights += 1
        return model.grammar

    def _consolidate(self, trees: List[Tree]):
        """Consolidate, then re-analyse (hard EM) while the total code shrinks.
        Returns the model, its code, the analyses and the code after each step."""
        model, bits = self._fit(trees)
        steps = [bits]
        for _ in range(self.reanalysis_steps):
            new = [Chart(model.grammar, s).viterbi_tree() for s in self.experiences]
            if all(a.brackets() == b.brackets() and a.roots == b.roots
                   for a, b in zip(trees, new)):
                break
            m2, b2 = self._fit(new)
            if b2 >= bits - 1e-6:
                break
            trees, model, bits = new, m2, b2
            steps.append(bits)
        return model, bits, trees, steps

    def _fit(self, trees: Sequence[Tree]) -> Tuple[Trellis2, float]:
        """Consolidate, starting from the analyses' own categories (the
        search's, or the grammar's after re-analysis)."""
        model = self.fit(trees)
        return model, model.grammar.info["total bits"]

    def _in_categories(self, model: Trellis2, trees: Sequence[Tree]) -> List[List[Node]]:
        """The analyses with every element labelled by its grammar symbol."""
        mem, symbol = model.memory, model.grammar.elem_symbol
        labels: List[Dict] = [{} for _ in self.experiences]
        for e, (sid, span) in enumerate(zip(mem.experience_of, mem.span)):
            labels[sid][span] = ("s", int(symbol[e]))
        return [from_tree(tokens, tree, labels[i].__getitem__)
                for i, (tokens, tree) in enumerate(zip(self.experiences, trees))]

    # Performance -------------------------------------------------------- #
    def parse(self, tokens: Sequence[str]) -> Tree:
        return Chart(self.grammar, tokens).mbr_tree()

    def analyse(self, tokens: Sequence[str]) -> Tree:
        """The shortest-code analysis (possibly partial: a forest of chunks)."""
        return Chart(self.grammar, tokens).viterbi_tree()

    def chart(self, tokens: Sequence[str]) -> Chart:
        return Chart(self.grammar, tokens)

    def log_prob(self, tokens: Sequence[str]) -> float:
        """ln P(sentence) under the grammar, summed over its analyses."""
        return Chart(self.grammar, tokens).log_prob
