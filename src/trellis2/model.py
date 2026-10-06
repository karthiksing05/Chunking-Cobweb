"""TRELLIS v2: learn, consolidate, parse, generate.

``Trellis2`` learns from analysed experiences; ``Learner`` from experiences
alone, by day and by night, with a domain's structure search."""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

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
                 composition_ref: bool = False, sentence_bags: bool = False, previous_word: bool = True,
                 sentence_parent: bool = False, fresh_pieces: bool = False,
                 alpha: float = 0.001, seed: int = 0,
                 search: bool = True, merge: bool = True, max_rounds: int = 6,
                 memory: Optional[Memory] = None):
        # A domain may bring its own memory (its own surface context and
        # relations, e.g. ``chess.BoardMemory``); sentences use the default.
        self.memory = memory if memory is not None else Memory(
            context_width=context_width, spine_depth=spine_depth, granularities=granularities,
            composition_ref=composition_ref, sentence_bags=sentence_bags, previous_word=previous_word,
            sentence_parent=sentence_parent, fresh_pieces=fresh_pieces)
        self.alpha = alpha
        self.seed = seed
        self.search = search
        self.merge = merge
        self.max_rounds = max_rounds
        self.history: List[Dict[str, float]] = []
        self._grammar: Optional[Grammar] = None

    # Learning ----------------------------------------------------------- #
    def learn(self, experience, analysis=None, weight: float = 1.0) -> None:
        """Record every element of an analysed experience: a sentence and its
        Tree, a character's relational tree, a position and its elements."""
        self.memory.add(experience, analysis, weight)
        self._grammar = None

    def consolidate(self, init_labels: Optional[List[np.ndarray]] = None) -> Grammar:
        """``init_labels`` (per granularity, one category per recorded
        element) warm-starts the re-description rounds from a previous
        solution instead of from blank chunk context."""
        labels = init_labels
        rules = None
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

    def log_prob(self, experience, analysis=None) -> float:
        """ln P(experience) under the grammar: a sentence summed over its trees
        (inside-outside); a character or a position given its analysis."""
        return self.memory.log_prob(self.grammar, experience, analysis)

    def generate(self, n: int, rng: Optional[np.random.Generator] = None,
                 max_len: int = 30, max_tries: int = 100, whole_only: bool = False
                 ) -> Tuple[List[Tuple[List[str], Tree]], int]:
        """Draw ``n`` experiences with their analyses from the grammar (the
        domain's ``sample``); returns (samples, number rejected and redrawn).
        For sentences, ``whole_only`` keeps to the sentences the grammar
        derives as one tree."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        out, rejected = [], 0
        while len(out) < n:
            for _ in range(max_tries):
                s = self.memory.sample(self.grammar, rng, max_len=max_len, whole_only=whole_only)
                if s is not None:
                    out.append(s)
                    break
                rejected += 1
            else:
                raise RuntimeError("grammar keeps producing over-long derivations")
        return out, rejected


class Learner:
    """TRELLIS v2 from experiences alone, by day and by night. ``observe``
    stores an experience; ``sleep`` runs the domain's structure search, which
    proposes the analyses that shorten the code, and consolidates them into
    the two hierarchies (``fit``); the grammar read off the hierarchies codes
    (``log_prob``) and draws (``generate``) experiences. A domain supplies
    the night (``sleep``) and how a new experience is analysed with what was
    learned (``analyse``): ``unsupervised.UnsupervisedLearner`` for
    sentences, ``chess.ChessLearner`` for positions."""

    def __init__(self, seed: int = 0, alpha: float = 0.001, **trellis_kwargs):
        self.seed, self.alpha = seed, alpha
        self.trellis_kwargs = trellis_kwargs
        self.experiences: List = []
        self.analyses: List = []
        self.model: Optional[Trellis2] = None
        self.history: List[Dict] = []

    def observe(self, experience):
        """Store an experience for the next night."""
        self.experiences.append(experience)

    def sleep(self) -> Grammar:
        raise NotImplementedError

    def analyse(self, experience):
        """An experience's analysis under what has been learned."""
        raise NotImplementedError

    def fit(self, analyses: Sequence, memory: Optional[Memory] = None) -> Trellis2:
        """Consolidate the stored experiences, so analysed, into the two
        hierarchies, starting from the analyses' own categories (each
        element's label) instead of from blank chunk context."""
        model = Trellis2(seed=self.seed, alpha=self.alpha, memory=memory, **self.trellis_kwargs)
        for experience, analysis in zip(self.experiences, analyses):
            model.learn(experience, analysis)
        ids: Dict[Hashable, int] = {}
        labels = np.array([ids.setdefault(lab, len(ids)) for lab in model.memory.label])
        model.consolidate(init_labels=[labels] * model.memory.granularities)
        return model

    @property
    def grammar(self) -> Grammar:
        if self.model is None:
            self.sleep()
        return self.model.grammar

    def log_prob(self, experience) -> float:
        """ln P(experience) under the grammar, given its analysis."""
        return self.model.log_prob(experience, self.analyse(experience))

    def generate(self, n: int, rng: Optional[np.random.Generator] = None, **kw):
        self.grammar  # ensure a night has passed
        return self.model.generate(n, rng, **kw)
