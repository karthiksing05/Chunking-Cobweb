"""Round 233: a graded concepts dial for the parser (opt-in subclass; trellis3.py's _gain is unchanged).

Trellis3R233(parse_min_bits=m): the parser's factor for a span (Trellis3._gain: the best over the span's sorting path of
log2 [share(C) P(x|C) / P(x|root)]) is taken only over concepts at information level >= m bits (-log2 of the concept's share of
the tree's root), i.e. concepts more general than 2^-m of memory are not consulted. m = None or 0 is the default (every concept);
m = inf leaves only the deepest node the span sorts to. Learning parses with the same factor, so it also changes what is stored
(as r225's concepts_off_parse).

chosen_mode ("mbr" default | "rb" | "random"): during learning only, the analysis memory commits for each experience (the chosen
tree, whose composites receive SENT / LPART / RPART / FILL and the chunk-context facets) is replaced by the right-branching tree or
by a random binary tree (seeded per experience); the posterior storage of every candidate is unchanged. This isolates what
generation gains from the composition the parser commits."""
from __future__ import annotations

import math
import random

from .trellis3 import LN2, log_prob
from .trellis3_r223 import Trellis3R223


class Trellis3R233(Trellis3R223):
    def __init__(self, *args, parse_min_bits=None, chosen_mode="mbr", **kw):
        super().__init__(*args, **kw)
        self.parse_min_bits = parse_min_bits
        self.chosen_mode = chosen_mode
        self._r233_learning = False
        self._r233_rng = random.Random(f"r233-chosen-{kw.get('seed', args[0] if args else 0)}")
        self.r233_replaced = 0

    def learn(self, tokens, *a, **k):
        self._r233_learning = True
        try:
            return super().learn(tokens, *a, **k)
        finally:
            self._r233_learning = False

    def parse(self, tokens, *a, **k):
        r = super().parse(tokens, *a, **k)
        if not self._r233_learning or self.chosen_mode in (None, "mbr"):
            return r
        n = len(list(tokens))
        if n < 2:
            return r
        if self.chosen_mode == "rb":
            keys = [(i, i + 1, n) for i in range(n - 1)]
        else:
            keys, rng = [], self._r233_rng

            def split(i, j):
                if j - i < 2:
                    return
                kk = rng.randint(i + 1, j - 1)
                keys.append((i, kk, j))
                split(i, kk)
                split(kk, j)
            split(0, n)
        if all(key in r["comps"] for key in keys):
            r = dict(r)
            r["chosen"] = keys
            self.r233_replaced += 1
        return r

    def _gain(self, tree, x):
        m = self.parse_min_bits
        if m is None or m <= 0:
            return super()._gain(tree, x)
        assert self.gain_alpha is None, "parse_min_bits does not support gain_alpha"
        root = tree.root
        if root.count <= 0:
            return 0.0
        path = tree.categorize(x)[1:]
        if not path:
            return 0.0
        keep = [n for n in path if n.count > 0 and -math.log2(n.count / root.count) >= m] or [path[-1]]
        base = log_prob(root, x, root, self.alpha)
        best = 0.0
        for node in keep:
            if node.count <= 0:
                continue
            best = max(best, (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - base) / LN2)
        return best
