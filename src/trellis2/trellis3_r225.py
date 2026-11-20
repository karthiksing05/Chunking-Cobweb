"""Round 225: ablation switches for the TinyStories recipe (opt-in subclass; trellis3.py is untouched).

Trellis3R225(concepts_off=False, concepts_off_parse=False) is Trellis3R204 (and so Trellis3) unless a flag is set.

Chunks off is not a flag here: it is Trellis3's own store="none" (no composite is stored, the content tree stays empty, only the
words and their windows sit in the context tree; the parser still runs, its gains then come from the context tree alone).

concepts_off (no generalisation: every read sees the instance's own leaf and nothing above it, i.e. memory of instances, no concepts):
  * `_gap_fills` (generation, the generator's score, edge-exact included): a stored record's likelihood of the query is
    prod_a (n_leaf(a=v) + beta P_root(v)) / (n_leaf + beta), i.e. the hierarchical back-off of Trellis3 with the path between the
    leaf and the root skipped (the leaf backs off straight to the Laplace root). The tree is still built (and still decides what shares
    a leaf), but no ancestor concept contributes a statistic.
  * `_gain` (the parser's factor; only with concepts_off_parse, which also changes what learning stores because learn() parses): the best share(C) P(x|C) / P(x|root) is read at the deepest concept x sorts to (its leaf) only,
    not maximised over the sorting path.
  * `type_licence` (the generator's pair gate): the licence of a pair a b is read from the ancestors of b's occurrences (their members),
    so with no ancestors it is 0 for every pair; the gate then passes exactly the pairs stored in the stream (the gate's first rule).
Everything else (learning, windows, edge-exact boundary reading) is the parent's. concepts_off=False is byte-identical to Trellis3R204.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from .trellis3 import LN2, Trellis3, log_prob
from .trellis3_r204 import Trellis3R204


class _LeafOnly(Trellis3):
    concepts_off = False
    concepts_off_parse = False

    def _gap_fills(self, q: dict, beta: float = 1.0, nbq: Optional[Dict[int, dict]] = None,
                   gamma: float = 1.0, backoff: str = "tokens") -> Dict[tuple, float]:
        if not self.concepts_off:
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        assert backoff == "tokens" and not self.__dict__.get("conj") and self.__dict__.get("pool") != "geo" and not nbq, \
            "concepts_off supports the default read only"
        root = self.ctx.root
        sizes = {a: len(root.av.get(a, {})) + 1 for a in q}
        memo: Dict[int, Dict[str, float]] = {}

        def probs(node) -> Dict[str, float]:
            if node.id in memo:
                return memo[node.id]
            if node.parent is None:
                out = {a: (node.av.get(a, {}).get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a])
                       for a, v in q.items()}
            else:
                up = probs(root)                       # leaf backs off to the root: no concept in between
                out = {a: (node.av.get(a, {}).get(v, 0.0) + beta * up[a]) / (node.count + beta) for a, v in q.items()}
            memo[node.id] = out
            return out

        cache = self.__dict__.setdefault("_gap_cache", {})
        key = (tuple(sorted(q.items())), repr(nbq), len(self.records), self.stats.get("experiences", 0), beta, gamma, backoff, "leaf")
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        mass: Dict[tuple, float] = {}
        wt: Dict[tuple, float] = {}
        for e in self.records:
            if e.leaf is None:
                continue
            for which, attr, value, w in e.pas:
                if which == "ctx" and attr == "fill":
                    lp = sum(math.log(max(v, 1e-300)) for v in probs(e.leaf).values())
                    mass[value] = mass.get(value, 0.0) + w * math.exp(lp)
                    wt[value] = wt.get(value, 0.0) + w
        if gamma != 1.0:
            mass = {v: m / wt[v] * wt[v] ** gamma for v, m in mass.items() if wt[v] > 0}
        cache[key] = mass
        return mass

    def _gain(self, tree, x: dict) -> float:
        if not self.concepts_off_parse:
            return super()._gain(tree, x)
        root = tree.root
        if root.count <= 0:
            return 0.0
        path = tree.categorize(x)
        node = path[-1]
        if node is root or node.count <= 0:
            return 0.0
        g = (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - log_prob(root, x, root, self.alpha)) / LN2
        return max(0.0, g)

    def type_licence(self, near: Optional[int] = None):
        out = super().type_licence(near)
        if not self.concepts_off:
            return out
        words, ix, M = out
        return words, ix, M * 0.0


class Trellis3R225(Trellis3R204, _LeafOnly):
    """MRO: R225, R204 (edge-exact reads), _LeafOnly (leaf-only reads), Trellis3."""

    def __init__(self, *args, concepts_off: bool = False, concepts_off_parse: bool = False, **kw):
        super().__init__(*args, **kw)
        self.concepts_off = concepts_off
        self.concepts_off_parse = concepts_off_parse
