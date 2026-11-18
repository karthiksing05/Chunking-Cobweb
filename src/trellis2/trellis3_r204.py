"""Round 204 (M1): the sentence boundary is observed, not inferred. Trellis3R204 is Trellis3 unless edge_exact is set.

edge_exact (default False): in _gap_fills, a query whose first-left slot is the start marker (l1 = BOS) or whose
first-right slot is the end marker (r1 = EOS) is read from the occurrences that REALLY stood at that boundary only.
Such an occurrence keeps its other attributes' back-off factors; the boundary attribute's factor is 1 (it matched).
An occurrence whose own value for the boundary slot differs is skipped, instead of contributing the hierarchical
back-off P(boundary | leaf) of its leaf's parents (which gives a mid-sentence occurrence about half its parent's
end/start mass at single-occurrence leaves: the r204 boundary leak). Interior attributes (l2.., r2.., neighbour bags)
keep the back-off, so novelty inside the sentence is untouched.

Implementation: the boundary attributes are dropped from the query and the committed records are filtered to the
ones whose own context holds the query's value for each of them, then the unchanged Trellis3._gap_fills runs on that
subset with a private cache (keys include the edge flag). With edge_exact False, or a query with no boundary slot,
the parent's method is called with the same arguments (byte-identical).
"""
from __future__ import annotations

from typing import Dict, Optional

from .trellis3 import BOS, EOS, Trellis3


def edge_slots(q: dict) -> list:
    """The gap-adjacent boundary attributes of a query: l1 when it is the start marker, r1 when it is the end marker."""
    return [a for a, v in q.items() if (a == "l1" and v == BOS) or (a == "r1" and v == EOS)]


class Trellis3R204(Trellis3):
    edge_exact = False

    def _gap_fills(self, q: dict, beta: float = 1.0, nbq: Optional[Dict[int, dict]] = None,
                   gamma: float = 1.0, backoff: str = "tokens") -> Dict[tuple, float]:
        if not self.edge_exact:
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        edge = edge_slots(q)
        if not edge:
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        assert not self.__dict__.get("conj"), "edge_exact does not support the joined histories"
        cache = self.__dict__.setdefault("_edge_cache", {})
        key = (tuple(sorted(q.items())), repr(nbq), len(self.records), self.stats.get("experiences", 0), beta, gamma, backoff)
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        q2 = {a: v for a, v in q.items() if a not in edge}
        keep = [e for e in self.records if e.leaf is not None and all(e.ctx.get(a) == q[a] for a in edge)]
        saved = self.records, self.__dict__.get("_gap_cache")
        self.records, self._gap_cache = keep, {}
        try:
            out = super()._gap_fills(q2, beta, nbq, gamma, backoff)
        finally:
            self.records = saved[0]
            if saved[1] is None:
                self.__dict__.pop("_gap_cache", None)
            else:
                self._gap_cache = saved[1]
        cache[key] = out
        return out
