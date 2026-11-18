"""Round 208 (M3 of r204): the window records identity. Trellis3R208 is Trellis3R207 unless store_same (learner) and repeat_read (read) are set.

store_same (constructor flag, default False): every stored word occurrence also records, per window slot k, whether its own token
equals the word at l_k: attributes s1..sW, "1" / "0" (a slot outside the sentence, and any chunk occurrence, leaves s_k blank).
The attributes join the context description, so the learner is relearned and its tree changes. With store_same False the class
describes, stores and reads exactly as Trellis3R207 (byte-identical).

repeat_read (default False; needs store_same): _gap_fills reads each committed occurrence with the query extended by
s_k = [occurrence's own word == q[l_k]] for every slot whose query word is a real word; the occurrence's factor is
P(s_k | its leaf) with the unchanged hierarchical back-off. The candidate filler is the occurrence's own word, so a filler that
repeats a window word reads the occurrences that did repeat there. Implemented by grouping the committed records by that pattern
and running the parent read on each group with the extended query (gamma must be 1); the group masses add. With repeat_read
False, or no real word in the window, the parent's method is called with the same arguments.
"""
from __future__ import annotations

from typing import Dict, Optional

from .trellis3 import BOS, Trellis3
from .trellis3_r207 import Trellis3R207


def same_names(window: int) -> tuple:
    return tuple(f"s{k}" for k in range(1, window + 1))


class Trellis3R208(Trellis3R207):
    store_same = False
    repeat_read = False

    def __init__(self, *args, store_same: bool = False, **kw):
        self.store_same = bool(store_same)          # set before the parent builds its attribute list
        super().__init__(*args, **kw)

    def _surface_names(self, window: int) -> tuple:
        base = Trellis3._surface_names(window)
        return base + same_names(window) if self.store_same else base

    def _surface(self, tokens, i, j) -> dict:
        x = super()._surface(tokens, i, j)
        if self.store_same and j - i == 1:
            for k in range(1, self.window + 1):
                if i - k >= 0:
                    x[f"s{k}"] = "1" if tokens[i] == tokens[i - k] else "0"
        return x

    def same_pattern(self, e, q: dict, ks) -> Optional[tuple]:
        """The occurrence's s_k values for the query's real window words ks; None for a chunk occurrence."""
        if e.token is None:
            return None
        return tuple("1" if e.token == q[f"l{k}"] else "0" for k in ks)

    def _gap_fills(self, q: dict, beta: float = 1.0, nbq: Optional[Dict[int, dict]] = None,
                   gamma: float = 1.0, backoff: str = "tokens") -> Dict[tuple, float]:
        if not (self.store_same and self.repeat_read):
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        ks = [k for k in range(1, self.window + 1) if q.get(f"l{k}") not in (None, BOS)]
        if not ks:
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        assert gamma == 1.0 and not self.__dict__.get("conj"), "repeat_read needs gamma 1 and no joined histories"
        cache = self.__dict__.setdefault("_r208_cache", {})
        key = (tuple(sorted(q.items())), repr(nbq), len(self.records), self.stats.get("experiences", 0), beta, backoff,
               bool(self.edge_exact))
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        groups: Dict[Optional[tuple], list] = {}
        for e in self.records:
            if e.leaf is not None:
                groups.setdefault(self.same_pattern(e, q, ks), []).append(e)
        mass: Dict[tuple, float] = {}
        saved = self.records, self.__dict__.get("_gap_cache")
        try:
            for pat, recs in groups.items():
                q2 = dict(q)
                if pat is not None:
                    q2.update({f"s{k}": p for k, p in zip(ks, pat)})
                self.records, self._gap_cache = recs, {}
                for v, m in super()._gap_fills(q2, beta, nbq, gamma, backoff).items():
                    mass[v] = mass.get(v, 0.0) + m
        finally:
            self.records = saved[0]
            if saved[1] is None:
                self.__dict__.pop("_gap_cache", None)
            else:
                self._gap_cache = saved[1]
        cache[key] = mass
        return mass
