"""Round 207 (M1 of r206): a novel continuation gets only its history's missing mass. Trellis3R207 is Trellis3R204 unless
escape_mass is set; the read itself (_gap_fills) is never changed, so the default path is byte-identical.

escape_split(q, mass, pair_stored) rescales one step's filler masses (the unchanged tree read's) so that the novel fillers
share exactly the history's missing mass e = N1 / (c + N1), and the attested ones share the rest:
  - history h = (l2, l1): c(h) = committed word occurrences whose context has exactly that history, N1(h) = fillers seen
    exactly once after it; a filler is novel when (l2, l1, w) was never committed. The tree read decides which novel words
    share the mass (and which attested words share the rest): each side keeps its relative masses.
  - pair level, nested inside: h1 = l1 with c(l1), N1(l1) from the stored pair counts. The fillers the type-licence gate
    admits (pair (l1, w) never stored) get at most the pair-level missing mass e1 = N1(l1) / (c(l1) + N1(l1)); the novel
    trigrams over a stored pair take what is left of e.
  - c(h) = 0 (history never seen), the start step (l1 = <s>), or no novel (or no attested) filler admitted: unchanged.
No constant. The word / end split of a filler is kept by multiplying both its masses by the same factor (done by the caller).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .trellis3 import BOS, FILL
from .trellis3_r204 import Trellis3R204


class Trellis3R207(Trellis3R204):
    escape_mass = False

    def _hist_index(self):
        """({(l2, l1): {word: count}}, {l1: {word: count}}) over the committed word occurrences, cached until memory changes
        (the flag is part of the key)."""
        key = (len(self.records), self.stats.get("experiences", 0), bool(self.escape_mass))
        got = self.__dict__.get("_r207_idx")
        if got is None or got[0] != key:
            tri: Dict[tuple, Dict[str, float]] = {}
            uni: Dict[str, Dict[str, float]] = {}
            for e in self.records:
                if e.leaf is None or e.ctx is None:
                    continue
                for which, attr, value, w in e.pas:
                    if which == "ctx" and attr == FILL and value[0] == "w":
                        l1, l2 = e.ctx.get("l1", BOS), e.ctx.get("l2", BOS)
                        d = tri.setdefault((l2, l1), {})
                        d[value[1]] = d.get(value[1], 0.0) + w
                        d = uni.setdefault(l1, {})
                        d[value[1]] = d.get(value[1], 0.0) + w
            self._r207_idx = got = (key, (tri, uni))
        return got[1]

    def history_mass(self, l2: str, l1: str):
        """(c, N1, attested counts) of the history (l2, l1); N1 counts fillers whose count is exactly one."""
        d = self._hist_index()[0].get((l2, l1), {})
        return sum(d.values()), sum(1 for x in d.values() if abs(x - 1.0) < 1e-9), d

    def escape_split(self, q: dict, mass: Dict[str, float],
                     pair_stored: Optional[Callable[[str], bool]] = None) -> Dict[str, float]:
        """Multiplier per filler word for one step. q: the step's query (l1, l2); mass: admitted word -> read mass (word + end
        mass together); pair_stored(w): (l1, w) was stored (default: every pair counts as stored, so only the trigram level acts).
        Returns {} (leave the read unchanged) when the rule does not apply."""
        l1, l2 = q.get("l1", BOS), q.get("l2", BOS)
        if l1 == BOS:
            return {}
        c, n1, att = self.history_mass(l2, l1)
        if c <= 0:
            return {}
        e = n1 / (c + n1)
        pair_stored = pair_stored or (lambda w: True)
        S = {w for w, m in mass.items() if att.get(w, 0.0) > 0 and m > 0}
        Ns = {w for w, m in mass.items() if w not in S and m > 0 and pair_stored(w)}
        Np = {w for w, m in mass.items() if w not in S and m > 0 and not pair_stored(w)}
        sS, sNs, sNp = (sum(mass[w] for w in X) for X in (S, Ns, Np))
        if sS <= 0 or sNs + sNp <= 0:
            return {}
        tot = sS + sNs + sNp
        pNp = e * sNp / (sNs + sNp) if sNp > 0 else 0.0
        if sNp > 0:
            uni = self._hist_index()[1].get(l1, {})
            c1, n11 = sum(uni.values()), sum(1 for x in uni.values() if abs(x - 1.0) < 1e-9)
            if c1 > 0:
                pNp = min(pNp, n11 / (c1 + n11))        # the pair-level missing mass caps the licence-admitted fillers
        pNs = (e - pNp) if sNs > 0 else 0.0
        pS = 1.0 - pNp - pNs
        out = {}
        for X, sX, pX in ((S, sS, pS), (Ns, sNs, pNs), (Np, sNp, pNp)):
            for w in X:
                out[w] = tot * pX / sX
        return out
