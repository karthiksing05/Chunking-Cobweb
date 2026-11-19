"""Round 216: a parser whose analyses are weighted by the generator's own conditional probabilities (opt-in subclass; trellis3.py is untouched).

Trellis3R216(r216=None, r216_mu=1.0) is Trellis3R204 (and so Trellis3) unless r216 is set. With r216 set, `_chart` computes a table of the
generator's word probabilities for the sentence being parsed and `_factor` adds one term per composite to the join's gain (so to the weight
2^(gain/tau) every analysis carries in `_inside_outside`; chosen tree, posteriors and logZ all see it).

Generator probability (the one `_gap_fills` gives generation, as r215b scored it): P(w | left context) = gated mass of w / sum of gated masses of
all words, the mass read with `_gap_fills` from a query holding only the attributes l1..lp of the p words to the left (p <= the window W; with the
sentence's own start marker padding when the context starts at word 0, i.e. exactly the full-window query of generation), edge-exact boundary,
pair gate = the median of the stored pairs' type licence (generation's "gate"). eps floor (p + eps) / (1 + eps V), V = 250.

D[i][t] = log P(w_t | w_i .. w_{t-1}) for a context that starts at word i (padded with the start marker only when i = 0). D[0][t] is the log
probability the generator gives w_t in the sentence.

r216 = "split" (the form used as (b)): composite (i, k, j) adds  mu * (D[i][k] - D[0][k])  nats: the generator's log probability of the first word
of the right child given only the words of the LEFT SIBLING (the left part of the composite) instead of the whole sentence's left window.
Principle: a tree defines a local language model in which each word k >= 1 is generated given the words of its parent chunk to its left; the
split points of a binary tree are exactly the words 1..n-1, each the split of exactly one composite, so every word gets one conditional and the
product over composites is that tree's likelihood P_T(s) = P(w_0) prod_k P(w_k | w_i(k)..w_{k-1}). Subtracting the whole-window term D[0][k]
(the same for every tree, because each k is a split once) makes the term the loss of context that this bracketing imposes: 0 for a composite that
starts at the sentence start, negative when the words to its left that the bracket hides mattered, positive when they hurt. It does not change the
posteriors (the subtracted term is the same in every analysis); it keeps the added score on the generator's scale. Words in the right child's
interior are not scored again, so nothing is counted twice, unlike the literal chunk-sum form below.

r216 = "chunk" (the literal form requested for (b)): composite (i, k, j) adds  mu * sum_{t=i}^{j-1} (D[i][t] - D[0][t])  nats: every word of the
chunk given the words of the chunk to its left, relative to the whole-sentence left window. A word inside d nested chunks is scored d times.

mu = 1 puts the term in nats against the join gain's gain * ln 2 / tau (the weights are 2^(gain/tau)); no fitting. The term is added to the join's
gain in bits as mu * term * tau / ln 2.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .trellis3 import BOS, LN2
from .trellis3_r204 import Trellis3R204


class Trellis3R216(Trellis3R204):
    r216: Optional[str] = None
    r216_mu: float = 1.0
    r216_eps: float = 1e-6
    r216_V: int = 250
    r216_gate: bool = True

    def __init__(self, *args, r216: Optional[str] = None, r216_mu: float = 1.0, **kw):
        super().__init__(*args, **kw)
        assert r216 in (None, "split", "chunk")
        self.r216, self.r216_mu = r216, r216_mu

    # ---- the generator's word distribution given a left context -------------------------------------------------
    def _r216_prep(self) -> None:
        """The gate of generation (r215b Gen): the median of the stored pairs' type licence."""
        if "_r216_gate" in self.__dict__:
            return
        if not self.r216_gate:
            self._r216_gate = None
            return
        w_, ix_, M_ = self.type_licence(near=None)
        st_ = set(getattr(self, "_pair_counts", {}))
        vals = sorted(float(M_[ix_[a], ix_[b]]) for a, b in st_ if a in ix_ and b in ix_)
        med = vals[min(len(vals) - 1, int(len(vals) * 0.5))] if vals else 0.0
        self._r216_gate = (ix_, M_, st_, med)

    def _r216_ok(self, l1: Optional[str], w: str) -> bool:
        g = self._r216_gate
        if g is None or l1 is None or l1 == BOS:
            return True
        ix_, M_, st_, med = g
        if (l1, w) in st_:
            return True
        return l1 in ix_ and w in ix_ and float(M_[ix_[l1], ix_[w]]) >= med > 0

    def r216_dist(self, left: Tuple[str, ...]) -> Dict[str, float]:
        """P(word | the words `left` (oldest first, at most the window of them, BOS-padded by the caller if it is the sentence start))."""
        self._r216_prep()
        cache = self.__dict__.setdefault("_r216_cache", {})
        if left in cache:
            return cache[left]
        q = {"l%d" % (m + 1): left[-1 - m] for m in range(len(left))}
        raw = self._gap_fills(q)
        self.__dict__.pop("_gap_cache", None)          # reduced cache (r215b: n = 4000 runs out of memory with it)
        self.__dict__.pop("_edge_cache", None)
        l1 = left[-1] if left else None
        d = {v[1]: m for v, m in raw.items() if v[0] == "w" and self._r216_ok(l1, v[1])}
        Z = sum(d.values())
        out = {w: m / Z for w, m in d.items()} if Z > 0 else {}
        cache[left] = out
        return out

    def r216_logp(self, left: Tuple[str, ...], w: str) -> float:
        d = self.r216_dist(left)
        if not d:                                           # no admissible word: uniform (as r215b)
            p = 1.0 / self.r216_V
        else:
            p = d.get(w, 0.0)
        return math.log((p + self.r216_eps) / (1 + self.r216_eps * self.r216_V))

    def r216_table(self, tokens: List[str]) -> List[List[float]]:
        """D[i][t] for 0 <= i <= t < n (see the module doc)."""
        W = max(self.window, 2)
        n = len(tokens)
        D = [[0.0] * n for _ in range(n)]
        for t in range(n):
            for i in range(t + 1):
                if i == 0:
                    left = tuple(([BOS] * W + tokens[:t])[-W:])
                else:
                    left = tuple(tokens[max(i, t - W):t])
                D[i][t] = self.r216_logp(left, tokens[t])
        return D

    # ---- the parser ----------------------------------------------------------------------------------------------
    def _chart(self, tokens):
        if self.r216:
            self._r216_D = self.r216_table(list(tokens))
        try:
            return super()._chart(tokens)
        finally:
            self.__dict__.pop("_r216_D", None)

    def _r216_term(self, e) -> float:
        D = self._r216_D
        i, j = e.span
        if self.r216 == "split":
            k = e.split
            return D[i][k] - D[0][k]
        return sum(D[i][t] - D[0][t] for t in range(i, j))

    def _factor(self, e) -> float:
        g = super()._factor(e)
        if self.r216 and "_r216_D" in self.__dict__ and e.token is None:
            g += self.r216_mu * self._r216_term(e) * self.tau / LN2
        return g
