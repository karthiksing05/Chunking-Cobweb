"""Round 223: abstract the CONTEXT the generator conditions on, keep the emission exact. Trellis3R223 is Trellis3R214 unless ctx_bits / ctx_func_bits is set.

ctx_bits (default None = unchanged, byte-identical): the content words of the query window (slots l1.. that survive read_reach W') are read at level b_ctx of
the word-category tree (WordTree, the class source of r214's emit_bits): a content word v stands for its class C_b(v), and the occurrence's factor for that slot is
P(slot in C | the occurrence's context leaf) = sum over u in C of P(slot = u | leaf), each term with the unchanged hierarchical Dirichlet back-off along the leaf's
path (the sum of the back-off estimates is the back-off estimate of the sum, so the read is the same read, marginalised over the class). b = inf / None is the exact word;
b = 0 is the root class: the slot is a wildcard (the factor is 1 and the attribute is dropped, exactly a smaller reach for that slot).
"Content" = not among the emit_freq_cut most frequent word types (the frequent / function-word set of r214's frequency cut); those stay exact unless ctx_func_bits is set,
which reads them at that level too (the function-word abstraction end). Emission is untouched: the filler's mass is still the occurrence's own word; the pair gate of
the driver is the unchanged word-level gate.

Implementation: a read-out change only. _reach_query (called by the r214 read) maps the abstracted slots to _Cls values; AbstractRead, a mixin whose MRO position sits just
above Trellis3, evaluates a query that holds _Cls values with Trellis3._gap_fills' own formula (hierarchical back-off, beta, tokens, edge_exact subsetting
and the caches of the parents are unchanged), and delegates to the parent when there is none. Stored contexts are not changed.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from .trellis3 import BOS, EOS, FILL, Trellis3
from .trellis3_r214 import Trellis3R214

DROP = object()


class _Cls:
    """A class-valued window slot: the set of word types of one class of the word tree at one level."""
    __slots__ = ("name", "members")

    def __init__(self, name, members):
        self.name, self.members = name, members

    def __hash__(self):
        return hash(("cls", self.name))

    def __eq__(self, o):
        return isinstance(o, _Cls) and o.name == self.name

    def __repr__(self):
        return "C(%s)" % self.name


class AbstractRead(Trellis3):
    """Trellis3._gap_fills for a query whose window slots may be classes (the same read, marginalised over the class)."""

    def _gap_fills(self, q: dict, beta: float = 1.0, nbq: Optional[Dict[int, dict]] = None,
                   gamma: float = 1.0, backoff: str = "tokens") -> Dict[tuple, float]:
        if not any(isinstance(v, _Cls) for v in q.values()):
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        assert backoff == "tokens" and not nbq and not self.__dict__.get("conj") and self.__dict__.get("pool") != "geo"
        root = self.ctx.root
        sizes = {a: len(root.av.get(a, {})) + 1 for a in q}
        stamp = (len(self.records), self.stats.get("experiences", 0))
        cm = self.__dict__.setdefault("_r223_cm", {})
        if self.__dict__.get("_r223_stamp") != stamp:
            cm.clear()
            self.__dict__["_r223_stamp"] = stamp
        nm_cache: Dict[tuple, int] = {}

        def cmass(node, a, v) -> float:
            if not isinstance(v, _Cls):
                return node.av.get(a, {}).get(v, 0.0)
            k = (node.id, a, v.name)
            if k not in cm:
                d = node.av.get(a, {})
                mem = v.members
                cm[k] = (sum(c for u, c in d.items() if u in mem) if len(d) < len(mem) else sum(d.get(u, 0.0) for u in mem))
            return cm[k]

        def nmem(a, v) -> int:
            if not isinstance(v, _Cls):
                return 1
            return len(v.members)          # each member, seen in this slot or not, carries the root smoothing alpha as in the exact read

        memo: Dict[int, Dict[str, float]] = {}

        def probs(node) -> Dict[str, float]:
            if node.id in memo:
                return memo[node.id]
            if node.parent is None:
                out = {a: (cmass(node, a, v) + self.alpha * nmem(a, v)) / (node.count + self.alpha * sizes[a]) for a, v in q.items()}
            else:
                up = probs(node.parent)
                out = {a: (cmass(node, a, v) + beta * up[a]) / (node.count + beta) for a, v in q.items()}
            memo[node.id] = out
            return out

        cache = self.__dict__.setdefault("_gap_cache", {})
        key = (tuple(sorted(q.items(), key=lambda kv: kv[0])), len(self.records), self.stats.get("experiences", 0), beta, gamma, backoff, "r223")
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
                if which == "ctx" and attr == FILL:
                    lp = sum(math.log(max(v, 1e-300)) for v in probs(e.leaf).values())
                    mass[value] = mass.get(value, 0.0) + w * math.exp(lp)
                    wt[value] = wt.get(value, 0.0) + w
        if gamma != 1.0:
            mass = {v: m / wt[v] * wt[v] ** gamma for v, m in mass.items() if wt[v] > 0}
        cache[key] = mass
        return mass


class Trellis3R223(Trellis3R214, AbstractRead):
    ctx_bits = None
    ctx_func_bits = None

    def _ctx_names(self, b) -> Dict[str, str]:
        cache = self.__dict__.setdefault("_r223_names", {})
        if b not in cache:
            names = self.word_tree.names(b=float(b))
            groups: Dict[str, set] = {}
            for w, c in names.items():
                groups.setdefault(c, set()).add(w)
            cache[b] = (names, {c: frozenset(m) for c, m in groups.items()}, len(names))
        return cache[b]

    def _ctx_class(self, v: str, b):
        """A slot word's class at b: the word itself (singleton / unknown word), DROP (the class holds every type: a wildcard), or a _Cls."""
        names, groups, n = self._ctx_names(b)
        c = names.get(v)
        if c is None:
            return v
        m = groups[c]
        if len(m) >= n:
            return DROP
        if len(m) <= 1:
            return v
        return _Cls("%s:%s" % (b, c), m)

    def _abstract_query(self, q: dict) -> dict:
        if self.ctx_bits is None and self.ctx_func_bits is None:
            return q
        top = self._top_words() if self.emit_freq_cut else frozenset()
        out = {}
        for a, v in q.items():
            if a[0] == "l" and a[1:].isdigit() and v not in (BOS, EOS) and isinstance(v, str):
                b = self.ctx_func_bits if v in top else self.ctx_bits
                if b is not None and not (not isinstance(b, str) and b == math.inf):
                    c = self._ctx_class(v, b)
                    if c is DROP:
                        continue
                    out[a] = c
                    continue
            out[a] = v
        return out

    def _reach_query(self, q: dict) -> dict:
        return self._abstract_query(super()._reach_query(q))
