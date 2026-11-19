"""Round 214: the information level of the read, and the read's reach, as opt-in dials. Trellis3R214 is Trellis3R208 unless a flag is set.

read_bits (default None = the context leaf, byte-identical; math.inf is the same; the string "basic" = the basic level):
    _gap_fills reads each committed occurrence at C_b(e), the deepest ancestor of its context leaf (the leaf included) whose
    information level -log2(count(C) / count(root)) is at most b bits, instead of at the leaf. b = 0 is the root. "basic" is
    the concept on the leaf's path with the highest category utility (the rule of naming="basic", Trellis3._path_ids; the root
    when the path holds no inner concept). The occurrences are replaced by light proxies whose leaf is the lifted concept, inside
    try / finally; every cache the parents keep is swapped out for the call (their keys do not hold b) and restored.
    The end read (r1 = </s>) is the same function and is lifted the same way.
read_reach (default None = the learner's whole window): the query keeps l1..lW' and drops l_k for k > W'. The product read
    marginalises a dropped slot exactly. One learned memory serves every W'.
parse_bits (default None = unchanged): _gain and _code take the maximum over the sorting-path concepts at level <= b only
    (b <= 0: _gain is 0 for every chunk; _code keeps the root).

word_classes(b) returns the modal C_b concept id per word type (the classes of the class n-gram baseline).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from .trellis3 import FILL, LN2, log_prob
from .trellis3_r208 import Trellis3R208


class _Proxy:
    """A committed occurrence read at another concept: only the leaf differs."""
    __slots__ = ("_e", "leaf", "pas", "span", "ctx", "token", "weight")

    def __init__(self, e, leaf):
        self._e, self.leaf = e, leaf
        self.pas, self.span, self.ctx, self.token, self.weight = e.pas, e.span, e.ctx, e.token, e.weight

    def __getattr__(self, name):
        return getattr(self._e, name)


def _is_off(b) -> bool:
    return b is None or (not isinstance(b, str) and b == math.inf)


class Trellis3R214(Trellis3R208):
    read_bits = None
    read_reach = None
    parse_bits = None

    # ------------------------------------------------------------------ levels
    def level_bits(self, node, root=None) -> float:
        root = root or self.ctx.root
        return -math.log2(max(node.count, 1e-9) / max(root.count, 1e-9))

    def lift(self, leaf, b):
        """The deepest ancestor of ``leaf`` (itself included) at level <= b bits; "basic": the category-utility maximum."""
        if leaf is None or _is_off(b):
            return leaf
        root = self.ctx.root
        key = (leaf.id, b)
        memo = self.__dict__.setdefault("_r214_lift", {})
        stamp = (len(self.records), self.stats.get("experiences", 0), root.count)
        if self.__dict__.get("_r214_lift_stamp") != stamp:
            memo.clear()
            self.__dict__["_r214_lift_stamp"] = stamp
        if key in memo:
            return memo[key]
        path = []
        n = leaf
        while n is not None:
            path.append(n)
            n = n.parent
        path.reverse()                                   # root ... leaf
        if b == "basic":
            inner = path[1:-1] if len(path) > 1 and not path[-1].children else path[1:]
            if inner:
                base = root.sstot / max(root.count, 1e-9) ** 2

                def cu(m):
                    return (m.count / max(root.count, 1e-9)) * (m.sstot / max(m.count, 1e-9) ** 2 - base)
                out = max(inner, key=cu)
            else:
                out = root
        else:
            out = root
            for m in path:
                if self.level_bits(m, root) <= b + 1e-12:
                    out = m
                else:
                    break                                # levels grow down the path
        memo[key] = out
        return out

    def word_classes(self, b) -> Dict[str, int]:
        """The modal C_b concept id per word type, over the committed word occurrences (weighted)."""
        cnt: Dict[str, Dict[int, float]] = {}
        for e in self.records:
            if e.leaf is None:
                continue
            for which, attr, value, w in e.pas:
                if which == "ctx" and attr == FILL and value[0] == "w":
                    c = self.lift(e.leaf, b)
                    d = cnt.setdefault(value[1], {})
                    d[c.id] = d.get(c.id, 0.0) + w
        return {w: max(d.items(), key=lambda kv: (kv[1], -kv[0]))[0] for w, d in cnt.items()}

    # ------------------------------------------------------------------ the read
    def _reach_query(self, q: dict) -> dict:
        W = self.read_reach
        if W is None:
            return q
        drop = [a for a in q if a[0] == "l" and a[1:].isdigit() and int(a[1:]) > W]
        return {a: v for a, v in q.items() if a not in drop} if drop else q

    def _named_query(self, q: dict) -> dict:
        """With window_names the stored contexts hold learned names in the window slots, so the query's words are named the
        same way (Trellis3._named_window); without window_names the query is unchanged."""
        if not self.window_names:
            return q
        win = {a: v for a, v in q.items() if a[0] == "l" and a[1:].isdigit()}
        return {**q, **self._named_window(win)} if win else q

    def _gap_fills(self, q: dict, beta: float = 1.0, nbq: Optional[Dict[int, dict]] = None,
                   gamma: float = 1.0, backoff: str = "tokens") -> Dict[tuple, float]:
        q = self._named_query(self._reach_query(q))
        b = self.read_bits
        if _is_off(b):
            return super()._gap_fills(q, beta, nbq, gamma, backoff)
        stamp = (len(self.records), self.stats.get("experiences", 0))
        cache = self.__dict__.setdefault("_r214_cache", {})
        if self.__dict__.get("_r214_stamp") != stamp:
            cache.clear()
            self.__dict__["_r214_stamp"] = stamp
            self.__dict__.pop("_r214_prox", None)
        key = (b, tuple(sorted(q.items())), repr(nbq), beta, gamma, backoff, bool(self.edge_exact))
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        prox = self.__dict__.setdefault("_r214_prox", {})
        if b not in prox:
            prox[b] = [_Proxy(e, self.lift(e.leaf, b)) if e.leaf is not None else e for e in self.records]
        names = ("_gap_cache", "_edge_cache", "_r208_cache")
        saved_recs = self.records
        saved = {k: self.__dict__.get(k) for k in names}
        self.records = prox[b]
        for k in names:
            self.__dict__[k] = {}
        try:
            out = super()._gap_fills(q, beta, nbq, gamma, backoff)
        finally:
            self.records = saved_recs
            for k in names:
                if saved[k] is None:
                    self.__dict__.pop(k, None)
                else:
                    self.__dict__[k] = saved[k]
        cache[key] = out
        return out

    # ------------------------------------------------------------------ the parse
    def _levels_ok(self, tree, nodes):
        b, root = self.parse_bits, tree.root
        return [n for n in nodes if n.count > 0 and -math.log2(n.count / root.count) <= b + 1e-12]

    def _code(self, tree, x: dict) -> float:
        if _is_off(self.parse_bits):
            return super()._code(tree, x)
        root = tree.root
        if root.count <= 0:
            return 0.0
        best = -math.inf
        for node in self._levels_ok(tree, tree.categorize(x)):
            best = max(best, (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha)) / LN2)
        return best

    def _gain(self, tree, x: dict) -> float:
        if _is_off(self.parse_bits):
            return super()._gain(tree, x)
        assert self.gain_alpha is None, "parse_bits does not support gain_alpha"
        root = tree.root
        if root.count <= 0 or self.parse_bits <= 0:
            return 0.0
        base = log_prob(root, x, root, self.alpha)
        best = 0.0
        for node in self._levels_ok(tree, tree.categorize(x)[1:]):
            g = (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - base) / LN2
            best = max(best, g)
        return best
