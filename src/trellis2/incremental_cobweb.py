"""Cobweb with removal: the hierarchy kept over a changing memory.

TRELLIS v2 by night replays every element into a fresh hierarchy. A learner
without nights keeps its hierarchies: an element enters once (``ifit``), and
when its description changes (its chunk-context categories were merged or
split) or its experience is re-analysed, it is taken out (``unfit``) and
re-inserted. ``cobweb_cu`` only inserts, so this module is the pure-Python
Cobweb of ``tests/trellis2/reference_cobweb.py`` (which ``cobweb_cu``
reproduces exactly: ``ifit`` here gives the same nodes, links and counts),
with removal added:

* ``unfit(leaf, x, w)``: decrement ``x``'s counts (weight ``w``) on the path
  from the leaf that stores it to the root. Values whose count reaches zero
  are dropped (so a leaf left with one kind of instance is again an exact
  match for it); a leaf whose count reaches zero is removed; an internal
  node left with a single child is spliced out (its child takes its place),
  so every internal node keeps two or more children, as insertion makes
  them. Leaves are never moved or merged by removal, so a leaf stays a
  stable handle for the instances stored at it.
* Sums of squares of the touched attributes are recomputed from the counts
  (not decremented), so category utility after a removal is computed from
  exactly the counts that remain.

Removal does not undo the restructuring an insertion may have done (merges
and splits of concepts): the counts are restored, the partition structure
is what Cobweb built. ``concept_codes`` gives each node's code length as
``cobweb_cu`` does (used by the evidence cut).
"""
from __future__ import annotations

import math
import random
from typing import Dict, Hashable, Iterator, List, Optional, Sequence

Instance = Dict[str, Hashable]
EPS = 1e-9


class CobwebNode:
    """A concept: weighted attribute-value counts plus taxonomy links."""

    __slots__ = ("id", "parent", "children", "count", "av", "ss", "sstot", "hidden", "pav", "na")

    def __init__(self, nid: int):
        self.id = nid
        self.parent: Optional[CobwebNode] = None
        self.children: List[CobwebNode] = []
        self.count = 0.0
        self.av: Dict[str, Dict[Hashable, float]] = {}
        self.ss: Dict[str, float] = {}
        self.sstot = 0.0
        # Hidden values (kept at the leaf that stores the instance; never used
        # for category utility, sorting or exact matching): attr -> value -> weight.
        self.hidden: Dict[str, Dict[Hashable, float]] = {}
        # Passive counts: aggregated over the subtree like ``av`` (every node
        # holds the sum of its children's), never used for category utility,
        # sorting or exact matching: attr -> value -> weight.
        self.pav: Dict[str, Dict[Hashable, float]] = {}
        # observed weight per attribute (instances that define it); equals
        # count for an attribute every instance defines
        self.na: Dict[str, float] = {}

    def __repr__(self):
        kind = "leaf" if not self.children else f"{len(self.children)} children"
        return f"CobwebNode(id={self.id}, count={self.count:g}, {kind})"

    def increment(self, x: Instance, w: float) -> None:
        for a in x:
            self.na[a] = self.na.get(a, 0.0) + w
        for a, v in x.items():
            d = self.av.get(a)
            if d is None:
                d = self.av[a] = {}
                self.ss[a] = 0.0
            for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                n = d.get(u, 0.0)
                d[u] = n + w * p
                inc = 2.0 * n * w * p + (w * p) ** 2
                self.ss[a] += inc
                self.sstot += inc
        self.count += w

    def decrement(self, x: Instance, w: float) -> None:
        """Remove ``x`` (weight ``w``) from the counts; values that reach
        zero are dropped and the sums of squares recomputed."""
        for a in x:
            self.na[a] = self.na.get(a, 0.0) - w
        for a, v in x.items():
            d = self.av[a]
            for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                n = d[u] - w * p
                if abs(n) <= EPS * max(1.0, w):
                    del d[u]
                else:
                    d[u] = n
            ss = math.fsum(m * m for m in d.values())
            self.sstot += ss - self.ss[a]
            self.ss[a] = ss
        self.count -= w
        if abs(self.count) <= EPS * max(1.0, w):
            self.count = 0.0
            self.sstot = 0.0
            for a in self.ss:
                self.ss[a] = 0.0
        else:
            self.sstot = math.fsum(self.ss.values())

    def absorb(self, other: "CobwebNode") -> None:
        for a, m in other.na.items():
            self.na[a] = self.na.get(a, 0.0) + m
        for a, od in other.pav.items():
            d = self.pav.setdefault(a, {})
            for v, m in od.items():
                d[v] = d.get(v, 0.0) + m
        for a, od in other.av.items():
            d = self.av.get(a)
            if d is None:
                d = self.av[a] = {}
                self.ss[a] = 0.0
            ss = self.ss[a]
            for v, m in od.items():
                n = d.get(v, 0.0)
                d[v] = n + m
                ss += 2.0 * n * m + m * m
            self.sstot += ss - self.ss[a]
            self.ss[a] = ss
        self.count += other.count


class IncrementalCobwebTree:
    """Cobweb (Fisher 1987) over nominal and bag-valued attributes, with
    insertion (``ifit``, identical to the reference) and removal (``unfit``)."""

    # fast_ec: score an instance against a node with the fast path of _ec_with
    # (same value up to rounding); False gives the reference arithmetic, which
    # cobweb_cu (cu_cobweb.CUCobwebTree with fast=False) reproduces bit for bit
    fast_ec = True

    def __init__(self, attrs: Sequence[str], seed: int = 0):
        self.attrs = tuple(attrs)
        self.n_attrs = len(self.attrs)
        # partial instances: an attribute an instance does not observe is left
        # out (it adds no value and so no shared similarity), as Cobweb treats
        # a missing value
        self.partial = False
        # sample_restructure: an instance of weight w (a posterior probability that
        # it occurs) may merge or split concepts only with probability w, so the
        # operators that change structure act, in expectation, as real
        # observations do; its counts stay fractional
        self.sample_restructure = False
        # concepts whose children a merge or split changed since the caller last
        # cleared this set (the partitions under them may now sort differently)
        self.restructured: set = set()
        self._next_id = 0
        self.rng = random.Random(seed)
        self.root = self._new_node()

    # Category utility (as the reference) -------------------------------- #
    def _new_node(self) -> CobwebNode:
        self._next_id += 1
        return CobwebNode(self._next_id - 1)

    def ec(self, node: CobwebNode) -> float:
        if node.count <= 0.0:
            return 0.0
        if self.partial:
            # each attribute over the instances that observe it, weighted by
            # how often it is observed: sum_a P(a observed) sum_v P(v | a)^2
            return sum(ss / (node.count * node.na[a]) for a, ss in node.ss.items()
                       if node.na.get(a, 0.0) > 0) / self.n_attrs
        return node.sstot / (node.count * node.count) / self.n_attrs

    @staticmethod
    def _added(d, v, w):
        """The change in an attribute's sum of squares when value(s) v of weight w are added."""
        inc = 0.0
        for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
            m = d.get(u, 0.0) if d else 0.0
            inc += 2.0 * m * w * p + (w * p) ** 2
        return inc

    @staticmethod
    def _sq(x: Instance, w: float) -> float:
        """sum_a sum_v (w p)^2: the part of _ec_with that does not depend on the node."""
        s = 0.0
        for v in x.values():
            if isinstance(v, dict):
                for p in v.values():
                    s += (w * p) ** 2
            else:
                s += w * w
        return s

    def _ec_with(self, node: CobwebNode, x: Instance, w: float, sq: Optional[float] = None) -> float:
        if sq is not None and not self.partial:
            # same value as below: sstot + 2w <counts, x> + sq, the dot product
            # over the smaller of the node's and the instance's value sets
            dot = 0.0
            for a, v in x.items():
                d = node.av.get(a)
                if not d:
                    continue
                if isinstance(v, dict):
                    if len(d) < len(v):
                        for u, m in d.items():
                            p = v.get(u)
                            if p:
                                dot += m * p
                    else:
                        for u, p in v.items():
                            m = d.get(u)
                            if m:
                                dot += m * p
                else:
                    dot += d.get(v, 0.0)
            n = node.count + w
            return (node.sstot + 2.0 * w * dot + sq) / (n * n) / self.n_attrs
        if self.partial:
            n = node.count + w
            tot = 0.0
            for a in set(node.ss) | set(x):
                ss = node.ss.get(a, 0.0); na = node.na.get(a, 0.0)
                if a in x:
                    ss += self._added(node.av.get(a), x[a], w); na += w
                if na > 0:
                    tot += ss / (n * na)
            return tot / self.n_attrs
        n = node.count + w
        tot = node.sstot
        for a, v in x.items():
            d = node.av.get(a)
            if isinstance(v, dict):
                for u, p in v.items():
                    m = d.get(u, 0.0) if d else 0.0
                    tot += 2.0 * m * w * p + (w * p) ** 2
            else:
                m = d.get(v, 0.0) if d else 0.0
                tot += 2.0 * m * w + w * w
        return tot / (n * n) / self.n_attrs

    def _ec_merge(self, b1: CobwebNode, b2: CobwebNode, x: Instance, w: float) -> float:
        if self.partial:
            n = b1.count + b2.count + w
            tot = 0.0
            for a in self.attrs:
                d1 = b1.av.get(a, {}); d2 = b2.av.get(a, {})
                small, big = (d1, d2) if len(d1) <= len(d2) else (d2, d1)
                ss = b1.ss.get(a, 0.0) + b2.ss.get(a, 0.0) + 2.0 * sum(m * big.get(v, 0.0) for v, m in small.items())
                na = b1.na.get(a, 0.0) + b2.na.get(a, 0.0)
                if a in x:
                    merged = dict(d1)
                    for v, m in d2.items():
                        merged[v] = merged.get(v, 0.0) + m
                    ss += self._added(merged, x[a], w); na += w
                if na > 0:
                    tot += ss / (n * na)
            return tot / self.n_attrs
        n = b1.count + b2.count + w
        tot = b1.sstot + b2.sstot
        for a in self.attrs:
            d1 = b1.av.get(a, {})
            d2 = b2.av.get(a, {})
            small, big = (d1, d2) if len(d1) <= len(d2) else (d2, d1)
            tot += 2.0 * sum(m * big.get(v, 0.0) for v, m in small.items())
            if a not in x:
                continue
            v = x[a]
            for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                mx = d1.get(u, 0.0) + d2.get(u, 0.0)
                tot += 2.0 * mx * w * p + (w * p) ** 2
        return tot / (n * n) / self.n_attrs

    def _ec_alone(self, x: Instance) -> float:
        """The expected correct guesses of a concept holding x alone (1 for a
        full instance of single values, as the reference has it)."""
        if not self.partial:
            return 1.0
        return sum(sum(p * p for p in v.values()) if isinstance(v, dict) else 1.0 for v in x.values()) / self.n_attrs

    def _best_operation(self, cur: CobwebNode, x: Instance, w: float, restructure: bool = True):
        kids = cur.children
        K = len(kids)
        n1 = cur.count + w
        sq = self._sq(x, w) if self.fast_ec else None
        ecp1 = self._ec_with(cur, x, w, sq)
        terms = [c.count * self.ec(c) for c in kids]
        S = sum(terms)
        ranked = []
        for c, t in zip(kids, terms):
            gain = (c.count + w) * self._ec_with(c, x, w, sq) - t
            ranked.append((gain, c.count, self.rng.random(), c))
        ranked.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        best1 = ranked[0][3]
        best2 = ranked[1][3] if K > 1 else None
        ops = [(((S + ranked[0][0]) / n1 - ecp1) / K, self.rng.random(), "best"),
               (((S + w * self._ec_alone(x)) / n1 - ecp1) / (K + 1), self.rng.random(), "new")]
        if restructure and K > 2 and best2 is not None:
            t1 = best1.count * self.ec(best1)
            t2 = best2.count * self.ec(best2)
            nm = best1.count + best2.count + w
            sm = S - t1 - t2 + nm * self._ec_merge(best1, best2, x, w)
            ops.append(((sm / n1 - ecp1) / (K - 1), self.rng.random(), "merge"))
        if restructure and best1.children:
            ss = (S - best1.count * self.ec(best1)
                  + sum(c.count * self.ec(c) for c in best1.children))
            ks = K - 1 + len(best1.children)
            ops.append(((ss / cur.count - self.ec(cur)) / ks, self.rng.random(), "split"))
        op = max(ops, key=lambda o: (o[0], o[1]))[2]
        return op, best1, best2

    # Structure ---------------------------------------------------------- #
    def _is_exact_match(self, leaf: CobwebNode, x: Instance) -> bool:
        if any(d and a not in x for a, d in leaf.av.items()):
            return False
        for a, v in x.items():
            d = leaf.av.get(a)
            if d is None:
                return False
            if isinstance(v, dict):
                if len(d) != len(v) or any(abs(d.get(u, 0.0) - p * leaf.count) > 1e-9
                                           for u, p in v.items()):
                    return False
            elif len(d) != 1 or v not in d or abs(d[v] - leaf.count) > 1e-9 * max(1.0, leaf.count):
                return False
        return True

    def _new_child(self, parent: CobwebNode, x: Instance, w: float) -> CobwebNode:
        node = self._new_node()
        node.increment(x, w)
        node.parent = parent
        parent.children.append(node)
        return node

    def _replace(self, old: CobwebNode, new: CobwebNode) -> None:
        new.parent = old.parent
        if old.parent is None:
            self.root = new
        else:
            sib = old.parent.children
            sib[sib.index(old)] = new

    def _merge(self, cur, b1, b2) -> CobwebNode:
        m = self._new_node()
        m.absorb(b1)
        m.absorb(b2)
        m.parent = cur
        m.children = [b1, b2]
        b1.parent = m
        b2.parent = m
        cur.children.remove(b1)
        cur.children.remove(b2)
        cur.children.append(m)
        self.restructured.add(m)
        return m

    def _split(self, cur, best) -> None:
        cur.children.remove(best)
        for c in best.children:
            c.parent = cur
            cur.children.append(c)
        best.children = []
        best.parent = None
        self.restructured.add(cur)

    # Public API --------------------------------------------------------- #
    def ifit(self, x: Instance, w: float = 1.0) -> CobwebNode:
        """Incorporate ``x`` (weight ``w``); return the leaf storing it."""
        if self.partial:
            if any(a not in self.attrs for a in x):
                raise ValueError(f"instance attributes must come from {self.attrs}; got {sorted(x)}")
        elif len(x) != self.n_attrs or any(a not in x for a in self.attrs):
            raise ValueError(f"instance must define exactly {self.attrs}; got {sorted(x)}")
        if w <= 0.0:
            raise ValueError("instance weight must be positive")
        restructure = not self.sample_restructure or self.rng.random() < w
        cur = self.root
        while True:
            if not cur.children:
                if cur.count == 0.0 or self._is_exact_match(cur, x):
                    cur.increment(x, w)
                    return cur
                p = self._new_node()
                p.absorb(cur)
                self._replace(cur, p)
                p.children = [cur]
                cur.parent = p
                p.increment(x, w)
                return self._new_child(p, x, w)
            op, best1, best2 = self._best_operation(cur, x, w, restructure)
            if op == "best":
                cur.increment(x, w)
                cur = best1
            elif op == "new":
                cur.increment(x, w)
                return self._new_child(cur, x, w)
            elif op == "merge":
                cur.increment(x, w)
                cur = self._merge(cur, best1, best2)
            else:
                self._split(cur, best1)

    # Hidden values -------------------------------------------------------- #
    @staticmethod
    def hide(leaf: CobwebNode, hidden: Dict[str, Hashable], w: float = 1.0) -> None:
        """Record hidden values of an instance stored at ``leaf`` (or remove
        them with a negative ``w``)."""
        for a, v in hidden.items():
            d = leaf.hidden.setdefault(a, {})
            n = d.get(v, 0.0) + w
            if abs(n) <= EPS * max(1.0, abs(w)):
                d.pop(v, None)
            else:
                d[v] = n

    def add_passive(self, leaf: CobwebNode, attr: str, value: Hashable, w: float = 1.0) -> None:
        """Add (or with a negative ``w`` remove) a passive count at ``leaf`` and
        every concept above it."""
        node = leaf
        while node is not None:
            d = node.pav.setdefault(attr, {})
            n = d.get(value, 0.0) + w
            if abs(n) <= EPS * max(1.0, abs(w)):
                d.pop(value, None)
            else:
                d[value] = n
            node = node.parent

    @staticmethod
    def passive(node: CobwebNode, attr: str) -> Dict[Hashable, float]:
        return node.pav.get(attr, {})

    def hidden_counts(self, node: CobwebNode, attr: str) -> Dict[Hashable, float]:
        """A hidden attribute's value counts over every leaf under ``node``:
        what the concept holds, inspected without having shaped it."""
        out: Dict[Hashable, float] = {}
        stack = [node]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children)
            else:
                for v, m in n.hidden.get(attr, {}).items():
                    out[v] = out.get(v, 0.0) + m
        return out

    def relabel(self, leaf: CobwebNode, old: Instance, new: Instance, w: float = 1.0) -> None:
        """Re-describe an instance in place: its stored description ``old``
        becomes ``new`` on the path from its leaf to the root, the instance
        staying where it is (as a stored leaf reference is re-read, not moved).
        Only the attributes that differ are touched."""
        if leaf.children:
            raise ValueError("instances are stored at leaves")
        a_old = {a: v for a, v in old.items() if new.get(a) != v}
        a_new = {a: new[a] for a in a_old if a in new}
        a_new.update({a: v for a, v in new.items() if a not in old})
        if not a_old and not a_new:
            return
        node = leaf
        while node is not None:
            for x, sign in ((a_old, -1.0), (a_new, 1.0)):
                for a in x:
                    node.na[a] = node.na.get(a, 0.0) + sign * w
                for a, v in x.items():
                    d = node.av.get(a)
                    if d is None:
                        d = node.av[a] = {}
                        node.ss[a] = 0.0
                    ss = node.ss[a]
                    for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                        m = d.get(u, 0.0)
                        n = m + sign * w * p
                        if abs(n) <= EPS * max(1.0, w):
                            d.pop(u, None)
                            n = 0.0
                        else:
                            d[u] = n
                        ss += n * n - m * m
                    node.sstot += ss - node.ss[a]
                    node.ss[a] = ss
            node = node.parent

    def unfit(self, leaf: CobwebNode, x: Instance, w: float = 1.0) -> None:
        """Remove ``x`` (weight ``w``), stored at ``leaf``, from the tree."""
        if leaf.children:
            raise ValueError("instances are stored at leaves")
        if leaf.count < w - EPS * max(1.0, w):
            raise ValueError("removing more weight than the leaf holds")
        node = leaf
        while node is not None:
            node.decrement(x, w)
            node = node.parent
        if leaf.count > 0.0:
            return
        parent = leaf.parent
        if parent is None:                     # the tree is empty again
            self.root = self._new_node()
            return
        parent.children.remove(leaf)
        leaf.parent = None
        if len(parent.children) == 1:          # splice the parent out
            child = parent.children[0]
            self._replace(parent, child)
            parent.children = []
            parent.parent = None

    def categorize(self, x: Instance, w: float = 1.0) -> List[CobwebNode]:
        """The path (root first) along which ``x`` would be sorted, without
        changing the tree: at each node the child that would best host it
        (the same gain as insertion; ties to the larger child, then the
        lower id)."""
        node, path = self.root, [self.root]
        sq = self._sq(x, w) if self.fast_ec else None
        while node.children:
            best = max(node.children, key=lambda c: ((c.count + w) * self._ec_with(c, x, w, sq)
                                                     - c.count * self.ec(c), c.count, -c.id))
            path.append(best)
            node = best
        return path

    def nodes(self) -> Iterator[CobwebNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> List[CobwebNode]:
        return [n for n in self.nodes() if not n.children]

    def ancestors(self, node: CobwebNode) -> List[CobwebNode]:
        out = []
        while node is not None:
            out.append(node)
            node = node.parent
        return out

    def concept_codes(self, alpha: float) -> List[float]:
        """Per node (pre-order), the code length (nats) of its attribute
        counts: per attribute a Dirichlet-multinomial over the values the root
        has seen, concentration ``alpha`` (as ``cobweb_cu``)."""
        n_values = {a: max(len(d), 1) for a, d in self.root.av.items()}
        out = []
        for node in self.nodes():
            tot = 0.0
            for a, d in node.av.items():
                A = n_values[a] * alpha
                tot += math.lgamma(sum(d.values()) + A) - math.lgamma(A)
                tot -= sum(math.lgamma(c + alpha) - math.lgamma(alpha) for c in d.values())
            out.append(tot)
        return out

    def check_invariants(self, tol: float = 1e-6) -> None:
        """Counts consistent with the children and the attribute tables; every
        internal node has two or more children."""
        for node in self.nodes():
            if node.children:
                assert len(node.children) >= 2, f"{node}: internal node with a single child"
                total = sum(c.count for c in node.children)
                assert abs(total - node.count) < tol * max(1.0, node.count), \
                    f"{node}: children sum {total} != {node.count}"
                for c in node.children:
                    assert c.parent is node, f"{c} has wrong parent"
                for a in self.attrs:
                    d = node.av.get(a, {})
                    agg: Dict[Hashable, float] = {}
                    for c in node.children:
                        for v, m in c.av.get(a, {}).items():
                            agg[v] = agg.get(v, 0.0) + m
                    keys = set(d) | set(agg)
                    assert all(abs(d.get(v, 0.0) - agg.get(v, 0.0)) < tol * max(1.0, node.count)
                               for v in keys), f"{node}: attr {a} differs from its children's"
            for a, d in node.av.items():
                s = sum(d.values())
                assert node.count == 0 or abs(s - node.count) < tol * max(1.0, node.count), \
                    f"{node}: attr {a} sums to {s}"
                assert all(m > 0 for m in d.values()), f"{node}: non-positive count in {a}"
                ss = sum(v * v for v in d.values())
                assert abs(ss - node.ss[a]) < 1e-6 * max(1.0, ss), f"{node}: stale sum of squares for {a}"
            if node.children:
                for a in set(node.pav) | {a for c in node.children for a in c.pav}:
                    agg: Dict[Hashable, float] = {}
                    for c in node.children:
                        for v, m in c.pav.get(a, {}).items():
                            agg[v] = agg.get(v, 0.0) + m
                    mine = node.pav.get(a, {})
                    assert all(abs(mine.get(v, 0.0) - agg.get(v, 0.0)) < tol * max(1.0, node.count)
                               for v in set(mine) | set(agg)), f"{node}: passive {a} differs from its children's"
            total_ss = sum(node.ss.values())
            assert abs(total_ss - node.sstot) < 1e-6 * max(1.0, total_ss), f"{node}: stale total"
