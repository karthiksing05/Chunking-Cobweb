"""A small discrete Cobweb (Fisher, 1987) used by both TRELLIS v2 hierarchies.

Instances are dicts mapping every attribute of the tree to one hashable value
or to a bag (a dict of values to weights summing to one), and each instance
may carry a (possibly fractional) weight. A bag spreads the instance's unit of
evidence for that attribute over several values, so every attribute's counts
still sum to the node's count and category utility applies unchanged. Learning follows
the classic algorithm and its four operators (best host, new child, merge,
split), scored with Fisher's category utility, mirroring the reference
implementation in ``concept_formation``.

Leaves are never deleted or moved by restructuring: a fringe split inserts a
new parent above the old leaf, merges add internal nodes, and splits remove
internal nodes only. A leaf is therefore a stable handle for the instances
stored at it, which is what lets the grammar read-out aggregate records
through any cut of the tree.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, Hashable, Iterator, List, Optional, Sequence

Instance = Dict[str, Hashable]


class CobwebNode:
    """A concept: weighted attribute-value counts plus taxonomy links."""

    __slots__ = ("id", "parent", "children", "count", "av", "ss", "sstot")

    def __init__(self, nid: int):
        self.id = nid
        self.parent: Optional[CobwebNode] = None
        self.children: List[CobwebNode] = []
        self.count = 0.0
        self.av: Dict[str, Dict[Hashable, float]] = {}
        # Per attribute, the sum of squared value counts, and their total
        # over attributes: expected correct guesses in O(1).
        self.ss: Dict[str, float] = {}
        self.sstot = 0.0

    def __repr__(self):
        kind = "leaf" if not self.children else f"{len(self.children)} children"
        return f"CobwebNode(id={self.id}, count={self.count:g}, {kind})"

    def increment(self, x: Instance, w: float) -> None:
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

    def absorb(self, other: "CobwebNode") -> None:
        """Add another node's counts to this one."""
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


class CobwebTree:
    """Incremental concept formation over nominal attributes.

    Parameters
    ----------
    attrs : the attribute names every instance must define.
    seed  : seeds the random tie-breaking between equally good operators.
    """

    def __init__(self, attrs: Sequence[str], seed: int = 0):
        self.attrs = tuple(attrs)
        self.n_attrs = len(self.attrs)
        self._ids = itertools.count()
        self.rng = random.Random(seed)
        self.root = self._new_node()

    # ------------------------------------------------------------------ #
    # Category utility pieces
    # ------------------------------------------------------------------ #
    def _new_node(self) -> CobwebNode:
        return CobwebNode(next(self._ids))

    def ec(self, node: CobwebNode) -> float:
        """Expected correct guesses, averaged over attributes."""
        if node.count <= 0.0:
            return 0.0
        return node.sstot / (node.count * node.count) / self.n_attrs

    def _ec_with(self, node: CobwebNode, x: Instance, w: float) -> float:
        """Expected correct guesses of ``node`` if ``x`` were added to it."""
        n = node.count + w
        tot = node.sstot                 # every instance defines every attribute
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

    def _ec_merge(self, b1: CobwebNode, b2: CobwebNode,
                  x: Instance, w: float) -> float:
        """Expected correct guesses of the union of b1, b2 and x."""
        n = b1.count + b2.count + w
        tot = b1.sstot + b2.sstot
        for a in self.attrs:
            d1 = b1.av.get(a, {})
            d2 = b2.av.get(a, {})
            small, big = (d1, d2) if len(d1) <= len(d2) else (d2, d1)
            tot += 2.0 * sum(m * big.get(v, 0.0) for v, m in small.items())
            v = x[a]
            for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                mx = d1.get(u, 0.0) + d2.get(u, 0.0)
                tot += 2.0 * mx * w * p + (w * p) ** 2
        return tot / (n * n) / self.n_attrs

    def _best_operation(self, cur: CobwebNode, x: Instance, w: float):
        """Score the four operators at ``cur``; return (op, best1, best2).

        CU(partition) = (1/K) [ sum_k P(C_k) EC(C_k) - EC(parent) ].
        As in concept_formation, best/new/merge are scored with the instance
        added and split is scored on the current counts.
        """
        kids = cur.children
        K = len(kids)
        n1 = cur.count + w
        ecp1 = self._ec_with(cur, x, w)
        terms = [c.count * self.ec(c) for c in kids]
        S = sum(terms)

        ranked = []
        for c, t in zip(kids, terms):
            gain = (c.count + w) * self._ec_with(c, x, w) - t
            ranked.append((gain, c.count, self.rng.random(), c))
        ranked.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        best1 = ranked[0][3]
        best2 = ranked[1][3] if K > 1 else None

        ops = [(((S + ranked[0][0]) / n1 - ecp1) / K, self.rng.random(), "best"),
               (((S + w) / n1 - ecp1) / (K + 1), self.rng.random(), "new")]
        if K > 2 and best2 is not None:
            t1 = best1.count * self.ec(best1)
            t2 = best2.count * self.ec(best2)
            nm = best1.count + best2.count + w
            sm = S - t1 - t2 + nm * self._ec_merge(best1, best2, x, w)
            ops.append(((sm / n1 - ecp1) / (K - 1), self.rng.random(), "merge"))
        if best1.children:
            ss = (S - best1.count * self.ec(best1)
                  + sum(c.count * self.ec(c) for c in best1.children))
            ks = K - 1 + len(best1.children)
            ops.append(((ss / cur.count - self.ec(cur)) / ks,
                        self.rng.random(), "split"))
        op = max(ops, key=lambda o: (o[0], o[1]))[2]
        return op, best1, best2

    # ------------------------------------------------------------------ #
    # Structural operators
    # ------------------------------------------------------------------ #
    def _is_exact_match(self, leaf: CobwebNode, x: Instance) -> bool:
        for a, v in x.items():
            d = leaf.av.get(a)
            if d is None:
                return False
            if isinstance(v, dict):
                if len(d) != len(v) or any(abs(d.get(u, 0.0) - p * leaf.count) > 1e-9
                                           for u, p in v.items()):
                    return False
            elif len(d) != 1 or v not in d:
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

    def _merge(self, cur: CobwebNode, b1: CobwebNode, b2: CobwebNode) -> CobwebNode:
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
        return m

    def _split(self, cur: CobwebNode, best: CobwebNode) -> None:
        cur.children.remove(best)
        for c in best.children:
            c.parent = cur
            cur.children.append(c)
        best.children = []
        best.parent = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ifit(self, x: Instance, w: float = 1.0) -> CobwebNode:
        """Incorporate ``x`` (with weight ``w``) and return the leaf storing it."""
        if len(x) != self.n_attrs or any(a not in x for a in self.attrs):
            raise ValueError(f"instance must define exactly {self.attrs}; got {sorted(x)}")
        if w <= 0.0:
            raise ValueError("instance weight must be positive")
        cur = self.root
        while True:
            if not cur.children:
                if cur.count == 0.0 or self._is_exact_match(cur, x):
                    cur.increment(x, w)
                    return cur
                # Fringe split: a new parent takes the leaf's place, so the
                # old leaf stays a leaf (stable handle for its instances).
                p = self._new_node()
                p.absorb(cur)
                self._replace(cur, p)
                p.children = [cur]
                cur.parent = p
                p.increment(x, w)
                return self._new_child(p, x, w)
            op, best1, best2 = self._best_operation(cur, x, w)
            if op == "best":
                cur.increment(x, w)
                cur = best1
            elif op == "new":
                cur.increment(x, w)
                return self._new_child(cur, x, w)
            elif op == "merge":
                cur.increment(x, w)
                cur = self._merge(cur, best1, best2)
            else:  # split, then reconsider the same node
                self._split(cur, best1)

    def nodes(self) -> Iterator[CobwebNode]:
        """All nodes in depth-first pre-order."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> List[CobwebNode]:
        return [n for n in self.nodes() if not n.children]

    def ancestors(self, node: CobwebNode) -> List[CobwebNode]:
        """``node`` and its ancestors, from the node up to the root."""
        out = []
        while node is not None:
            out.append(node)
            node = node.parent
        return out

    def check_invariants(self, tol: float = 1e-6) -> None:
        """Raise AssertionError if counts are inconsistent (used in tests)."""
        for node in self.nodes():
            if node.children:
                total = sum(c.count for c in node.children)
                assert abs(total - node.count) < tol * max(1.0, node.count), \
                    f"{node}: children sum {total} != {node.count}"
                for c in node.children:
                    assert c.parent is node, f"{c} has wrong parent"
            for a, d in node.av.items():
                s = sum(d.values())
                assert abs(s - node.count) < tol * max(1.0, node.count), \
                    f"{node}: attr {a} sums to {s}"
                ss = sum(v * v for v in d.values())
                assert abs(ss - node.ss[a]) < 1e-6 * max(1.0, ss), \
                    f"{node}: stale sum of squares for {a}"
            total_ss = sum(node.ss.values())
            assert abs(total_ss - node.sstot) < 1e-6 * max(1.0, total_ss), \
                f"{node}: stale total sum of squares"
