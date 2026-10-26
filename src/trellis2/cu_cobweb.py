"""TRELLIS v2 on cobweb-private's compiled Fisher Cobweb (``cobweb_cu``).

The same interface as ``incremental_cobweb.IncrementalCobwebTree`` (``ifit``,
``unfit``, ``relabel``, ``categorize``, passive counts, hidden values,
``ancestors``, ``nodes``, ``leaves``, ``root``, ``restructured``,
``sample_restructure``; nodes with ``id``, ``count``, ``children``, ``parent``,
``av``, ``pav``, ``ss``, ``sstot``) over the C++ tree of branch
``trellis-discrete`` (module ``cobweb_cu_trellis``: ``cobweb_cu`` under its own
name and nanobind domain).

With ``fast=False`` the trees are identical to the Python tree's with
``fast_ec = False`` (same nodes, ids, links, counts and sums of squares, and
the same random tie-breaking); with ``fast=True`` (the default) the
instance-node scores use the fast path of ``IncrementalCobwebTree._ec_with``
(as the Python tree with its default ``fast_ec = True``) and ``unfit`` updates
sums of squares by each count's change instead of recomputing them with fsum:
the same algorithm, scores equal up to rounding (so trees can drift apart
from the Python ones after a near-tie). ``partial`` mode is not supported.

Attribute names and values are encoded in C++ (Python dicts, so values compare
as dict keys do). A node has one wrapper per id (nodes are never freed), and a
node's decoded ``av``/``pav`` tables are cached until its counts change (a
generation counter per node). The tables returned are the cache: read only.
The compiled module is imported from ``$TRELLIS_CWP_BUILD`` (default: the
``build-trellis`` directory of the ``cobweb-private-trellis`` worktree).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Hashable, Iterator, List, Sequence

_BUILD = os.environ.get("TRELLIS_CWP_BUILD", "/coc/flash10/ksingara3/cobweb-private-trellis/build-trellis")
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
import cobweb_cu_trellis as _cu  # noqa: E402

EPS = 1e-9


class _Counts:
    """Read-only view of a node's value counts per attribute, decoded per
    attribute on first read (valid until the node's counts change)."""
    __slots__ = ("c", "i", "passive", "d")

    def __init__(self, c, i: int, passive: bool):
        self.c, self.i, self.passive, self.d = c, i, passive, {}

    def get(self, a, default=None):
        d = self.d
        try:
            r = d[a]
        except KeyError:
            r = d[a] = (self.c.pav_attr if self.passive else self.c.av_attr)(self.i, a)
        return default if r is None else r

    def __getitem__(self, a):
        r = self.get(a)
        if r is None:
            raise KeyError(a)
        return r

    def __contains__(self, a):
        return self.get(a) is not None

    def keys(self):
        return (self.c.pav_keys if self.passive else self.c.av_keys)(self.i)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def items(self):
        return [(a, self[a]) for a in self.keys()]

    def values(self):
        return [self[a] for a in self.keys()]


class CUNode:
    """A wrapper with a stable identity per compiled node (one per id)."""
    __slots__ = ("id", "t", "c", "_g", "_av", "_pav", "hidden", "__weakref__")

    def __init__(self, t: "CUCobwebTree", i: int):
        self.id, self.t, self.c = i, t, t.tree
        self._g = -1
        self._av = self._pav = None
        # hidden values (attr -> value -> weight), kept at the leaf as in the Python tree
        self.hidden: Dict[str, Dict[Hashable, float]] = {}

    def _fresh(self) -> None:
        g = self.c.gen_of(self.id)
        if g != self._g:
            self._g = g
            self._av = _Counts(self.c, self.id, False)
            self._pav = _Counts(self.c, self.id, True)

    @property
    def count(self) -> float:
        return self.c.count_of(self.id)

    @property
    def children(self) -> List["CUNode"]:
        return self.t._nodes(self.c.children_of(self.id))

    @property
    def parent(self):
        p = self.c.parent_of(self.id)
        return None if p < 0 else self.t._node(p)

    @property
    def av(self) -> _Counts:
        self._fresh()
        return self._av

    @property
    def pav(self) -> _Counts:
        self._fresh()
        return self._pav

    @property
    def ss(self) -> Dict[str, float]:
        return self.c.ss_of(self.id)

    @property
    def sstot(self) -> float:
        return self.c.sstot_of(self.id)

    def __repr__(self):
        k = self.c.n_children(self.id)
        kind = "leaf" if not k else f"{k} children"
        return f"CUNode(id={self.id}, count={self.count:g}, {kind})"


class _Restructured:
    """The concepts whose children a merge or split changed since the last
    ``clear()`` (a set-like view of IncrementalCobwebTree.restructured)."""
    __slots__ = ("t",)

    def __init__(self, t: "CUCobwebTree"):
        self.t = t

    def __len__(self):
        return self.t.tree.n_restructured()

    def __bool__(self):
        return self.t.tree.n_restructured() > 0

    def __iter__(self):
        return iter(self.t._nodes(self.t.tree.restructured_ids()))

    def __contains__(self, node):
        return node in set(self)

    def clear(self) -> None:
        self.t.tree.clear_restructured()


class CUCobwebTree:
    def __init__(self, attrs: Sequence[str], seed: int = 0, fast: bool = True):
        self.attrs = tuple(attrs)
        self.n_attrs = len(self.attrs)
        self.tree = _cu.CobwebCUTree(list(self.attrs), seed)
        self.tree.fast = bool(fast)
        self._w: List[CUNode] = []
        self.restructured = _Restructured(self)

    # nodes ---------------------------------------------------------------- #
    def _node(self, i: int) -> CUNode:
        w = self._w
        if i >= len(w):
            w.extend(CUNode(self, k) for k in range(len(w), self.tree.n_nodes()))
        return w[i]

    def _nodes(self, ids) -> List[CUNode]:
        w = self._w
        if ids and max(ids) >= len(w):
            self._node(max(ids))
        return [w[i] for i in ids]

    # settings ------------------------------------------------------------- #
    @property
    def partial(self) -> bool:
        return False

    @partial.setter
    def partial(self, on: bool) -> None:
        if on:
            raise NotImplementedError("cobweb_cu has no partial-instance mode")

    @property
    def sample_restructure(self) -> bool:
        return self.tree.sample_restructure

    @sample_restructure.setter
    def sample_restructure(self, on: bool) -> None:
        self.tree.sample_restructure = bool(on)

    @property
    def fast(self) -> bool:
        return self.tree.fast

    @fast.setter
    def fast(self, on: bool) -> None:
        self.tree.fast = bool(on)

    fast_ec = fast

    @property
    def root(self) -> CUNode:
        return self._node(self.tree.root_id)

    # learning ------------------------------------------------------------- #
    def ifit(self, x: dict, w: float = 1.0) -> CUNode:
        """Incorporate ``x`` (weight ``w``); return the leaf storing it."""
        return self._node(self.tree.ifit_id(x, float(w)))

    def unfit(self, leaf: CUNode, x: dict, w: float = 1.0) -> None:
        """Remove ``x`` (weight ``w``), stored at ``leaf``, from the tree."""
        self.tree.unfit_id(leaf.id, x, float(w))

    def relabel(self, leaf: CUNode, old: dict, new: dict, w: float = 1.0) -> None:
        """Re-describe an instance in place (IncrementalCobwebTree.relabel):
        only the attributes that differ are touched."""
        if self.tree.n_children(leaf.id):
            raise ValueError("instances are stored at leaves")
        a_old = {a: v for a, v in old.items() if new.get(a) != v}
        a_new = {a: new[a] for a in a_old if a in new}
        a_new.update({a: v for a, v in new.items() if a not in old})
        if not a_old and not a_new:
            return
        self.tree.relabel_id(leaf.id, a_old, a_new, float(w))

    # passive and hidden values ------------------------------------------- #
    def add_passive(self, leaf: CUNode, attr: str, value: Hashable, w: float = 1.0) -> None:
        self.tree.add_passive_id(leaf.id, attr, value, float(w))

    @staticmethod
    def passive(node: CUNode, attr: str) -> Dict[Hashable, float]:
        return node.pav.get(attr, {})

    @staticmethod
    def hide(leaf: CUNode, hidden: Dict[str, Hashable], w: float = 1.0) -> None:
        for a, v in hidden.items():
            d = leaf.hidden.setdefault(a, {})
            n = d.get(v, 0.0) + w
            if abs(n) <= EPS * max(1.0, abs(w)):
                d.pop(v, None)
            else:
                d[v] = n

    def hidden_counts(self, node: CUNode, attr: str) -> Dict[Hashable, float]:
        out: Dict[Hashable, float] = {}
        stack = [node]
        while stack:
            n = stack.pop()
            kids = n.children
            if kids:
                stack.extend(kids)
            else:
                for v, m in n.hidden.get(attr, {}).items():
                    out[v] = out.get(v, 0.0) + m
        return out

    # reading -------------------------------------------------------------- #
    def ec(self, node: CUNode) -> float:
        return self.tree.ec_of(node.id)

    def categorize(self, x: dict, w: float = 1.0) -> List[CUNode]:
        """The path (root first) along which ``x`` would be sorted, without
        changing the tree (ties to the larger child, then the lower id)."""
        return self._nodes(self.tree.categorize_ids(x, float(w)))

    def ancestors(self, node: CUNode) -> List[CUNode]:
        return self._nodes(self.tree.ancestors_ids(node.id))

    def nodes(self) -> Iterator[CUNode]:
        return iter(self._nodes(self.tree.nodes_ids()))

    def leaves(self) -> List[CUNode]:
        return self._nodes(self.tree.leaves_ids())

    def concept_codes(self, alpha: float) -> List[float]:
        """As IncrementalCobwebTree.concept_codes (same arithmetic)."""
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
        self.tree.check_invariants(tol)
