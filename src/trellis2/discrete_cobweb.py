"""TRELLIS v2 on cobweb-private's ``CobwebDiscreteTree``.

The same interface as ``incremental_cobweb.IncrementalCobwebTree`` (``ifit``,
``unfit``, ``relabel``, ``categorize``, passive counts, ``ancestors``, ``nodes``,
``leaves``, ``root``, nodes with ``id``, ``count``, ``children``, ``parent``,
``av``, ``pav``), over the compiled discrete tree of branch ``trellis-discrete``
(weighted ``ifit``, ``unfit``, ``relabel``, ``add_passive``, sampled
restructuring, retained deleted nodes, stable ``uid``).

What differs from the Python tree is the learning criterion: category utility
from the discrete tree's smoothed entropy (``alpha`` pseudo-counts per value
the attribute has ever had) instead of Fisher's sum of squared probabilities.

Attributes and values are encoded as integers: active attributes as 0..n-1 in
the order given, passive (hidden) attributes as negative integers, values by a
table shared by all attributes of the tree. The compiled module is imported
from ``$TRELLIS_CWP_BUILD`` (default: the ``build-trellis`` directory of the
``cobweb-private-trellis`` worktree).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Hashable, Iterator, List, Sequence

_BUILD = os.environ.get("TRELLIS_CWP_BUILD", "/coc/flash10/ksingara3/cobweb-private-trellis/build-trellis")
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
import cobweb_discrete as _cd  # noqa: E402


class _Counts:
    """Read-only view of a node's value counts per attribute (decoded on read)."""
    __slots__ = ("t", "n", "passive")

    def __init__(self, t: "DiscreteCobwebTree", n, passive: bool):
        self.t, self.n, self.passive = t, n, passive

    def _aid(self, a):
        return (self.t._pid.get(a) if self.passive else self.t._aid.get(a))

    def get(self, a, default=None):
        k = self._aid(a)
        if k is None:
            return default
        d = self.n.av(k)
        if not d:
            return default
        dec = self.t._dec
        return {dec[v]: c for v, c in d.items()}

    def __getitem__(self, a):
        out = self.get(a)
        if out is None:
            raise KeyError(a)
        return out

    def __contains__(self, a):
        return self.get(a) is not None

    def keys(self):
        names = self.t._pnames if self.passive else self.t.attrs
        return [a for a in names if a in self]

    def items(self):
        return [(a, self[a]) for a in self.keys()]

    def values(self):
        return [self[a] for a in self.keys()]


class DNode:
    """A wrapper with a stable identity per compiled node (one per ``uid``)."""
    __slots__ = ("n", "t", "__weakref__")

    def __init__(self, t: "DiscreteCobwebTree", n):
        self.t, self.n = t, n

    @property
    def id(self) -> int:
        return self.n.uid

    @property
    def count(self) -> float:
        return self.n.count

    @property
    def children(self) -> List["DNode"]:
        return [self.t._wrap(c) for c in self.n.children]

    @property
    def parent(self):
        p = self.n.parent
        return None if p is None else self.t._wrap(p)

    @property
    def av(self) -> _Counts:
        return _Counts(self.t, self.n, False)

    @property
    def pav(self) -> _Counts:
        return _Counts(self.t, self.n, True)

    @property
    def deleted(self) -> bool:
        return self.n.deleted

    def __repr__(self):
        kind = "leaf" if not self.n.children else f"{len(self.n.children)} children"
        return f"DNode(id={self.id}, count={self.count:g}, {kind})"


class DiscreteCobwebTree:
    def __init__(self, attrs: Sequence[str], seed: int = 0, alpha: float = 1e-4):
        self.attrs = tuple(attrs)
        self.n_attrs = len(self.attrs)
        self._aid = {a: i for i, a in enumerate(self.attrs)}
        self._pid: Dict[str, int] = {}
        self._pnames: List[str] = []
        self._enc: Dict[Hashable, int] = {}
        self._dec: List[Hashable] = []
        self._nodes: Dict[int, DNode] = {}
        _cd.set_random_seed(seed)
        self.tree = _cd.CobwebDiscreteTree(alpha=alpha, weight_attr=False)
        self.tree.retain_deleted = True
        self.partial = False

    # encoding ------------------------------------------------------------- #
    def _v(self, v) -> int:
        k = self._enc.get(v)
        if k is None:
            k = self._enc[v] = len(self._dec)
            self._dec.append(v)
        return k

    def _p(self, a: str) -> int:
        k = self._pid.get(a)
        if k is None:
            k = self._pid[a] = -(len(self._pnames) + 1)
            self._pnames.append(a)
        return k

    def _x(self, x: dict) -> dict:
        out = {}
        for a, v in x.items():
            if isinstance(v, dict):
                out[self._aid[a]] = {self._v(u): float(p) for u, p in v.items()}
            else:
                out[self._aid[a]] = {self._v(v): 1.0}
        return out

    def _wrap(self, n) -> DNode:
        w = self._nodes.get(n.uid)
        if w is None:
            w = self._nodes[n.uid] = DNode(self, n)
        return w

    # learning ------------------------------------------------------------- #
    @property
    def sample_restructure(self) -> bool:
        return self.tree.sample_restructure

    @sample_restructure.setter
    def sample_restructure(self, on: bool) -> None:
        self.tree.sample_restructure = bool(on)

    @property
    def root(self) -> DNode:
        return self._wrap(self.tree.root)

    def ifit(self, x: dict, w: float = 1.0) -> DNode:
        if any(a not in self._aid for a in x):
            raise ValueError(f"instance attributes must come from {self.attrs}; got {sorted(x)}")
        leaf, _ = self.tree.ifit(self._x(x), False, float(w))
        return self._wrap(leaf)

    def unfit(self, leaf: DNode, x: dict, w: float = 1.0) -> None:
        self.tree.unfit(leaf.n, self._x(x), float(w))

    def relabel(self, leaf: DNode, old: dict, new: dict, w: float = 1.0) -> None:
        if old == new:
            return
        self.tree.relabel(leaf.n, self._x(old), self._x(new), float(w))

    def add_passive(self, leaf: DNode, attr: str, value: Hashable, w: float = 1.0) -> None:
        self.tree.add_passive(leaf.n, self._p(attr), self._v(value), float(w))

    def passive(self, node: DNode, attr: str) -> Dict[Hashable, float]:
        return node.pav.get(attr, {})

    # reading -------------------------------------------------------------- #
    def categorize(self, x: dict, w: float = 1.0) -> List[DNode]:
        """The path (root first) along which ``x`` would be sorted by category
        utility, without changing the tree."""
        leaf = self.tree.categorize_with_cu(self._x(x))
        return [self._wrap(n) for n in reversed(self.tree.ancestors(leaf))]

    def ancestors(self, node: DNode) -> List[DNode]:
        return [self._wrap(n) for n in self.tree.ancestors(node.n)]

    def nodes(self) -> Iterator[DNode]:
        stack = [self.tree.root]
        while stack:
            n = stack.pop()
            yield self._wrap(n)
            stack.extend(reversed(n.children))

    def leaves(self) -> List[DNode]:
        return [self._wrap(n) for n in self.tree.leaves()]

    def concept_codes(self, alpha: float) -> List[float]:
        """Per node (pre-order), the code length (nats) of its attribute counts:
        per attribute a Dirichlet-multinomial over the values the root has seen."""
        root = self.root
        n_values = {a: max(len(d), 1) for a, d in root.av.items()}
        out = []
        for node in self.nodes():
            tot = 0.0
            for a, d in node.av.items():
                A = n_values[a] * alpha
                tot += math.lgamma(sum(d.values()) + A) - math.lgamma(A)
                tot -= sum(math.lgamma(m + alpha) - math.lgamma(alpha) for m in d.values())
            out.append(tot)
        return out
