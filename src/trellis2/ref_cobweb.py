"""Cobweb over reference-valued attributes, updated on read (docs/UPDATE_ON_READ.md).

An attribute value may be a ``Ref``: a stable handle (a word, a stored record)
whose current value, a bag of concept ids, lives in a shared ``NameTable``.
TRELLIS changes a reference's entry when its name changes. A node keeps, per
reference attribute, the weight of every reference it counts and the bag it
last counted it with; its value counts (``av``, ``ss``, ``sstot``) are those
bags summed, so category utility is computed exactly as in the base tree.

A node is brought up to date when a sort reaches it (``sync``): every reference
whose entry changed since the node's stamp has its weight times (new bag - old
bag) applied, and the result is saved with a new stamp. Nodes no sort visits
wait until one does. An instance is translated with the current table when it
is scored or stored, so stored descriptions never go stale.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence

from .incremental_cobweb import EPS, CobwebNode, IncrementalCobwebTree, Instance

BLANK_BAG = {"-": 1.0}


class Ref:
    """A stable handle whose current value is read from a NameTable."""
    __slots__ = ("key",)

    def __init__(self, key: Hashable):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, Ref) and other.key == self.key

    def __hash__(self):
        return hash(("Ref", self.key))

    def __repr__(self):
        return f"Ref({self.key!r})"


class NameTable:
    """key -> current bag (a dict, replaced, never mutated) plus a log of changed keys."""

    def __init__(self):
        self.cur: Dict[Hashable, dict] = {}
        self.log: List[Hashable] = []

    def set(self, key: Hashable, bag) -> bool:
        if not isinstance(bag, dict):
            bag = {bag: 1.0}
        old = self.cur.get(key)
        if old is not None and (old == bag or (len(old) == len(bag) and
                                               all(abs(bag.get(u, -1.0) - p) <= 1e-12 for u, p in old.items()))):
            return False
        self.cur[key] = dict(bag)
        if old is not None:
            self.log.append(key)
        return True

    def get(self, key: Hashable) -> dict:
        return self.cur.get(key, BLANK_BAG)

    @property
    def version(self) -> int:
        return len(self.log)


class RefNode(CobwebNode):
    __slots__ = ("rb", "stamp")

    def __init__(self, nid: int):
        super().__init__(nid)
        self.rb: Dict[str, Dict[Hashable, list]] = {}   # attr -> key -> [weight, bag counted]
        self.stamp = 0


class RefCobwebTree(IncrementalCobwebTree):
    def __init__(self, attrs: Sequence[str], seed: int = 0, table: Optional[NameTable] = None):
        self.table = table if table is not None else NameTable()
        self.n_synced = 0                # (reference, node) corrections applied
        super().__init__(attrs, seed)

    def _new_node(self) -> RefNode:
        self._next_id += 1
        node = RefNode(self._next_id - 1)
        node.stamp = self.table.version
        return node

    # Translation and synchronisation ----------------------------------- #
    def translate(self, x: Instance) -> Instance:
        get = self.table.get
        return {a: (get(v.key) if isinstance(v, Ref) else v) for a, v in x.items()}

    def sync(self, node: RefNode) -> None:
        """Apply the name changes since the node's stamp to its counts and save them."""
        ver = self.table.version
        if node.stamp == ver:
            return
        if node.rb:
            changed = self.table.log[node.stamp:]
            nref = sum(len(r) for r in node.rb.values())
            keys = set(changed) if len(changed) < nref else None
            cur = self.table.cur
            for a, refs in node.rb.items():
                todo = refs.items() if keys is None else ((k, refs[k]) for k in keys if k in refs)
                d = node.av.get(a)
                if d is None:
                    d = node.av[a] = {}
                    node.ss[a] = 0.0
                ss = node.ss[a]
                for k, ent in list(todo):
                    new = cur.get(k, BLANK_BAG)
                    old = ent[1]
                    if new is old:
                        continue
                    w = ent[0]
                    for bag, sign in ((old, -1.0), (new, 1.0)):
                        for u, p in bag.items():
                            m = d.get(u, 0.0)
                            n = m + sign * w * p
                            if abs(n) <= EPS * max(1.0, abs(w)):
                                d.pop(u, None)
                                n = 0.0
                            else:
                                d[u] = n
                            ss += n * n - m * m
                    ent[1] = new
                    self.n_synced += 1
                node.sstot += ss - node.ss[a]
                node.ss[a] = ss
        node.stamp = ver

    def _rb_add(self, node: RefNode, x: Instance, w: float) -> None:
        for a, v in x.items():
            if not isinstance(v, Ref):
                continue
            refs = node.rb.setdefault(a, {})
            ent = refs.get(v.key)
            if ent is None:
                refs[v.key] = [w, self.table.get(v.key)]
            else:
                ent[0] += w
                if abs(ent[0]) <= EPS * max(1.0, abs(w)):
                    del refs[v.key]

    def _inc(self, node: RefNode, x: Instance, w: float, xt: Optional[Instance] = None) -> None:
        self.sync(node)
        node.increment(xt if xt is not None else self.translate(x), w)
        self._rb_add(node, x, w)

    def _dec(self, node: RefNode, x: Instance, w: float) -> None:
        self.sync(node)
        node.decrement(self.translate(x), w)
        self._rb_add(node, x, -w)

    def _absorb(self, node: RefNode, other: RefNode) -> None:
        self.sync(other)
        self.sync(node)
        node.absorb(other)
        for a, refs in other.rb.items():
            mine = node.rb.setdefault(a, {})
            for k, (w, bag) in refs.items():
                ent = mine.get(k)
                if ent is None:
                    mine[k] = [w, bag]
                else:
                    ent[0] += w

    # Structure (the base operations, with syncing and reference counts) -- #
    def _new_child(self, parent, x, w, xt=None):
        node = self._new_node()
        self._inc(node, x, w, xt)
        node.parent = parent
        parent.children.append(node)
        return node

    def _merge(self, cur, b1, b2):
        m = self._new_node()
        self._absorb(m, b1)
        self._absorb(m, b2)
        m.parent = cur
        m.children = [b1, b2]
        b1.parent = m
        b2.parent = m
        cur.children.remove(b1)
        cur.children.remove(b2)
        cur.children.append(m)
        self.restructured.add(m)
        return m

    def ifit(self, x: Instance, w: float = 1.0):
        if self.partial:
            if any(a not in self.attrs for a in x):
                raise ValueError(f"instance attributes must come from {self.attrs}; got {sorted(x)}")
        elif len(x) != self.n_attrs or any(a not in x for a in self.attrs):
            raise ValueError(f"instance must define exactly {self.attrs}; got {sorted(x)}")
        if w <= 0.0:
            raise ValueError("instance weight must be positive")
        xt = self.translate(x)
        restructure = not self.sample_restructure or self.rng.random() < w
        cur = self.root
        while True:
            self.sync(cur)
            if not cur.children:
                if cur.count == 0.0 or self._is_exact_match(cur, xt):
                    self._inc(cur, x, w, xt)
                    return cur
                p = self._new_node()
                self._absorb(p, cur)
                self._replace(cur, p)
                p.children = [cur]
                cur.parent = p
                self._inc(p, x, w, xt)
                return self._new_child(p, x, w, xt)
            for c in cur.children:
                self.sync(c)
            op, best1, best2 = self._best_operation(cur, xt, w, restructure)
            if op == "best":
                self._inc(cur, x, w, xt)
                cur = best1
            elif op == "new":
                self._inc(cur, x, w, xt)
                return self._new_child(cur, x, w, xt)
            elif op == "merge":
                self._inc(cur, x, w, xt)
                cur = self._merge(cur, best1, best2)
            else:
                self._split(cur, best1)

    def unfit(self, leaf, x: Instance, w: float = 1.0) -> None:
        if leaf.children:
            raise ValueError("instances are stored at leaves")
        if leaf.count < w - EPS * max(1.0, w):
            raise ValueError("removing more weight than the leaf holds")
        node = leaf
        while node is not None:
            self._dec(node, x, w)
            node = node.parent
        if leaf.count > 0.0:
            return
        parent = leaf.parent
        if parent is None:
            self.root = self._new_node()
            return
        parent.children.remove(leaf)
        leaf.parent = None
        if len(parent.children) == 1:
            child = parent.children[0]
            self._replace(parent, child)
            parent.children = []
            parent.parent = None

    def relabel(self, leaf, old: Instance, new: Instance, w: float = 1.0) -> None:
        if leaf.children:
            raise ValueError("instances are stored at leaves")
        a_old = {a: v for a, v in old.items() if new.get(a) != v}
        a_new = {a: new[a] for a in a_old if a in new}
        a_new.update({a: v for a, v in new.items() if a not in old})
        if not a_old and not a_new:
            return
        t_old, t_new = self.translate(a_old), self.translate(a_new)
        node = leaf
        while node is not None:
            self.sync(node)
            for x, sign in ((t_old, -1.0), (t_new, 1.0)):
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
            self._rb_add(node, a_old, -w)
            self._rb_add(node, a_new, w)
            node = node.parent

    def categorize(self, x: Instance, w: float = 1.0) -> List[CobwebNode]:
        xt = self.translate(x)
        node, path = self.root, [self.root]
        self.sync(node)
        sq = self._sq(xt, w) if self.fast_ec else None
        while node.children:
            for c in node.children:
                self.sync(c)
            best = max(node.children, key=lambda c: ((c.count + w) * self._ec_with(c, xt, w, sq)
                                                     - c.count * self.ec(c), c.count, -c.id))
            path.append(best)
            node = best
        return path

    def sync_all(self) -> None:
        """Bring every node up to date (for inspection: generation, tests)."""
        for n in self.nodes():
            self.sync(n)
