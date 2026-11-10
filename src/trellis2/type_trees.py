"""Two Cobweb trees over unit types, read by types, licensing pairs never stored.

Tree "R" holds each type described by one bag-valued attribute: the set of types seen just
after it. Tree "L" does the same with the set seen just before it. A type weighs as often as
it occurred. Types that share followers (or predecessors) end up under common concepts.

Reading by types: to ask whether b may follow a, walk the concepts on a's path in the R tree
(every concept except a's own leaf). For each concept, take the types grouped under it and
ask what share of them were seen followed by b; sum the shares over the path. The same is
done on b's path in the L tree (what share of its group was seen preceded by a). The licence
of the pair (a, b) is the geometric mean of the two readings.

Why types and not occurrences: every member of a group counts once, so one frequent
collocation does not stand for its whole group.

The trees stay current without rebuilding: when a type's description (neighbour set) or
weight (count) changes, it leaves the tree with its old description and enters with its new one.
"""
import math
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Set, Tuple

try:
    from trellis2.cu_cobweb import CUCobwebTree as _Tree
except Exception:  # pragma: no cover - fallback with the same API
    from trellis2.incremental_cobweb import IncrementalCobwebTree as _Tree

NONE = "<none>"   # description of a type with no neighbour on a side


class SideTrees:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self._order: List[Hashable] = []                       # types by first occurrence
        self._count: Counter = Counter()
        self._pairs: Set[Tuple[Hashable, Hashable]] = set()
        # per side: type -> ordered neighbour dict (insertion order keeps results deterministic)
        self._nb: Dict[str, Dict[Hashable, dict]] = {"R": defaultdict(dict), "L": defaultdict(dict)}
        self._tree = {}
        for side in ("R", "L"):
            t = _Tree((side,), seed=seed)
            t.exact_alone = True
            self._tree[side] = t
        self._leaf: Dict[str, dict] = {"R": {}, "L": {}}       # type -> leaf node
        self._cur: Dict[str, dict] = {"R": {}, "L": {}}        # type -> (description, weight) in the tree
        self._dirty = True
        self._chain: Dict[str, dict] = {}
        self._members: Dict[str, dict] = {}
        self._memo: Dict[str, dict] = {}

    def _describe(self, side: str, w: Hashable) -> dict:
        d = self._nb[side].get(w)
        return {side: {k: 1.0 / len(d) for k in d} if d else {NONE: 1.0}}

    def observe(self, seq: Iterable[Hashable]) -> None:
        """One experience: a sequence of unit types. Counts each type, records each adjacent pair,
        and keeps both trees current: a type whose description (neighbour set) or weight (count)
        changed leaves the tree with its old description and enters with its new one."""
        seq = list(seq)
        touched: List[Hashable] = []
        for w in seq:
            if w not in self._count:
                self._order.append(w)
            if w not in touched:
                touched.append(w)
            self._count[w] += 1
        for a, b in zip(seq, seq[1:]):          # boundaries are not neighbours
            self._pairs.add((a, b))
            self._nb["R"][a][b] = True
            self._nb["L"][b][a] = True
        for side in ("R", "L"):
            for w in touched:                    # once per type per tree, after the whole sequence
                if w in self._cur[side]:
                    self._tree[side].unfit(self._leaf[side][w], *self._cur[side][w])
                x, wt = self._describe(side, w), float(self._count[w])
                self._leaf[side][w] = self._tree[side].ifit(x, wt)
                self._cur[side][w] = (x, wt)
        self._dirty = True

    def _index(self) -> None:
        if not self._dirty:
            return
        for side in ("R", "L"):
            chains, members = {}, defaultdict(set)
            for w in self._order:
                nd, chain = self._leaf[side][w], []
                while nd is not None:
                    chain.append(nd.id)
                    members[nd.id].add(w)
                    nd = nd.parent
                chains[w] = chain
            self._chain[side], self._members[side], self._memo[side] = chains, members, {}
        self._dirty = False

    def pair_stored(self, a: Hashable, b: Hashable) -> bool:
        return (a, b) in self._pairs

    def _read(self, w: Hashable, side: str, v: Hashable) -> float:
        memo, members = self._memo[side], self._members[side]
        tot = 0.0
        for c in self._chain[side][w][1:]:       # skip w's own leaf
            k = (c, v)
            if k not in memo:
                ms = members[c]
                hit = sum(1 for m in ms if ((m, v) if side == "R" else (v, m)) in self._pairs)
                memo[k] = hit / len(ms)
            tot += memo[k]
        return tot

    def licence(self, a: Hashable, b: Hashable) -> float:
        self._index()
        if a not in self._count or b not in self._count:
            return 0.0
        return math.sqrt(self._read(a, "R", b) * self._read(b, "L", a))

    def next_distribution(self, a: Hashable) -> dict:
        """{b: licence(a, b) / total}; types with zero licence are left out; {} if none."""
        if a not in self._count:
            return {}
        lic = {b: self.licence(a, b) for b in self._order}
        lic = {b: v for b, v in lic.items() if v > 0}
        tot = sum(lic.values())
        return {b: v / tot for b, v in lic.items()} if tot > 0 else {}

    def votes(self, w: Hashable, fillers: Iterable[Hashable]) -> float:
        """Frame vote: over both trees and every concept on w's path except its own leaf,
        the share of the concept's types that are fillers."""
        self._index()
        if w not in self._count:
            return 0.0
        fs, tot = set(fillers), 0.0
        for side in ("R", "L"):
            for c in self._chain[side][w][1:]:
                ms = self._members[side][c]
                tot += len(ms & fs) / len(ms)
        return tot

    def types(self) -> list:
        return list(self._order)
