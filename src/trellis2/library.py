"""TRELLIS v3: a chunk library in two Cobweb trees, read by the shortest code.

Learning stores every candidate: each experience's every span enters the
context tree once (described only by its context window), and every composite
(every span with every split) enters the content tree (its parts' context paths
and its own). Nothing depends on how the experience is parsed, so what is built
once is not favoured to be built again.

At runtime an experience is read by its shortest description under the trees,
found exactly by dynamic programming over spans:

* a top-level piece: its context concept (log share), one stop bit;
* an element at context concept v: primitive or composite, from what v holds;
  a primitive: its token among v's tokens; a composite: its parts' context paths
  given its own, through the content concept that codes them best;

so a chunk library is good exactly when it codes new experiences in few bits.
The same code generates: sampled top-down it produces experiences.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS = "<s>", "</s>"
BLANK = "-"
LN2 = math.log(2.0)
KIND, TOK = "kind", "tok"


def log_prob(node: CobwebNode, x: dict, root: CobwebNode, alpha: float) -> float:
    """ln P(x | node): Laplace-smoothed value probabilities, attributes independent;
    each attribute's alphabet is the root's values plus one unseen."""
    total = 0.0
    for a, v in x.items():
        d = node.av.get(a, {})
        size = len(root.av.get(a, {})) + 1
        total += math.log((d.get(v, 0.0) + alpha) / (node.count + alpha * size))
    return total


class ChunkLibrary:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, window: int = 2,
                 redescribe_budget: Optional[int] = 2000):
        self.depth, self.alpha, self.window = depth, alpha, window
        self.ctx_attrs = tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))
        d = range(1, depth + 1)
        # L1..LD etc. shape the content tree's clustering; the code reads one
        # attribute per part (LC, RC, SC: the concept at depth D, or the deepest),
        # since the levels of a path are nested and coding each would count the
        # same information several times.
        self.cnt_attrs = (tuple(f"L{k}" for k in d) + tuple(f"R{k}" for k in d) + tuple(f"S{k}" for k in d)
                          + ("LC", "RC", "SC"))
        self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
        self.cnt = IncrementalCobwebTree(self.cnt_attrs, seed=seed + 1)
        self.rng = random.Random(seed)
        self.redescribe_budget = redescribe_budget
        self.records: List[list] = []     # [left leaf, right leaf, own leaf, description, content leaf]
        self._rr = 0
        self.stats = {"experiences": 0, "spans stored": 0, "composites stored": 0, "re-described": 0}

    # Descriptions --------------------------------------------------------- #
    def _ctx_instance(self, tokens: Sequence[str], i: int, j: int) -> dict:
        x = {}
        for k in range(1, self.window + 1):
            x[f"l{k}"] = tokens[i - k] if i - k >= 0 else BOS
            x[f"r{k}"] = tokens[j + k - 1] if j + k - 1 < len(tokens) else EOS
        return x

    def _ids(self, path: List[CobwebNode]) -> List[str]:
        ids = [f"c{n.id}" for n in path[1:self.depth + 1]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _path_of_leaf(self, leaf: CobwebNode) -> List[CobwebNode]:
        return self.ctx.ancestors(leaf)[::-1]

    def _content(self, lp: List[CobwebNode], rp: List[CobwebNode], sp: List[CobwebNode]) -> dict:
        x = {}
        for p, path in (("L", lp), ("R", rp), ("S", sp)):
            for d, v in enumerate(self._ids(path), 1):
                x[f"{p}{d}"] = v
            x[f"{p}C"] = f"c{self._node(path).id}"
        return x

    @staticmethod
    def _coded(x: dict, keys=("LC", "RC", "SC")) -> dict:
        return {k: x[k] for k in keys}

    # Learning ------------------------------------------------------------- #
    def learn(self, tokens: Sequence[str]) -> None:
        """Store every span's context (once) and every composite's content."""
        tokens = list(tokens)
        n = len(tokens)
        leaf: Dict[Tuple[int, int], CobwebNode] = {}
        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                lf = self.ctx.ifit(self._ctx_instance(tokens, i, j))
                self.ctx.add_passive(lf, KIND, "P" if length == 1 else "C")
                if length == 1:
                    self.ctx.add_passive(lf, TOK, tokens[i])
                leaf[(i, j)] = lf
                self.stats["spans stored"] += 1
        first = len(self.records)
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    a, b, s = leaf[(i, k)], leaf[(k, j)], leaf[(i, j)]
                    x = self._content(self._path_of_leaf(a), self._path_of_leaf(b), self._path_of_leaf(s))
                    self.records.append([a, b, s, x, self.cnt.ifit(x)])
                    self.stats["composites stored"] += 1
        self.stats["experiences"] += 1
        self._redescribe(first)

    def _redescribe(self, first: int) -> None:
        """Bring stored content descriptions up to date with the context tree
        (its concepts move as it grows): the newest records and a round-robin
        slice of the rest, ``redescribe_budget`` per experience (None: all)."""
        recs = self.records
        b = self.redescribe_budget
        if b is None or first <= b:
            todo = recs
        else:
            old = recs[:first]
            start = self._rr % len(old)
            todo = recs[first:] + (old[start:] + old[:start])[:b]
            self._rr = (start + b) % len(old)
        for rec in todo:
            a, b_, s, x, cl = rec
            new = self._content(self._path_of_leaf(a), self._path_of_leaf(b_), self._path_of_leaf(s))
            if new != x:
                self.cnt.unfit(cl, x)
                rec[3], rec[4] = new, self.cnt.ifit(new)
                self.stats["re-described"] += 1

    # Code ----------------------------------------------------------------- #
    def _node(self, path: List[CobwebNode]) -> CobwebNode:
        """An element's context concept: its path's node at depth D (or deepest)."""
        return path[min(self.depth, len(path) - 1)]

    def _kind_bits(self, v: CobwebNode, kind: str) -> float:
        d = v.pav.get(KIND, {})
        tot = sum(d.values())
        return -math.log2((d.get(kind, 0.0) + self.alpha) / (tot + 2 * self.alpha))

    def _tok_bits(self, v: CobwebNode, w: str) -> float:
        d = v.pav.get(TOK, {})
        tot = sum(d.values())
        V = len(self.ctx.root.pav.get(TOK, {})) + 1
        return -math.log2((d.get(w, 0.0) + self.alpha) / (tot + self.alpha * V))

    def _best_concept(self, x: dict, coded: dict) -> Tuple[float, CobwebNode]:
        """max over x's sorting path (root included) of log2 [share(C) P(coded|C)]."""
        root = self.cnt.root
        best, arg = -math.inf, root
        for node in self.cnt.categorize(x):
            if node.count <= 0:
                continue
            v = (math.log(node.count / root.count) + log_prob(node, coded, root, self.alpha)) / LN2
            if v > best:
                best, arg = v, node
        return best, arg

    def _comp_bits(self, lp, rp, sp) -> Tuple[float, CobwebNode]:
        """Bits for a composite's parts given its own context path:
        -log2 P(L, R, S) + log2 P(S), each through its best concept."""
        x = self._content(lp, rp, sp)
        joint, node = self._best_concept(x, self._coded(x))
        own = {k: v for k, v in x.items() if k[0] == "S"}
        marg, _ = self._best_concept(own, self._coded(x, ("SC",)))
        return marg - joint, node

    def _top_bits(self, v: CobwebNode) -> float:
        root = self.ctx.root
        return -math.log2((v.count + self.alpha) / (root.count + self.alpha)) + 1.0

    def read(self, tokens: Sequence[str]) -> dict:
        """The shortest description of an experience under the trees (exact, by
        dynamic programming over spans): its bits, its tree, and the content
        concept each composite is coded through."""
        tokens = list(tokens)
        n = len(tokens)
        if self.cnt.root.count <= 0 or self.ctx.root.count <= 0:
            raise ValueError("empty library")
        path = {}
        for i in range(n):
            for j in range(i + 1, n + 1):
                path[(i, j)] = self.ctx.categorize(self._ctx_instance(tokens, i, j))
        E: Dict[Tuple[int, int], float] = {}
        back: Dict[Tuple[int, int], Tuple[int, CobwebNode]] = {}
        for i in range(n):
            v = self._node(path[(i, i + 1)])
            E[(i, i + 1)] = self._kind_bits(v, "P") + self._tok_bits(v, tokens[i])
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                best = (math.inf, -1, None)
                for k in range(i + 1, j):
                    bits, node = self._comp_bits(path[(i, k)], path[(k, j)], path[(i, j)])
                    tot = bits + E[(i, k)] + E[(k, j)]
                    if tot < best[0]:
                        best = (tot, k, node)
                E[(i, j)] = best[0] + self._kind_bits(self._node(path[(i, j)]), "C")
                back[(i, j)] = (best[1], best[2])
        F = [0.0] + [math.inf] * n
        cut = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(j):
                t = F[i] + self._top_bits(self._node(path[(i, j)])) + E[(i, j)]
                if t < F[j]:
                    F[j], cut[j] = t, i
        roots, j = [], n
        while j > 0:
            roots.append((cut[j], j))
            j = cut[j]
        roots.reverse()
        split, label, concepts = {}, {}, []

        def walk(i, j):
            if j - i < 2:
                return
            k, node = back[(i, j)]
            split[(i, j)] = k
            label[(i, j)] = f"n{node.id}"
            concepts.append(node.id)
            walk(i, k)
            walk(k, j)
        for i, j in roots:
            walk(i, j)
        return {"bits": F[n], "tree": Tree(n, split, label, roots), "concepts": concepts}

    # Generation ----------------------------------------------------------- #
    def _descend(self, tree: IncrementalCobwebTree, partial: dict, start: Optional[CobwebNode] = None) -> CobwebNode:
        node = start or tree.root
        while node.children:
            ws = [c.count * math.exp(log_prob(c, partial, tree.root, self.alpha)) for c in node.children]
            node = self.rng.choices(node.children, weights=ws)[0]
        return node

    def _pick(self, d: Dict) -> Optional[object]:
        if not d:
            return None
        vals, ws = zip(*sorted(d.items(), key=lambda kv: str(kv[0])))
        return self.rng.choices(vals, weights=ws)[0]

    def generate(self, max_depth: int = 14, max_len: int = 40) -> Optional[List[str]]:
        """Sample an experience top-down from the same code: a context concept at
        the experience's boundaries; at each element's concept, primitive or
        composite; a token, or a composition sampled given the element's own
        context path, each part expanded at the concept its path names."""
        ids = {f"c{n.id}": n for n in self.ctx.nodes()}
        out: List[str] = []

        def expand(path_ids: List[str], depth: int) -> bool:
            if depth > max_depth or len(out) > max_len:
                return False
            v = self.ctx.root
            for x in path_ids:
                if x in ids:
                    v = ids[x]
            p_c = (v.pav.get(KIND, {}).get("C", 0.0) + self.alpha) / (sum(v.pav.get(KIND, {}).values()) + 2 * self.alpha)
            if self.rng.random() >= p_c:
                w = self._pick(v.pav.get(TOK, {})) or self._pick(self.ctx.root.pav.get(TOK, {}))
                out.append(w)
                return True
            leaf = self._descend(self.cnt, {f"S{d}": x for d, x in enumerate(path_ids, 1)})
            for side in ("L", "R"):
                sub = [self._pick(leaf.av.get(f"{side}{d}", {})) or BLANK for d in range(1, self.depth + 1)]
                sub.append(self._pick(leaf.av.get(f"{side}C", {})) or BLANK)
                if not expand(sub, depth + 1):
                    return False
            return True
        whole = {f"l{k}": BOS for k in range(1, self.window + 1)}
        whole.update({f"r{k}": EOS for k in range(1, self.window + 1)})
        top = self._descend(self.ctx, whole)
        return out if expand(self._ids(self._path_of_leaf(top)), 0) and out else None
