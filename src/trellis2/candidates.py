"""TRELLIS v3: candidate trees (docs/V3_DESIGN.md).

Learning is greedy parsing that stores every candidate: at each step every new
adjacent pair on the frontier is stored in both Cobweb trees (context and
content), then each pair is scored by the concepts it now sits under, and the
best positive one is committed. Nothing is distilled: scores, parses and
generation are computed from the trees' counts when needed, and stored
descriptions are recomputed from stable leaf handles as the trees change.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS = "<s>", "</s>"
COMPOSITE, PRIMITIVE = "<C>", "<P>"
BLANK = "-"
LN2 = math.log(2.0)
CTX_ATTRS = ("l1", "r1", "first", "last", "kind")


def content_attrs(depth: int) -> Tuple[str, ...]:
    """W: whether the composite is a whole experience; L, R: each part's kind;
    L1..LD, R1..RD: each part's path in the context tree (how it behaves);
    LC1..LCD, RC1..RCD: a composite part's path in the content tree (what it is
    made of; blank for a primitive); S1..SD: the composite's own context path."""
    d = range(1, depth + 1)
    return (("W", "L", "R") + tuple(f"L{k}" for k in d) + tuple(f"R{k}" for k in d)
            + tuple(f"LC{k}" for k in d) + tuple(f"RC{k}" for k in d) + tuple(f"S{k}" for k in d))


@dataclass(eq=False)
class Element:
    span: Tuple[int, int]
    ctx: dict
    token: Optional[str] = None
    left: Optional["Element"] = None
    right: Optional["Element"] = None
    content: Optional[dict] = None
    concept: Optional[CobwebNode] = None   # the proposed non-terminal (content tree)
    score: float = 0.0
    leaf: Optional[CobwebNode] = None      # its context instance's leaf (a stable handle)
    cleaf: Optional[CobwebNode] = None     # its content instance's leaf

    @property
    def is_primitive(self) -> bool:
        return self.token is not None


def log_prob(node: CobwebNode, x: dict, root: CobwebNode, alpha: float) -> float:
    """ln P(x | node): Laplace-smoothed value probabilities, attributes independent;
    each attribute's alphabet is the root's values plus one unseen."""
    total = 0.0
    for a, v in x.items():
        d = node.av.get(a, {})
        size = len(root.av.get(a, {})) + 1
        total += math.log((d.get(v, 0.0) + alpha) / (node.count + alpha * size))
    return total


def gain_along(path: List[CobwebNode], x: dict, root: CobwebNode, alpha: float,
               min_count: float = 0.0) -> Tuple[float, Optional[CobwebNode]]:
    """Best gain (bits) along a root-first path, and its concept:
    log2 share of the concept (strength) + log2 P(x|C) - log2 P(x|root) (fit);
    0 at the root, so a positive gain means coding x through C compresses."""
    if root.count <= 0:
        return 0.0, None
    base = log_prob(root, x, root, alpha)
    best, arg = 0.0, None
    for node in path[1:]:
        if node.count < min_count or node.count <= 0:
            continue
        g = (math.log(node.count / root.count) + log_prob(node, x, root, alpha) - base) / LN2
        if g > best:
            best, arg = g, node
    return best, arg


class CandidateLearner:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0,
                 refresh: bool = True, min_count: float = 0.0, lookahead: bool = False,
                 context_gain: bool = True, store_first: bool = True):
        self.depth, self.alpha = depth, alpha
        self.ctx = IncrementalCobwebTree(CTX_ATTRS, seed=seed)
        self.cnt = IncrementalCobwebTree(content_attrs(depth), seed=seed + 1)
        self.rng = random.Random(seed)
        # refresh: after each experience, every stored content instance is
        # re-described from its parts' current paths (dynamic recomputation over
        # stored leaves). min_count: the support a concept needs to be proposed.
        # lookahead: each step commits the candidate whose greedy completion of
        # the parse compresses the experience most. context_gain: candidates are
        # scored against both trees (P4). store_first: candidates are stored
        # before they are scored, so each is judged by the concept it ends up in.
        self.refresh, self.min_count, self.lookahead = refresh, min_count, lookahead
        self.context_gain, self.store_first = context_gain, store_first
        self.records: List[Element] = []
        self.stats = {"experiences": 0, "candidates stored": 0, "committed": 0, "re-described": 0}

    # Descriptions --------------------------------------------------------- #
    @staticmethod
    def _ctx_instance(tokens: Sequence[str], span: Tuple[int, int], kind: str) -> dict:
        i, j = span
        return {"l1": tokens[i - 1] if i > 0 else BOS, "r1": tokens[j] if j < len(tokens) else EOS,
                "first": tokens[i], "last": tokens[j - 1], "kind": kind}

    def _ids(self, path: List[CobwebNode]) -> List[str]:
        ids = [f"c{n.id}" for n in path[1:self.depth + 1]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _ctx_path(self, el: Element) -> List[str]:
        """The element's path in the context tree, now."""
        path = self.ctx.ancestors(el.leaf)[::-1] if el.leaf is not None else self.ctx.categorize(el.ctx)
        return self._ids(path)

    def _cnt_path(self, el: Element) -> List[str]:
        """A composite's path in the content tree, now (blank for a primitive)."""
        if el.is_primitive:
            return [BLANK] * self.depth
        if el.cleaf is not None:
            return self._ids(self.cnt.ancestors(el.cleaf)[::-1])
        return self._ids(self.cnt.categorize(el.content))

    def _content(self, c: Element) -> dict:
        left, right = c.left, c.right
        x = {"W": c.ctx["l1"] == BOS and c.ctx["r1"] == EOS,
             "L": PRIMITIVE if left.is_primitive else COMPOSITE,
             "R": PRIMITIVE if right.is_primitive else COMPOSITE}
        for p, ids in (("L", self._ctx_path(left)), ("R", self._ctx_path(right)),
                       ("LC", self._cnt_path(left)), ("RC", self._cnt_path(right)), ("S", self._ctx_path(c))):
            for d, v in enumerate(ids, 1):
                x[f"{p}{d}"] = v
        return x

    # Scoring -------------------------------------------------------------- #
    def _score(self, c: Element) -> None:
        """Content gain (with its concept, the proposed non-terminal) plus, with
        context_gain, the context gain of the candidate's surroundings. A stored
        candidate is judged along its own leaves' paths; an unstored one by
        sorting it."""
        cpath = self.cnt.ancestors(c.cleaf)[::-1] if c.cleaf is not None else self.cnt.categorize(c.content)
        g, node = gain_along(cpath, c.content, self.cnt.root, self.alpha, self.min_count)
        if self.context_gain:
            xpath = self.ctx.ancestors(c.leaf)[::-1] if c.leaf is not None else self.ctx.categorize(c.ctx)
            g += gain_along(xpath, c.ctx, self.ctx.root, self.alpha, self.min_count)[0]
        c.score, c.concept = g, node

    def _candidate(self, tokens, a: Element, b: Element) -> Element:
        span = (a.span[0], b.span[1])
        c = Element(span, self._ctx_instance(tokens, span, "C"), left=a, right=b)
        c.content = self._content(c)
        return c

    # Parsing -------------------------------------------------------------- #
    def parse(self, tokens: Sequence[str], learn: bool = False) -> Tuple[Tree, List[Element]]:
        tokens = list(tokens)
        frontier = [Element((k, k + 1), self._ctx_instance(tokens, (k, k + 1), "P"), token=w)
                    for k, w in enumerate(tokens)]
        if learn:
            for el in frontier:
                el.leaf = self.ctx.ifit(el.ctx)
        made: Dict[Tuple[int, int], Element] = {}
        while len(frontier) > 1:
            new = []
            for a, b in zip(frontier, frontier[1:]):
                if (id(a), id(b)) not in made:
                    c = self._candidate(tokens, a, b)
                    made[(id(a), id(b))] = c
                    new.append(c)
            if learn:
                if not self.store_first:
                    for c in new:
                        self._score(c)
                for c in new:
                    c.leaf = self.ctx.ifit(c.ctx)
                for c in new:
                    c.content = self._content(c)
                    c.cleaf = self.cnt.ifit(c.content)
                    self.records.append(c)
                    self.stats["candidates stored"] += 1
            pairs = [(made[(id(a), id(b))], t) for t, (a, b) in enumerate(zip(frontier, frontier[1:]))]
            if learn and self.store_first:
                for c, _ in pairs:                 # every pair, by the trees as they are now
                    self._score(c)
            elif not learn:
                for c in new:
                    self._score(c)
            if self.lookahead:
                pos = [(c, t) for c, t in pairs if c.score > 0.0]
                if not pos:
                    break
                value = {t: c.score + self._completion(tokens, frontier[:t] + [c] + frontier[t + 2:], made)
                         for c, t in pos}
                best, t = max(pos, key=lambda p: (value[p[1]], -p[1]))
            else:
                best, t = max(pairs, key=lambda p: (p[0].score, -p[1]))
                if best.score <= 0.0:
                    break
            frontier[t:t + 2] = [best]
            if learn:
                self.stats["committed"] += 1
        if learn:
            self.stats["experiences"] += 1
            if self.refresh:
                self._redescribe()
        return self._tree(len(tokens), frontier), frontier

    def _completion(self, tokens, frontier: List[Element], made: Dict) -> float:
        """Total gain of completing a frontier greedily (the trees unchanged;
        hypothetical candidates are memoised in ``made``, unstored)."""
        frontier, total = list(frontier), 0.0
        while len(frontier) > 1:
            best, bt = None, -1
            for t, (a, b) in enumerate(zip(frontier, frontier[1:])):
                key = (id(a), id(b))
                if key not in made:
                    made[key] = self._candidate(tokens, a, b)
                    self._score(made[key])
                c = made[key]
                if best is None or c.score > best.score:
                    best, bt = c, t
            if best.score <= 0.0:
                break
            total += best.score
            frontier[bt:bt + 2] = [best]
        return total

    def _redescribe(self) -> None:
        """Re-describe every stored content instance whose parts' paths changed
        (remove the old description, insert the new one)."""
        for c in self.records:
            new = self._content(c)
            if new != c.content:
                self.cnt.unfit(c.cleaf, c.content)
                c.content, c.cleaf = new, self.cnt.ifit(new)
                self.stats["re-described"] += 1

    def learn(self, tokens: Sequence[str]) -> Tree:
        return self.parse(tokens, learn=True)[0]

    @staticmethod
    def _tree(n: int, roots: List[Element]) -> Tree:
        split, label = {}, {}

        def walk(el: Element):
            if el.is_primitive:
                label[el.span] = el.token
                return
            split[el.span] = el.left.span[1]
            label[el.span] = f"n{el.concept.id}" if el.concept is not None else "?"
            walk(el.left)
            walk(el.right)
        for r in roots:
            walk(r)
        return Tree(n, split, label, [r.span for r in roots])

    # Generation ----------------------------------------------------------- #
    def _sample_leaf(self, tree: IncrementalCobwebTree, partial: dict,
                     start: Optional[CobwebNode] = None) -> CobwebNode:
        """Descend from ``start`` choosing each child with probability
        proportional to its count times P(partial | child)."""
        node = start or tree.root
        while node.children:
            ws = [c.count * math.exp(log_prob(c, partial, tree.root, self.alpha)) for c in node.children]
            node = self.rng.choices(node.children, weights=ws)[0]
        return node

    def _value(self, node: CobwebNode, a: str):
        d = node.av.get(a, {})
        if not d:
            return BLANK
        vals, ws = zip(*sorted(d.items(), key=lambda kv: str(kv[0])))
        return self.rng.choices(vals, weights=ws)[0]

    @staticmethod
    def _deepest(tree: IncrementalCobwebTree, ids: List[str]) -> Optional[CobwebNode]:
        nodes = {f"c{n.id}": n for n in tree.nodes()}
        out = None
        for v in ids:
            if v in nodes:
                out = nodes[v]
        return out

    def generate(self, max_depth: int = 12, max_len: int = 40) -> Optional[List[str]]:
        """One experience sampled top-down: a whole-experience composite from the
        content tree; each composite part expanded under its content path, each
        primitive part's token drawn under its context path."""
        out: List[str] = []

        def expand(leaf: CobwebNode, depth: int) -> bool:
            if depth > max_depth or len(out) > max_len:
                return False
            for side in ("L", "R"):
                kind = self._value(leaf, side)
                if kind == COMPOSITE:
                    ids = [self._value(leaf, f"{side}C{d}") for d in range(1, self.depth + 1)]
                    start = self._deepest(self.cnt, ids)
                    sub = self._sample_leaf(self.cnt, {"W": False}, start)
                    if not expand(sub, depth + 1):
                        return False
                else:
                    ids = [self._value(leaf, f"{side}{d}") for d in range(1, self.depth + 1)]
                    start = self._deepest(self.ctx, ids)
                    out.append(self._value(self._sample_leaf(self.ctx, {"kind": "P"}, start), "first"))
            return True
        top = self._sample_leaf(self.cnt, {"W": True})
        return out if expand(top, 0) and len(out) <= max_len else None
