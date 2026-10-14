"""TRELLIS v3: candidate trees (docs/V3_DESIGN.md).

Learning is greedy parsing that stores every candidate: each adjacent pair on
the frontier is scored against the two Cobweb trees as they stand, then stored
in both (context and content), and the best positive one is committed. Nothing
is distilled: scores, parses and generation are computed from the trees'
counts when needed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS = "<s>", "</s>"
COMPOSITE = "<C>"
BLANK = "-"
LN2 = math.log(2.0)
CTX_ATTRS = ("l1", "r1", "first", "last", "kind")


def content_attrs(depth: int, tokens: bool = False) -> Tuple[str, ...]:
    """L, R: each part's kind (primitive or composite), or with ``tokens`` its
    token (<C> for a composite); L1..LD, R1..RD: each part's path in the
    context tree; S1..SD: the composite's own path there."""
    return (("L", "R") + tuple(f"L{d}" for d in range(1, depth + 1))
            + tuple(f"R{d}" for d in range(1, depth + 1)) + tuple(f"S{d}" for d in range(1, depth + 1)))


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
    leaf: Optional[CobwebNode] = None      # where its context instance is stored (a stable handle)

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


class CandidateLearner:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, gen_depth: int = 3,
                 tokens: bool = False, refresh: bool = True, concepts_only: bool = False,
                 min_count: float = 0.0, lookahead: bool = False):
        # tokens: a part is named by its token too (else only by its kind and
        # its path in the context tree, i.e. by category).
        self.depth, self.alpha, self.gen_depth, self.tokens = depth, alpha, gen_depth, tokens
        self.ctx = IncrementalCobwebTree(CTX_ATTRS, seed=seed)
        self.cnt = IncrementalCobwebTree(content_attrs(depth), seed=seed + 1)
        self.rng = random.Random(seed)
        # refresh: after each experience, every stored content instance is
        # re-described from its parts' current paths (dynamic recomputation over
        # stored leaves), so a concept is always named by the context tree as it is.
        self.refresh = refresh
        # The proposed non-terminal is a concept: with concepts_only, an internal
        # node (leaves are stored instances); and one holding at least min_count.
        self.concepts_only, self.min_count = concepts_only, min_count
        # lookahead: each step commits the candidate whose greedy completion of
        # the parse compresses the experience most (the sum of the gains).
        self.lookahead = lookahead
        self.records: List[list] = []          # [left, right, self element, description, content leaf]
        self.stats = {"experiences": 0, "candidates stored": 0, "committed": 0, "re-described": 0}

    # Descriptions --------------------------------------------------------- #
    @staticmethod
    def _ctx_instance(tokens: Sequence[str], span: Tuple[int, int], kind: str) -> dict:
        i, j = span
        return {"l1": tokens[i - 1] if i > 0 else BOS, "r1": tokens[j] if j < len(tokens) else EOS,
                "first": tokens[i], "last": tokens[j - 1], "kind": kind}

    def _path(self, el: Element) -> List[str]:
        """Concept ids at depths 1..D of the element's path in the context tree,
        computed now: from its stored leaf, or by sorting its context instance."""
        if el.leaf is not None:
            path = self.ctx.ancestors(el.leaf)[::-1][1:]
        else:
            path = self.ctx.categorize(el.ctx)[1:]
        ids = [f"c{n.id}" for n in path[:self.depth]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _content(self, left: Element, right: Element, own: Element) -> dict:
        def head(el):
            if not el.is_primitive:
                return COMPOSITE
            return el.token if self.tokens else "<P>"
        x = {"L": head(left), "R": head(right)}
        for p, el in (("L", left), ("R", right), ("S", own)):
            ids = self._path(el)
            for d, v in enumerate(ids, 1):
                x[f"{p}{d}"] = v
        return x

    # Scoring -------------------------------------------------------------- #
    def score(self, x: dict) -> Tuple[float, Optional[CobwebNode]]:
        """Best gain (bits) along x's sorting path in the content tree, and its
        concept: log2 share of the concept (strength) + log2 P(x|C) - log2 P(x|root)."""
        root = self.cnt.root
        if root.count <= 0:
            return 0.0, None
        base = log_prob(root, x, root, self.alpha)
        best, arg = 0.0, None
        for node in self.cnt.categorize(x)[1:]:
            if (self.concepts_only and not node.children) or node.count < self.min_count:
                continue
            g = (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - base) / LN2
            if g > best:
                best, arg = g, node
        return best, arg

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
                key = (id(a), id(b))
                if key in made:
                    continue
                span = (a.span[0], b.span[1])
                ctx = self._ctx_instance(tokens, span, "C")
                c = Element(span, ctx, left=a, right=b)
                c.content = self._content(a, b, c)
                c.score, c.concept = self.score(c.content)
                made[key] = c
                new.append(c)
            if learn:
                for c in new:
                    c.leaf = self.ctx.ifit(c.ctx)
                for c in new:
                    desc = self._content(c.left, c.right, c)
                    self.records.append([c.left, c.right, c, desc, self.cnt.ifit(desc)])
                    self.stats["candidates stored"] += 1
            pairs = [(made[(id(a), id(b))], t) for t, (a, b) in enumerate(zip(frontier, frontier[1:]))]
            if self.lookahead:
                pairs = [(c, t) for c, t in pairs if c.score > 0.0]
                if not pairs:
                    break
                value = {t: c.score + self._completion(tokens, frontier[:t] + [c] + frontier[t + 2:], made)
                         for c, t in pairs}
                best, t = max(pairs, key=lambda p: (value[p[1]], -p[1]))
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

    def _redescribe(self) -> None:
        """Re-describe every stored content instance whose parts' paths changed
        (remove the old description, insert the new one)."""
        for rec in self.records:
            left, right, own, desc, leaf = rec
            new = self._content(left, right, own)
            if new != desc:
                self.cnt.unfit(leaf, desc)
                rec[3], rec[4] = new, self.cnt.ifit(new)
                self.stats["re-described"] += 1

    def _completion(self, tokens, frontier: List[Element], made: Dict) -> float:
        """Total gain of completing a frontier greedily (scores only; the trees
        are not changed). Candidates are memoised in ``made``."""
        frontier, total = list(frontier), 0.0
        while len(frontier) > 1:
            best, bt = None, -1
            for t, (a, b) in enumerate(zip(frontier, frontier[1:])):
                key = (id(a), id(b))
                if key not in made:
                    span = (a.span[0], b.span[1])
                    c = Element(span, self._ctx_instance(tokens, span, "C"), left=a, right=b)
                    c.content = self._content(a, b, c)
                    c.score, c.concept = self.score(c.content)
                    made[key] = c
                c = made[key]
                if best is None or c.score > best.score:
                    best, bt = c, t
            if best.score <= 0.0:
                break
            total += best.score
            frontier[bt:bt + 2] = [best]
        return total

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

    @staticmethod
    def _value(node: CobwebNode, a: str, rng: random.Random):
        d = node.av.get(a, {})
        if not d:
            return BLANK
        vals, ws = zip(*sorted(d.items(), key=lambda kv: str(kv[0])))
        return rng.choices(vals, weights=ws)[0]

    def _word(self, path: List[str]) -> str:
        """A token for a primitive part named by its context path: a primitive
        leaf sampled under the deepest node of that path still in the tree."""
        nodes = {f"c{n.id}": n for n in self.ctx.nodes()}
        start = None
        for v in path[:self.gen_depth]:
            if v in nodes:
                start = nodes[v]
        leaf = self._sample_leaf(self.ctx, {"kind": "P"}, start)
        return self._value(leaf, "first", self.rng)

    def generate(self, max_depth: int = 12, max_len: int = 40) -> Optional[List[str]]:
        """One experience sampled top-down from the trees (None if it runs past
        the length or depth limit)."""
        rng = self.rng
        top = self._sample_leaf(self.ctx, {"l1": BOS, "r1": EOS})
        if self._value(top, "kind", rng) == "P":
            return [self._value(top, "first", rng)]
        path = [f"c{n.id}" for n in self.ctx.ancestors(top)[::-1][1:]][:self.depth]
        out: List[str] = []

        def expand(own_path: List[str], depth: int) -> bool:
            if depth > max_depth or len(out) > max_len:
                return False
            partial = {f"S{d}": v for d, v in enumerate(own_path[:self.gen_depth], 1)}
            leaf = self._sample_leaf(self.cnt, partial)
            for side in ("L", "R"):
                v = self._value(leaf, side, rng)
                sub = [self._value(leaf, f"{side}{d}", rng) for d in range(1, self.depth + 1)]
                if v == COMPOSITE:
                    if not expand(sub, depth + 1):
                        return False
                elif v != "<P>":
                    out.append(v)
                else:
                    out.append(self._word(sub))
            return True
        return out if expand(path, 0) and len(out) <= max_len else None
