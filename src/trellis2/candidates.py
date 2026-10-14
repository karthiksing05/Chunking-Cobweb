"""TRELLIS v3: candidate trees (docs/V3_DESIGN.md).

Learning is greedy parsing that stores every candidate: at each step every new
adjacent pair on the frontier is stored in both Cobweb trees, then each pair is
scored by the concepts it now sits under, and the best positive one is
committed. Nothing is distilled: scores, parses and generation are computed
from the trees' counts when needed.

The representation follows the ACS paper's unsupervised one. An element's
context instance is only its context: the two tokens on either side of its
span. A hidden content reference (the token of a primitive, the content leaf of
a composite) is stored with it, never used to sort, so a context concept's type
(word-like or phrase-like) is read off the references under it. A composite's
content instance names its two parts by their paths in the context tree.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS = "<s>", "</s>"
BLANK = "-"
LN2 = math.log(2.0)
CTX_ATTRS = ("l2", "l1", "r1", "r2")
REF = "ref"
USED = "used"      # hidden: references used by committed parses (what generation regenerates from)


def content_attrs(depth: int) -> Tuple[str, ...]:
    """L1..LD, R1..RD: the context-tree path of each part (depths 1..D)."""
    return tuple(f"L{k}" for k in range(1, depth + 1)) + tuple(f"R{k}" for k in range(1, depth + 1))


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
    ref: Optional[tuple] = None            # the hidden content reference stored with its context
    used: bool = False                     # committed in its experience's parse

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
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, refresh: bool = True,
                 min_count: float = 0.0, lookahead: bool = False, context_gain: bool = True,
                 window: int = 2, gen_support: int = 2):
        self.depth, self.alpha, self.window = depth, alpha, window
        self.ctx_attrs = tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))
        self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
        self.cnt = IncrementalCobwebTree(content_attrs(depth), seed=seed + 1)
        self.rng = random.Random(seed)
        # refresh: after each experience, every stored content instance is
        # re-described from its parts' current paths (dynamic recomputation over
        # stored leaves). min_count: the support a concept needs to be proposed.
        # lookahead: each step commits the candidate whose greedy completion of
        # the parse compresses the experience most. context_gain: candidates are
        # scored against both trees (P4).
        self.refresh, self.min_count = refresh, min_count
        self.lookahead, self.context_gain = lookahead, context_gain
        self.records: List[Element] = []
        self.cleaves: Dict[int, CobwebNode] = {}
        self.xleaves: Dict[int, CobwebNode] = {}
        # Generation expands a part from the smallest concept above the context
        # leaf it came from that holds at least gen_support distinct used references.
        self.gen_support = gen_support
        self.stats = {"experiences": 0, "candidates stored": 0, "committed": 0, "re-described": 0}

    # Descriptions --------------------------------------------------------- #
    def _ctx_instance(self, tokens: Sequence[str], span: Tuple[int, int]) -> dict:
        """The window around the span's outermost positions."""
        i, j = span
        x = {}
        for k in range(1, self.window + 1):
            x[f"l{k}"] = tokens[i - k] if i - k >= 0 else BOS
            x[f"r{k}"] = tokens[j + k - 1] if j + k - 1 < len(tokens) else EOS
        return x

    def _ids(self, path: List[CobwebNode]) -> List[str]:
        ids = [f"c{n.id}" for n in path[1:self.depth + 1]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _ctx_path(self, el: Element) -> List[str]:
        """The element's path in the context tree, now."""
        path = self.ctx.ancestors(el.leaf)[::-1] if el.leaf is not None else self.ctx.categorize(el.ctx)
        return self._ids(path)

    def _content(self, c: Element) -> dict:
        x = {}
        for p, el in (("L", c.left), ("R", c.right)):
            for d, v in enumerate(self._ctx_path(el), 1):
                x[f"{p}{d}"] = v
        return x

    def node_type(self, node: CobwebNode) -> float:
        """The share of a context concept's elements that are primitives (read
        off the hidden references under it)."""
        refs = self.ctx.hidden_counts(node, REF)
        tot = sum(refs.values())
        return sum(m for r, m in refs.items() if r[0] == "tok") / tot if tot else 0.0

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
        c = Element(span, self._ctx_instance(tokens, span), left=a, right=b)
        c.content = self._content(c)
        return c

    @staticmethod
    def _parts(c: Element) -> dict:
        """Hidden at the content leaf: the context leaves its parts came from."""
        return {"Lpart": (c.left.leaf.id, c.left.ref), "Rpart": (c.right.leaf.id, c.right.ref)}

    def _store(self, c: Element) -> None:
        """A candidate into both trees: its content, then its context with the
        hidden reference to the content leaf."""
        c.content = self._content(c)
        c.cleaf = self.cnt.ifit(c.content)
        self.cleaves[c.cleaf.id] = c.cleaf
        self.cnt.hide(c.cleaf, self._parts(c))
        c.ref = ("cnt", c.cleaf.id)
        c.leaf = self.ctx.ifit(c.ctx)
        self.xleaves[c.leaf.id] = c.leaf
        self.ctx.hide(c.leaf, {REF: c.ref})
        self.records.append(c)
        self.stats["candidates stored"] += 1

    # Parsing -------------------------------------------------------------- #
    def parse(self, tokens: Sequence[str], learn: bool = False) -> Tuple[Tree, List[Element]]:
        tokens = list(tokens)
        frontier = [Element((k, k + 1), self._ctx_instance(tokens, (k, k + 1)), token=w)
                    for k, w in enumerate(tokens)]
        if learn:
            for el in frontier:
                el.ref = ("tok", el.token)
                el.leaf = self.ctx.ifit(el.ctx)
                self.xleaves[el.leaf.id] = el.leaf
                self.ctx.hide(el.leaf, {REF: el.ref, USED: el.ref})
        made: Dict[Tuple[int, int], Element] = {}
        while len(frontier) > 1:
            new = []
            for a, b in zip(frontier, frontier[1:]):
                if (id(a), id(b)) not in made:
                    c = self._candidate(tokens, a, b)
                    made[(id(a), id(b))] = c
                    new.append(c)
            if learn:
                for c in new:
                    self._store(c)
            pairs = [(made[(id(a), id(b))], t) for t, (a, b) in enumerate(zip(frontier, frontier[1:]))]
            for c, _ in (pairs if learn else [(c, None) for c in new]):
                self._score(c)                     # by the trees as they are now
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
                self.ctx.hide(best.leaf, {USED: best.ref})
                self.cnt.hide(best.cleaf, {USED: True})
                best.used = True
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
        """Re-describe every stored content instance whose parts' paths changed;
        its context's hidden reference follows it to its new content leaf."""
        for c in self.records:
            new = self._content(c)
            if new == c.content:
                continue
            self.cnt.unfit(c.cleaf, c.content)
            self.cnt.hide(c.cleaf, self._parts(c), -1.0)
            if c.used:
                self.cnt.hide(c.cleaf, {USED: True}, -1.0)
            c.content, c.cleaf = new, self.cnt.ifit(new)
            self.cnt.hide(c.cleaf, self._parts(c))
            if c.used:
                self.cnt.hide(c.cleaf, {USED: True})
            self.cleaves[c.cleaf.id] = c.cleaf
            ref = ("cnt", c.cleaf.id)
            if ref != c.ref:
                self.ctx.hide(c.leaf, {REF: c.ref}, -1.0)
                self.ctx.hide(c.leaf, {REF: ref})
                if c.used:
                    self.ctx.hide(c.leaf, {USED: c.ref}, -1.0)
                    self.ctx.hide(c.leaf, {USED: ref})
                c.ref = ref
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

    def _sample_used_leaf(self, partial: dict, start: Optional[CobwebNode]) -> CobwebNode:
        """Like _sample_leaf, with each child weighted by how much of it the
        committed parses used (hidden counts), times P(partial | child)."""
        node = start or self.ctx.root
        while node.children:
            ws = [sum(self.ctx.hidden_counts(c, USED).values()) * math.exp(log_prob(c, partial, self.ctx.root, self.alpha))
                  for c in node.children]
            if sum(ws) <= 0:
                ws = [c.count * math.exp(log_prob(c, partial, self.ctx.root, self.alpha)) for c in node.children]
            node = self.rng.choices(node.children, weights=ws)[0]
        return node

    def _choice(self, d: Dict) :
        if not d:
            return None
        vals, ws = zip(*sorted(d.items(), key=lambda kv: str(kv[0])))
        return self.rng.choices(vals, weights=ws)[0]

    def _basic(self, leaf: Optional[CobwebNode]) -> Optional[CobwebNode]:
        """The smallest concept above a part's context leaf holding at least
        gen_support distinct used references (the part's basic level)."""
        node = leaf
        while node is not None and node.parent is not None and \
                len(self.ctx.hidden_counts(node, USED)) < self.gen_support:
            node = node.parent
        return node

    def _deepest(self, ids: List[str]) -> Optional[CobwebNode]:
        nodes = {f"c{n.id}": n for n in self.ctx.nodes()}
        out = None
        for v in ids:
            if v in nodes:
                out = nodes[v]
        return out

    def _used(self, tree: IncrementalCobwebTree, node: CobwebNode) -> Dict:
        return tree.hidden_counts(node, USED)

    def _climb(self, tree: IncrementalCobwebTree, leaf: CobwebNode, enough) -> CobwebNode:
        """The smallest concept above ``leaf`` for which ``enough(node)`` holds
        (the part's basic level for recombination)."""
        node = leaf
        while node.parent is not None and not enough(node):
            node = node.parent
        return node

    def _sample_word(self, xleaf: CobwebNode, left: dict) -> Optional[str]:
        """Another word for a word part: from the basic level above its context
        leaf (enough distinct used words), leaves weighted by used words times
        P(left context | leaf)."""
        words = lambda n: {r: m for r, m in self._used(self.ctx, n).items() if r[0] == "tok"}
        node = self._climb(self.ctx, xleaf, lambda n: len(words(n)) >= self.gen_support)
        leaves = [n for n in self._leaves(node) if words(n)]
        if not leaves:
            return None
        ws = [sum(words(n).values()) * math.exp(log_prob(n, left, self.ctx.root, self.alpha)) for n in leaves]
        leaf = self.rng.choices(leaves, weights=ws)[0]
        return self._choice(words(leaf))[1]

    def _sample_phrase(self, cleaf: CobwebNode) -> Optional[CobwebNode]:
        """Another composition for a phrase part: from the basic level above its
        content leaf (enough committed compositions), a used content leaf."""
        used = lambda n: sum(self._used(self.cnt, n).values())
        node = self._climb(self.cnt, cleaf, lambda n: len([x for x in self._leaves(n) if used(x)]) >= self.gen_support)
        leaves = [n for n in self._leaves(node) if used(n)]
        if not leaves:
            return None
        return self.rng.choices(leaves, weights=[used(n) for n in leaves])[0]

    @staticmethod
    def _leaves(node: CobwebNode) -> List[CobwebNode]:
        out, stack = [], [node]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children)
            else:
                out.append(n)
        return out

    def generate(self, max_depth: int = 12, max_len: int = 40) -> Optional[List[str]]:
        """One experience sampled top-down, left to right, by recombination: a
        composition used as a whole experience; each of its parts replaced by
        material of the same type from the part's basic level (a word from the
        context tree given the words already generated, a phrase from the
        content tree, expanded the same way)."""
        out: List[str] = []

        def left() -> dict:
            return {f"l{k}": out[-k] if len(out) >= k else BOS for k in range(1, self.window + 1)}

        def expand(cleaf: CobwebNode, depth: int) -> bool:
            if depth > max_depth or len(out) > max_len:
                return False
            for side in ("L", "R"):
                part = self._choice(cleaf.hidden.get(f"{side}part", {}))
                if part is None:
                    return False
                xleaf_id, ref = part
                if ref[0] == "tok":
                    w = self._sample_word(self.xleaves[xleaf_id], left())
                    if w is None:
                        return False
                    out.append(w)
                else:
                    sub = self._sample_phrase(self.cleaves[ref[1]])
                    if sub is None or not expand(sub, depth + 1):
                        return False
            return True
        whole = {f"l{k}": BOS for k in range(1, self.window + 1)}
        whole.update({f"r{k}": EOS for k in range(1, self.window + 1)})
        top = self._sample_used_leaf(whole, None)
        ref = self._choice(top.hidden.get(USED) or top.hidden.get(REF, {}))
        if ref is None:
            return None
        if ref[0] == "tok":
            return [ref[1]]
        return out if expand(self.cleaves[ref[1]], 0) and len(out) <= max_len else None
