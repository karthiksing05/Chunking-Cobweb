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
    score: float = 0.0                     # gain against the root: positive = the chunk compresses
    rank: float = 0.0                      # what the parser chooses by (objective)
    leaf: Optional[CobwebNode] = None      # its context instance's leaf (a stable handle)
    cleaf: Optional[CobwebNode] = None     # its content instance's leaf
    ref: Optional[tuple] = None            # the hidden content reference stored with its context
    used: bool = False                     # committed in its experience's parse
    idx: int = -1                          # its index among the stored records (stable)

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
               min_count: float = 0.0, weight=None) -> Tuple[float, Optional[CobwebNode]]:
    """Best gain (bits) along a root-first path, and its concept:
    log2 share of the concept (strength) + log2 P(x|C) - log2 P(x|root) (fit);
    0 at the root, so a positive gain means coding x through C compresses."""
    if root.count <= 0:
        return 0.0, None
    base = log_prob(root, x, root, alpha)
    best, arg = 0.0, None
    total = weight(root) if weight else root.count
    if total <= 0:
        return 0.0, None
    for node in path[1:]:
        share = weight(node) if weight else node.count
        if node.count < min_count or node.count <= 0 or share <= 0:
            continue
        g = (math.log(share / total) + log_prob(node, x, root, alpha) - base) / LN2
        if g > best:
            best, arg = g, node
    return best, arg


class CandidateLearner:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, refresh: bool = True,
                 min_count: float = 0.0, context_gain: bool = True,
                 window: int = 2, gen_support: int = 2, substitution: bool = True,
                 rehearse: int = 0, beam: int = 1, enrich: bool = False, support: float = 3.0,
                 enrich_weight: float = 1.0, objective: str = "gain", candidates: str = "frontier",
                 redescribe_budget: Optional[int] = None):
        self.depth, self.alpha, self.window = depth, alpha, window
        self.ctx_attrs = tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))
        self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
        self.cnt = IncrementalCobwebTree(content_attrs(depth), seed=seed + 1)
        self.rng = random.Random(seed)
        # refresh: after each experience, every stored content instance is
        # re-described from its parts' current paths (dynamic recomputation over
        # stored leaves). min_count: the support a concept needs to be proposed.
        # context_gain: candidates are
        # scored against both trees (P4).
        self.refresh, self.min_count = refresh, min_count
        self.context_gain = context_gain
        self.records: List[Element] = []
        self.cleaves: Dict[int, CobwebNode] = {}
        self.xleaves: Dict[int, CobwebNode] = {}
        # Generation expands a part from the smallest concept above the context
        # leaf it came from that holds at least gen_support distinct used references.
        self.gen_support = gen_support
        # substitution (Harris): a candidate's surroundings count as evidence for
        # it in proportion to the units in use (words and committed chunks) that
        # occupy the same context concept, not to every stored candidate.
        self.substitution = substitution
        # rehearse: earlier experiences re-parsed after each new one (their use
        # follows the trees as they grow). beam: parses kept per step (1: greedy).
        self.rehearse_n, self.beam = rehearse, beam
        # enrich: the substitution test as a ratio: log2 of how over-represented
        # units in use are, against all stored candidates, in the most specific
        # context concept on the candidate's path holding at least ``support``
        # elements (negative where the surroundings mostly host candidates never used).
        self.enrich, self.support, self.enrich_weight = enrich, support, enrich_weight
        # objective: "gain" chooses by the gain against the root (summed over a tree
        # it rewards parts rare at the root); "code" chooses by the composite's code
        # length through its concept, -log2 [share(C) P(x|C)], whose sum over a
        # tree is the tree's description length (every full binary tree over n
        # tokens has n - 1 composites); the gain still decides whether to chunk.
        self.objective = objective
        # candidates: "frontier" stores the pairs on the parser's own frontier;
        # "all" stores every composite of an experience (every span, every split),
        # so what is stored does not depend on the parser's own early choices.
        # redescribe_budget: stored records re-described per experience (the new
        # ones plus a round-robin slice of the rest; None: all).
        self.candidates, self.redescribe_budget = candidates, redescribe_budget
        self._rr = 0
        self.experiences: List[dict] = []
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
        if self.enrich:
            xpath = self.ctx.ancestors(c.leaf)[::-1] if c.leaf is not None else self.ctx.categorize(c.ctx)
            g += self.enrich_weight * self._enrichment(xpath)
        elif self.context_gain:
            xpath = self.ctx.ancestors(c.leaf)[::-1] if c.leaf is not None else self.ctx.categorize(c.ctx)
            used = (lambda n: sum(n.pav.get(USED, {}).values())) if self.substitution else None
            g += gain_along(xpath, c.ctx, self.ctx.root, self.alpha, self.min_count, used)[0]
        c.score, c.concept = g, node
        if self.objective == "code":
            root = self.cnt.root
            best = -math.inf
            for n in cpath[1:]:
                if n.count <= 0 or n.count < self.min_count:
                    continue
                r = (math.log(n.count / root.count) + log_prob(n, c.content, root, self.alpha)) / LN2
                if r > best:
                    best, c.concept = r, n
            c.rank = best if best > -math.inf else (log_prob(root, c.content, root, self.alpha) / LN2 if root.count > 0 else 0.0)
            if self.context_gain and not self.enrich:
                c.rank += g - gain_along(cpath, c.content, root, self.alpha, self.min_count)[0]
        else:
            c.rank = g

    def _enrichment(self, path: List[CobwebNode]) -> float:
        """log2 [(used_C / used_root) / (n_C / n_root)] at the deepest concept on
        the path with at least ``support`` elements (Laplace-smoothed)."""
        root = self.ctx.root
        u = lambda n: sum(n.pav.get(USED, {}).values())
        node = root
        for n in path[1:]:
            if n.count >= self.support:
                node = n
        if root.count <= 0:
            return 0.0
        return math.log2(((u(node) + 1.0) / (u(root) + 1.0)) / ((node.count + 1.0) / (root.count + 1.0)))

    def _candidate(self, tokens, a: Element, b: Element) -> Element:
        span = (a.span[0], b.span[1])
        c = Element(span, self._ctx_instance(tokens, span), left=a, right=b)
        c.content = self._content(c)
        return c

    @staticmethod
    def _parts(c: Element) -> dict:
        """Hidden at the content leaf: the context leaves its parts came from."""
        def part(el: Element):
            # a word by its token; a composite by its record (its content leaf
            # moves when it is re-described, the record does not)
            return (el.leaf.id, ("tok", el.token) if el.is_primitive else ("rec", el.idx))
        return {"Lpart": part(c.left), "Rpart": part(c.right)}

    def _store(self, c: Element, share: Optional[Element] = None) -> None:
        """A candidate into both trees: its content, then its context with the
        hidden reference to the content leaf (with ``share``, the context leaf of
        another composite of the same span: a span's context is stored once)."""
        c.idx = len(self.records)
        c.content = self._content(c)
        c.cleaf = self.cnt.ifit(c.content)
        self.cleaves[c.cleaf.id] = c.cleaf
        self.cnt.hide(c.cleaf, self._parts(c))
        c.ref = ("cnt", c.cleaf.id)
        c.leaf = share.leaf if share is not None else self.ctx.ifit(c.ctx)
        self.xleaves[c.leaf.id] = c.leaf
        self.ctx.hide(c.leaf, {REF: c.ref})
        self.records.append(c)
        self.stats["candidates stored"] += 1

    # Parsing -------------------------------------------------------------- #
    def _primitives(self, tokens: List[str], store: bool) -> List[Element]:
        prims = [Element((k, k + 1), self._ctx_instance(tokens, (k, k + 1)), token=w) for k, w in enumerate(tokens)]
        if store:
            for el in prims:
                el.ref = ("tok", el.token)
                el.leaf = self.ctx.ifit(el.ctx)
                self.xleaves[el.leaf.id] = el.leaf
                self.ctx.hide(el.leaf, {REF: el.ref, USED: el.ref})
                self.ctx.add_passive(el.leaf, USED, "word")
        return prims

    def _mark(self, c: Element, w: float) -> None:
        """Record (w = +1) or withdraw (w = -1) a composite's use in a committed parse."""
        self.ctx.hide(c.leaf, {USED: c.ref}, w)
        self.ctx.add_passive(c.leaf, USED, "unit", w)
        self.cnt.add_passive(c.cleaf, USED, "unit", w)
        c.used = w > 0

    def parse(self, tokens: Sequence[str], learn: bool = False, exp: Optional[dict] = None,
              force: Optional[Tree] = None) -> Tuple[Tree, List[Element]]:
        """Greedy parsing (or a beam, ``beam`` > 1, by the total gain). With
        ``learn``, every candidate met is stored (once per experience: a
        candidate is its span and split) and the committed parse is marked as
        used; ``exp`` re-parses a stored experience with the trees as they are."""
        tokens = list(tokens)
        if learn:
            if exp is None:
                exp = {"tokens": tokens, "prims": self._primitives(tokens, True), "cands": {}, "used": [],
                       "first": len(self.records)}
                self.experiences.append(exp)
                if self.candidates == "all":
                    self._store_all(tokens, exp)
            cands = exp["cands"]
            start = list(exp["prims"])
        else:
            cands = {}
            start = self._primitives(tokens, False)

        def candidate(a: Element, b: Element) -> Element:
            k = (a.span[0], a.span[1], b.span[1])
            c = cands.get(k)
            if c is None:
                c = self._candidate(tokens, a, b)
                cands[k] = c
                if learn:
                    self._store(c)
                else:
                    self._score(c)
            return c

        states = [(0.0, start, [])]             # (total gain, frontier, committed)
        finals = []
        width = max(1, self.beam)
        while states:
            grown, fresh = [], []
            for total, frontier, done in states:
                opts = [(candidate(a, b), t) for t, (a, b) in enumerate(zip(frontier, frontier[1:]))]
                fresh.append((total, frontier, done, opts))
            if learn:                           # stored first, then judged by the trees as they are
                for _, _, _, opts in fresh:
                    for c, _ in opts:
                        self._score(c)
            for total, frontier, done, opts in fresh:
                if force is not None:           # diagnostics only: commit a given tree's composites
                    gold = force.brackets()
                    pos = [(c, t) for c, t in opts if c.span in gold]
                else:
                    pos = [(c, t) for c, t in opts if c.score > 0.0]
                if not pos:
                    finals.append((total, frontier, done))
                    continue
                pos.sort(key=lambda p: (-p[0].rank, p[1]))
                for c, t in pos[:width]:
                    grown.append((total + c.rank, frontier[:t] + [c] + frontier[t + 2:], done + [c]))
            grown.sort(key=lambda s_: -s_[0])
            seen, states = set(), []
            for st in grown:
                sig = tuple(el.span for el in st[1])
                if sig not in seen:
                    seen.add(sig)
                    states.append(st)
                if len(states) >= width:
                    break
        total, frontier, done = max(finals, key=lambda s_: s_[0])
        if learn:
            for c in done:
                self._mark(c, 1.0)
            exp["used"] = list(done)
            self.stats["committed"] += len(done)
        split = {c.span: c.left.span[1] for c in done}
        label = {c.span: (f"n{c.concept.id}" if c.concept is not None else "?") for c in done}
        return Tree(len(tokens), split, label, [el.span for el in frontier]), frontier

    def _store_all(self, tokens: List[str], exp: dict) -> None:
        """Every composite of the experience, shortest spans first: each span's
        context stored once (its first split), every split's content stored."""
        n = len(tokens)
        rep_of = {el.span: el for el in exp["prims"]}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    c = self._candidate(tokens, rep_of[(i, k)], rep_of[(k, j)])
                    first = rep_of.get((i, j))
                    self._store(c, share=first)
                    exp["cands"][(i, k, j)] = c
                    if first is None:
                        rep_of[(i, j)] = c

    def rehearse(self, i: int) -> None:
        """Re-parse stored experience i with the trees as they are now: its old
        parse's use is withdrawn and the new parse's recorded (bounded work)."""
        exp = self.experiences[i]
        for c in exp["used"]:
            self._mark(c, -1.0)
        exp["used"] = []
        self.parse(exp["tokens"], learn=True, exp=exp)
        self.stats["rehearsed"] = self.stats.get("rehearsed", 0) + 1

    def _redescribe(self) -> None:
        """Re-describe every stored content instance whose parts' paths changed;
        its context's hidden reference follows it to its new content leaf."""
        todo = self.records
        b = self.redescribe_budget
        if b is not None and len(self.records) > b:
            first = self.experiences[-1]["first"] if self.experiences else len(self.records)
            old = self.records[:first]
            start = self._rr % max(len(old), 1)
            slice_ = (old[start:] + old[:start])[:b]
            self._rr = (start + b) % max(len(old), 1)
            todo = self.records[first:] + slice_
        for c in todo:
            new = self._content(c)
            if new == c.content:
                continue
            if c.used:
                self.cnt.add_passive(c.cleaf, USED, "unit", -1.0)
            self.cnt.hide(c.cleaf, self._parts(c), -1.0)
            self.cnt.unfit(c.cleaf, c.content)
            c.content, c.cleaf = new, self.cnt.ifit(new)
            self.cnt.hide(c.cleaf, self._parts(c))
            if c.used:
                self.cnt.add_passive(c.cleaf, USED, "unit")
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

    def learn(self, tokens: Sequence[str], force: Optional[Tree] = None) -> Tree:
        """One experience: parsed and stored; then ``rehearse`` earlier ones
        re-parsed; then stored descriptions brought up to date. ``force`` commits
        a given tree's composites instead (diagnostics only)."""
        tree = self.parse(tokens, learn=True, force=force)[0]
        self.stats["experiences"] += 1
        n = len(self.experiences)
        for _ in range(self.rehearse_n if n > 1 else 0):
            self.rehearse(self.rng.randrange(n - 1))
        if self.refresh:
            self._redescribe()
        return tree

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
            ws = [sum(c.pav.get(USED, {}).values()) * math.exp(log_prob(c, partial, self.ctx.root, self.alpha))
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
        used = lambda n: sum(n.pav.get(USED, {}).values())
        node = self._climb(self.cnt, cleaf, lambda n: used(n) >= self.gen_support)
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

        fail = self.stats.setdefault("generation failures", {})

        def why(reason: str) -> bool:
            fail[reason] = fail.get(reason, 0) + 1
            return False

        def expand(cleaf: CobwebNode, depth: int) -> bool:
            if depth > max_depth:
                return why("depth")
            if len(out) > max_len:
                return why("length")
            for side in ("L", "R"):
                part = self._choice(cleaf.hidden.get(f"{side}part", {}))
                if part is None:
                    return why("no parts recorded")
                xleaf_id, ref = part
                if ref[0] == "tok":
                    w = self._sample_word(self.xleaves[xleaf_id], left())
                    if w is None:
                        return why("no word")
                    out.append(w)
                else:
                    sub = self._sample_phrase(self.records[ref[1]].cleaf)
                    if sub is None:
                        return why("no phrase")
                    if not expand(sub, depth + 1):
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
