"""TRELLIS v3: v1's two hierarchies with chunk context, representations donated
between them, and inside-outside parsing.

Iterates directly on TRELLIS v1 (ACS paper): every element has a context
description (how it sits in the experience) and, if composite, a content
description (what it is made of); each kind is organised by its own Cobweb
hierarchy. v3 strengthens what each description holds and lets each hierarchy
name things in the other's terms:

* context instance: the surface window (two tokens either side, over the
  element's outermost positions); chunk context — the context concepts of its
  two children and, when the element is in the experience's analysis, of its
  parent and sibling; and its own content concept (donated from the content
  hierarchy);
* content instance: each part's context concept (as in v1), each part's content
  concept (donated back), and each part's height (v1's complexity tag).

Concepts are named at depths 1..D of their hierarchy, so a description can
match coarsely or finely. A composite's parts are the best sub-analyses of
their spans, found bottom-up.

Parsing is inside-outside over every analysis: each composite (span and split)
carries a factor 2^(gain / tau), its gain being how much better its content and
context descriptions are coded by the concepts they sort to than by the roots;
the posterior of every composite follows, and the reported analysis maximises
the expected number of correct composites. Descriptions that need an analysis
(parent, sibling) are left out while parsing: Cobweb sorts partial instances.

Learning stores, per experience, every word and (``store``) every composite
weighted by its posterior, every composite equally, or the best analysis only.
Stored descriptions are refreshed as the concepts they name move (bounded per
experience).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS, BLANK = "<s>", "</s>", "-"
LN2 = math.log(2.0)
CONST = "const"      # passive: constituent (posterior) mass of the stored elements


def log_prob(node: CobwebNode, x: dict, root: CobwebNode, alpha: float) -> float:
    """ln P(x | node), Laplace-smoothed, attributes independent; x may be partial."""
    total = 0.0
    for a, v in x.items():
        d = node.av.get(a, {})
        size = len(root.av.get(a, {})) + 1
        if isinstance(v, dict):        # a bag: the expected log probability of its members
            total += sum(p * math.log((d.get(u, 0.0) + alpha) / (node.count + alpha * size)) for u, p in v.items())
        else:
            total += math.log((d.get(v, 0.0) + alpha) / (node.count + alpha * size))
    return total


def logsumexp(xs: List[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Elem:
    __slots__ = ("span", "split", "token", "left", "right", "ctx", "cnt", "leaf", "cleaf", "weight",
                 "height", "gain", "par", "sib", "memo", "const")

    def __init__(self, span, split=None, token=None, left=None, right=None):
        self.span, self.split, self.token, self.left, self.right = span, split, token, left, right
        self.ctx = self.cnt = None
        self.leaf = self.cleaf = None
        self.weight = 0.0
        self.height = 0 if token is not None else 1 + max(left.height, right.height)
        self.gain = 0.0
        self.par = self.sib = None          # context concept ids of parent / sibling in the analysis
        self.memo = {}                      # concept ids found by sorting (valid while unstored)
        self.const = 0.0                    # its posterior of being a constituent (store "ccm")


class Trellis3:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, window: int = 2,
                 store: str = "posterior", tau: float = 1.0, chunk_context: bool = True,
                 donate: bool = True, refresh_budget: Optional[int] = 3000, min_weight: float = 1e-3,
                 bags: bool = True, children: Optional[bool] = None, spine: Optional[bool] = None,
                 neighbour_concepts: bool = False, donate_ctx: Optional[bool] = None,
                 donate_cnt: Optional[bool] = None, factor: str = "gain", support: float = 3.0):
        self.depth, self.alpha, self.window, self.tau = depth, alpha, window, tau
        self.store, self.chunk_context, self.donate = store, chunk_context, donate
        self.refresh_budget, self.min_weight = refresh_budget, min_weight
        # bags: each facet that names a concept (a part, a child, the parent, the
        # sibling, the own content) is ONE attribute whose value is the bag of its
        # concept ids at depths 1..D, so every facet counts once in category
        # utility however many levels it names (else one attribute per depth).
        self.bags = bags
        # chunk context in two facets: the children's context concepts (fixed by
        # the chunk itself) and the spine, parent and sibling (from the analysis)
        self.children = chunk_context if children is None else children
        self.spine = chunk_context if spine is None else spine
        # the window at the level of categories: the context concepts of the
        # tokens just left and right of the span (known while parsing)
        self.neighbour_concepts = neighbour_concepts
        # donation's two directions: the element's own content concept into its
        # context; the parts' content concepts into the content
        self.donate_ctx = donate if donate_ctx is None else donate_ctx
        self.donate_cnt = donate if donate_cnt is None else donate_cnt
        self._word_ids: Dict[Tuple[tuple, int], List[str]] = {}
        # a chunk's factor in inside-outside: "gain" (against the roots) or
        # "code": log2 of the chunk's probability under memory, max over its
        # sorting path of share(C) P(x|C), in both hierarchies. Every full binary
        # analysis has n - 1 chunks, so a tree's weight is then the joint
        # probability of its chunks' descriptions, with no per-chunk baseline.
        self.factor = factor
        # "ccm" (Klein & Manning's constituent-context contrast, over Cobweb's
        # concepts): log2 [(constituent share of C) / (non-constituent share of
        # C)] at the deepest concept on the description's path holding at least
        # ``support`` elements, in both hierarchies; words count as constituents
        self.support = support
        d = [""] if bags else [str(k) for k in range(1, depth + 1)]
        surface = tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))
        chunk = (tuple(f"{p}{k}" for p in ("cl", "cr") for k in d) if self.children else ()) + \
            (tuple(f"{p}{k}" for p in ("par", "sib") for k in d) if self.spine else ())
        own = tuple(f"cc{k}" for k in d) if self.donate_ctx else ()
        neigh = tuple(f"{p}{k}" for p in ("lc", "rc") for k in d) if neighbour_concepts else ()
        self.ctx_attrs = surface + neigh + chunk + own
        parts = tuple(f"{p}{k}" for p in ("L", "R") for k in d)
        given = tuple(f"{p}{k}" for p in ("LC", "RC") for k in d) if self.donate_cnt else ()
        self.cnt_attrs = parts + given + ("LH", "RH")
        self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
        self.cnt = IncrementalCobwebTree(self.cnt_attrs, seed=seed + 1)
        self.rng = random.Random(seed)
        self.records: List[Elem] = []
        self._rr = 0
        self.stats = {"experiences": 0, "words stored": 0, "composites stored": 0, "refreshed": 0}

    # ------------------------------------------------------------------ #
    # Descriptions
    # ------------------------------------------------------------------ #
    def _surface(self, tokens, i, j) -> dict:
        x = {}
        for k in range(1, self.window + 1):
            x[f"l{k}"] = tokens[i - k] if i - k >= 0 else BOS
            x[f"r{k}"] = tokens[j + k - 1] if j + k - 1 < len(tokens) else EOS
        return x

    def _ids(self, path: List[CobwebNode]) -> List[str]:
        ids = [f"c{n.id}" for n in path[1:self.depth + 1]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _ctx_ids(self, e: Elem) -> List[str]:
        """An element's context concept, now: from its stored leaf, or by sorting."""
        if e.leaf is not None:
            return self._ids(self.ctx.ancestors(e.leaf)[::-1])
        if "ctx" not in e.memo:
            e.memo["ctx"] = self._ids(self.ctx.categorize(e.ctx))
        return e.memo["ctx"]

    def _cnt_ids(self, e: Elem) -> List[str]:
        if e.token is not None:
            return [BLANK] * self.depth
        if e.cleaf is not None:
            return self._ids(self.cnt.ancestors(e.cleaf)[::-1])
        if "cnt" not in e.memo:
            e.memo["cnt"] = self._ids(self.cnt.categorize(e.cnt))
        return e.memo["cnt"]

    def _facet(self, x: dict, name: str, ids: List[str]) -> None:
        """Write a facet's concept ids: one bag attribute, or one attribute per depth."""
        if self.bags:
            real = [v for v in ids if v != BLANK]
            x[name] = {v: 1.0 / len(real) for v in real} if real else BLANK
        else:
            for k, v in enumerate(ids, 1):
                x[f"{name}{k}"] = v

    def _content(self, e: Elem) -> dict:
        x = {}
        for p, part in (("L", e.left), ("R", e.right)):
            self._facet(x, p, self._ctx_ids(part))
            if self.donate_cnt:
                self._facet(x, f"{p}C", self._cnt_ids(part))
            x[f"{p}H"] = str(min(part.height, 4))
        return x

    def _context(self, tokens, e: Elem, analysed: bool) -> dict:
        """The context description; with ``analysed`` False the parts that need an
        analysis (parent, sibling) are left out (a partial instance)."""
        x = self._surface(tokens, *e.span)
        if self.neighbour_concepts:
            i, j = e.span
            for p, pos in (("lc", i - 1), ("rc", j)):
                self._facet(x, p, self._word_concept(tokens, pos))
        if self.children:
            for p, part in (("cl", e.left), ("cr", e.right)):
                self._facet(x, p, self._ctx_ids(part) if part is not None else [BLANK] * self.depth)
        if self.spine:
            if analysed:
                for p, ids in (("par", e.par), ("sib", e.sib)):
                    self._facet(x, p, ids or [BLANK] * self.depth)
        if self.donate_ctx:
            self._facet(x, "cc", self._cnt_ids(e))
        return x

    def _word_concept(self, tokens, pos: int) -> List[str]:
        """The context concept of the token at ``pos`` (sorting its surface window);
        blank outside the experience."""
        if pos < 0 or pos >= len(tokens):
            return [BLANK] * self.depth
        x = self._surface(tokens, pos, pos + 1)
        return self._ids(self.ctx.categorize(x))

    def _full(self, x: dict, attrs) -> dict:
        """Complete an instance for storage (missing attributes blank)."""
        return {a: x.get(a, BLANK) for a in attrs}

    # ------------------------------------------------------------------ #
    # Scoring and inside-outside
    # ------------------------------------------------------------------ #
    def _code(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """max over x's sorting path (root included) of log2 [share(C) P(x|C)]."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        best = -math.inf
        for node in tree.categorize(x):
            if node.count <= 0:
                continue
            best = max(best, (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha)) / LN2)
        return best

    def _contrast(self, tree: IncrementalCobwebTree, x: dict) -> float:
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for n in tree.categorize(x)[1:]:
            if n.count >= self.support:
                node = n
        c = lambda n: sum(n.pav.get(CONST, {}).values())
        cC, cR = c(node), c(root)
        dC, dR = max(node.count - cC, 0.0), max(root.count - cR, 0.0)
        eps = 0.5
        return math.log2(((cC + eps) / (cR + eps)) / ((dC + eps) / (dR + eps)))

    def _gain(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """Best over x's sorting path of log2 [share(C) P(x|C) / P(x|root)]; 0 at the root."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        base = log_prob(root, x, root, self.alpha)
        best = 0.0
        for node in tree.categorize(x)[1:]:
            if node.count <= 0:
                continue
            g = (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - base) / LN2
            best = max(best, g)
        return best

    def _chart(self, tokens: List[str]):
        """All elements of an experience, bottom-up: words, then every composite
        (span, split) with its parts the best (Viterbi) sub-analyses; each
        composite's gain from both hierarchies."""
        n = len(tokens)
        words = []
        for i, w in enumerate(tokens):
            e = Elem((i, i + 1), token=w)
            e.ctx = self._context(tokens, e, analysed=False)
            words.append(e)
        best: Dict[Tuple[int, int], Tuple[float, Elem]] = {(i, i + 1): (0.0, words[i]) for i in range(n)}
        comps: Dict[Tuple[int, int, int], Elem] = {}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    a, b = best[(i, k)][1], best[(k, j)][1]
                    e = Elem((i, j), split=k, left=a, right=b)
                    e.cnt = self._content(e)
                    e.ctx = self._context(tokens, e, analysed=False)
                    if self.factor == "ccm":
                        e.gain = self._contrast(self.cnt, e.cnt) + self._contrast(self.ctx, e.ctx)
                    elif self.factor == "code":
                        e.gain = self._code(self.cnt, e.cnt) + self._code(self.ctx, e.ctx)
                    else:
                        e.gain = self._gain(self.cnt, e.cnt) + self._gain(self.ctx, e.ctx)
                    comps[(i, k, j)] = e
                    v = best[(i, k)][0] + best[(k, j)][0] + e.gain
                    if (i, j) not in best or v > best[(i, j)][0]:
                        best[(i, j)] = (v, e)
        return words, comps

    def _inside_outside(self, n: int, comps):
        """Log inside/outside over binary analyses with factors 2^(gain/tau);
        returns the posterior of every composite and log Z."""
        f = {key: e.gain * LN2 / self.tau for key, e in comps.items()}
        ins = {(i, i + 1): 0.0 for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                ins[(i, j)] = logsumexp([f[(i, k, j)] + ins[(i, k)] + ins[(k, j)] for k in range(i + 1, j)])
        out = {(0, n): 0.0}
        for length in range(n, 1, -1):
            for i in range(0, n - length + 1):
                j = i + length
                o = out.get((i, j), -math.inf)
                for k in range(i + 1, j):
                    t = o + f[(i, k, j)]
                    for span, other in (((i, k), ins[(k, j)]), ((k, j), ins[(i, k)])):
                        out[span] = logsumexp([out.get(span, -math.inf), t + other])
        logZ = ins[(0, n)]
        post = {key: math.exp(f[key] + ins[(key[0], key[1])] + ins[(key[1], key[2])] + out.get((key[0], key[2]), -math.inf) - logZ)
                for key in comps}
        return post, logZ

    def _mbr(self, n: int, post) -> Tuple[Tree, List[Tuple[int, int, int]]]:
        """The analysis maximising the expected number of correct composites."""
        span_post: Dict[Tuple[int, int], float] = {}
        for (i, k, j), p in post.items():
            span_post[(i, j)] = span_post.get((i, j), 0.0) + p
        best = {(i, i + 1): (0.0, None) for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                k = max(range(i + 1, j), key=lambda k: best[(i, k)][0] + best[(k, j)][0] + post[(i, k, j)])
                best[(i, j)] = (best[(i, k)][0] + best[(k, j)][0] + span_post[(i, j)], k)
        split, chosen = {}, []

        def walk(i, j):
            if j - i < 2:
                return
            k = best[(i, j)][1]
            split[(i, j)] = k
            chosen.append((i, k, j))
            walk(i, k)
            walk(k, j)
        walk(0, n)
        return Tree(n, split, {}, [(0, n)]), chosen

    def parse(self, tokens: Sequence[str]) -> dict:
        tokens = list(tokens)
        n = len(tokens)
        words, comps = self._chart(tokens)
        if n < 2:
            return {"tree": Tree(n, {}, {}, [(0, n)] if n else []), "words": words, "comps": comps,
                    "post": {}, "chosen": []}
        post, logZ = self._inside_outside(n, comps)
        tree, chosen = self._mbr(n, post)
        return {"tree": tree, "words": words, "comps": comps, "post": post, "chosen": chosen, "logZ": logZ}

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def learn(self, tokens: Sequence[str]) -> Tree:
        tokens = list(tokens)
        r = self.parse(tokens)
        words, comps, post, chosen = r["words"], r["comps"], r["post"], set(r["chosen"])
        # chunk context from the analysis: parent and sibling of its elements
        node_of = {(e.span[0], e.span[0] + 1): e for e in words}
        for key in chosen:
            node_of[(key[0], key[2])] = comps[key]
        for (i, k, j) in chosen:
            parent = comps[(i, k, j)]
            for child, sib in ((node_of[(i, k)], node_of[(k, j)]), (node_of[(k, j)], node_of[(i, k)])):
                child.par, child.sib = self._ctx_ids(parent), self._ctx_ids(sib)
        if self.store == "ccm":
            weights = {key: 1.0 for key in comps}
            for key, e in comps.items():
                e.const = post.get(key, 0.0)
        elif self.store == "posterior":
            weights = {key: p for key, p in post.items()}
        elif self.store == "all":
            weights = {key: 1.0 for key in comps}
        else:
            weights = {key: 1.0 for key in chosen}
        # words first, then composites bottom-up: content (so a context can name it), then context
        for e in words:
            e.weight = 1.0
            e.ctx = self._full(self._context(tokens, e, analysed=True), self.ctx_attrs)
            e.leaf = self.ctx.ifit(e.ctx, 1.0)
            if self.store == "ccm":
                e.const = 1.0
                self.ctx.add_passive(e.leaf, CONST, "c", 1.0)
            self.records.append(e)
            self.stats["words stored"] += 1
        first = len(self.records)
        for key in sorted(comps, key=lambda t: (t[2] - t[0], t)):
            w = weights.get(key, 0.0)
            if w < self.min_weight:
                continue
            e = comps[key]
            e.weight = w
            e.cnt = self._content(e)
            e.cleaf = self.cnt.ifit(e.cnt, w)
            e.ctx = self._full(self._context(tokens, e, analysed=key in chosen), self.ctx_attrs)
            e.leaf = self.ctx.ifit(e.ctx, w)
            if e.const > 0:
                self.cnt.add_passive(e.cleaf, CONST, "c", e.const)
                self.ctx.add_passive(e.leaf, CONST, "c", e.const)
            self.records.append(e)
            self.stats["composites stored"] += 1
        self.stats["experiences"] += 1
        self._refresh(first, tokens)
        return r["tree"]

    def _refresh(self, first: int, tokens) -> None:
        """Re-describe stored instances whose named concepts moved: the newest
        and a round-robin slice of the rest (``refresh_budget``)."""
        recs = self.records
        b = self.refresh_budget
        if b is None or first <= b:
            todo = recs[first:] if b is not None and first > b else recs
        else:
            old = recs[:first]
            s = self._rr % len(old)
            todo = recs[first:] + (old[s:] + old[:s])[:b]
            self._rr = (s + b) % len(old)
        for e in todo:
            if e.token is None:
                new = self._content(e)
                if new != e.cnt:
                    if e.const > 0:
                        self.cnt.add_passive(e.cleaf, CONST, "c", -e.const)
                    self.cnt.unfit(e.cleaf, e.cnt, e.weight)
                    e.cnt, e.cleaf = new, self.cnt.ifit(new, e.weight)
                    if e.const > 0:
                        self.cnt.add_passive(e.cleaf, CONST, "c", e.const)
                    self.stats["refreshed"] += 1
            # the context's named concepts (children, own content) follow too
            new = dict(e.ctx)
            if self.children and e.token is None:
                for p, part in (("cl", e.left), ("cr", e.right)):
                    self._facet(new, p, self._ctx_ids(part))
            if self.donate_ctx:
                self._facet(new, "cc", self._cnt_ids(e))
            if new != e.ctx:
                if e.const > 0:
                    self.ctx.add_passive(e.leaf, CONST, "c", -e.const)
                self.ctx.unfit(e.leaf, e.ctx, e.weight)
                e.ctx, e.leaf = new, self.ctx.ifit(new, e.weight)
                if e.const > 0:
                    self.ctx.add_passive(e.leaf, CONST, "c", e.const)
                self.stats["refreshed"] += 1
