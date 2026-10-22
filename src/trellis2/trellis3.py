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
WORD = "word"        # passive: the word a context concept's word elements are (v1's content reference)


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
                 donate_cnt: Optional[bool] = None, factor: str = "gain", support: float = 3.0,
                 naming: str = "depth", heights: bool = True, donate_own: bool = False,
                 refresh: str = "refit", readout: str = "mbr", cc_naming: Optional[str] = None,
                 word_cc: bool = False, boundary: str = "marker"):
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
        # how a facet names a concept: "depth" (its concepts at depths 1..D) or
        # "path" (every concept on its path below the root, the own leaf left
        # out): shared ancestry at every granularity, no fixed depth; or "leaf":
        # the D concepts just above its own leaf (the substitutable granularity)
        self.naming = naming
        # the donated own-content facet in the context may be named differently
        # from the parts (it keeps fine detail; parts take one clean name)
        self.cc_naming = cc_naming or naming
        # a word's own content is the word itself: with word_cc its identity is
        # donated into its context description as the cc facet, as a chunk's
        # content concept is (otherwise a word's cc is blank)
        self.word_cc = word_cc
        # the sentence boundary in a window: "marker" (<s> / </s>, a value many
        # instances share) or "missing" (the slot is simply not observed)
        self.boundary = boundary
        # the parts' heights in the content (v1's complexity tag); without them a
        # part that recurses (an N' where an N stands) is named only by its concepts
        self.heights = heights
        # donation of the element's OWN context concept into its content (the
        # mirror of ``cc``): the context concept of its surface window, so a
        # content concept groups chunks by makeup and by where they occur
        self.donate_own = donate_own
        # how a stored instance whose named concepts moved is re-described:
        # "refit" (taken out and re-inserted) or "relabel" (its counts updated
        # in place on its own path; it stays where it was stored)
        self.refresh = refresh
        # the analysis read out of an experience: "mbr" (most expected correct
        # composites) or "regen" (the analysis whose chunks memory regenerates
        # most probably: the sum over its composites of log2 P(parts | content
        # concept), so unpacking the chunks gives back the experience)
        self.readout = readout
        d = [""] if bags else [str(k) for k in range(1, depth + 1)]
        surface = tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))
        chunk = (tuple(f"{p}{k}" for p in ("cl", "cr") for k in d) if self.children else ()) + \
            (tuple(f"{p}{k}" for p in ("par", "sib") for k in d) if self.spine else ())
        own = tuple(f"cc{k}" for k in d) if self.donate_ctx else ()
        neigh = tuple(f"{p}{k}" for p in ("lc", "rc") for k in d) if neighbour_concepts else ()
        self.ctx_attrs = surface + neigh + chunk + own + (("le", "re") if boundary == "edge" else ())
        parts = tuple(f"{p}{k}" for p in ("L", "R") for k in d)
        given = tuple(f"{p}{k}" for p in ("LC", "RC") for k in d) if self.donate_cnt else ()
        self.cnt_attrs = parts + given + (("LH", "RH") if heights else ()) + \
            (tuple(f"oc{k}" for k in d) if donate_own else ())
        self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
        self.cnt = IncrementalCobwebTree(self.cnt_attrs, seed=seed + 1)
        self.ctx.partial = self.cnt.partial = (boundary in ("missing", "edge"))
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
            if i - k >= 0:
                x[f"l{k}"] = tokens[i - k]
            elif self.boundary == "marker":
                x[f"l{k}"] = BOS
            if j + k - 1 < len(tokens):
                x[f"r{k}"] = tokens[j + k - 1]
            elif self.boundary == "marker":
                x[f"r{k}"] = EOS
        if self.boundary == "edge":
            # whether a neighbour exists on each side, once per side; the
            # neighbour slots past an edge stay unobserved
            x["le"] = "edge" if i == 0 else "in"
            x["re"] = "edge" if j == len(tokens) else "in"
        return x

    def is_sentence(self, e: Elem) -> bool:
        """A sentence-level element: nothing on either side."""
        if self.boundary == "edge":
            return e.ctx.get("le") == "edge" and e.ctx.get("re") == "edge"
        if self.boundary == "missing":
            return "l1" not in e.ctx and "r1" not in e.ctx
        return e.ctx.get("l1") == BOS and e.ctx.get("r1") == EOS

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

    def _ctx_name(self, e: Elem) -> List[str]:
        """The ids a facet names an element's context concept by."""
        if self.naming == "depth":
            return self._ctx_ids(e)
        if e.leaf is not None:
            path = self.ctx.ancestors(e.leaf)[::-1]
        else:
            if "ctxp" not in e.memo:
                e.memo["ctxp"] = self.ctx.categorize(e.ctx)
            path = e.memo["ctxp"]
        return self._path_ids(path)

    def _cnt_name(self, e: Elem, naming: Optional[str] = None) -> List[str]:
        naming = naming or self.naming
        if naming == "depth":
            return self._cnt_ids(e)
        if e.token is not None:
            return [BLANK]
        if e.cleaf is not None:
            path = self.cnt.ancestors(e.cleaf)[::-1]
        else:
            if "cntp" not in e.memo:
                e.memo["cntp"] = self.cnt.categorize(e.cnt)
            path = e.memo["cntp"]
        return self._path_ids(path, naming)

    def _path_ids(self, path, naming: Optional[str] = None) -> List[str]:
        naming = naming or self.naming
        inner = path[1:-1] if len(path) > 1 and not path[-1].children else path[1:]
        if naming == "leaf":
            # the D concepts nearest the element's own leaf (finest first)
            inner = inner[::-1][:self.depth]
        if naming == "basic" and inner:
            # the basic level (Corter & Gluck 1992): the concept on the path
            # maximising P(c) [sum P(v|c)^2 - sum P(v|root)^2], one categorical name
            root = path[0]
            base = root.sstot / max(root.count, 1e-9) ** 2

            def cu(n):
                return (n.count / max(root.count, 1e-9)) * (n.sstot / max(n.count, 1e-9) ** 2 - base)
            return [f"c{max(inner, key=cu).id}"]
        if naming == "ipath" and inner:
            # each concept weighted by its information, -log2 of its share of
            # the tree in whole bits: shared coarse ancestry counts little
            total = max(path[0].count, 1e-9)
            return [f"c{n.id}" for n in inner
                    for _ in range(max(1, round(-math.log2(max(n.count, 1e-9) / total))))]
        return [f"c{n.id}" for n in inner] or [BLANK]

    def _cc_ids(self, e: Elem) -> List[str]:
        if e.token is not None and self.word_cc:
            return [f"w:{e.token}"]
        return self._cnt_name(e, self.cc_naming)

    def _facet(self, x: dict, name: str, ids: List[str]) -> None:
        """Write a facet's concept ids: one bag attribute, or one attribute per depth."""
        if self.bags:
            real = [v for v in ids if v != BLANK]
            bag: Dict[str, float] = {}
            for v in real:
                bag[v] = bag.get(v, 0.0) + 1.0 / len(real)
            x[name] = bag if real else BLANK
        else:
            ids = (list(ids) + [BLANK] * self.depth)[:self.depth]
            for k, v in enumerate(ids, 1):
                x[f"{name}{k}"] = v

    def _own_ctx(self, e: Elem, tokens=None) -> List[str]:
        """The context concept of the element's surface window (no donated facet)."""
        if "own" not in e.memo:
            x = {a: v for a, v in e.ctx.items() if a[0] in "lr" and a[1:].isdigit()}
            path = self.ctx.categorize(x)
            e.memo["own"] = self._path_ids(path) if self.naming in ("path", "ipath", "leaf", "basic") else self._ids(path)
        return e.memo["own"]

    def _content(self, e: Elem) -> dict:
        x = {}
        if self.donate_own:
            self._facet(x, "oc", self._own_ctx(e))
        for p, part in (("L", e.left), ("R", e.right)):
            self._facet(x, p, self._ctx_name(part))
            if self.donate_cnt:
                self._facet(x, f"{p}C", self._cnt_name(part))
            if self.heights:
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
            self._facet(x, "cc", self._cc_ids(e))
        return x

    def _word_concept(self, tokens, pos: int) -> List[str]:
        """The context concept of the token at ``pos`` (sorting its surface window);
        blank outside the experience."""
        if pos < 0 or pos >= len(tokens):
            return [BLANK] * self.depth
        x = self._surface(tokens, pos, pos + 1)
        return self._ids(self.ctx.categorize(x))

    def _full(self, x: dict, attrs) -> dict:
        """Complete an instance for storage (missing attributes blank; with
        boundary="missing" the surface slots beyond the experience stay out)."""
        if self.boundary in ("missing", "edge"):
            return {a: x.get(a, BLANK) for a in attrs if a in x or not (a[0] in "lr" and a[1:].isdigit())}
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

    def _climb(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """v1's recognition at the climbing ancestor: log2 P(x|A) / P(x|root),
        A the deepest concept on x's sorting path holding at least ``support``
        instances (the root when none does: an unrecognised chunk scores 0)."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for n in tree.categorize(x)[1:]:
            if n.count >= self.support:
                node = n
        if node is root:
            return 0.0
        return (log_prob(node, x, root, self.alpha) - log_prob(root, x, root, self.alpha)) / LN2

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
                    e.ctx = self._surface(tokens, i, j)
                    e.cnt = self._content(e)
                    e.ctx = self._context(tokens, e, analysed=False)
                    if self.factor == "ccm":
                        e.gain = self._contrast(self.cnt, e.cnt) + self._contrast(self.ctx, e.ctx)
                    elif self.factor == "regen":
                        # content scored by how probably memory unpacks it
                        e.gain = self._regen_bits(e) + self._gain(self.ctx, e.ctx)
                    elif self.factor == "deriv":
                        e.gain = self._deriv_bits(e)
                    elif self.factor == "regen2":
                        e.gain = self._regen_bits(e) + self._regen_tree(self.ctx, e.ctx)
                    elif self.factor == "climb":
                        e.gain = self._climb(self.cnt, e.cnt) + self._climb(self.ctx, e.ctx)
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

    def _regen_bits(self, e: Elem) -> float:
        """log2 P(e's content | its content concept): how probably memory
        unpacks this chunk into these parts (the deepest concept on its sorting
        path holding at least ``support`` instances; the root when none does)."""
        if "regen" not in e.memo:
            root = self.cnt.root
            if root.count <= 0:
                e.memo["regen"] = 0.0
            else:
                node = root
                for c in self.cnt.categorize(e.cnt)[1:]:
                    if c.count >= self.support:
                        node = c
                e.memo["regen"] = log_prob(node, e.cnt, root, self.alpha) / LN2
        return e.memo["regen"]

    def _deriv_bits(self, e: Elem) -> float:
        """One node of a derivation through both trees: log2 P(parts | content
        concept) plus log2 P(own content concept | context concept), the second
        read through the donated cc facet (the context concept sorted from the
        rest of the context instance, at the deepest supported level)."""
        if "deriv" not in e.memo:
            bits = self._regen_bits(e)
            cc = {a: v for a, v in e.ctx.items() if a.startswith("cc")}
            root = self.ctx.root
            if cc and root.count > 0:
                rest = {a: v for a, v in e.ctx.items() if not a.startswith("cc")}
                node = root
                for c in self.ctx.categorize(rest)[1:]:
                    if c.count >= self.support:
                        node = c
                bits += log_prob(node, cc, root, self.alpha) / LN2
            e.memo["deriv"] = bits
        return e.memo["deriv"]

    def _regen_tree(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """log2 P(x | the deepest concept on x's path with at least ``support`` instances)."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for c in tree.categorize(x)[1:]:
            if c.count >= self.support:
                node = c
        return log_prob(node, x, root, self.alpha) / LN2

    def _regen(self, n: int, comps, score=None) -> Tuple[Tree, List[Tuple[int, int, int]]]:
        """The analysis maximising the summed regeneration log probability of its chunks."""
        score = score or self._regen_bits
        best = {(i, i + 1): (0.0, None) for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                k = max(range(i + 1, j), key=lambda k: best[(i, k)][0] + best[(k, j)][0] + score(comps[(i, k, j)]))
                best[(i, j)] = (best[(i, k)][0] + best[(k, j)][0] + score(comps[(i, k, j)]), k)
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
        if self.readout in ("regen", "deriv"):
            tree, chosen = self._regen(n, comps, self._deriv_bits if self.readout == "deriv" else self._regen_bits)
        else:
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
            # each span is one observation (as in the constituent-context model):
            # weight 1 shared among its splits by P(split | span), of which the
            # constituent share is the span's posterior
            span_post: Dict[Tuple[int, int], float] = {}
            for (i, k, j), p in post.items():
                span_post[(i, j)] = span_post.get((i, j), 0.0) + p
            weights = {}
            for (i, k, j), e in comps.items():
                sp = span_post.get((i, j), 0.0)
                weights[(i, k, j)] = post[(i, k, j)] / sp if sp > 1e-12 else 1.0 / (j - i - 1)
                e.const = post.get((i, k, j), 0.0)
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
            self.ctx.add_passive(e.leaf, WORD, e.token, 1.0)
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
                e.memo.pop("own", None)
                new = self._content(e)
                if new != e.cnt and self.refresh == "relabel":
                    self.cnt.relabel(e.cleaf, e.cnt, new, e.weight)
                    e.cnt = new
                    self.stats["refreshed"] += 1
                elif new != e.cnt:
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
                self._facet(new, "cc", self._cc_ids(e))
            if new != e.ctx and self.refresh == "relabel":
                self.ctx.relabel(e.leaf, e.ctx, new, e.weight)
                e.ctx = new
                self.stats["refreshed"] += 1
            elif new != e.ctx:
                if e.const > 0:
                    self.ctx.add_passive(e.leaf, CONST, "c", -e.const)
                if e.token is not None:
                    self.ctx.add_passive(e.leaf, WORD, e.token, -1.0)
                self.ctx.unfit(e.leaf, e.ctx, e.weight)
                e.ctx, e.leaf = new, self.ctx.ifit(new, e.weight)
                if e.const > 0:
                    self.ctx.add_passive(e.leaf, CONST, "c", e.const)
                if e.token is not None:
                    self.ctx.add_passive(e.leaf, WORD, e.token, 1.0)
                self.stats["refreshed"] += 1

    # ------------------------------------------------------------------ #
    # Generation: top-down through the two hierarchies (no word-to-word step)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _index(tree):
        idx, stack = {}, [(tree.root, 0)]
        while stack:
            n, d = stack.pop()
            idx[f"c{n.id}"] = (n, d)
            stack.extend((c, d + 1) for c in n.children)
        return idx

    def _pick(self, rng, counts: Dict, idx=None, skip=()):
        """Sample a value by its count; concept ids only at their deepest named depth."""
        items = [(v, m) for v, m in counts.items() if m > 0 and v not in skip]
        if idx is not None:
            items = [(v, m) for v, m in items if v in idx]
            if items:
                deep = max(idx[v][1] for v, _ in items)
                items = [(v, m) for v, m in items if idx[v][1] == deep]
        if not items:
            return None
        r = rng.random() * sum(m for _, m in items)
        for v, m in items:
            r -= m
            if r <= 0:
                return v
        return items[-1][0]

    def generate(self, rng: Optional[random.Random] = None, max_depth: int = 12) -> Optional[List[str]]:
        """A sentence from memory, top-down: the sentence context (BOS ... EOS)
        names, through its donated own-content facet, the content concepts that
        fill it; a content concept names its parts' context concepts, content
        concepts and heights; a part of height 0 is a word drawn from its
        context concept's word counts, a taller part expands its content concept."""
        rng = rng or self.rng
        if not self.donate_ctx or not self.donate_cnt:
            raise ValueError("generation reads the donated facets (donate_ctx and donate_cnt)")
        cidx, nidx = self._index(self.ctx), self._index(self.cnt)
        x = {f"l{k}": BOS for k in range(1, self.window + 1)}
        x.update({f"r{k}": EOS for k in range(1, self.window + 1)})
        path = self.ctx.categorize(x)
        node = next((n for n in reversed(path) if self._pick(rng, n.av.get("cc", {}), nidx, (BLANK,))), None)
        if node is None:
            return None
        top = self._pick(rng, node.av.get("cc", {}), nidx, (BLANK,))
        out: List[str] = []

        def word(ctx_id):
            w = self._pick(rng, self.ctx.passive(cidx[ctx_id][0], WORD)) if ctx_id in cidx else None
            if w is None:
                raise LookupError
            out.append(w)

        def expand(cnt_id, d):
            c = nidx[cnt_id][0]
            for p in ("L", "R"):
                h = self._pick(rng, c.av.get(f"{p}H", {}))
                sub = self._pick(rng, c.av.get(f"{p}C", {}), nidx, (BLANK,))
                if h == "0" or sub is None or d >= max_depth:
                    word(self._pick(rng, c.av.get(p, {}), cidx, (BLANK,)))
                else:
                    expand(sub, d + 1)
        try:
            expand(top, 0)
        except LookupError:
            return None
        return out

    def _members(self, tree: IncrementalCobwebTree, which: str):
        """concept id -> (stored records under it, their weights), for one hierarchy."""
        at: Dict[int, List[Elem]] = {}
        for e in self.records:
            leaf = e.leaf if which == "ctx" else e.cleaf
            if leaf is not None:
                at.setdefault(id(leaf), []).append(e)
        out: Dict[str, List[Elem]] = {}
        # post-order without recursion (trees can be deeper than Python's stack)
        stack, order = [tree.root], []
        while stack:
            n = stack.pop()
            order.append(n)
            stack.extend(n.children)
        for n in reversed(order):
            out[f"c{n.id}"] = (list(at.get(id(n), [])) if not n.children
                               else [r for c in n.children for r in out[f"c{c.id}"]])
        return out

    def generate_sub(self, rng: Optional[random.Random] = None, depth: Optional[int] = None,
                     via: str = "ctx", max_depth: int = 12, members=None,
                     up: Optional[int] = None) -> Optional[List[str]]:
        """A sentence by substitution (v1's substitutability): a stored sentence-
        spanning element is expanded part by part, each part replaced by a stored
        element drawn (by weight) from the part's concept at ``depth``: its
        context concept (``via='ctx'``), or for a composite part its content
        concept (``via='cnt'``); drawn words are emitted, drawn composites expand.
        With ``up`` the concept is ``up`` levels above the part's own leaf
        (leaf-relative granularity) instead of at a fixed depth."""
        rng = rng or self.rng
        d = depth or self.depth
        if members is None:
            members = (self._members(self.ctx, "ctx"), self._members(self.cnt, "cnt"))
        mctx, mcnt = members
        roots = [e for e in self.records if e.token is None and self.is_sentence(e)]
        if not roots:
            return None

        def draw(pool):
            pool = [r for r in pool if r.weight > 0]
            if not pool:
                return None
            t = rng.random() * sum(r.weight for r in pool)
            for r in pool:
                t -= r.weight
                if t <= 0:
                    return r
            return pool[-1]
        out: List[str] = []

        def concept(tree, leaf, x):
            path = tree.ancestors(leaf)[::-1] if leaf is not None else tree.categorize(x)
            if up is not None:
                return f"c{path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)].id}"
            return f"c{path[min(d, len(path) - 1)].id}"

        def expand(e, k):
            for part in (e.left, e.right):
                if via == "cnt" and part.token is None:
                    pool = mcnt.get(concept(self.cnt, part.cleaf, part.cnt), [])
                else:
                    pool = mctx.get(concept(self.ctx, part.leaf, part.ctx), [])
                r = draw(pool) if k < max_depth else None
                if r is None:
                    r = part
                if r.token is not None:
                    out.append(r.token)
                elif r.left is None:
                    raise LookupError
                else:
                    expand(r, k + 1)
        try:
            expand(draw(roots), 0)
        except (LookupError, RecursionError):
            return None
        return out

    def generate_expand(self, rng: Optional[random.Random] = None, members=None, up: int = 1,
                        max_depth: int = 12, kind: bool = True) -> Optional[List[str]]:
        """Expansive generation. Sample a sentence-level chunk from memory as the
        root, then expand top-down, left to right. Each slot carries metadata
        (its context concept, ``up`` levels above the stored part's leaf, and
        whether the part was a word or a chunk) and chunk context (the tokens
        already generated to its left; the sentence end when the slot is last).
        Every stored member of the slot's concept is weighted by its stored
        weight times the probability its concept gives those surroundings, and
        one is drawn in proportion (probability matching), then expanded the
        same way: a word is emitted, a chunk's parts become new slots."""
        rng = rng or self.rng
        if members is None:
            members = self._members(self.ctx, "ctx")
        roots = [e for e in self.records if e.token is None and e.weight > 0 and self.is_sentence(e)]
        if not roots:
            return None
        sizes = {a: max(len(self.ctx.root.av.get(a, {})), 1) for a in self.ctx_attrs}
        out: List[str] = []

        def surround(m, at_end):
            node = m.leaf.parent if m.leaf is not None and m.leaf.parent is not None else m.leaf
            if node is None:
                return 1.0
            p = 1.0
            if self.boundary == "edge":
                known = [(f"l{k}", out[-k]) for k in range(1, min(self.window, len(out)) + 1)]
                known += [("le", "edge" if not out else "in"), ("re", "edge" if at_end else "in")]
            elif self.boundary == "missing":
                known = [(f"l{k}", out[-k]) for k in range(1, min(self.window, len(out)) + 1)]
                # a slot past an edge is unobserved: weight by how often the
                # member's concept leaves that slot unobserved
                edge = [f"l{k}" for k in range(len(out) + 1, self.window + 1)] + \
                       ([f"r{k}" for k in range(1, self.window + 1)] if at_end else [])
                for a in edge:
                    seen_a = sum(node.av.get(a, {}).values())
                    p *= (max(node.count - seen_a, 0.0) + self.alpha) / (node.count + 2 * self.alpha)
            else:
                known = ([(f"l{k}", out[-k] if k <= len(out) else BOS) for k in range(1, self.window + 1)] +
                         ([(f"r{k}", EOS) for k in range(1, self.window + 1)] if at_end else []))
            for a, v in known:
                d = node.av.get(a, {})
                p *= (d.get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a])
            return p

        def draw(pool, at_end):
            scored = [(m, m.weight * surround(m, at_end)) for m in pool if m.weight > 0]
            tot = sum(w for _, w in scored)
            if tot <= 0:
                return None
            t = rng.random() * tot
            for m, w in scored:
                t -= w
                if t <= 0:
                    return m
            return scored[-1][0]

        def concept(part):
            path = self.ctx.ancestors(part.leaf)[::-1] if part.leaf is not None else self.ctx.categorize(part.ctx)
            return f"c{path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)].id}"

        def expand(e, d, at_end):
            for side, part in (("L", e.left), ("R", e.right)):
                last = at_end and side == "R"
                pool = members.get(concept(part), [])
                if kind:
                    word = part.token is not None or d >= max_depth
                    pool = [m for m in pool if (m.token is not None) == word]
                m = draw(pool, last) or part
                if m.token is not None:
                    out.append(m.token)
                elif m.left is None or d >= max_depth:
                    raise LookupError
                else:
                    expand(m, d + 1, last)
        try:
            expand(draw(roots, True), 0, True)
        except (LookupError, RecursionError):
            return None
        return out

    def generate_best(self, rng: Optional[random.Random] = None, members=None, k: int = 8,
                      up: int = 1) -> Optional[List[str]]:
        """Expand ``k`` candidate sentences, then draw one in proportion to how
        well memory recognises it: exp of the parser's log total weight over all
        analyses, per composite (so length does not decide). The parser acts as
        the critic of the generator; drawing in proportion keeps it novel."""
        rng = rng or self.rng
        if members is None:
            members = self._members(self.ctx, "ctx")
        cands = [s for s in (self.generate_expand(rng, members=members, up=up) for _ in range(k)) if s]
        if not cands:
            return None
        scores = []
        for s in cands:
            if len(s) < 2:
                scores.append(0.0)
                continue
            r = self.parse(s)
            scores.append(r["logZ"] / (len(s) - 1))
        m = max(scores)
        w = [math.exp(v - m) for v in scores]
        t = rng.random() * sum(w)
        for s, v in zip(cands, w):
            t -= v
            if t <= 0:
                return s
        return cands[-1]
