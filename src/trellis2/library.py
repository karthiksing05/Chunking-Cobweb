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
    def __init__(self, depth: int = 6, alpha: float = 1.0, seed: int = 0, window: int = 2,
                 redescribe_budget: Optional[int] = 2000, confidence: float = 0.95,
                 prediction: str = "descent", match_alpha: Optional[float] = None):
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
        # prediction descends while the left context picks a child this clearly
        self.confidence = confidence
        # prediction: "descent" (along the tree while the left context decides)
        # or "mixture" (every leaf holding tokens, weighted by its count times how
        # well its context matches the left context: Cobweb's prediction from the
        # stored instances, nothing about later tokens used)
        self.prediction = prediction
        # the prior with which a leaf's context is matched to the left context
        # (smaller: sharper matching); defaults to alpha
        self.match_alpha = alpha if match_alpha is None else match_alpha
        self._word_leaves = None
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
        # the end of the experience, as an element after the last token: its
        # window is the last tokens, and what it "is" is the end marker
        lf = self.ctx.ifit(self._ctx_instance(tokens, n, n))
        self.ctx.add_passive(lf, KIND, "E")
        self.ctx.add_passive(lf, TOK, EOS)
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
        self._word_leaves = None
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

    def _ref_bits(self, v: CobwebNode, w: Optional[str]) -> float:
        """One code for what an element at concept v is: a token (w), or a
        composite (w = None), from v's tokens and composites together."""
        toks = v.pav.get(TOK, {})
        comp = v.pav.get(KIND, {}).get("C", 0.0)
        V = len(self.ctx.root.pav.get(TOK, {})) + 1
        n = sum(toks.values()) + comp
        c = comp if w is None else toks.get(w, 0.0)
        return -math.log2((c + self.alpha) / (n + self.alpha * (V + 1)))

    def _attr_bits(self, C: CobwebNode, attr: str, value) -> float:
        root = self.cnt.root
        d = C.av.get(attr, {})
        size = len(root.av.get(attr, {})) + 1
        return -math.log2((d.get(value, 0.0) + self.alpha) / (C.count + self.alpha * size))

    def _share_bits(self, C: CobwebNode) -> float:
        return -math.log2(C.count / self.cnt.root.count)

    def _reach_bits(self, path: List[CobwebNode], partial: dict) -> List[float]:
        """Bits to name each node of a context path (depth 0..) given what the
        decoder already knows (``partial``): the probability of descending to it
        when the context tree is sorted on that partial context."""
        out, bits = [0.0], 0.0
        root = self.ctx.root
        for parent, child in zip(path, path[1:]):
            ws = [c.count * math.exp(log_prob(c, partial, root, self.alpha)) for c in parent.children]
            w = child.count * math.exp(log_prob(child, partial, root, self.alpha))
            bits -= math.log2(w / sum(ws)) if sum(ws) > 0 and w > 0 else 0.0
            out.append(bits)
        return out

    def read(self, tokens: Sequence[str]) -> dict:
        """The shortest description of an experience under the trees, exact by
        dynamic programming over spans and naming depths:

        * E[span, d]: bits for a span's element once its name (its context
          concept at depth d; 0 = the root) is known: primitive or composite
          (from what the concept holds); a token among the concept's tokens; or a
          composite through the content concept C coding it best:
          -log2 share(C) + bits for its own name under C + for each part the
          naming depth that is cheapest under C plus that part's E, minus the
          bits of its own name alone (it is given);
        * a top-level piece: its name given the words before it (sorting the
          context tree on that left context), its E, one stop bit.

        Every naming depth is chosen by the code, decision by decision."""
        tokens = list(tokens)
        n = len(tokens)
        if self.cnt.root.count <= 0 or self.ctx.root.count <= 0:
            raise ValueError("empty library")
        D = self.depth
        path, ids = {}, {}
        for i in range(n):
            for j in range(i + 1, n + 1):
                pth = self.ctx.categorize(self._ctx_instance(tokens, i, j))[:D + 1]
                path[(i, j)] = pth
                ids[(i, j)] = [f"c{nd.id}" for nd in pth]
        # naming an element from the words before it (known to a left-to-right decoder)
        reach = {}
        for (i, j), pth in path.items():
            left = {f"l{k}": tokens[i - k] if i - k >= 0 else BOS for k in range(1, self.window + 1)}
            reach[(i, j)] = self._reach_bits(pth, left)

        def name_bits(C, attr, span, d):
            """A part's name: an even mixture of the code under the composite's
            concept and the code from its left context."""
            if d == 0:
                return 0.0
            a_, b_ = self._attr_bits(C, attr, ids[span][d]), reach[span][d]
            return -math.log2(0.5 * 2.0 ** -a_ + 0.5 * 2.0 ** -b_)

        E: Dict[Tuple[Tuple[int, int], int], float] = {}
        back: Dict[Tuple[Tuple[int, int], int], tuple] = {}
        for i in range(n):
            for d, v in enumerate(path[(i, i + 1)]):
                E[((i, i + 1), d)] = self._ref_bits(v, tokens[i])
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                own = path[(i, j)]
                # -log2 P(own name at depth d), through its best concept
                marg = [0.0]
                for d in range(1, len(own)):
                    part = {f"S{d}": ids[(i, j)][d]}
                    marg.append(min(self._share_bits(C) + self._attr_bits(C, f"S{d}", ids[(i, j)][d])
                                    for C in self.cnt.categorize(part) if C.count > 0))
                best = [(math.inf, None)] * len(own)
                for k in range(i + 1, j):
                    lp, rp = path[(i, k)], path[(k, j)]
                    x = self._content(lp, rp, own)
                    for C in self.cnt.categorize(x):
                        if C.count <= 0:
                            continue
                        lb = min((name_bits(C, f"L{d1}", (i, k), d1) + E[((i, k), d1)], d1) for d1 in range(len(lp)))
                        rb = min((name_bits(C, f"R{d2}", (k, j), d2) + E[((k, j), d2)], d2) for d2 in range(len(rp)))
                        base = self._share_bits(C) + lb[0] + rb[0]
                        for d in range(len(own)):
                            name = 0.0 if d == 0 else self._attr_bits(C, f"S{d}", ids[(i, j)][d])
                            tot = base + name - marg[d]
                            if tot < best[d][0]:
                                best[d] = (tot, (k, C, lb[1], rb[1]))
                for d, v in enumerate(own):
                    E[((i, j), d)] = best[d][0] + self._ref_bits(v, None)
                    back[((i, j), d)] = best[d][1]
        F = [0.0] + [math.inf] * n
        cut = [None] * (n + 1)
        for j in range(1, n + 1):
            for i in range(j):
                for d in range(len(path[(i, j)])):
                    t = F[i] + 1.0 + reach[(i, j)][d] + E[((i, j), d)]
                    if t < F[j]:
                        F[j], cut[j] = t, (i, d)
        roots, j = [], n
        while j > 0:
            i, d = cut[j]
            roots.append(((i, j), d))
            j = i
        roots.reverse()
        split, label, concepts = {}, {}, []

        def walk(span, d):
            i, j = span
            if j - i < 2:
                return
            k, C, d1, d2 = back[(span, d)]
            split[span] = k
            label[span] = f"n{C.id}"
            concepts.append(C.id)
            walk((i, k), d1)
            walk((k, j), d2)
        for span, d in roots:
            walk(span, d)
        return {"bits": F[n], "tree": Tree(n, split, label, [sp for sp, _ in roots]), "concepts": concepts,
                "depths": [d for _, d in roots]}

    # Predictive code ------------------------------------------------------ #
    def _mixture(self, left: dict, w: str) -> float:
        if self._word_leaves is None:
            self._word_leaves = [n for n in self.ctx.leaves() if n.pav.get(TOK)]
        root = self.ctx.root
        V = len(root.pav.get(TOK, {})) + 1
        num = den = 0.0
        for leaf in self._word_leaves:
            d = leaf.pav[TOK]
            n = sum(d.values())
            wt = n * math.exp(log_prob(leaf, left, root, self.match_alpha))
            num += wt * (d.get(w, 0.0) + 1.0 / V) / (n + 1.0)
            den += wt
        return num / den if den > 0 else 1.0 / V

    def predict(self, tokens: Sequence[str], i: int, w: str) -> float:
        """P(token w at position i | the tokens before it), read off the context
        tree: the left context is sorted down the tree and each concept's token
        counts are smoothed toward its parent's prediction (Witten-Bell), from
        the root to the deepest concept. Nothing about later tokens is used."""
        left = {f"l{k}": tokens[i - k] if i - k >= 0 else BOS for k in range(1, self.window + 1)}
        if self.prediction == "mixture":
            return self._mixture(left, w)
        root = self.ctx.root
        V = len(root.pav.get(TOK, {})) + 1
        p = 1.0 / V
        node = root
        while True:
            d = node.pav.get(TOK, {})
            n = sum(d.values())
            if n <= 0:
                break
            beta = len(d)
            p = (d.get(w, 0.0) + beta * p) / (n + beta)
            if not node.children:
                break
            # go deeper only while the left context decides the branch
            ws = [c.count * math.exp(log_prob(c, left, root, self.alpha)) for c in node.children]
            tot = sum(ws)
            if tot <= 0:
                break
            k = max(range(len(ws)), key=ws.__getitem__)
            if ws[k] / tot < self.confidence:
                break
            node = node.children[k]
        return p

    def code_bits(self, tokens: Sequence[str]) -> float:
        """Bits for an experience coded word by word from the left (with the end)."""
        tokens = list(tokens)
        bits = 0.0
        for i, w in enumerate(tokens + [EOS]):
            bits -= math.log2(self.predict(tokens, i, w))
        return bits

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
