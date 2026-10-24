"""TRELLIS v2 on relational experiences: an element is a connected set of
primitives, not a contiguous span (docs/GRAPH.md).

An experience is a set of primitives (tokens) in a read order, with typed
relations the domain supplies. A composite joins two disjoint elements A and B
when B's anchor (its first primitive in the read order) comes after A's and some
primitive of A sees it under a relation; its anchor is A's, and its content
gains one attribute, ``rel``: the bag of relations by which A sees B. A sentence
is the case with one relation (the next word), where these composites are
exactly the spans and splits of ``Trellis3``.

Context, as for sentences, is the element's surface (the domain's window around
its anchor, its own primitives left out: for chess the star of rays and knight
jumps), with the same chunk-context, donated and neighbour facets.

Parsing. The analyses of a board are forests of chunks of at most ``max_size``
primitives, too many to sum over, so the posterior of each composite is
estimated from ``samples`` analyses drawn bottom-up: at each step one of the
possible joins of the current top-level elements is taken with probability
proportional to 2^(gain / tau), or the analysis stops with weight 1 (a join that
does not pay is less likely than stopping). The stored analysis (chunk context,
generation statistics) is the greedy one: the best-paying join while one pays.
The rest of learning (storage weighted by the posterior, refresh,
reassignment, names) is ``Trellis3``'s.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .trellis3 import BLANK, FILL, LPART, RPART, WORD, CONST, LN2, Elem, Trellis3
from . import chess as C

TOP = "top"          # at a content concept: committed top-level chunks of whole experiences (record id)


class Experience:
    """A prepared experience: primitives in read order and the domain's own data."""
    __slots__ = ("tokens", "data")

    def __init__(self, tokens: List[str], data=None):
        self.tokens, self.data = tokens, data


class SentenceDomain:
    """Sentences as relational experiences (one relation, the next word):
    composites are spans; the surface is Trellis3's two-word window."""
    def __init__(self, window: int = 2):
        self.window = window
        self.surface_attrs = Trellis3._surface_names(window)

    def prepare(self, tokens: Sequence[str]) -> Experience:
        return Experience(list(tokens))

    def links(self, exp: Experience, a: frozenset, b: frozenset) -> List[str]:
        return ["next"] if min(b) == max(a) + 1 else []

    def surface(self, exp: Experience, ext: frozenset) -> dict:
        t, i, j, x = exp.tokens, min(ext), max(ext) + 1, {}
        for k in range(1, self.window + 1):
            x[f"l{k}"] = t[i - k] if i - k >= 0 else "<s>"
            x[f"r{k}"] = t[j + k - 1] if j + k - 1 < len(t) else "</s>"
        return x

    def neighbours(self, exp: Experience, ext: frozenset) -> List[int]:
        i, j = min(ext), max(ext) + 1
        return [p for p in (i - 1, j) if 0 <= p < len(exp.tokens)]


class ChessDomain:
    """Chess positions: pieces read square by square (a1 b1 ... h8); B attaches
    to A when a piece of A sees B's anchor forward along a ray (first piece, any
    distance, A's own pieces not blocking) or a knight's jump (``chess.forward_neighbours``).
    The surface is the anchor's star (two bags: rays and jumps, each value
    ``direction:piece``) and the anchor's square."""
    surface_attrs = ("rays", "jumps", "sq")

    def prepare(self, position: C.Position) -> Experience:
        sqs = sorted(position, key=lambda s: (s[1], s[0]))
        return Experience([position[s] for s in sqs], (position, sqs, {s: i for i, s in enumerate(sqs)}))

    def links(self, exp: Experience, a: frozenset, b: frozenset) -> List[str]:
        pos, sqs, _ = exp.data
        target, own = sqs[min(b)], frozenset(sqs[i] for i in a)
        out = []
        for i in sorted(a):
            for rel, t in C.forward_neighbours(pos, sqs[i], own):
                if t == target:
                    out.append(rel)
        return out

    def surface(self, exp: Experience, ext: frozenset) -> dict:
        pos, sqs, _ = exp.data
        sq = sqs[min(ext)]
        st = C.star(pos, sq, frozenset(sqs[i] for i in ext))
        rays = {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.RAYS}
        jumps = {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.JUMPS}
        return {"rays": rays, "jumps": jumps, "sq": C.square_name(sq)}

    def neighbours(self, exp: Experience, ext: frozenset) -> List[int]:
        pos, sqs, index = exp.data
        sq, own = sqs[min(ext)], frozenset(sqs[i] for i in ext)
        out = []
        for d, (dx, dy) in C.RAYS:
            f, r = sq[0] + dx, sq[1] + dy
            while 0 <= f < 8 and 0 <= r < 8:
                if (f, r) in pos and (f, r) not in own:
                    out.append(index[(f, r)])
                    break
                f, r = f + dx, r + dy
        for d, (dx, dy) in C.JUMPS:
            t = (sq[0] + dx, sq[1] + dy)
            if t in pos and t not in own:
                out.append(index[t])
        return out


class GraphTrellis(Trellis3):
    def __init__(self, domain=None, max_size: int = 4, samples: int = 16, **kw):
        self.dom = domain or ChessDomain()
        self.max_size, self.samples = max_size, samples
        super().__init__(**kw)
        self._sample_rng = random.Random(kw.get("seed", 0) + 11)

    # attribute names ------------------------------------------------------ #
    def _surface_names(self, window: int) -> tuple:
        return tuple(self.dom.surface_attrs)

    def _neigh_names(self, d) -> tuple:
        return ("nb",)

    def _extra_cnt_names(self) -> tuple:
        return ("rel",)

    # descriptions --------------------------------------------------------- #
    def _context(self, exp: Experience, e: Elem, analysed: bool) -> dict:
        x = self.dom.surface(exp, e.span)
        if self.neighbour_types:
            e.nbt = tuple(exp.tokens[p] for p in self.dom.neighbours(exp, e.span))
            self._nb_facets(x, e)
        if self.children:
            for p, part in (("cl", e.left), ("cr", e.right)):
                self._facet(x, p, self._ctx_ids(part) if part is not None else [BLANK] * self.depth)
        if self.spine and analysed:
            for p, ids in (("par", e.par), ("sib", e.sib)):
                self._facet(x, p, ids or [BLANK] * self.depth)
        if self.donate_ctx:
            self._facet(x, "cc", self._cc_ids(e))
        return x

    def _nb_facets(self, x: dict, e: Elem) -> None:
        """One bag: the mean of the neighbours' learned names."""
        bag: Dict[str, float] = {}
        named = [b for b in (self._nb_name(t) for t in e.nbt) if b]
        for b in named:
            tot = sum(b.values()) or 1.0
            for v, w in b.items():
                bag[v] = bag.get(v, 0.0) + w / tot / len(named)
        self._facet(x, "nb", dict(sorted(bag.items(), key=lambda kv: -kv[1])[:self.type_top]) if bag else [BLANK])

    def _content(self, e: Elem) -> dict:
        x = super()._content(e)
        rels = list(e.split)
        x["rel"] = {r: rels.count(r) / len(rels) for r in set(rels)}
        return x

    # parsing -------------------------------------------------------------- #
    def _join(self, exp: Experience, a: Elem, b: Elem, memo: dict) -> Optional[Elem]:
        key = (id(a), id(b))
        if key in memo:
            return memo[key]
        e = None
        if len(a.span) + len(b.span) <= self.max_size:
            rels = self.dom.links(exp, a.span, b.span)
            if rels:
                e = Elem(a.span | b.span, split=tuple(rels), left=a, right=b)
                e.cnt = self._content(e)
                e.ctx = self._context(exp, e, analysed=False)
                e.gain = self._gain(self.cnt, e.cnt) + self._gain(self.ctx, e.ctx)
        memo[key] = e
        return e

    def _analysis(self, exp: Experience, words: List[Elem], memo: dict, rng=None) -> Tuple[List[Elem], List[Elem]]:
        """One analysis, bottom-up: greedy (rng None) or sampled. Returns its
        top-level elements and its composites."""
        units, made = list(words), []
        while True:
            cands = []
            for a in units:
                for b in units:
                    if a is not b and min(a.span) < min(b.span):
                        e = self._join(exp, a, b, memo)
                        if e is not None:
                            cands.append(e)
            if not cands:
                break
            if rng is None:
                e = max(cands, key=lambda c: c.gain)
                if e.gain <= 0:
                    break
            else:
                f = [c.gain * LN2 / self.tau for c in cands]
                m = max(max(f), 0.0)
                w = [math.exp(v - m) for v in f]
                stop = math.exp(-m)
                r = rng.random() * (sum(w) + stop)
                e = None
                for c, wc in zip(cands, w):
                    r -= wc
                    if r <= 0:
                        e = c
                        break
                if e is None:
                    break
            units = [u for u in units if u is not e.left and u is not e.right] + [e]
            made.append(e)
        return units, made

    def parse(self, experience) -> dict:
        exp = experience if isinstance(experience, Experience) else self.dom.prepare(experience)
        self._wc_cache, self._rel_cache, self._nb_cache = {}, {}, {}
        words = []
        for i, t in enumerate(exp.tokens):
            e = Elem(frozenset([i]), token=t)
            e.ctx = self._context(exp, e, analysed=False)
            words.append(e)
        memo: dict = {}
        top, chosen = self._analysis(exp, words, memo)
        count: Dict[int, int] = {}
        for _ in range(self.samples):
            _, made = self._analysis(exp, words, memo, self._sample_rng)
            for e in made:
                count[id(e)] = count.get(id(e), 0) + 1
        comps = {id(e): e for e in memo.values() if e is not None}
        post = {k: count.get(k, 0) / max(self.samples, 1) for k in comps}
        return {"exp": exp, "words": words, "comps": comps, "post": post,
                "chosen": [id(e) for e in chosen], "top": top}

    # learning ------------------------------------------------------------- #
    def learn(self, experience) -> dict:
        r = self.parse(experience)
        exp, words, comps, post = r["exp"], r["words"], r["comps"], r["post"]
        chosen = set(r["chosen"])
        for k in chosen:
            comps[k].used = 1.0
        for w in words:
            w.used = 1.0
        for k in chosen:
            parent = comps[k]
            for child, sib in ((parent.left, parent.right), (parent.right, parent.left)):
                child.par, child.sib = self._ctx_ids(parent), self._ctx_ids(sib)
        if self.store == "posterior":
            weights = dict(post)
            for k in chosen:                    # the stored analysis is always stored
                weights[k] = max(weights.get(k, 0.0), self.min_weight)
        elif self.store == "all":
            weights = {k: 1.0 for k in comps}
        else:
            weights = {k: 1.0 for k in chosen}
        for e in words:
            e.weight = 1.0
            e.ctx = self._full(self._context(exp, e, analysed=True), self.ctx_attrs)
            e.leaf = self.ctx.ifit(e.ctx, 1.0)
            e.rid = len(self.records)
            self._pas(e, "ctx", WORD, e.token, 1.0)
            if self.type_names or self.neighbour_types:
                self._add_type(e)
            self.records.append(e)
            self.stats["words stored"] += 1
        first = len(self.records)
        for k in sorted(comps, key=lambda k: (len(comps[k].span), min(comps[k].span), k)):
            w = weights.get(k, 0.0)
            if w < self.min_weight:
                continue
            e = comps[k]
            if (e.left.token is None and e.left.rid is None) or (e.right.token is None and e.right.rid is None):
                continue                        # a part too improbable to be stored itself
            e.weight = w
            e.cnt = self._content(e)
            e.cleaf = self.cnt.ifit(e.cnt, w)
            e.ctx = self._full(self._context(exp, e, analysed=k in chosen), self.ctx_attrs)
            e.leaf = self.ctx.ifit(e.ctx, w)
            e.rid = len(self.records)
            if self.type_names or self.neighbour_types:
                self._add_type(e)
            if self.word_content:
                self._add_word_content(e)
            self.records.append(e)
            self.stats["composites stored"] += 1
        for e in words:
            self._pas(e, "ctx", FILL, ("w", e.token), 1.0)
        for k in chosen:
            e = comps[k]
            if e.cleaf is None or e.leaf is None:
                continue
            self._pas(e, "ctx", FILL, ("c", e.rid), 1.0)
            for a, part in ((LPART, e.left), (RPART, e.right)):
                if part.leaf is not None:
                    self._pas(e, "cnt", a, (part.rid, part.token is not None), 1.0)
        for e in r["top"]:
            if e.token is None and e.cleaf is not None:
                self._pas(e, "cnt", TOP, e.rid, 1.0)
        self.stats["experiences"] += 1
        self._refresh(first, exp.tokens)
        return r

    # reading -------------------------------------------------------------- #
    def chunks(self, r: dict, exp: Optional[Experience] = None) -> List[str]:
        """The stored analysis's top-level chunks, rendered (chess: [a rel b])."""
        exp = exp or r["exp"]

        def show(e):
            if e.token is not None:
                return e.token
            return f"[{show(e.left)} {'/'.join(sorted(set(e.split)))} {show(e.right)}]"
        return [show(e) for e in r["top"] if e.token is None]
