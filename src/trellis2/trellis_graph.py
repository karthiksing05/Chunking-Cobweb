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

import bisect
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
    ``direction:piece``) and the anchor's square.

    ``join="any"``: B attaches to A when any piece of A sees any piece of B, in any
    direction (the first piece along a ray, A's own pieces not blocking, or a
    knight's jump), so two chunks can join as well as a chunk and a piece.
    ``join="interact"``: B attaches to A when a piece of A attacks or defends a
    piece of B or is attacked or defended by one (``chess.interactions``).

    ``flat``: chunks are one level up from pieces and described by what they hold
    (``flat_content``: the pieces, their offsets from the anchor, and how they
    interact with each other); every element's surface also holds how its pieces
    interact with the pieces outside it (``ext``), the anchor-context. ``ctx="ext"``:
    the context is only that and the anchor square (the anchor's first-piece star
    blank), so a chunk keeps the same context wherever its pattern recurs.
    ``absolute``: a flat chunk's content also holds each piece on its own square
    (``abs``), beside the offsets from the anchor (``tpl``); category utility
    weighs the two by how well each predicts, so a concept keeps whichever is
    stable for it (a castled king its squares, a pawn chain its shape).

    ``causal`` (learner config "chess_causal"): an occurrence's context is what a reader
    has already decoded when the unit arrives in reading order: the backward half of the
    anchor's star (the rays SE, S, SW, W and the jumps ESE, SSE, SSW, WSW, whose squares all
    come before the anchor, so they hold no piece of the element itself and every piece on
    them belongs to an earlier unit), the anchor square, and (once the analysis is known)
    the previous top unit, ``pu`` (a chunk by its content concept's band names, a piece by
    its kind, "<s>" for none) and the gap ``gp`` from its anchor (square indices y*8+x; from
    -1 for the first unit). No forward star, no outside ties, no own content (cc)."""
    surface_attrs = ("rays", "jumps", "sq")
    BACK_RAYS = ("SE", "S", "SW", "W")
    BACK_JUMPS = ("ESE", "SSE", "SSW", "WSW")

    def __init__(self, join: str = "anchor", flat: bool = False, ctx: str = "star", absolute: bool = False,
                 causal: bool = False):
        self.join, self.flat, self.ctx, self.absolute, self.causal = join, flat, ctx, absolute, causal
        if flat:
            self.surface_attrs = ("rays", "jumps", "sq", "ext")
        if causal:
            self.surface_attrs = ("rays", "jumps", "sq", "pu", "gp")

    @classmethod
    def causal_star(cls, board, sq) -> dict:
        """The backward half of the star at ``sq`` on ``board`` (square -> piece) and the
        square: only squares before ``sq`` in the reading order are read, so a board holding
        just the units decoded so far gives the same description as the full position."""
        st = C.star(board, sq)
        return {"rays": {f"{d}:{st[d]}": 1.0 / 4 for d in cls.BACK_RAYS},
                "jumps": {f"{d}:{st[d]}": 1.0 / 4 for d in cls.BACK_JUMPS},
                "sq": C.square_name(sq)}

    def prepare(self, position: C.Position) -> Experience:
        sqs = sorted(position, key=lambda s: (s[1], s[0]))
        return Experience([position[s] for s in sqs], (position, sqs, {s: i for i, s in enumerate(sqs)}))

    def links(self, exp: Experience, a: frozenset, b: frozenset) -> List[str]:
        pos, sqs, _ = exp.data
        target, own = sqs[min(b)], frozenset(sqs[i] for i in a)
        out = []
        if self.join == "interact":
            targets = {sqs[i] for i in b}
            for i in sorted(a):
                for t, fwd, back in C.interactions(pos, sqs[i]):
                    if t in targets and (fwd != "sees" or back):
                        out.append(fwd if fwd != "sees" else back)
            return out
        if self.join == "any":
            targets = {sqs[i] for i in b}
            for i in sorted(a):
                sq = sqs[i]
                for d, (dx, dy) in C.RAYS:
                    for k in range(1, 8):
                        t = (sq[0] + dx * k, sq[1] + dy * k)
                        if not C.on_board(t):
                            break
                        if t in pos and t not in own:
                            if t in targets:
                                out.append(f"{d}{k}")
                            break
                for d, (dx, dy) in C.JUMPS:
                    if (sq[0] + dx, sq[1] + dy) in targets:
                        out.append(d)
            return out
        for i in sorted(a):
            for rel, t in C.forward_neighbours(pos, sqs[i], own):
                if t == target:
                    out.append(rel)
        return out

    def surface(self, exp: Experience, ext: frozenset) -> dict:
        pos, sqs, _ = exp.data
        sq = sqs[min(ext)]
        if self.causal:
            return self.causal_star(pos, sq)
        st = C.star(pos, sq, frozenset(sqs[i] for i in ext))
        rays = {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.RAYS}
        jumps = {f"{d}:{st[d]}": 1.0 / 8 for d, _ in C.JUMPS}
        x = {"rays": rays, "jumps": jumps, "sq": C.square_name(sq)}
        if self.flat and self.ctx == "ext":      # the anchor-context: where it stands and what it does
            x["rays"], x["jumps"] = {"-": 1.0}, {"-": 1.0}
        if self.flat:
            x["ext"] = self._bag(self._ties(pos, [sqs[i] for i in ext], inside=False))
        return x

    @staticmethod
    def _bag(items: List[str]) -> dict:
        if not items:
            return {"-": 1.0}
        out: Dict[str, float] = {}
        for v in items:
            out[v] = out.get(v, 0.0) + 1.0 / len(items)
        return out

    @staticmethod
    def _ties(pos, members, inside: bool) -> List[str]:
        """How the pieces on ``members`` interact with the pieces inside (or outside)
        their set: "member does other" for every attack or defence either way."""
        own, out = set(members), []
        for sq in members:
            for t, fwd, back in C.interactions(pos, sq):
                if (t in own) != inside:
                    continue
                if fwd != "sees":
                    out.append(f"{pos[sq]} {fwd} {pos[t]}")
                if back and not inside:      # inside, the pair is met from both ends
                    out.append(f"{pos[sq]} {back} {pos[t]}")
        return out

    def flat_content(self, exp: Experience, ext: frozenset) -> dict:
        """A chunk as what it holds: its pieces, each piece at its offset from the
        anchor, and how its pieces interact with one another."""
        pos, sqs, _ = exp.data
        members = [sqs[i] for i in sorted(ext)]
        a = members[0]
        out = {"pc": self._bag([pos[s] for s in members]),
               "tpl": self._bag([f"{s[0] - a[0]},{s[1] - a[1]}:{pos[s]}" for s in members]),
               "int": self._bag(self._ties(pos, members, inside=True))}
        if self.absolute:
            out["abs"] = self._bag([f"{C.square_name(s)}:{pos[s]}" for s in members])
        return out

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
        join, flat = kw.pop("chess_join", "anchor"), kw.pop("chess_flat", False)
        self.join_gain = kw.pop("join_gain", "both")
        causal = kw.pop("chess_causal", False)
        if causal:                              # the own content is not known before the unit: no cc facet
            kw["donate_ctx"] = False
        self.dom = domain or ChessDomain(join, flat, kw.pop("chess_ctx", "star"), kw.pop("chess_abs", False), causal)
        self.causal = bool(getattr(self.dom, "causal", False))
        self._exp: Optional[Experience] = None
        self.max_size, self.samples = max_size, samples
        super().__init__(**kw)
        self._sample_rng = random.Random(kw.get("seed", 0) + 11)

    # attribute names ------------------------------------------------------ #
    def _surface_names(self, window: int) -> tuple:
        return tuple(self.dom.surface_attrs)

    def _neigh_names(self, d) -> tuple:
        return ("nb",)

    def _extra_cnt_names(self) -> tuple:
        if not self.dom.flat:
            return ("rel",)
        return ("pc", "tpl", "int", "abs") if self.dom.absolute else ("pc", "tpl", "int")

    # descriptions --------------------------------------------------------- #
    # causal contexts (chess_causal) -------------------------------------- #
    def _prev_name(self, prev: Optional[Elem]) -> List[str]:
        """The previous unit as a reader names it: a piece by its kind, a chunk by its
        content concept at the learner's naming (band names), "<s>" before the first."""
        if prev is None:
            return ["<s>"]
        if prev.token is not None:
            return [f"w:{prev.token}"]
        return self._cnt_name(prev)

    def causal_context(self, board, sq, prev: Optional[Elem], gap: Optional[int]) -> dict:
        """The causal context of a unit anchored at ``sq``: the backward half-star on
        ``board``, the square, and (gap not None) the previous unit and the gap from it.
        Learning calls it with the full position, the coder with the decoded board."""
        x = self.dom.causal_star(board, sq)
        if gap is not None:
            self._facet(x, "pu", self._prev_name(prev))
            x["gp"] = str(gap)
        return x

    def _set_prev(self, exp: Experience, elems, top) -> None:
        """Each element's previous unit: the top unit of the analysis with the greatest
        anchor before its anchor (for a top unit, the unit read just before it)."""
        sqs = exp.data[1]
        order = sorted(top, key=lambda u: min(u.span))
        anchors = [min(u.span) for u in order]
        for e in elems:
            i = min(e.span)
            k = bisect.bisect_left(anchors, i) - 1
            prev = order[k] if k >= 0 else None
            pa = sqs[min(prev.span)][1] * 8 + sqs[min(prev.span)][0] if prev is not None else -1
            e.memo["cprev"] = (prev, sqs[i][1] * 8 + sqs[i][0] - pa)

    def _refresh_todo(self, todo, b, recs, first) -> None:
        if self.causal:                         # the previous unit's name follows its concept
            for e in todo:
                cp = e.memo.get("cprev")
                if cp is None or e.leaf is None or cp[0] is None or cp[0].token is not None:
                    continue
                new = dict(e.ctx)
                self._facet(new, "pu", self._prev_name(cp[0]))
                if new != e.ctx:
                    self.ctx.relabel(e.leaf, e.ctx, new, e.weight)
                    e.ctx = new
                    self.stats["refreshed"] += 1
        super()._refresh_todo(todo, b, recs, first)

    def _context(self, exp: Experience, e: Elem, analysed: bool) -> dict:
        if self.causal:                         # what is decoded when the unit arrives, nothing else
            cp = e.memo.get("cprev") if analysed else None
            return self.causal_context(exp.data[0], exp.data[1][min(e.span)],
                                       cp[0] if cp else None, cp[1] if cp else None)
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

    def _content(self, e: Elem, sides: str = "LR") -> dict:
        if self.dom.flat:                       # what the chunk holds, kept with the element
            if "flat" not in e.memo:
                e.memo["flat"] = self.dom.flat_content(self._exp, e.span)
            # the part facets stay blank (constant, so they carry nothing)
            return {a: e.memo["flat"].get(a, BLANK) for a in self.cnt_attrs}
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
                # join_gain="content": a join is scored by what the chunk holds alone, so the
                # same pieces group the same way whatever surrounds them
                e.gain = self._gain(self.cnt, e.cnt) + (self._gain(self.ctx, e.ctx) if self.join_gain != "content" else 0.0)
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
        self._exp = exp
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
    def learn(self, experience, top=None) -> dict:
        r = self.parse(experience)
        exp, words, comps, post = r["exp"], r["words"], r["comps"], r["post"]
        if top is not None and self.dom.flat:   # the units a code chose (spans): stored as the analysis
            by_span = {}
            for e in comps.values():
                by_span.setdefault(e.span, e)
            units = []
            for s in top:
                if len(s) < 2:
                    continue
                e = by_span.get(s)
                if e is None:                   # a stored chunk recognised on the board
                    e = Elem(s, token="-")
                    e.token = None
                    comps[id(e)] = by_span[s] = e
                units.append(e)
            covered = set().union(*(e.span for e in units))
            r["top"] = sorted(units + [w for w in words if not (w.span & covered)], key=lambda e: min(e.span))
            r["chosen"] = [id(e) for e in units]
        if self.causal:                         # each element's previous unit in the final analysis
            self._set_prev(exp, list(words) + list(comps.values()), r["top"])
        chosen = set(r["chosen"])
        if self.dom.flat:                       # chunks one level up: only the top chunks are stored
            chosen &= {id(e) for e in r["top"] if e.token is None}
        for k in chosen:
            comps[k].used = 1.0
        for w in words:
            w.used = 1.0
        for k in chosen:
            parent = comps[k]
            if parent.left is None:
                continue
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
            if not self.dom.flat and ((e.left.token is None and e.left.rid is None) or
                                      (e.right.token is None and e.right.rid is None)):
                continue                        # a part too improbable to be stored itself
            e.weight = w
            e.cnt = self._content(e)
            e.cleaf = self.cnt.ifit(e.cnt, w)
            e.ctx = self._full(self._context(exp, e, analysed=k in chosen or self.causal), self.ctx_attrs)
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
            if self.dom.flat:
                continue
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
