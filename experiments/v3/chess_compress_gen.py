"""Chess state compression with TRELLIS's own generator as the probability model:
a position's code is the sequence of choices generation makes to produce exactly
that position, each costing -log2 of its probability (arithmetic coding), so a
decoder recovers the board by running generation on the coded choices. The same
statistics generation reads in every domain are used; nothing is added for chess.

For each top unit of the position's parse (a chunk, or a single piece):
  root      a chunk's content concept (``kc`` levels above its leaf), coded by TOP:
            the share of stored top chunks under that concept; a single piece by
            its piece-square frequency among the training parses' single tops (one
            alphabet, plus a stop item per position)
  anchor    the chunk's anchor square, by the anchor squares of the stored top
            chunks under its concept (Laplace over 64)
  parts     each part by generation's draw: a mixture over the concept's stored
            part slots (LPART / RPART) of FILL at the slot's context concept (``kx``
            levels above its leaf): a piece by its token, a chunk by its content
            concept (then its own parts, recursively)
  relation  the right part's anchor: the relation from the concept's ``rel``
            distribution, and which left piece it starts from (uniform)
  copy      (leaf match) alternatively the whole chunk as one stored instance under its
            concept: P(exact pieces at their offsets from the anchor | concept), by the
            stored top chunks under it; the chunk's code mixes the copy and generate
            routes (at most 1 bit above the better one); rows tagged "copy"
  absolute  (flat chunks with ``chess_abs``) the chunk's pieces may instead be coded on their
            own squares by the concept's ``abs`` distribution (the anchor then comes free);
            the chunk's code mixes the relative route (anchor + offsets) and the absolute
            route evenly (at most 1 bit above the better one)
  escape    a part is generated from memory with probability 1 - eps or written as
            a piece-square list with eps (both routes count), so every position
            has a code
  context   (flat chunks, CTXR=1 in the environment) alternatively a chunk's concept from what is
            already on the board, as generation fills a slot from its neighbours: its anchor
            by the anchor squares of all stored top chunks (occupied squares left out), then
            the star seen from the anchor over the units decoded so far, sorted in the context
            tree, and the content concept by FILL along that context path (backed off from the
            root as Cobweb predicts: P(c | X) = (n_X(c) + P(c | parent)) / (n_X + 1)); the
            chunk's code mixes this head and the concept-first head evenly
            (CTXR=2: a diagnostic only, not decodable: the chunk's own full context instead)
            (CTXR=3: the even mixture of each path concept's prediction, so a partial board
            that sorts too deep still reads the general concepts above)
  abs copy  (ACOPY=1 in the environment) the copy route also offers the stored instance on its own
            squares: P(exact pieces on exact squares | concept), by the stored top chunks under it
            (the anchor then comes free); the copy route is the even mixture of the two
  routes    (ROUTEW=1, flat chunks with copy) the routes (generate by offsets, generate on squares,
            copy by offsets, copy on squares) are mixed by learned weights instead of evenly: each
            route's share of the chunks coded so far that it coded best (Krichevsky-Trofimov,
            counts updated after each chunk, so a decoder keeps the same counts); with ROUTEW=2 a
            fifth route writes the chunk as its pieces, each a single top unit (no concept charged)
  cover     (COVER=1, flat chunks) the units coded are chosen for the code, as TRELLIS parses by
            compression gain: among every chunk the parse built (greedy and sampled analyses),
            non-overlapping chunks are taken greedily by how many bits they save over their pieces
            written singly (their cheapest route); the other pieces are single units. The encoder
            chooses the cover, so it costs nothing to send; COVER=2 also offers every stored chunk
            whose exact pieces stand on the board (recognition), described and coded as any chunk
  used      (USED=<b>, prequential) a unit's probability (concept, single piece, stop) also counts
            how often the code itself used it: P(u) = (n_code(u) + b P_top(u)) / (N_code + b), where
            P_top is the estimate from learning's top units; a decoder sees the same cover and keeps
            the same counts
Bits are code bits only: the memory itself (the model) is not charged.

PREQ=<block> (environment) instead codes prequentially: each of the first n_learn
positions is coded by the model learned from the positions before it (code models
rebuilt every <block> positions; the trees are live), then learned. The sum is the
whole description of the data, model included, with no separate memory term. Each
position's code is the even mixture of the copy codes at every level and the
piece-square list (at most log2 6 bits above the best).

Usage: chess_compress_gen.py <cfg json> <seed> <tag> [n_rep n_learn n_code]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis3 import BLANK, FILL, LPART, RPART, Elem  # noqa: E402
from trellis2.trellis_graph import TOP, GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n_rep, n_learn, n_code = [int(x) for x in sys.argv[4:7]] if len(sys.argv) > 6 else [4000, 2000, 1000]
max_size = cfg.pop("max_size", 4)
EPS, ALPHA = 0.05, 0.5
PREQ = int(os.environ.get("PREQ", 0))
CTXR = int(os.environ.get("CTXR", 0))
ACOPY = int(os.environ.get("ACOPY", 0))
ROUTEW = int(os.environ.get("ROUTEW", 0))
COVER = int(os.environ.get("COVER", 0))
ROUTES = {}                                          # (content level, context level) -> wins per route
USED = float(os.environ.get("USED", 0))
USES = {}                                            # (content level, context level) -> units the code used
LEVELS = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]

fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
learn_fens, code_fens = fens[:n_learn], fens[n_rep:n_rep + n_code]
preq_fens, learn_fens = (learn_fens, []) if PREQ else ([], learn_fens)
KINDS = sorted(C.TOKENS)


def psq(tok, sq):
    return KINDS.index(tok) * 64 + sq[1] * 8 + sq[0]


def offsets(span, exp):
    sqs = exp.data[1]
    ax, ay = sqs[min(span)]
    return frozenset((exp.tokens[i], sqs[i][0] - ax, sqs[i][1] - ay) for i in span)


L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
single_tops = Counter()
stored = Counter()                                   # distinct stored chunks (exact pieces and squares): the memory
rel_key = {}                                         # record id -> its pieces at their offsets from the anchor
abs_key = {}                                         # record id -> its pieces on their own squares
stored_by_first = {}                                 # a stored chunk's least piece-square -> the chunks
n_pos = 0
raw_use, raw_n = Counter(), 0                        # piece-square counts of the positions learned so far


def learn_one(f):
    global n_pos, raw_n
    pos = C.parse_fen(f)
    raw_use.update(psq(t, sq) for sq, t in pos.items())
    raw_n += len(pos)
    r = L.learn(pos)
    sqs = r["exp"].data[1]
    for k in r["chosen"]:
        e = r["comps"][k]
        if e.rid is not None:
            key = frozenset((sqs[i], r["exp"].tokens[i]) for i in e.span)
            if key not in stored:
                stored_by_first.setdefault(min(key), []).append(key)
            stored[key] += 1
            rel_key[e.rid] = offsets(e.span, r["exp"])
            abs_key[e.rid] = frozenset((sqs[i], r["exp"].tokens[i]) for i in e.span)
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            single_tops[psq(e.token, sqs[i])] += 1
    n_pos += 1


t0 = time.time()
for f in learn_fens:
    learn_one(f)
learn_s = time.time() - t0
L._quiet = True


def raw_bits(tok, sq):
    return -math.log2((raw_use.get(psq(tok, sq), 0) + 1) / (raw_n + 768))


def lvl(path, k):
    return path[max(0, len(path) - 1 - k)]


cpath_cache = {}


def cpath_of_record(rid):
    if rid not in cpath_cache:
        cpath_cache[rid] = L.cnt.categorize(L.records[rid].cnt)
    return cpath_cache[rid]


def up(node, k):
    for _ in range(k):
        node = node.parent or node
    return node


class Model:
    """The code at one pair of levels (kc content, kx context)."""

    def __init__(self, kc, kx):
        self.kc, self.kx = kc, kx
        tops = L.cnt.passive(L.cnt.root, TOP)
        self.top_c, self.anchor, self.size, self.keys, self.akeys = Counter(), {}, {}, {}, {}
        for rid, c in tops.items():
            cid = lvl(cpath_of_record(rid), kc).id
            self.top_c[cid] += c
            sq = (L.records[rid].ctx or {}).get("sq")
            if sq:
                self.anchor.setdefault(cid, Counter())[sq] += c
            self.size.setdefault(cid, Counter())[len(L.records[rid].span)] += c
            if rid in rel_key:
                self.keys.setdefault(cid, Counter())[rel_key[rid]] += c
                self.akeys.setdefault(cid, Counter())[abs_key[rid]] += c
        self.n_concepts = len(self.top_c) + 1
        self.anchor_all = Counter()
        for d in self.anchor.values():
            self.anchor_all.update(d)
        self.ctx_fill = {}
        self.top_tot = sum(self.top_c.values()) + sum(single_tops.values()) + n_pos + ALPHA * (self.n_concepts + 768) + 1
        self.fill_cache = {}
        self.pending = []

    def template_bits(self, cnode, e, attr="tpl"):
        """A flat chunk from its concept, as Cobweb predicts attributes: how many
        pieces (by the sizes of the top chunks stored under it), then each piece at
        its offset from the anchor by the concept's ``tpl`` distribution, backed off
        along its path (P(v | C) = (n_C(v) + P(v | parent)) / (n_C + 1), from a
        Laplace-smoothed root, as the generator's other look-ups)."""
        sizes = self.size.get(cnode.id, {})
        n = len(e.span)
        bits = -math.log2((sizes.get(n, 0) + ALPHA) / (sum(sizes.values()) + ALPHA * max_size))
        path, node = [], cnode
        while node is not None:
            path.append(node)
            node = node.parent
        n_values = 15 * 15 * 12 if attr == "tpl" else 64 * 12
        for v in e.cnt[attr]:
            p = None
            for node in reversed(path):
                d = node.av.get(attr, {})
                tot = sum(d.values())
                p = (d.get(v, 0.0) + ALPHA) / (tot + ALPHA * n_values) if p is None else (d.get(v, 0.0) + p) / (tot + 1.0)
            bits -= math.log2(p)
        return bits

    def used(self, key, p):
        """P(unit) from the code's own use, backed off to learning's estimate p."""
        if not USED:
            return -math.log2(p)
        u =USES.setdefault((self.kc, self.kx), Counter())
        return -math.log2((u.get(key, 0) + USED * p) / (u.get(None, 0) + USED))

    def top_bits(self, cid):
        return self.used(("c", cid), (self.top_c.get(cid, 0) + ALPHA) / self.top_tot)

    def single_bits(self, tok, sq):
        return self.used(("s", psq(tok, sq)), (single_tops.get(psq(tok, sq), 0) + ALPHA) / self.top_tot)

    def stop_bits(self):
        return self.used("stop", (n_pos + 1) / self.top_tot)

    def anchor_bits(self, cid, sq):
        d = self.anchor.get(cid, {})
        return -math.log2((d.get(sq, 0) + ALPHA) / (sum(d.values()) + 64 * ALPHA))

    def fill_prob(self, xnode, key, prim):
        """P(filler | the context concept xnode), within its kind (piece or chunk)."""
        ck = (xnode.id, key, prim)
        if ck not in self.fill_cache:
            fills = L.ctx.passive(xnode, FILL)
            tot = sum(c for f, c in fills.items() if (f[0] == "w") == prim)
            if tot <= 0:
                p = 0.0
            elif prim:
                p = fills.get(("w", key), 0.0) / tot
            else:
                p = sum(c for f, c in fills.items() if f[0] == "c" and lvl(cpath_of_record(f[1]), self.kc).id == key) / tot
            self.fill_cache[ck] = p
        return self.fill_cache[ck]

    def part_prob(self, cnode, side, key, prim):
        """Generation's draw of a part: a slot of the concept, then FILL at its context concept."""
        slots = L.cnt.passive(cnode, side)
        tot = sum(slots.values())
        if tot <= 0:
            return 0.0
        p = 0.0
        for (rid, is_tok), c in slots.items():
            if bool(is_tok) != prim or L.records[rid].leaf is None:
                continue
            p += c / tot * self.fill_prob(up(L.records[rid].leaf, self.kx), key, prim)
        return p

    def chunk_fills(self, xnode):
        """The content concepts (at level kc) of the chunks filled at a context concept."""
        if xnode.id not in self.ctx_fill:
            out = Counter()
            for f, c in L.ctx.passive(xnode, FILL).items():
                if f[0] == "c":
                    out[lvl(cpath_of_record(f[1]), self.kc).id] += c
            self.ctx_fill[xnode.id] = out
        return self.ctx_fill[xnode.id]

    def context_bits(self, xpath, cid, anchor, occupied):
        """A chunk's head from the board decoded so far: that a chunk comes, its anchor
        among the free squares, then its content concept by FILL along the context path."""
        tot_c = sum(self.top_c.values())
        bits = -math.log2((tot_c + ALPHA) / self.top_tot)
        free = sum(c for s, c in self.anchor_all.items() if s not in occupied)
        bits -= math.log2((self.anchor_all.get(anchor, 0) + ALPHA) / (free + ALPHA * (64 - len(occupied))))
        p, ps = None, []
        for node in xpath:
            d = self.chunk_fills(node)
            n = sum(d.values())
            p = (d.get(cid, 0) + ALPHA) / (n + ALPHA * self.n_concepts) if p is None else (d.get(cid, 0) + p) / (n + 1.0)
            ps.append(p)
        if CTXR == 3:                        # every concept on the path predicts; their even mixture
            return bits - math.log2(sum(ps) / len(ps))
        return bits - math.log2(p)

    def abs_copy_bits(self, cid, key):
        d = self.akeys.get(cid, {})
        n = d.get(key, 0)
        return -math.log2(n / sum(d.values())) if n else float("inf")

    def copy_bits(self, cid, key):
        d = self.keys.get(cid, {})
        n = d.get(key, 0)
        return -math.log2(n / sum(d.values())) if n else float("inf")

    def rel_bits(self, cnode, rel):
        d = cnode.av.get("rel", {})
        n = sum(d.values())
        return -math.log2((d.get(rel, 0.0) + ALPHA) / (n + ALPHA * len(C.OFFSET)))


ACC = Counter()                                      # bits by component, for the current model


def pieces(e, exp):
    sqs = exp.data[1]
    return [(exp.tokens[i], sqs[i]) for i in sorted(e.span)]


def source(e, exp):
    """(relation, number of left pieces) by which the right part's anchor is reached."""
    pos, sqs, _ = exp.data
    target, own = sqs[min(e.right.span)], frozenset(sqs[i] for i in e.left.span)
    for i in sorted(e.left.span):
        for rel, t in C.forward_neighbours(pos, sqs[i], own):
            if t == target:
                return rel, len(e.left.span)
    return None, len(e.left.span)


def inside_bits(M, e, exp, cnode, memo):
    """Bits for a chunk's parts given its content concept (generation from cnode)."""
    total = 0.0
    for side, part in ((LPART, e.left), (RPART, e.right)):
        total += part_bits(M, part, exp, cnode, side, memo)
    rel, n_left = source(e, exp)
    rb = (M.rel_bits(cnode, rel) if rel else 0.0) + math.log2(max(n_left, 1))
    ACC["relation + source"] += rb
    total += rb
    return total


def part_bits(M, part, exp, cnode, side, memo):
    raw = sum(raw_bits(t, sq) for t, sq in pieces(part, exp))
    if part.token is not None:
        p = M.part_prob(cnode, side, part.token, True)
        b = -math.log2((1 - EPS) * p + EPS * 2.0 ** -raw)
        ACC["piece parts"] += b
        ACC["piece parts generated (count)"] += (1 - EPS) * p > EPS * 2.0 ** -raw
        ACC["piece parts (count)"] += 1
        return b
    path = memo.setdefault(id(part), L.cnt.categorize(part.cnt))
    sub = lvl(path, M.kc)
    p = M.part_prob(cnode, side, sub.id, False)
    gen = -math.log2(p) + inside_bits(M, part, exp, sub, memo) if p > 0 else float("inf")
    # both routes: generated from memory (1 - eps) or written out (eps)
    a, b = math.log2(1 - EPS) - gen, math.log2(EPS) - raw
    ACC["chunk parts generated (count)"] += a > b
    ACC["chunk parts (count)"] += 1
    ACC["chunk parts, choosing the concept"] += -math.log2(p) if p > 0 else 0.0
    m = max(a, b)
    return -(m + math.log2(2 ** (a - m) + 2 ** (b - m)))


def either(a, b):
    """-log2 of the even mixture of two routes with these code lengths."""
    m = min(a, b)
    if m == float("inf"):
        return m
    return m - math.log2((2.0 ** (m - a) + 2.0 ** (m - b)) / 2)


def flat_bits(M, cn, e, anchor):
    """A flat chunk given its concept: anchor + offsets, or (with ``abs``) the even
    mixture of that and its pieces on their own squares."""
    rel = M.anchor_bits(cn.id, anchor) + M.template_bits(cn, e)
    if "abs" not in e.cnt:
        return rel
    ab = M.template_bits(cn, e, "abs")
    ACC["absolute route cheaper (count)"] += ab < rel
    return either(rel, ab)


def partial_context(board, anchor_sq):
    """The star seen from an empty anchor square over the pieces decoded so far: only the
    pieces seen count (a ray or jump that meets nothing yet is left out, not EDGE/EMPTY)."""
    st = C.star(board, anchor_sq)
    x = {"sq": C.square_name(anchor_sq)}
    for name, dirs in (("rays", C.RAYS), ("jumps", C.JUMPS)):
        hits = {f"{d}:{st[d]}": 1.0 / 8 for d, _ in dirs if st[d] not in (C.EDGE, C.EMPTY)}
        if hits:
            x[name] = hits
    return x


def cover(M, r, copy):
    """The units to code: chunks the parse built, chosen greedily by bits saved."""
    exp, memo = r["exp"], r["memo"]
    sqs = exp.data[1]
    single = {i: M.single_bits(exp.tokens[i], sqs[i]) for i in range(len(exp.tokens))}
    by_span = {}
    for e in r["comps"].values():
        by_span.setdefault(e.span, e)
    if COVER == 2:                                   # recognition: stored chunks present on the board
        index, here = exp.data[2], {(sqs[i], exp.tokens[i]) for i in range(len(sqs))}
        L._exp = exp
        for ps in here:
            for key in stored_by_first.get(ps, ()):
                if key <= here:
                    span = frozenset(index[sq] for sq, _ in key)
                    if span not in by_span:
                        e = Elem(span, token="-")
                        e.token = None
                        flat = L.dom.flat_content(exp, span)
                        e.cnt = {a: flat.get(a, BLANK) for a in L.cnt_attrs}
                        by_span[span] = e
    scored = []
    for span, e in by_span.items():
        path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
        cn = lvl(path, M.kc)
        anchor = C.square_name(sqs[min(span)])
        ls = [M.anchor_bits(cn.id, anchor) + M.template_bits(cn, e)]
        if "abs" in e.cnt:
            ls.append(M.template_bits(cn, e, "abs"))
        if copy:
            ls.append(M.anchor_bits(cn.id, anchor) + M.copy_bits(cn.id, offsets(span, exp)))
            if ACOPY:
                ls.append(M.abs_copy_bits(cn.id, frozenset((sqs[i], exp.tokens[i]) for i in span)))
        save = sum(single[i] for i in span) - (M.top_bits(cn.id) + min(ls))
        if save > 0:
            scored.append((save, e))
    scored.sort(key=lambda t: -t[0])
    used, out = set(), []
    for _, e in scored:
        if not (e.span & used):
            used |= e.span
            out.append(e)
    out += [w for w in r["words"] if not (w.span & used)]
    return sorted(out, key=lambda e: min(e.span))


def code_bits(M, r, copy=False):
    exp, memo = r["exp"], r["memo"]
    sqs = exp.data[1]
    bits = M.stop_bits()
    board, occupied = {}, set()
    M.pending = ["stop"]                     # the units this code used, counted once the position is coded
    for e in (cover(M, r, copy) if COVER and L.dom.flat else r["top"]):
        units = [(exp.tokens[i], sqs[i]) for i in e.span]
        if e.token is not None:
            (i,) = tuple(e.span)
            M.pending.append(("s", psq(e.token, sqs[i])))
            ACC["single tops"] += M.single_bits(e.token, sqs[i])
            bits += M.single_bits(e.token, sqs[i])
            board.update({sq: t for t, sq in units})
            occupied.add(C.square_name(sqs[i]))
            continue
        path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
        cn = lvl(path, M.kc)
        M.pending.append(("c", cn.id))
        ACC["chunk roots"] += M.top_bits(cn.id)
        ACC["anchors"] += M.anchor_bits(cn.id, C.square_name(sqs[min(e.span)]))
        ACC["chunk tops (count)"] += 1
        anchor = C.square_name(sqs[min(e.span)])
        if L.dom.flat:                       # one level up: the concept predicts the whole chunk (anchor included)
            inner = flat_bits(M, cn, e, anchor)
            ACC["templates"] += inner
        else:
            inner = M.anchor_bits(cn.id, anchor) + inside_bits(M, e, exp, cn, memo)
        if copy:
            cb = M.anchor_bits(cn.id, anchor) + M.copy_bits(cn.id, offsets(e.span, exp))
            if ACOPY:
                ab = M.abs_copy_bits(cn.id, frozenset((sqs[i], exp.tokens[i]) for i in e.span))
                ACC["abs copy cheaper (count)"] += ab < cb
                cb = either(cb, ab)
            ACC["chunk tops copied (count)"] += cb < inner
            ACC["copy, choosing the instance"] += cb if cb < inner else 0.0
            inner = either(cb, inner)
            if ROUTEW and L.dom.flat:
                key = frozenset((sqs[i], exp.tokens[i]) for i in e.span)
                ls = [M.anchor_bits(cn.id, anchor) + M.template_bits(cn, e),
                      M.template_bits(cn, e, "abs") if "abs" in e.cnt else float("inf"),
                      M.anchor_bits(cn.id, anchor) + M.copy_bits(cn.id, offsets(e.span, exp)),
                      M.abs_copy_bits(cn.id, key) if ACOPY else float("inf")]
                top = M.top_bits(cn.id)
                ls = [top + v for v in ls] + [sum(M.single_bits(exp.tokens[i], sqs[i]) for i in e.span)]
                wins = ROUTES.setdefault((M.kc, M.kx), [0.0] * 5)
                live = [k for k in range(5) if not (k == 1 and "abs" not in e.cnt) and not (k == 3 and not ACOPY)
                        and not (k == 4 and ROUTEW < 2)]
                n = sum(wins[k] for k in live)
                m = min(ls)
                inner = m - math.log2(sum((wins[k] + 0.5) / (n + 0.5 * len(live)) * 2.0 ** (m - ls[k]) for k in live))
                wins[min(live, key=lambda k: ls[k])] += 1
                ACC["written as pieces (count)"] += min(live, key=lambda k: ls[k]) == 4
                inner -= M.top_bits(cn.id)           # the concept is already inside each route
        head = M.top_bits(cn.id) + inner
        if CTXR and L.dom.flat:
            xk = ("ctx", id(e))
            if xk not in memo:
                memo[xk] = L.ctx.categorize(e.ctx if CTXR == 2 else partial_context(board, sqs[min(e.span)]))
            xpath = memo[xk][:max(1, len(memo[xk]) - M.kx)]
            hb = M.context_bits(xpath, cn.id, anchor, occupied)
            body = M.template_bits(cn, e)
            if "abs" in e.cnt:
                body = either(body, M.template_bits(cn, e, "abs"))
            if copy:
                body = either(M.copy_bits(cn.id, offsets(e.span, exp)), body)
            ACC["context head cheaper (count)"] += hb < M.top_bits(cn.id) + M.anchor_bits(cn.id, anchor)
            ACC["context head"] += hb
            head = either(head, hb + body)
        bits += head
        board.update({sq: t for t, sq in units})
        occupied.update(C.square_name(sq) for _, sq in units)
    return bits


def position_raw_bits(pos):
    """A position as a piece-square list with a stop item."""
    return sum(raw_bits(t, sq) for sq, t in pos.items()) - math.log2((n_pos + 1) / (raw_n + n_pos + 769))


if PREQ:
    out, Ms, tot, tot_raw, t0 = [], None, 0.0, 0.0, time.time()
    marks = {300, 1000, 2000, 4000, len(preq_fens)}
    for i, f in enumerate(preq_fens):
        pos = C.parse_fen(f)
        raw = position_raw_bits(pos)
        if i and i % PREQ == 0:
            cpath_cache.clear()
            Ms = [Model(kc, kx) for kc, kx in LEVELS]
        if Ms:
            L._quiet = True
            r = L.parse(pos)
            L._quiet = False
            r["memo"] = {}
            costs = [code_bits(M, r, True) for M in Ms] + [raw]
            for M in Ms if USED else ():
                u = USES.setdefault((M.kc, M.kx), Counter())
                u.update(M.pending)
                u[None] += len(M.pending)
            m = min(costs)
            bits = m - math.log2(sum(2.0 ** (m - c) for c in costs) / len(costs))
        else:
            bits = raw
        tot, tot_raw = tot + bits, tot_raw + raw
        learn_one(f)
        if i + 1 in marks:
            row = {"positions": i + 1, "total bits": round(tot), "mean bits per position": round(tot / (i + 1), 1),
                   "piece-square total bits": round(tot_raw), "piece-square mean": round(tot_raw / (i + 1), 1),
                   "stored chunks": len(stored), "s": round(time.time() - t0)}
            out.append(row)
            print(json.dumps(row), flush=True)
    summary = {"tag": tag, "seed": seed, "prequential block": PREQ, "eps": EPS, "curve": out}
    out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
    json.dump(summary, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary), flush=True)
    sys.exit(0)

t1 = time.time()
parsed = []
for f in code_fens:
    r = L.parse(C.parse_fen(f))
    r["memo"] = {}
    parsed.append(r)
parse_s = time.time() - t1
rbits = sum(position_raw_bits(C.parse_fen(f)) for f in code_fens) / len(code_fens)
stop_raw = -math.log2((n_pos + 1) / (raw_n + n_pos + 769))   # one stop item per listed chunk, as the dictionaries

summary = {"tag": tag, "seed": seed, "learn s": round(learn_s), "parse s": round(parse_s), "positions coded": len(code_fens),
           "piece-square bits mean": round(rbits, 1), "eps": EPS,
           # the memory as the dictionaries are charged: every distinct stored chunk written as a
           # piece-square list (a lower bound: TRELLIS also keeps repeated records and concept statistics)
           "stored chunks": len(stored), "stored records": sum(stored.values()),
           "memory bits (distinct chunks as piece-square lists)": round(sum(sum(raw_bits(t, sq) for sq, t in k) + stop_raw for k in stored)),
           "by levels": []}
for (kc, kx), copy in [(lv, cp) for lv in LEVELS for cp in (False, True)]:
    t2 = time.time()
    M = Model(kc, kx)
    ACC.clear()
    b = [code_bits(M, r, copy) for r in parsed]
    row = {"content level": kc, "context level": kx, "copy": copy, "bits per position": round(sum(b) / len(b), 1),
           "top concepts": len(M.top_c), "s": round(time.time() - t2),
           "per position": {k: round(v / len(parsed), 2) for k, v in sorted(ACC.items())}}
    summary["by levels"].append(row)
    print(json.dumps(row), flush=True)


def mix(costs):
    """-log2 of the uniform mixture of codes with these lengths."""
    m = min(costs)
    return m - math.log2(sum(2.0 ** (m - c) for c in costs) / len(costs))


if L.dom.flat:
    # every top item coded by a mixture of the codes at content levels 0..3 of its
    # concept path, so each chunk is predicted from whichever concept on its path
    # predicts it best (at most log2 4 bits above that level)
    Ms = [Model(k, 0) for k in range(4)]
    b = []
    for r in parsed:
        exp, memo = r["exp"], r["memo"]
        sqs = exp.data[1]
        bits = mix([M.stop_bits() for M in Ms])
        for e in r["top"]:
            if e.token is not None:
                (i,) = tuple(e.span)
                bits += mix([M.single_bits(e.token, sqs[i]) for M in Ms])
                continue
            path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
            anchor = C.square_name(sqs[min(e.span)])
            bits += mix([M.top_bits(lvl(path, M.kc).id) + flat_bits(M, lvl(path, M.kc), e, anchor) for M in Ms])
        b.append(bits)
    row = {"content level": "mixture of 0-3", "bits per position": round(sum(b) / len(b), 1)}
    summary["by levels"].append(row)
    print(json.dumps(row), flush=True)
out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
json.dump(summary, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(summary), flush=True)
