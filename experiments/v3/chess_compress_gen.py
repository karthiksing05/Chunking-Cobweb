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
  gap       (GAP=1 or 2, flat chunks) positions coded relatively: units in increasing anchor order,
            each anchor by its gap from the previous unit's (see GAP below)
  causal    (learner config "chess_causal", needs GAP) the context tree holds each occurrence by what
            a reader has decoded when the unit arrives (the anchor's backward half-star, its square,
            the previous unit's name and the gap from it: ChessDomain causal), and every unit gains a
            causal route: that a unit comes (the top alphabet's non-stop mass), its gap (pooled over
            all top units), its symbol (piece kind, or content concept at level kc) by P(symbol |
            context concept), the concept found by sorting the causal description computed on the
            board decoded so far (the earlier units), and a chunk's body given its concept and anchor
            (generate by offsets / on squares, copy by offsets / on squares at that anchor, learned
            weights); the route and the existing code are mixed by learned weights, one pair of counts
            for all units. P(symbol | X) backs off from the root along the path (n_X(v) + P(v |
            parent)) / (n_X + 1); with CAUSALD=1 every concept on the path predicts, mixed by learned
            weights per depth; CAUSALD=3 also codes the gap from the context: the partial description
            {pu} sorted in the context tree, P(gap | each concept on its path) by the stored top units'
            gp (backed off from the pooled gaps), mixed by learned depth weights and renormalised over
            the gaps that stay on the board; CAUSALD=4 adds the previous unit's own context path,
            predicting the gap to the unit after it
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
TOPC = int(os.environ.get("TOPC", 0))            # learning stores the code-chosen cover as its top units
ANCH = int(os.environ.get("ANCH", 0))            # anchors backed off along the concept path (1), and renormalised to where the shape fits (2)
ROUTES = {}                                          # (content level, context level) -> wins per route
USED = float(os.environ.get("USED", 0))
USES = {}                                            # (content level, context level) -> units the code used
# GAP=1: positions coded relatively. Units go in increasing anchor order along the reading
# order (square index y*8+x, as the experience reads pieces); each unit's anchor is coded as the
# gap from the previous unit's anchor (-1 before the first), renormalised over the gaps that keep
# it on the board: a single piece by P(piece, gap), a chunk by P(gap | concept) backed off along
# the concept's path to a gap distribution pooled over all units (then uniform). Every route of a
# chunk then codes its body given the decoded anchor (copy on squares: P(instance | concept, anchor)).
# GAP=2: as 1, but the routes on squares code the anchor themselves (no gap charged), the copy on
# squares renormalised over the stored instances whose anchor lies after the previous unit's.
# Counts come from the learner's stored top units of the training positions, in reading order.
GAP = int(os.environ.get("GAP", 0))
GAP_DEBUG = int(os.environ.get("GAP_DEBUG", 0))      # assert the GAP distributions sum to 1 on their supports
gap_single, gap_chunk, gap_all = Counter(), Counter(), Counter()   # (token, gap), (record id, gap), gap
LEVELS = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]

fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
learn_fens, code_fens = fens[:n_learn], fens[n_rep:n_rep + n_code]
if os.environ.get("LEARN_FILE"):                     # transfer: learn one set, code another
    learn_fens = [l.strip() for l in open(os.environ["LEARN_FILE"], encoding="utf-8") if l.strip()][:n_learn]
if os.environ.get("CODE_FILE"):
    code_fens = [l.strip() for l in open(os.environ["CODE_FILE"], encoding="utf-8") if l.strip()][:n_code]
preq_fens, learn_fens = (learn_fens, []) if PREQ else ([], learn_fens)
KINDS = sorted(C.TOKENS)


SQUARES = [C.square_name((x, y)) for x in range(8) for y in range(8)]
_fits = {}


def shape_of(e):
    """A chunk's pieces as offsets from its anchor (from its tpl bag), or None."""
    tpl = (e.cnt or {}).get("tpl") if e.cnt else None
    if not isinstance(tpl, dict):
        return None
    return frozenset(tuple(int(v) for v in k.split(":")[0].split(",")) for k in tpl)


def fits(shape):
    """The anchor squares at which every piece of the shape lands on the board."""
    if shape not in _fits:
        _fits[shape] = [C.square_name((x, y)) for x in range(8) for y in range(8)
                        if all(0 <= x + dx < 8 and 0 <= y + dy < 8 for dx, dy in shape)]
    return _fits[shape]


def psq(tok, sq):
    return KINDS.index(tok) * 64 + sq[1] * 8 + sq[0]


def ridx(sq):
    """A square's index in the reading order (a1, b1, ..., h1, a2, ..., h8)."""
    return sq[1] * 8 + sq[0]


def offsets(span, exp):
    sqs = exp.data[1]
    ax, ay = sqs[min(span)]
    return frozenset((exp.tokens[i], sqs[i][0] - ax, sqs[i][1] - ay) for i in span)


L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
# chess_causal (learner config): the context tree holds each occurrence by what a reader has decoded
# when the unit arrives (ChessDomain causal: backward half-star, square, previous unit, gap), and the
# code gains a causal route per unit: P(go) P(gap | prev) P(symbol | context concept) P(body | symbol,
# anchor), the symbol a piece kind or a content concept, mixed with the existing code by learned weights
CAUSAL = bool(getattr(L, "causal", False))
if CAUSAL and not GAP:
    raise SystemExit("chess_causal codes units in reading order: set GAP=1 or 2")
top_units = []                                       # CAUSAL: (record, "w" token | "c" record id) of every stored top unit
CTXW, CBODY, DEPTHW = {}, {}, {}                     # CAUSAL: (kc, kx, copy) -> wins per route (existing, causal); per body route; per path depth
CAUSALD = int(os.environ.get("CAUSALD", 0))          # CAUSAL: 1 = the symbol by a learned mixture over the context path's concepts; 2 = root only (ablation)
# 3 = 1 plus the gap from the context: the partial description {pu} (all a reader knows of the new unit's
# description before its anchor) sorted in the context tree, P(gap | each concept on that path) from the stored
# top units' gp, backed off from the pooled gaps, mixed along the path by learned depth weights;
# 4 = 3 plus the previous unit's own (full, decoded) context path predicting the gap that follows it
top_next = []                                        # CAUSALD=4: (record, gap to the next top unit) of every stored top unit
DEPTHG = {}                                          # CAUSALD>=3: (kc, kx, copy) -> wins per (path, depth) of the gap mixture
# GAPW=1 (CAUSALD>=3): the gap's experts are the concepts at depths GAP_DEPTHS from the root of each path (a
# shorter path's leaf standing in for the deeper ones), mixed by Bayesian weights (each expert's probability of
# the gaps coded so far, uniform prior) instead of win counts
GAPW = int(os.environ.get("GAPW", 0))
GAP_DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24)
single_tops = Counter()
stored = Counter()                                   # distinct stored chunks (exact pieces and squares): the memory
rel_key = {}                                         # record id -> its pieces at their offsets from the anchor
abs_key = {}                                         # record id -> its pieces on their own squares
stored_by_first = {}                                 # a stored chunk's least piece-square -> the chunks
n_pos = 0
raw_use, raw_n = Counter(), 0                        # piece-square counts of the positions learned so far


def learn_one(f, top=None):
    global n_pos, raw_n
    pos = C.parse_fen(f)
    raw_use.update(psq(t, sq) for sq, t in pos.items())
    raw_n += len(pos)
    r = L.learn(pos, top)
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
        if CAUSAL and e.leaf is not None:            # its context leaf is read when the code is built (records move)
            if e.token is not None:
                top_units.append((e, "w", e.token))
            elif e.rid is not None and getattr(e, "cleaf", None) is not None:
                top_units.append((e, "c", e.rid))
        if e.token is not None:
            (i,) = tuple(e.span)
            single_tops[psq(e.token, sqs[i])] += 1
    if CAUSAL and CAUSALD == 4:                      # each stored top unit and the gap to the unit after it
        order = sorted(r["top"], key=lambda e: min(e.span))
        for u, v in zip(order, order[1:]):
            if u.leaf is not None and (u.token is not None or (u.rid is not None and getattr(u, "cleaf", None) is not None)):
                top_next.append((u, ridx(sqs[min(v.span)]) - ridx(sqs[min(u.span)])))
    if GAP:                                          # the stored top units in reading order, as gaps
        prev = -1
        for e in sorted(r["top"], key=lambda e: min(e.span)):
            a = ridx(sqs[min(e.span)])
            gap_all[a - prev] += 1
            if e.token is not None:
                gap_single[(e.token, a - prev)] += 1
            elif e.rid is not None and getattr(e, "cleaf", None) is not None:   # as TOP counts it
                gap_chunk[(e.rid, a - prev)] += 1
            prev = a
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
        self.nodes, self.anchor_up, self.avec = {}, {}, {}
        for rid, c in tops.items():
            node = lvl(cpath_of_record(rid), kc)
            cid = node.id
            self.nodes[cid] = node
            self.top_c[cid] += c
            sq = (L.records[rid].ctx or {}).get("sq")
            if sq:
                self.anchor.setdefault(cid, Counter())[sq] += c
                while node is not None:                  # ANCH: anchors counted up the concept's path
                    self.anchor_up.setdefault(node.id, Counter())[sq] += c
                    node = node.parent
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
        if GAP:
            self.init_gaps()
        if CAUSAL:
            self.init_causal()

    # CAUSAL: the unit's top symbol from its causal context concept --------- #
    def init_causal(self):
        """Counts of each stored top unit's symbol (piece kind, or content concept at level kc)
        at every concept on its context leaf's current path; the alphabet: the 12 kinds, the
        top concepts and one slot for a concept never stored as a top unit."""
        self.cx = {}
        for e, k, v in top_units:
            sym = ("w", v) if k == "w" else ("c", lvl(cpath_of_record(v), self.kc).id)
            node = e.leaf
            while node is not None:
                self.cx.setdefault(node.id, Counter())[sym] += 1
                node = node.parent
        self.cx_n = {nid: sum(d.values()) for nid, d in self.cx.items()}
        self.n_sym = len(KINDS) + len(self.top_c) + 1
        self.p_go = 1.0 - (n_pos + 1) / self.top_tot     # the top alphabet's non-stop mass
        self.pool_cum = [0.0]
        for g in range(1, 65):
            self.pool_cum.append(self.pool_cum[-1] + self.gap_pool[g])
        if CAUSALD >= 3:                                 # gap counts at every concept on each stored top unit's context path
            self.gcx, self.gvc = {"own": {}, "next": {}}, {}
            src = [("own", e, int(e.ctx["gp"])) for e, _, _ in top_units]
            src += [("next", e, g) for e, g in top_next]
            for which, e, g in src:
                node, tab = e.leaf, self.gcx[which]
                while node is not None:
                    tab.setdefault(node.id, Counter())[g] += 1
                    node = node.parent
        if GAP_DEBUG:
            self.check_causal()

    def sym_probs(self, xpath, sym):
        """P(symbol | X) at every concept X on the path (root first), each backed off from the
        root as Cobweb predicts: P(v | X) = (n_X(v) + P(v | parent)) / (n_X + 1), the root
        Laplace-smoothed; the last is P(symbol | the context concept)."""
        p, ps = None, []
        for node in xpath:
            d, n = self.cx.get(node.id, {}), self.cx_n.get(node.id, 0)
            c = d.get(sym, 0)
            p = (c + ALPHA) / (n + ALPHA * self.n_sym) if p is None else (c + p) / (n + 1.0)
            ps.append(p)
        return ps

    def sym_prob(self, xpath, sym):
        return self.sym_probs(xpath, sym)[-1]

    def causal_gap_bits(self, g, prev):
        """P(gap) pooled over all top units, renormalised over the gaps that stay on the board."""
        return -math.log2(self.gap_pool[g] / self.pool_cum[63 - prev])

    def gap_ctx_vec(self, which, node):
        """CAUSALD>=3: P(gap | context concept) over gaps 1..64 and its cumulative sums, backed off
        from the pooled gaps along the path: P(g | C) = (n_C(g) + P(g | parent)) / (n_C + 1), the
        root's parent the pooled distribution (``which``: "own" the unit's own gap, "next" the gap
        to the unit after it)."""
        key = (which, node.id)
        if key not in self.gvc:
            p0 = self.gap_ctx_vec(which, node.parent)[0] if node.parent is not None else self.gap_pool
            d = self.gcx[which].get(node.id, {})
            tot = sum(d.values())
            p = [0.0] + [(d.get(g, 0) + p0[g]) / (tot + 1.0) for g in range(1, 65)]
            cum = [0.0]
            for g in range(1, 65):
                cum.append(cum[-1] + p[g])
            self.gvc[key] = (p, cum)
        return self.gvc[key]

    def ctx_gap_bits(self, paths, g, prev, wins):
        """CAUSALD>=3: the gap by the learned mixture over every concept on the given context paths
        (each concept's P(gap | C) renormalised over the gaps that stay on the board)."""
        for which, path in paths:                        # diagnostic: each capped depth alone
            for d in GAP_DEPTHS:
                p, cum = self.gap_ctx_vec(which, path[min(d, len(path) - 1)])
                ACC[f"gap alone, {which} depth {d}"] += -math.log2(p[g] / cum[63 - prev])
        if GAPW:                                         # Bayesian weights over the capped depths (always live)
            ls = {}
            for which, path in paths:
                for d in GAP_DEPTHS:
                    p, cum = self.gap_ctx_vec(which, path[min(d, len(path) - 1)])
                    ls[(which, d)] = -math.log2(p[g] / cum[63 - prev])
            return bayes_mix(wins, ls)
        ls, live = {}, []
        for which, path in paths:
            for d, node in enumerate(path):
                p, cum = self.gap_ctx_vec(which, node)
                k = (which, d)
                ls[k] = -math.log2(p[g] / cum[63 - prev])
                live.append(k)
        return kt_mix(wins, ls, live)

    def check_causal(self):
        """GAP_DEBUG: P(symbol | X) sums to 1 over the alphabet along sampled context paths."""
        alphabet = [("w", t) for t in KINDS] + [("c", c) for c in self.top_c] + [("c", "novel")]
        assert len(alphabet) == self.n_sym
        leaves = [e.leaf for e, _, _ in top_units[:40]]
        for leaf in leaves + [L.ctx.root]:
            path = L.ctx.ancestors(leaf)[::-1]
            for k in (0, 1, 2):
                xp = path[:max(1, len(path) - k)]
                s = sum(self.sym_prob(xp, v) for v in alphabet)
                assert abs(s - 1) < 1e-9, (leaf.id, k, s)
        for prev in (-1, 0, 7, 31, 62):
            s = sum(2.0 ** -self.causal_gap_bits(g, prev) for g in range(1, 64 - prev))
            assert abs(s - 1) < 1e-9, (prev, s)
            if CAUSALD >= 3:                             # every concept's gap distribution, and their mixture
                for leaf in leaves + [L.ctx.root]:
                    path = L.ctx.ancestors(leaf)[::-1]
                    for which in ("own", "next"):
                        for node in path:
                            p, cum = self.gap_ctx_vec(which, node)
                            s = sum(p[g] / cum[63 - prev] for g in range(1, 64 - prev))
                            assert abs(s - 1) < 1e-9, (which, node.id, prev, s)
                    w = Counter({("own", 0): 3, ("next", 1): 1})
                    s = sum(2.0 ** -self.ctx_gap_bits([("own", path), ("next", path)], g, prev, Counter(w))
                            for g in range(1, 64 - prev))
                    assert abs(s - 1) < 1e-9, ("mixture", prev, s)
        print(json.dumps({"CAUSAL_DEBUG ok": [self.kc, self.kx], "alphabet": self.n_sym}), flush=True)

    # GAP: relative positions ------------------------------------------------ #
    def init_gaps(self):
        self.gap_up, self.gvec, self.acum = {}, {}, {}
        for (rid, g), c in gap_chunk.items():            # each top chunk's gap, counted up its concept's path
            node = lvl(cpath_of_record(rid), self.kc)
            while node is not None:
                self.gap_up.setdefault(node.id, Counter())[g] += c
                node = node.parent
        n_all = sum(gap_all.values())
        self.gap_pool = [0.0] + [(gap_all.get(g, 0) + ALPHA) / (n_all + 64 * ALPHA) for g in range(1, 65)]
        # single pieces: the top alphabet's single-piece mass, split over (piece, gap) by Laplace counts
        self.single_mass = (sum(single_tops.values()) + 768 * ALPHA) / self.top_tot
        per_g = [0.0] * 65
        for (_, g), c in gap_single.items():
            per_g[g] += c
        self.single_z = [0.0]                            # single_z[m]: the (piece, gap) mass over gaps 1..m
        for g in range(1, 65):
            self.single_z.append(self.single_z[-1] + per_g[g] + len(KINDS) * ALPHA)
        self.akey_anchor = {}                            # concept -> stored instances' weight by anchor index
        for cid, d in self.akeys.items():
            out = Counter()
            for key, n in d.items():
                out[min(ridx(sq) for sq, _ in key)] += n
            self.akey_anchor[cid] = out
        if GAP_DEBUG:
            self.check_gaps()

    def gap_vec(self, node):
        """P(gap | concept) over gaps 1..64, backed off along the concept's path from the root
        (P(g | C) = (n_C(g) + P(g | parent)) / (n_C + 1)), the root's parent the pooled gaps."""
        if node.id not in self.gvec:
            path, up_ = [], node
            while up_ is not None:
                path.append(up_)
                up_ = up_.parent
            p = self.gap_pool
            for n in reversed(path):
                d = self.gap_up.get(n.id, {})
                tot = sum(d.values())
                p = [0.0] + [(d.get(g, 0) + p[g]) / (tot + 1.0) for g in range(1, 65)]
            cum = [0.0]
            for g in range(1, 65):
                cum.append(cum[-1] + p[g])
            self.gvec[path[0].id] = (p, cum)
        return self.gvec[node.id]

    def chunk_gap_bits(self, cnode, g, prev):
        """-log2 P(gap | concept), renormalised over the gaps 1..63-prev that stay on the board."""
        p, cum = self.gap_vec(cnode)
        return -math.log2(p[g] / cum[63 - prev])

    def single_gap_bits(self, tok, g, prev):
        """A single top piece: P(single) P(piece, gap | single), over the feasible gaps."""
        p = self.single_mass * (gap_single.get((tok, g), 0) + ALPHA) / self.single_z[63 - prev]
        return self.used(("s", KINDS.index(tok) * 64 + prev + g), p)

    def pieces_gap_bits(self, toks_idx, prev):
        """A chunk written as its pieces, each a single top unit by its gap from the one before."""
        bits = 0.0
        for tok, a in toks_idx:
            bits += self.single_gap_bits(tok, a - prev, prev)
            prev = a
        return bits

    def abs_copy_gap_bits(self, cid, key, a, prev):
        """Copy on squares: GAP=1 given the decoded anchor a (the stored instances at a); GAP=2
        anchor included, over the stored instances whose anchor lies after prev."""
        n = self.akeys.get(cid, {}).get(key, 0)
        if not n:
            return float("inf")
        by_a = self.akey_anchor[cid]
        if GAP == 1:
            return -math.log2(n / by_a[a])
        ck = (cid, prev)
        if ck not in self.acum:
            self.acum[ck] = sum(v for b, v in by_a.items() if b > prev)
        return -math.log2(n / self.acum[ck])

    def check_gaps(self):
        """GAP_DEBUG: every new distribution sums to 1 over its feasible support."""
        nodes = list(self.nodes.values())[:50] + [L.cnt.root]
        for prev in (-1, 0, 7, 31, 50, 62):
            for node in nodes:
                s = sum(2.0 ** -self.chunk_gap_bits(node, g, prev) for g in range(1, 64 - prev))
                assert abs(s - 1) < 1e-9, (node.id, prev, s)
            s = sum(2.0 ** -self.single_gap_bits(t, g, prev) for t in KINDS for g in range(1, 64 - prev))
            assert USED or abs(s - self.single_mass) < 1e-9, (prev, s, self.single_mass)
            for cid in list(self.akeys)[:50]:
                keys = self.akeys[cid]
                anchors = {k: min(ridx(sq) for sq, _ in k) for k in keys}
                if GAP == 1:
                    for a in set(anchors.values()):
                        s = sum(2.0 ** -self.abs_copy_gap_bits(cid, k, a, prev) for k in keys if anchors[k] == a)
                        assert abs(s - 1) < 1e-9, (cid, a, s)
                else:
                    ks = [k for k in keys if anchors[k] > prev]
                    if ks:
                        s = sum(2.0 ** -self.abs_copy_gap_bits(cid, k, anchors[k], prev) for k in ks)
                        assert abs(s - 1) < 1e-9, (cid, prev, s)
        # the top alphabet (stop, every concept, the single-piece mass) stays within 1
        top = (n_pos + 1 + sum(self.top_c.values()) + ALPHA * len(self.top_c)) / self.top_tot + self.single_mass
        assert top <= 1 + 1e-12, top
        print(json.dumps({"GAP_DEBUG ok": [self.kc, self.kx], "top alphabet mass": top,
                          "nodes checked": len(nodes)}), flush=True)

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

    def anchor_bits(self, cid, sq, shape=None):
        if not ANCH or cid not in self.nodes:
            d = self.anchor.get(cid, {})
            return -math.log2((d.get(sq, 0) + ALPHA) / (sum(d.values()) + 64 * ALPHA))
        if cid not in self.avec:                         # P(sq | C) backed off along C's path, as template_bits
            path, node = [], self.nodes[cid]
            while node is not None:
                path.append(node)
                node = node.parent
            p = None
            for node in reversed(path):
                d = self.anchor_up.get(node.id, {})
                tot = sum(d.values())
                p = {s: (d.get(s, 0) + ALPHA) / (tot + 64 * ALPHA) for s in SQUARES} if p is None else \
                    {s: (d.get(s, 0) + p[s]) / (tot + 1.0) for s in SQUARES}
            self.avec[cid] = p
        p = self.avec[cid]
        if ANCH >= 2 and shape:                          # only anchors where the shape fits on the board
            return -math.log2(p[sq] / sum(p[s] for s in fits(shape)))
        return -math.log2(p[sq])

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


def flat_bits(M, cn, e, anchor, pb=None):
    """A flat chunk given its concept: anchor + offsets, or (with ``abs``) the even
    mixture of that and its pieces on their own squares. ``pb``: GAP, the anchor's gap bits."""
    rel = (M.anchor_bits(cn.id, anchor, shape_of(e)) if pb is None else pb) + M.template_bits(cn, e)
    if "abs" not in e.cnt:
        return rel
    ab = M.template_bits(cn, e, "abs") + (pb if GAP == 1 else 0.0)
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
    if GAP:                                          # the encoder's estimate: each piece's gap from the piece before it
        idx = [ridx(s) for s in sqs]
        before = [idx[i - 1] if i else -1 for i in range(len(idx))]
        single = {i: M.single_gap_bits(exp.tokens[i], idx[i] - before[i], before[i]) for i in range(len(idx))}
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
    # memo is keyed by id(e): every element offered here must outlive the memo, or a freed one's
    # address can be reused by a later chunk, which then reads a stale concept path
    r.setdefault("alive", []).extend(by_span.values())
    scored = []
    for span, e in by_span.items():
        path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
        cn = lvl(path, M.kc)
        anchor = C.square_name(sqs[min(span)])
        if GAP:
            i0 = min(span)
            pb = M.chunk_gap_bits(cn, idx[i0] - before[i0], before[i0])
            key = frozenset((sqs[i], exp.tokens[i]) for i in span)
            ls = [pb + M.template_bits(cn, e)]
            if "abs" in e.cnt:
                ls.append(M.template_bits(cn, e, "abs") + (pb if GAP == 1 else 0.0))
            if copy:
                ls.append(pb + M.copy_bits(cn.id, offsets(span, exp)))
                if ACOPY:
                    ls.append((pb if GAP == 1 else 0.0) + M.abs_copy_gap_bits(cn.id, key, idx[i0], before[i0]))
            save = sum(single[i] for i in span) - (M.top_bits(cn.id) + min(ls))
            if save > 0:
                scored.append((save, e))
            continue
        ls = [M.anchor_bits(cn.id, anchor, shape_of(e)) + M.template_bits(cn, e)]
        if "abs" in e.cnt:
            ls.append(M.template_bits(cn, e, "abs"))
        if copy:
            ls.append(M.anchor_bits(cn.id, anchor, shape_of(e)) + M.copy_bits(cn.id, offsets(span, exp)))
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


def kt_mix(wins, ls, live):
    """-log2 of the mixture of routes ``live`` with code lengths ``ls``, weighted by each
    route's share of the units it coded best so far (Krichevsky-Trofimov); the winner's
    count goes up after the unit, so a decoder keeps the same counts."""
    n = sum(wins[k] for k in live)
    m = min(ls[k] for k in live)
    if m == float("inf"):
        return m
    out = m - math.log2(sum((wins[k] + 0.5) / (n + 0.5 * len(live)) * 2.0 ** (m - ls[k]) for k in live))
    wins[min(live, key=lambda k: ls[k])] += 1
    return out


def bayes_mix(cum, ls):
    """-log2 of the Bayesian mixture of experts with code lengths ``ls`` (dict), each weighted by its
    probability of everything it coded before (``cum``: expert -> past bits, updated after the unit)."""
    keys = list(ls)
    m0 = min(cum[k] for k in keys)
    m1 = min(cum[k] + ls[k] for k in keys)
    num = sum(2.0 ** (m1 - cum[k] - ls[k]) for k in keys)
    den = sum(2.0 ** (m0 - cum[k]) for k in keys)
    for k in keys:
        cum[k] += ls[k]
    return m1 - math.log2(num) - (m0 - math.log2(den))


def causal_bits(M, e, exp, prev_e, prev, board, copy, memo):
    """CAUSAL: the unit by the causal route: that a unit comes (the top alphabet's non-stop mass),
    its gap from the previous unit (pooled), its symbol from its causal context concept, then (a
    chunk) its body given its concept and its now known anchor. The context is the learner's own
    causal description computed on the board decoded so far (``board``: the earlier units' pieces),
    sorted in the context tree, so a decoder that has decoded the earlier units finds the same
    concept before reading the symbol."""
    sqs = exp.data[1]
    sq = sqs[min(e.span)]
    a = ridx(sq)
    ck = ("cx", id(e), id(prev_e))
    if ck not in memo:
        x = L._full(L.causal_context(board, sq, prev_e, a - prev), L.ctx_attrs)
        if GAP_DEBUG:                        # the decoded board gives the description of the full position
            assert x == L._full(L.causal_context(exp.data[0], sq, prev_e, a - prev), L.ctx_attrs), (x, sq)
        memo[ck] = L.ctx.categorize(x)
    path = memo[ck]
    xpath = path[:max(1, len(path) - M.kx)]
    go, gb = -math.log2(M.p_go), M.causal_gap_bits(a - prev, prev)
    if CAUSALD >= 3:                         # the gap from what is known before the anchor
        ACC["causal: gaps, pooled"] += gb
        gk = ("cg", id(prev_e))
        if gk not in memo:                   # the partial description {pu}: missing attributes are left out
            x = {}
            L._facet(x, "pu", L._prev_name(prev_e))
            memo[gk] = L.ctx.categorize(x)
        paths = [("own", memo[gk][:max(1, len(memo[gk]) - M.kx)])]
        if CAUSALD == 4 and M.last_cx is not None:   # the previous unit's own decoded context path
            paths.append(("next", M.last_cx[:max(1, len(M.last_cx) - M.kx)]))
        gb = M.ctx_gap_bits(paths, a - prev, prev, DEPTHG.setdefault((M.kc, M.kx, copy), Counter()))
    M.last_cx = path

    def sym_bits(sym):
        if CAUSALD == 2:                     # ablation: the root only (the context tree does no work)
            return -math.log2(M.sym_prob(xpath[:1], sym))
        if not CAUSALD:                      # the context concept's own prediction
            return -math.log2(M.sym_prob(xpath, sym))
        # CAUSALD=1: every concept on the path predicts (root to context concept), mixed by
        # learned weights per depth (one set for all units, so the mixture is one distribution)
        ls = [-math.log2(p) for p in M.sym_probs(xpath, sym)]
        return kt_mix(DEPTHW.setdefault((M.kc, M.kx, copy), Counter()), ls, range(len(ls)))
    if e.token is not None:
        sb = sym_bits(("w", e.token))
        ACC["causal: single symbols"] += sb
        return go + gb + sb, go, gb, 0.0
    cn = lvl(memo[id(e)], M.kc)
    sb = sym_bits(("c", cn.id))
    ACC["causal: chunk symbols"] += sb
    ls = [M.template_bits(cn, e),
          M.template_bits(cn, e, "abs") if "abs" in e.cnt else float("inf"),
          M.copy_bits(cn.id, offsets(e.span, exp)) if copy else float("inf"),
          float("inf")]
    if copy and ACOPY:                       # the stored instance on its squares, given the anchor
        key = frozenset((sqs[i], exp.tokens[i]) for i in e.span)
        n = M.akeys.get(cn.id, {}).get(key, 0)
        ls[3] = -math.log2(n / M.akey_anchor[cn.id][a]) if n else float("inf")
    live = [k for k in range(4) if not (k == 1 and "abs" not in e.cnt) and not (k >= 2 and not copy)
            and not (k == 3 and not ACOPY)]
    body = kt_mix(CBODY.setdefault((M.kc, M.kx, copy), [0.0] * 4), ls, live)
    return go + gb + sb + body, go, gb, body


def causal_mix(M, old, new, go, gb, copy):
    """One unit's code: the existing code and the causal route mixed by learned weights (one
    pair of counts for every unit, single or chunk, so the mixture is one distribution)."""
    ACC["existing route, unit bits"] += old
    ACC["causal route, unit bits"] += new
    ACC["causal: go"] += go
    ACC["causal: gaps"] += gb
    ACC["causal route cheaper (count)"] += new < old
    ACC["units (count)"] += 1
    b = kt_mix(CTXW.setdefault((M.kc, M.kx, copy), [0.0, 0.0]), [old, new], [0, 1])
    ACC["mixed, unit bits"] += b
    return b


def code_bits(M, r, copy=False):
    exp, memo = r["exp"], r["memo"]
    sqs = exp.data[1]
    bits = M.stop_bits()
    board, occupied = {}, set()
    prev_e = None                            # CAUSAL: the unit coded just before
    if CAUSAL:
        M.last_cx = None                     # CAUSALD=4: that unit's context path
    M.pending = ["stop"]                     # the units this code used, counted once the position is coded
    todo = cover(M, r, copy) if COVER and L.dom.flat else r["top"]
    M.units = [e.span for e in todo]        # TOPC: learning stores these as its analysis
    if GAP:                                  # units in increasing anchor order; anchors as gaps
        todo = sorted(todo, key=lambda e: min(e.span))
    prev = -1
    for e in todo:
        units = [(exp.tokens[i], sqs[i]) for i in e.span]
        a = ridx(sqs[min(e.span)])
        if e.token is not None:
            (i,) = tuple(e.span)
            M.pending.append(("s", psq(e.token, sqs[i])))
            if GAP:
                sb = M.single_gap_bits(e.token, a - prev, prev)
                ACC["single tops"] += sb
                if CAUSAL:                   # the existing code and the causal route, mixed by learned weights
                    cb, go, gb, _ = causal_bits(M, e, exp, prev_e, prev, board, copy, memo)
                    sb = causal_mix(M, sb, cb, go, gb, copy)
                    prev_e = e
                bits += sb
                prev = a
            else:
                ACC["single tops"] += M.single_bits(e.token, sqs[i])
                bits += M.single_bits(e.token, sqs[i])
            board.update({sq: t for t, sq in units})
            occupied.add(C.square_name(sqs[i]))
            continue
        path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
        cn = lvl(path, M.kc)
        M.pending.append(("c", cn.id))
        ACC["chunk roots"] += M.top_bits(cn.id)
        ACC["anchors"] += M.anchor_bits(cn.id, C.square_name(sqs[min(e.span)]), shape_of(e))
        ACC["chunk tops (count)"] += 1
        anchor = C.square_name(sqs[min(e.span)])
        pb = None
        if GAP:                              # the anchor's code on the gap routes, in place of anchor_bits
            pb = M.chunk_gap_bits(cn, a - prev, prev)
            ACC["chunk gaps"] += pb
            akey = frozenset((sqs[i], exp.tokens[i]) for i in e.span)
            abc = (pb if GAP == 1 else 0.0) + M.abs_copy_gap_bits(cn.id, akey, a, prev) if ACOPY else float("inf")
        if L.dom.flat:                       # one level up: the concept predicts the whole chunk (anchor included)
            inner = flat_bits(M, cn, e, anchor, pb)
            ACC["templates"] += inner
        else:
            inner = (M.anchor_bits(cn.id, anchor, shape_of(e)) if pb is None else pb) + inside_bits(M, e, exp, cn, memo)
        if copy:
            cb = (M.anchor_bits(cn.id, anchor, shape_of(e)) if pb is None else pb) + M.copy_bits(cn.id, offsets(e.span, exp))
            if ACOPY:
                ab = M.abs_copy_bits(cn.id, frozenset((sqs[i], exp.tokens[i]) for i in e.span)) if pb is None else abc
                ACC["abs copy cheaper (count)"] += ab < cb
                cb = either(cb, ab)
            ACC["chunk tops copied (count)"] += cb < inner
            ACC["copy, choosing the instance"] += cb if cb < inner else 0.0
            inner = either(cb, inner)
            if ROUTEW and L.dom.flat:
                key = frozenset((sqs[i], exp.tokens[i]) for i in e.span)
                if GAP:
                    ls = [pb + M.template_bits(cn, e),
                          M.template_bits(cn, e, "abs") + (pb if GAP == 1 else 0.0) if "abs" in e.cnt else float("inf"),
                          pb + M.copy_bits(cn.id, offsets(e.span, exp)),
                          abc]
                    top = M.top_bits(cn.id)
                    ls = [top + v for v in ls] + [M.pieces_gap_bits([(exp.tokens[i], ridx(sqs[i])) for i in sorted(e.span)], prev)]
                else:
                    ls = [M.anchor_bits(cn.id, anchor, shape_of(e)) + M.template_bits(cn, e),
                          M.template_bits(cn, e, "abs") if "abs" in e.cnt else float("inf"),
                          M.anchor_bits(cn.id, anchor, shape_of(e)) + M.copy_bits(cn.id, offsets(e.span, exp)),
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
            ACC["context head cheaper (count)"] += hb < M.top_bits(cn.id) + M.anchor_bits(cn.id, anchor, shape_of(e))
            ACC["context head"] += hb
            head = either(head, hb + body)
        if CAUSAL:
            cb, go, gb, body = causal_bits(M, e, exp, prev_e, prev, board, copy, memo)
            ACC["causal: chunk bodies"] += body
            head = causal_mix(M, head, cb, go, gb, copy)
            prev_e = e
        bits += head
        prev = a
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
            best = min(range(len(Ms)), key=lambda j: costs[j])
            top = Ms[best].units if TOPC and costs[best] <= raw else None
        else:
            bits, top = raw, None
        tot, tot_raw = tot + bits, tot_raw + raw
        learn_one(f, top)
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
if GAP:
    summary["gap"] = GAP
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
        prev = -1
        for e in (sorted(r["top"], key=lambda e: min(e.span)) if GAP else r["top"]):
            a = ridx(sqs[min(e.span)])
            if e.token is not None:
                (i,) = tuple(e.span)
                if GAP:
                    bits += mix([M.single_gap_bits(e.token, a - prev, prev) for M in Ms])
                    prev = a
                else:
                    bits += mix([M.single_bits(e.token, sqs[i]) for M in Ms])
                continue
            path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
            anchor = C.square_name(sqs[min(e.span)])
            if GAP:
                bits += mix([M.top_bits(lvl(path, M.kc).id) + flat_bits(M, lvl(path, M.kc), e, anchor,
                             M.chunk_gap_bits(lvl(path, M.kc), a - prev, prev)) for M in Ms])
                prev = a
            else:
                bits += mix([M.top_bits(lvl(path, M.kc).id) + flat_bits(M, lvl(path, M.kc), e, anchor) for M in Ms])
        b.append(bits)
    row = {"content level": "mixture of 0-3", "bits per position": round(sum(b) / len(b), 1)}
    summary["by levels"].append(row)
    print(json.dumps(row), flush=True)
out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
json.dump(summary, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(summary), flush=True)
