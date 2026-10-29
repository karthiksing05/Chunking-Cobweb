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
Bits are code bits only: the memory itself (the model) is not charged.

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
from trellis2.trellis3 import FILL, LPART, RPART  # noqa: E402
from trellis2.trellis_graph import TOP, GraphTrellis  # noqa: E402

cfg, seed, tag = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n_rep, n_learn, n_code = [int(x) for x in sys.argv[4:7]] if len(sys.argv) > 6 else [4000, 2000, 1000]
max_size = cfg.pop("max_size", 4)
EPS, ALPHA = 0.05, 0.5
LEVELS = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]

fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
learn_fens, code_fens = fens[:n_learn], fens[n_rep:n_rep + n_code]
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
n_pos = 0
t0 = time.time()
for f in learn_fens:
    r = L.learn(C.parse_fen(f))
    sqs = r["exp"].data[1]
    for k in r["chosen"]:
        e = r["comps"][k]
        if e.rid is not None:
            stored[frozenset((sqs[i], r["exp"].tokens[i]) for i in e.span)] += 1
            rel_key[e.rid] = offsets(e.span, r["exp"])
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            single_tops[psq(e.token, sqs[i])] += 1
    n_pos += 1
learn_s = time.time() - t0
L._quiet = True

raw_use = Counter(psq(t, sq) for f in learn_fens for sq, t in C.parse_fen(f).items())
raw_tot = sum(raw_use.values()) + 768


def raw_bits(tok, sq):
    return -math.log2((raw_use.get(psq(tok, sq), 0) + 1) / raw_tot)


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
        self.top_c, self.anchor, self.size, self.keys = Counter(), {}, {}, {}
        for rid, c in tops.items():
            cid = lvl(cpath_of_record(rid), kc).id
            self.top_c[cid] += c
            sq = (L.records[rid].ctx or {}).get("sq")
            if sq:
                self.anchor.setdefault(cid, Counter())[sq] += c
            self.size.setdefault(cid, Counter())[len(L.records[rid].span)] += c
            if rid in rel_key:
                self.keys.setdefault(cid, Counter())[rel_key[rid]] += c
        self.n_concepts = len(self.top_c) + 1
        self.top_tot = sum(self.top_c.values()) + sum(single_tops.values()) + n_pos + ALPHA * (self.n_concepts + 768) + 1
        self.fill_cache = {}

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

    def top_bits(self, cid):
        return -math.log2((self.top_c.get(cid, 0) + ALPHA) / self.top_tot)

    def single_bits(self, tok, sq):
        return -math.log2((single_tops.get(psq(tok, sq), 0) + ALPHA) / self.top_tot)

    def stop_bits(self):
        return -math.log2((n_pos + 1) / self.top_tot)

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


def code_bits(M, r, copy=False):
    exp, memo = r["exp"], r["memo"]
    sqs = exp.data[1]
    bits = M.stop_bits()
    for e in r["top"]:
        if e.token is not None:
            (i,) = tuple(e.span)
            ACC["single tops"] += M.single_bits(e.token, sqs[i])
            bits += M.single_bits(e.token, sqs[i])
            continue
        path = memo.setdefault(id(e), L.cnt.categorize(e.cnt))
        cn = lvl(path, M.kc)
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
            ACC["chunk tops copied (count)"] += cb < inner
            ACC["copy, choosing the instance"] += cb if cb < inner else 0.0
            inner = either(cb, inner)
        bits += M.top_bits(cn.id) + inner
    return bits


t1 = time.time()
parsed = []
for f in code_fens:
    r = L.parse(C.parse_fen(f))
    r["memo"] = {}
    parsed.append(r)
parse_s = time.time() - t1
rbits = sum(sum(raw_bits(t, sq) for sq, t in C.parse_fen(f).items()) - math.log2((n_pos + 1) / (sum(raw_use.values()) + n_pos + 769))
            for f in code_fens) / len(code_fens)
stop_raw = -math.log2((n_pos + 1) / (sum(raw_use.values()) + n_pos + 769))   # one stop item per listed chunk, as the dictionaries

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
