"""Chunking baselines for chess compression on the place-transfer sets
(chess_transfer_sets.py): each learns the first n_learn positions of the learn set
and codes every test set; held-out code bits per position, the model not charged
(its size is reported beside).

  bpe-abs   co-occurrence merges of pieces on squares (chess_compress_bpe.py, encoded
            the standard BPE way: repeatedly merge the pair of lowest merge rank)
  bpe-rel   the same merges of shapes: a unit is pieces at offsets from an anchor
            square plus that anchor, and a merge rule is the shape two units form,
            wherever they stand, so the dictionary is translation-invariant. A unit
            costs -log2 P(shape) - log2 P(anchor | shape), the anchor backed off to
            the anchors of all units (a single piece: its kind, then its square)
  krimp     MDL itemsets (Vreeken, van Leeuwen & Siebes 2011): frequent itemsets of
            pieces on squares, taken in the standard candidate order when they lower
            the total description (code table + data), positions covered in the
            standard cover order
  xz        a general-purpose compressor: xz(learn + one test position) - xz(learn), on
            board FENs, averaged over the first 200 positions of each set

Usage: chess_baselines.py <seed> <tag> <n_learn> <method> [max_size n_merges]
"""
import heapq
import json
import lzma
import math
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2 import chess as C  # noqa: E402

seed, tag, n_learn, method = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
max_size, n_merges = [int(x) for x in sys.argv[5:7]] if len(sys.argv) > 6 else [4, 3000]
T = os.environ.get("TRANSFER_DIR", os.path.join(ROOT, "data", "chess", "transfer"))
SETS = ["control", "dropped", "family", "wing", "shift"]
KS = [k for k in (100, 300, 1000, 3000) if k <= n_merges]


def read(name, n=None):
    fens = [l.strip() for l in open(os.path.join(T, f"{name}.fen"), encoding="utf-8") if l.strip()]
    return fens[:n] if n else fens


learn_fens = read("learn", n_learn)
tests = {s: read(s) for s in SETS}
out = {"tag": tag, "seed": seed, "method": method, "positions learned": n_learn, "max size": max_size, "sets": {}}
t0 = time.time()


def emit():
    out["s"] = round(time.time() - t0)
    path = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress", f"{tag}_s{seed}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out), flush=True)


if method == "xz":
    def xz(fens):
        return 8 * len(lzma.compress(("\n".join(fens) + "\n").encode(), preset=9 | lzma.PRESET_EXTREME))
    base = xz(learn_fens)
    n_xz = int(os.environ.get("XZ_N", 200))             # each position coded alone given the learn set
    for s, fens in tests.items():
        out["sets"][s] = {"bits per position": round(sum(xz(learn_fens + [f]) - base for f in fens[:n_xz]) / n_xz, 1),
                          "positions coded": n_xz}
    emit()
    sys.exit(0)

learn = [C.parse_fen(f) for f in learn_fens]
held = {s: [C.parse_fen(f) for f in fens] for s, fens in tests.items()}

# ---------------------------------------------------------------- BPE (both)
REL = method == "bpe-rel"


def unit_of(sq, tok):
    return (frozenset([(0, 0, tok)]), sq) if REL else frozenset([(sq, tok)])


def size(u):
    return len(u[0]) if REL else len(u)


def points(u):
    if REL:
        (af, ar) = u[1]
        return [((af + df, ar + dr), t) for df, dr, t in u[0]]
    return list(u)


def join(a, b):
    """The unit two units form, and its merge key (the key ignores place for bpe-rel)."""
    if not REL:
        u = a | b
        return u, u
    pts = points(a) + points(b)
    anchor = min(sq for sq, _ in pts)
    shape = frozenset((sq[0] - anchor[0], sq[1] - anchor[1], t) for sq, t in pts)
    return (shape, anchor), shape


def pairs_of(units):
    c = Counter()
    for a, b in combinations(units, 2):
        if size(a) + size(b) <= max_size:
            c[join(a, b)[1]] += 1
    return c


def merge_key(units, key):
    """Merge pairs forming ``key`` until none is left."""
    units = list(units)
    while True:
        hit = None
        for i, j in combinations(range(len(units)), 2):
            if size(units[i]) + size(units[j]) <= max_size:
                u, k = join(units[i], units[j])
                if k == key:
                    hit = (i, j, u)
                    break
        if hit is None:
            return units
        i, j, u = hit
        units = [x for n, x in enumerate(units) if n not in (i, j)] + [u]


def encode(pos, rank, K):
    """Standard BPE encoding: repeatedly merge the pair of lowest rank below K."""
    units = [unit_of(sq, t) for sq, t in pos.items()]
    while True:
        best = None
        for i, j in combinations(range(len(units)), 2):
            if size(units[i]) + size(units[j]) <= max_size:
                u, k = join(units[i], units[j])
                r = rank.get(k)
                if r is not None and r < K and (best is None or r < best[0]):
                    best = (r, i, j, u)
        if best is None:
            return units
        _, i, j, u = best
        units = [x for n, x in enumerate(units) if n not in (i, j)] + [u]


if method in ("bpe-abs", "bpe-rel"):
    tr = [[unit_of(sq, t) for sq, t in pos.items()] for pos in learn]
    pp = [pairs_of(u) for u in tr]
    count, where, ids, heap = Counter(), defaultdict(set), {}, []
    for pid, c in enumerate(pp):
        count.update(c)
        for k in c:
            where[k].add(pid)

    def kid(k):
        if k not in ids:
            ids[k] = len(ids)
        return ids[k]

    for k, c in count.items():
        heapq.heappush(heap, (-c, kid(k), k))
    merges = []
    while heap and len(merges) < n_merges:
        c, _, k = heapq.heappop(heap)
        if -c != count.get(k, 0):
            continue                                   # stale entry
        if -c < 2:
            break
        merges.append(k)
        for pid in list(where[k]):
            new_units = merge_key(tr[pid], k)
            if new_units == tr[pid]:
                continue
            tr[pid] = new_units
            old, new = pp[pid], pairs_of(new_units)
            for kk in set(old) | set(new):
                d = new.get(kk, 0) - old.get(kk, 0)
                if d:
                    count[kk] += d
                    heapq.heappush(heap, (-count[kk], kid(kk), kk))
                if kk in new:
                    where[kk].add(pid)
            pp[pid] = new
        count[k] = 0
    rank = {k: i for i, k in enumerate(merges)}
    out["merges made"] = len(merges)
    out["learn s"] = round(time.time() - t0)
    for K in [0] + [k for k in KS if k <= len(merges)] + ([len(merges)] if len(merges) not in KS else []):
        enc = [encode(pos, rank, K) for pos in learn]
        n_alpha = 768 if not REL else 12
        if REL:
            shapes = Counter(u[0] for us in enc for u in us)
            anch = defaultdict(Counter)
            for us in enc:
                for u in us:
                    anch[u[0]][u[1]] += 1
            glob = Counter(u[1] for us in enc for u in us)
            n_glob = sum(glob.values())
            tot = sum(shapes.values()) + n_learn + n_alpha + len([s for s in shapes if len(s) > 1]) + 1

            def cost(u):
                p_s = (shapes.get(u[0], 0) + 1) / tot
                pg = (glob.get(u[1], 0) + 1) / (n_glob + 64)
                a = anch.get(u[0], {})
                p_a = (a.get(u[1], 0) + pg) / (sum(a.values()) + 1)
                return -math.log2(p_s) - math.log2(p_a)
            n_chunks = len([s for s in shapes if len(s) > 1])
            dict_bits = sum(len(s) * math.log2(15 * 15 * 12) + math.log2(max_size) for s in shapes if len(s) > 1)
        else:
            use = Counter(u for us in enc for u in us)
            n_chunks = len([u for u in use if len(u) > 1])
            tot = sum(use.values()) + n_learn + n_alpha + n_chunks + 1

            def cost(u):
                return -math.log2((use.get(u, 0) + 1) / tot)
            dict_bits = sum(len(u) * math.log2(768) + math.log2(max_size) for u in use if len(u) > 1)
        stop = -math.log2((n_learn + 1) / tot)
        for s, ps in held.items():
            b = sum(sum(cost(u) for u in encode(pos, rank, K)) + stop for pos in ps) / len(ps)
            out["sets"].setdefault(s, {})[f"merges {K}"] = round(b, 1)
        out.setdefault("chunks", {})[f"merges {K}"] = n_chunks
        out.setdefault("dictionary bits", {})[f"merges {K}"] = round(dict_bits)
        print(json.dumps({"merges": K, "chunks": n_chunks, **{s: out["sets"][s][f"merges {K}"] for s in held}}), flush=True)
    for s in held:
        out["sets"][s]["best"] = min(v for k, v in out["sets"][s].items() if k.startswith("merges") and k != "merges 0")
    emit()
    sys.exit(0)

# ---------------------------------------------------------------- Krimp
if method == "krimp":
    MINSUP = max(2, n_learn // 200)
    MAXCAND = int(os.environ.get("KRIMP_CAND", 4000))
    trans = [frozenset((sq, t) for sq, t in pos.items()) for pos in learn]
    item_sup = Counter(i for x in trans for i in x)
    # frequent itemsets up to max_size, level-wise (Apriori)
    level = {frozenset([i]): c for i, c in item_sup.items() if c >= MINSUP}
    freq = {}
    occ = defaultdict(set)
    for tid, x in enumerate(trans):
        for i in x:
            occ[frozenset([i])].add(tid)
    for k in range(2, max_size + 1):
        cands = Counter()
        keys = list(level)
        singles = sorted({i for s in keys for i in s}, key=str)
        for s in keys:
            for i in singles:
                if i not in s:
                    u = s | {i}
                    if len(u) == k and u not in cands and all((u - {j}) in level for j in u):
                        cands[u] = 0
        nxt = {}
        for u in cands:
            a, b = sorted(u, key=str)[:2]
            tids = occ[u - {a}] & occ[u - {b}] if k > 2 else occ[frozenset([a])] & occ[frozenset([b])]
            if len(tids) >= MINSUP:
                nxt[u] = len(tids)
                occ[u] = tids
        freq.update(nxt)
        level = nxt
        if not level:
            break
    order = sorted(freq, key=lambda s: (-freq[s], -len(s), sorted(map(str, s))))[:MAXCAND]
    out["candidates"] = len(order)
    std = {i: -math.log2(c / sum(item_sup.values())) for i, c in item_sup.items()}

    ct = [frozenset([i]) for i in item_sup]                # the code table, in standard cover order

    def cover_order(table):
        return sorted(table, key=lambda s: (-len(s), -(freq.get(s) or item_sup[next(iter(s))]), sorted(map(str, s))))

    def cover(x, table):
        rest, used = set(x), []
        for s in table:
            if s <= rest:
                used.append(s)
                rest -= s
            if not rest:
                break
        return used + [frozenset([i]) for i in rest]       # pieces never seen: singletons

    def total(usage):
        n = sum(usage.values())
        data = sum(c * -math.log2(c / n) for c in usage.values() if c)
        model = sum(sum(std.get(i, 10.0) for i in s) - math.log2(c / n) for s, c in usage.items() if c)
        return data + model

    table = cover_order(ct)
    covers = [cover(x, table) for x in trans]
    usage = Counter(s for cv in covers for s in cv)
    best = total(usage)
    for cand in order:
        trial = cover_order(table + [cand])
        tids = occ[cand]
        new_cov = {tid: cover(trans[tid], trial) for tid in tids}
        u2 = usage.copy()
        for tid, cv in new_cov.items():
            u2.subtract(covers[tid])
            u2.update(cv)
        u2 = +u2
        L = total(u2)
        if L < best:
            best, table, usage = L, trial, u2
            for tid, cv in new_cov.items():
                covers[tid] = cv
    chunks = [s for s in table if len(s) > 1 and usage.get(s, 0) > 0]
    out["chunks"] = len(chunks)
    out["total description bits (learn)"] = round(best)
    n = sum(usage.values())
    tot = n + n_learn + len(table) + 768 + 1
    stop = -math.log2((n_learn + 1) / tot)
    for s, ps in held.items():
        b = sum(sum(-math.log2((usage.get(u, 0) + 1) / tot) for u in cover(frozenset(p.items()), table)) + stop
                for p in ps)
        out["sets"][s] = {"bits per position": round(b / len(ps), 1)}
    emit()
