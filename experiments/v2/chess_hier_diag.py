"""Step 1 diagnostic: do larger board configurations pay as nested chunks, and
what stops the greedy search from building them?

For one data setting (minimum rating, ply; the split of run_chess_chunks.py:
seed-13 permutation, first N learned), counts on, one alpha:

1. Mine the most frequent connected configurations of 3-6 pieces under the
   board's relations: a configuration is an anchor piece (the first of its
   pieces in the read) and pieces each seen from the anchor along a forward
   ray or knight jump, the configuration's own pieces not blocking (as a
   chunk sees past its own pieces). Translation-invariant: (anchor token,
   {(offset, token)}). Its binary build is left-branching from the anchor,
   members in read order of their offsets: [[[a r1 x1] r2 x2] r3 x3].
2. For each configuration, the exact BoardSearch code change (plain code,
   counts on) of making it nested binary chunks, from two starts: the read
   alone (flat) and the state the greedy chunk search converges to (greedy):
   (a) step by step, each step a global chunk move (every non-overlapping
       pair of those labels in that relation, as the search would apply it);
   (b) as whole configurations: the same steps applied only to the
       configuration's own occurrences, scored step by step and in total;
   (c) with oracle categories: the configuration's variants (same anchor and
       same chain up to the last part, which differs) applied as in (b), then
       their whole-configuration labels merged into one category (a merge
       move, scored exactly); and labels of either side (white / black
       versions of a shape) merged.
   Every change is split into the read's answers, the top-level labels and
   the definition rows.

    python chess_hier_diag.py --elo 1800 --ply 16 --alpha 1e-3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chess_search import HierBoardSearch  # noqa: E402
from run_chess_chunks import positions_path  # noqa: E402
from run_chess_memory import fen_to_board, to_position  # noqa: E402
from trellis2.chess import OFFSET, forward_neighbours  # noqa: E402

OUT = os.path.join(HERE, "results", "chess_hier", "diag")
REL_OF = {v: k for k, v in OFFSET.items()}
FILES = "abcdefgh"


def sqname(sq):
    return FILES[sq[0]] + str(sq[1] + 1)


def load(elo, ply, n, seed=13):
    with open(positions_path(elo, ply)) as f:
        fens = [line.strip() for line in f if line.strip()]
    order = np.random.default_rng(seed).permutation(len(fens))
    return [to_position(fen_to_board(fens[i])) for i in order[:n]]


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #
def mine(positions, max_size=6, minsup=40, keep=200):
    """{size: [(key, instances)]}, most frequent first. key = (anchor token,
    tuple of (offset, token) in read order); instance = (position index,
    anchor square, tuple of member squares in read order)."""
    out = {}
    level = defaultdict(dict)       # key -> {(pi, anchor, frozenset): members}
    for pi, p in enumerate(positions):
        for s, tok in p.items():
            for rel, t in forward_neighbours(p, s, frozenset([s])):
                off = (t[0] - s[0], t[1] - s[1])
                key = (tok, ((off, p[t]),))
                level[key][(pi, s, frozenset([t]))] = (t,)
    size = 2
    while True:
        ranked = sorted(((k, v) for k, v in level.items() if len(v) >= minsup),
                        key=lambda kv: (-len(kv[1]), str(kv[0])))[:keep]
        out[size] = [(k, list(v.items())) for k, v in ranked]
        if size == max_size:
            break
        nxt = defaultdict(dict)
        for key, inst in ranked:
            for (pi, s, mem), _ in inst.items():
                p = positions[pi]
                own = frozenset(mem | {s})
                for rel, t in forward_neighbours(p, s, own):
                    m2 = mem | {t}
                    members = tuple(sorted(m2, key=lambda q: (q[1] - s[1], q[0] - s[0])))
                    k2 = (key[0], tuple(((q[0] - s[0], q[1] - s[1]), p[q]) for q in members))
                    nxt[k2].setdefault((pi, s, m2), members)
        level = nxt
        size += 1
    return {sz: [(k, [(pi, s, members) for (pi, s, _), members in inst]) for k, inst in lst]
            for sz, lst in out.items()}


def render_key(key):
    a, mem = key
    node = a
    for off, tok in mem:
        node = f"[{node} {REL_OF[off]} {tok}]"
    return node


def classify(key):
    a, mem = key
    toks = [a] + [t for _, t in mem]
    colours = {t[0] for t in toks}
    kinds = Counter(t[1] for t in toks)
    names = []
    if len(colours) == 1 and kinds["K"] and kinds["R"] and kinds["P"]:
        names.append("castled king (K+R+pawns)")
    offs = {(0, 0): a, **{o: t for o, t in mem}}
    for o, t in offs.items():
        if t[1] == "B":
            front = (o[0], o[1] + (1 if t[0] == "w" else -1))
            if offs.get(front) == t[0] + "P" and any(u == t[0] + "K" for u in offs.values()):
                names.append("fianchetto with king")
                break
    if kinds["P"] == len(toks) and len(colours) == 1 and len(toks) >= 3:
        if all(o[0] == o[1] for o, _ in mem) or all(-o[0] == o[1] for o, _ in mem):
            names.append("pawn chain")
    return names


# --------------------------------------------------------------------------- #
# Applying a configuration's chain
# --------------------------------------------------------------------------- #
def chain(key):
    """[(relation, token)] of the left-branching build."""
    return [(REL_OF[off], tok) for off, tok in key[1]]


def _valid(search, pi, s, members, step):
    """Whether the instance can take step ``step`` (0-based) in this state:
    the anchor's top-level element holds exactly the members before it and
    the member is a top-level piece seen in that relation."""
    p = search.pos[pi]
    t = members[step]
    if p["owner"].get(t) != t or t not in p["tops"] or len(p["tops"][t][1]) == 3:
        return False
    own = frozenset(m for m, o in p["owner"].items() if o == s)
    if own != frozenset((s,) + members[:step]):
        return False
    return any(q == t for _, q in search._seen(p, s))


def parts_delta(before, after):
    return {k: after[k] - before[k] for k in before}


def apply_joint(search, key, instances, label0=None):
    """Steps applied to the configuration's own occurrences only. Returns
    per-step [(bits change, parts change, n)] and the label made at each
    step (the last is the whole configuration's)."""
    steps, lab, made = [], label0 or key[0], []
    live = []
    used = defaultdict(set)
    for pi, s, members in sorted(instances, key=lambda x: (x[0], x[1][1], x[1][0])):
        sqs = {s, *members}
        if used[pi] & sqs:
            continue
        if search.pos[pi]["tops"].get(s, (None,))[0] != lab or not _valid(search, pi, s, members, 0):
            continue
        used[pi] |= sqs
        live.append((pi, s, members))
    for j, (rel, tok) in enumerate(chain(key)):
        live = [x for x in live if _valid(search, x[0], x[1], x[2], j)
                and search.pos[x[0]]["tops"][x[2][j]][0] == tok]
        if len(live) < 2:
            return steps, None
        b0, p0 = search.bits(), search.parts()
        chosen = [(pi, s, members[j]) for pi, s, members in live]
        made.append(search.resolve(("chunk", search.fresh)))
        predicted = search.score((lab, rel, tok), chosen, {})
        lab = search.apply((lab, rel, tok), chosen)
        b1 = search.bits()
        assert abs(b1 - predicted) < 1e-5
        steps.append((b1 - b0, parts_delta(p0, search.parts()), len(chosen)))
    return steps, made


def apply_global(search, key):
    """Steps applied as the search would: each a global chunk move."""
    steps, lab = [], key[0]
    for rel, tok in chain(key):
        pairs = search.candidates().get((lab, rel, tok), [])
        chosen = search.select(pairs)
        if len(chosen) < 2:
            return steps, None
        b0, p0 = search.bits(), search.parts()
        lab = search.apply((lab, rel, tok), chosen)
        steps.append((search.bits() - b0, parts_delta(p0, search.parts()), len(chosen)))
    return steps, lab


def fmt_steps(steps):
    return [{"bits": round(d, 2), "n": n, **{k: round(v, 2) for k, v in pd.items()}} for d, pd, n in steps]


def mirror(key):
    """The same shape for the other side, as the read sees it: colours
    swapped and the board flipped top to bottom (the anchor is then the
    configuration's first piece in the read again)."""
    a, mem = key
    pts = [((0, 0), a)] + list(mem)
    flipped = [((o[0], -o[1]), ("b" if t[0] == "w" else "w") + t[1]) for o, t in pts]
    anchor = min(flipped, key=lambda x: (x[0][1], x[0][0]))
    rest = sorted(((o[0] - anchor[0][0], o[1] - anchor[0][1]), t) for o, t in flipped if o != anchor[0])
    rest = sorted(rest, key=lambda x: (x[0][1], x[0][0]))
    return (anchor[1], tuple(rest))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default="1800")
    ap.add_argument("--ply", type=int, default=16)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--top", type=int, default=6, help="most frequent configurations per size evaluated")
    ap.add_argument("--named", type=int, default=4, help="most frequent per named class and size")
    ap.add_argument("--minsup", type=float, default=0.01)
    args = ap.parse_args()
    t0 = time.time()
    positions = load(args.elo, args.ply, args.train)
    name = f"e{args.elo}_p{args.ply}_n{args.train}_a{args.alpha:g}"
    mined = mine(positions, minsup=max(2, int(args.minsup * len(positions))))
    print(f"mined in {time.time() - t0:.0f}s:", {k: len(v) for k, v in mined.items()}, flush=True)
    index = {k: inst for sz in mined for k, inst in mined[sz]}

    flat = HierBoardSearch(positions, args.alpha, True, merges=False)
    greedy = copy.deepcopy(flat)
    greedy.run(500)
    print(f"flat {flat.bits() / len(positions):.3f} greedy {greedy.bits() / len(positions):.3f} bits/pos, "
          f"{len(greedy.moves)} moves, {time.time() - t0:.0f}s", flush=True)
    starts = {"flat": flat, "greedy": greedy}

    chosen = []
    for sz in range(3, 7):
        lst = mined.get(sz, [])
        picks = [k for k, _ in lst[:args.top]]
        per_class = Counter()
        for k, _ in lst:
            for c in classify(k):
                if per_class[c] < args.named and k not in picks:
                    picks.append(k)
                    per_class[c] += 1
        chosen += [(sz, k) for k in picks]

    results = []
    for sz, key in chosen:
        inst = index[key]
        row = {"size": sz, "configuration": render_key(key), "classes": classify(key),
               "occurrences": len(inst), "positions": len({i[0] for i in inst}),
               "anchors": dict(Counter(sqname(i[1]) for i in inst).most_common(4))}
        for sname, base in starts.items():
            b0 = base.bits()
            s = copy.deepcopy(base)
            steps, _ = apply_global(s, key)
            row[f"(a) step by step, global [{sname}]"] = {"steps": fmt_steps(steps),
                                                          "total": round(s.bits() - b0, 2),
                                                          "complete": len(steps) == sz - 1}
            s = copy.deepcopy(base)
            steps, _ = apply_joint(s, key, inst)
            row[f"(b) whole configuration [{sname}]"] = {"steps": fmt_steps(steps),
                                                         "total": round(s.bits() - b0, 2),
                                                         "complete": len(steps) == sz - 1}
        # (c) oracle categories, from the flat start.
        fam = [k for k, _ in mined[sz] if k[0] == key[0] and k[1][:-1] == key[1][:-1]]
        if len(fam) >= 2:
            s = copy.deepcopy(flat)
            b0 = s.bits()
            made = []
            for k in fam:
                steps, labs = apply_joint(s, k, index[k])
                if labs and len(labs) == sz - 1:
                    made.append((k, labs))
            after_chunks = s.bits()
            levels = []
            if len(made) >= 2:
                # Level by level from the bottom: the variants' identical
                # prefixes first (they collide into one row), then the
                # whole configurations (the category of variants).
                for lev in range(sz - 1):
                    pb, bb = s.parts(), s.bits()
                    first = made[0][1][lev]
                    for _, labs in made[1:]:
                        other = labs[lev]
                        if other == first or other not in s.rows or first not in s.rows:
                            continue
                        pred = s.merge_score(first, other, {})
                        s.apply_merge(first, other)
                        assert abs(s.bits() - pred) < 1e-5
                    levels.append({"level": lev + 1, "bits": round(s.bits() - bb, 2),
                                   **{q: round(v, 2) for q, v in parts_delta(pb, s.parts()).items()}})
                row["(c) variants as one category [flat]"] = {
                    "variants": [(render_key(k), len(index[k])) for k, _ in made],
                    "chunks of all variants (b)": round(after_chunks - b0, 2),
                    "merges by level": levels, "total": round(s.bits() - b0, 2)}
        mk = mirror(key)
        if mk in index and mk != key:
            s = copy.deepcopy(flat)
            b0 = s.bits()
            _, la = apply_joint(s, key, inst)
            _, lb = apply_joint(s, mk, index[mk])
            if la and lb and len(la) == len(lb) == sz - 1:
                la, lb = la[-1], lb[-1]
                after_chunks = s.bits()
                pb = s.parts()
                s.apply_merge(la, lb)
                row["(c) either side as one category [flat]"] = {
                    "other side": render_key(mk), "chunks of both (b)": round(after_chunks - b0, 2),
                    "merge": round(s.bits() - after_chunks, 2),
                    **{q: round(v, 2) for q, v in parts_delta(pb, s.parts()).items()}}
        results.append(row)
        print(json.dumps(row), flush=True)

    os.makedirs(OUT, exist_ok=True)
    res = {"setting": name, "N": len(positions), "alpha": args.alpha,
           "flat bits": flat.bits(), "greedy bits": greedy.bits(), "flat parts": flat.parts(),
           "greedy parts": greedy.parts(),
           "greedy moves": [f"[{m[0]} {m[1]} {m[2]}]" for m in greedy.moves],
           "mined": {sz: [(render_key(k), len(v)) for k, v in lst[:30]] for sz, lst in mined.items()},
           "configurations": results, "seconds": time.time() - t0}
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"done {name} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
