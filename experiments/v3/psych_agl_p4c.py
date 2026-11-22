"""P4c: position-sensitive context scores for artificial grammar learning (KS96 selection, MVdL97 2a/2b confirmation).

Binding design: docs/PLAN_PSYCH.md "P4c pre-registration". Stimuli, training orders and subjects are those of psych_agl.py
(same RNG strings, so subject s sees the same training sequence as in P4 / P4b).

Usage:
  psych_agl_p4c.py run --exp ks96|mvdl2a|mvdl2b --arm full|conceptsoff|acs|anchor [--seeds 13,14,...]
  psych_agl_p4c.py summary            -> results/psych_agl/p4c/SUMMARY_P4C.txt (stage 1 always; stage 2 if MVdL runs exist)

Scores (endorsement = top half by score, ties at random; one RNG per score and subject):
  pos    : mean over sub-spans t (length >= 2) at (i, j) of the summed posterior weight of stored records of t whose span had the same
           boundary status in its training string (begins it iff i = 0, ends it iff j = n). Chunks plus position, no concepts.
  fit    : parse (learning off); for each chosen composite e, c / x = deepest supported content / context concept (climb_path);
           fit(e) = W(c and x) / W(x), W = summed weight of stored composite records with content leaf under c and context leaf under x.
           Score = mean fit over chosen composites (0 if none).
  acs    : KS96 printed chunk strength; MVdL recomputed global ACS (as psych_agl).
  anchor : MVdL positional anchor ACS: mean of #training strings (distinct list) beginning with the item's first bigram, first trigram,
           ending with its last bigram, last trigram (AGL_PROTOCOL 2.5).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, os.path.join("experiments", "v3"))
from psych_segment import CFG, Trellis3Seg  # noqa: E402
import psych_agl as A  # noqa: E402

OUT = os.path.join("experiments", "v3", "results", "psych_agl", "p4c")


def climb_path(L, tree, x):
    """Sorting path, root first, up to the deepest ancestor with count >= support (psych_digits.climb_path)."""
    path = tree.categorize(x)
    k = 0
    for j, n in enumerate(path[1:], 1):
        if n.count >= L.support:
            k = j
    return path[:k + 1]


def leaf_ids(tree, node):
    cu = getattr(tree, "tree", None)
    out = []
    if cu is not None and hasattr(cu, "children_of"):
        stack = [node.id]
        while stack:
            i = stack.pop()
            kids = cu.children_of(i)
            if kids:
                stack.extend(kids)
            else:
                out.append(i)
        return out
    stack = [node]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            out.append(n.id)
    return out


def anchor_acs(item, train):
    s = item
    v = [sum(1 for t in train if t.startswith(s[:2])), sum(1 for t in train if t.endswith(s[-2:]))]
    if len(s) >= 3:
        v += [sum(1 for t in train if t.startswith(s[:3])), sum(1 for t in train if t.endswith(s[-3:]))]
    return sum(v) / len(v)


def learn_all(L, seq):
    span_pos = defaultdict(float)
    for s in seq:
        toks = list(s)
        n = len(toks)
        a0 = len(L.records)
        L.learn_buf(toks)
        for e in L.records[a0:]:
            if e.token is None:
                i, j = e.span
                span_pos[(tuple(toks[i:j]), i == 0, j == n)] += e.weight
    return span_pos


def score_items(L, span_pos, strings):
    by_ctx = defaultdict(list)                   # context leaf id -> [(content leaf id, weight)] of composite records
    for e in L.records:
        if e.token is None and e.leaf is not None and e.cleaf is not None:
            by_ctx[e.leaf.id].append((e.cleaf.id, e.weight))
    leafcache = {}

    def leaves(tree, node):
        k = (id(tree), node.id)
        if k not in leafcache:
            leafcache[k] = set(leaf_ids(tree, node))
        return leafcache[k]
    pos, fit, nfit = [], [], []
    for s in strings:
        toks = list(s)
        n = len(toks)
        spans = [(i, j) for i in range(n) for j in range(i + 2, n + 1)]
        pos.append(sum(span_pos.get((tuple(toks[i:j]), i == 0, j == n), 0.0) for i, j in spans) / len(spans))
        L._quiet = True
        try:
            r = L.parse(toks)
            fs = []
            for k in r["chosen"]:
                e = r["comps"][k]
                c = climb_path(L, L.cnt, L._content(e))[-1]
                x = climb_path(L, L.ctx, e.ctx)[-1]
                lc, lx = leaves(L.cnt, c), leaves(L.ctx, x)
                wx = wcx = 0.0
                for lid in lx:
                    for cl, w in by_ctx.get(lid, ()):
                        wx += w
                        if cl in lc:
                            wcx += w
                fs.append(wcx / wx if wx > 0 else 0.0)
        finally:
            L._quiet = False
        fit.append(sum(fs) / len(fs) if fs else 0.0)
        nfit.append(len(fs))
    return pos, fit, nfit


def setup(exp, seed):
    if exp == "ks96":
        rng = random.Random(f"agl-{seed}")
        seq = [s for s in A.TRAIN23 for _ in range(2)]
        rng.shuffle(seq)
        items = [(c, s, cs) for c, s, cs in A.ITEMS]
        return seq, items, A.TRAIN23, list(A.HUMAN)
    e = exp[4:]
    rng = random.Random(f"mvdl{e}-{seed}")
    order = list(A.MV_TRAIN[e])
    rng.shuffle(order)
    seq = order * (2 if e == "2a" else 1)
    return seq, list(A.MV_ITEMS[e]), list(A.MV_TRAIN[e]), list(A.MV_CELLS)


def run(exp, arm, seed, outdir):
    t0 = time.time()
    seq, items, train, cells = setup(exp, seed)
    res = {"experiment": "psych_agl P4c", "exp": exp, "arm": arm, "seed": seed, "cfg": CFG, "n_presentations": len(seq),
           "items": [{"cell": c, "string": s} for c, s, _ in items]}
    strings = [s for _, s, _ in items]
    scores = {}
    if arm == "acs":
        scores["acs"] = [a for _, _, a in items]
    elif arm == "anchor":
        scores["anchor"] = [anchor_acs(s, train) for s in strings]
    else:
        kw = {"concepts_off": True, "concepts_off_parse": True} if arm == "conceptsoff" else {}
        L = Trellis3Seg(seed=seed, **CFG, **kw)
        span_pos = learn_all(L, seq)
        scores["pos"], scores["fit"], res["n_chosen"] = score_items(L, span_pos, strings)
        res["n_records"] = len(L.records)
    res["scores"] = scores
    res["endorse"] = {k: A.endorse(v, random.Random(f"p4c-{exp}-{k}-{seed}")) for k, v in scores.items()}
    res["cell_endorse"] = {k: {c: 100 * sum(x for x, it in zip(v, items) if it[0] == c) / 8 for c in cells}
                           for k, v in res["endorse"].items()}
    res["time_s"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print(json.dumps({"exp": exp, "arm": arm, "seed": seed, "cells": res["cell_endorse"], "time_s": round(res["time_s"], 1)}), flush=True)


def _load(exp, arm):
    d = os.path.join(OUT, exp, arm)
    return A.load(d) if os.path.isdir(d) else []


def summary():
    out = ["P4c (psych_agl_p4c.py): position-sensitive context scores. Pre-registration: docs/PLAN_PSYCH.md P4c.", ""]
    # ---------------- stage 1: KS96 selection
    out.append("=== Stage 1: KS96 Exp 1 (selection), 18 subjects. Human G-NG 22.0; cells " +
               "  ".join(f"{c} {v}" for c, v in A.HUMAN.items()))
    keep = {}
    for arm, keys in (("full", ("pos", "fit")), ("conceptsoff", ("pos", "fit")), ("acs", ("acs",)), ("anchor", ("anchor",))):
        subs = _load("ks96", arm)
        if not subs:
            continue
        for k in keys:
            lines, kk = A.block(subs, k)
            out += [f"[{arm}] score = {k}, n = {len(subs)}"] + lines + [""]
            keep[(arm, k)] = kk["gram_within"]
    selected = None
    passing = []
    for k in ("pos", "fit"):
        if ("full", k) not in keep:
            continue
        f = keep[("full", k)]
        rivals = {"conceptsoff": keep.get(("conceptsoff", k)), "acs": keep.get(("acs", "acs")), "anchor": keep.get(("anchor", "anchor"))}
        ok = all(v is not None and f > v for v in rivals.values())
        out.append(f"criterion {k}: full G-NG within CS {f:.1f} vs " + ", ".join(f"{a} {v:.1f}" if v is not None else f"{a} -" for a, v in rivals.items())
                   + f" -> {'PASSES' if ok else 'fails'}")
        if ok:
            passing.append((f, k))
    if passing:
        selected = max(passing)[1]
        out.append(f"SELECTED score: {selected}")
    else:
        out.append("SELECTED score: none -> P4c stops negative; MVdL not run (pre-registered)")
    out.append("")
    # ---------------- stage 2: MVdL confirmation
    if any(_load(e, "full") for e in ("mvdl2a", "mvdl2b")):
        out.append("=== Stage 2: MVdL 2a/2b (confirmation), 20 subjects each. Human shift (2b - 2a): G-NG +9.2, A-NA -22.0")
        H = A.MV_HUMAN
        eff = {}
        for arm, keys in (("full", ("pos", "fit")), ("conceptsoff", ("pos", "fit")), ("acs", ("acs",)), ("anchor", ("anchor",))):
            for k in keys:
                row = {}
                for e in ("mvdl2a", "mvdl2b"):
                    subs = _load(e, arm)
                    if subs:
                        ce, m, gram, assoc = A._mv_effects(subs, k)
                        row[e] = (m, gram, assoc)
                if len(row) < 2:
                    continue
                (ma, ga, aa), (mb, gb, ab) = row["mvdl2a"], row["mvdl2b"]
                shift = A.mean(gb) - A.mean(ga)
                se = math.sqrt(A.sd(ga) ** 2 / len(ga) + A.sd(gb) ** 2 / len(gb))
                ashift = A.mean(ab) - A.mean(aa)
                r8 = A.pearson([ma[c] for c in A.MV_CELLS] + [mb[c] for c in A.MV_CELLS],
                               [H["2a"][c] for c in A.MV_CELLS] + [H["2b"][c] for c in A.MV_CELLS])
                eff[(arm, k)] = shift
                tag = "COUNTED" if (k == selected or arm in ("acs", "anchor")) else "reported, not counted"
                out.append(f"[{arm}] {k} ({tag}): 2a cells " + "/".join(f"{ma[c]:.1f}" for c in A.MV_CELLS) +
                           f"  2b cells " + "/".join(f"{mb[c]:.1f}" for c in A.MV_CELLS) +
                           f"  | G-NG 2a {A.mean(ga):+.1f} 2b {A.mean(gb):+.1f} shift {shift:+.1f} +- {se:.1f} SE"
                           f" | A-NA 2a {A.mean(aa):+.1f} 2b {A.mean(ab):+.1f} shift {ashift:+.1f} | r8 {r8:.2f}")
        if selected and ("full", selected) in eff:
            f = eff[("full", selected)]
            rivals = {"conceptsoff": eff.get(("conceptsoff", selected)), "acs": eff.get(("acs", "acs")), "anchor": eff.get(("anchor", "anchor"))}
            ok = f > 0 and all(v is not None and f > v for v in rivals.values())
            out.append(f"criterion (selected {selected}): full shift {f:+.1f} > 0 and > " +
                       ", ".join(f"{a} {v:+.1f}" if v is not None else f"{a} -" for a, v in rivals.items()) +
                       f" -> {'SUPPORTED' if ok else 'NOT supported'}")
    path = os.path.join(OUT, "SUMMARY_P4C.txt")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--exp", default="ks96", choices=["ks96", "mvdl2a", "mvdl2b"])
    ap.add_argument("--arm", default="full", choices=["full", "conceptsoff", "acs", "anchor"])
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    if a.mode == "summary":
        summary()
    else:
        seeds = A.SEEDS if a.exp == "ks96" else A.MV_SEEDS
        if a.seeds:
            seeds = [int(x) for x in a.seeds.split(",")]
        for s in seeds:
            run(a.exp, a.arm, s, os.path.join(OUT, a.exp, a.arm))
