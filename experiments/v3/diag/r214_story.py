"""Round 214 driver: t3_story.py with the read-level dials, without editing t3_story.py.

Usage: r214_story.py <the same arguments as t3_story.py>   (repo root, compute node, env chunks)

t3_story.py is read, patched in memory at four anchors (each must match exactly once) and executed as __main__:
  1. the learner is trellis2.trellis3_r214.Trellis3R214 (Trellis3R208 with store_same False; every flag None = the same learner);
  2. after a causal arm's keys are read, the arm keys  "bits"  (b: a number of bits, "basic", or "inf"/null = the leaf),
     "wreach" (W': the query keeps l1..lW')  and  "pbits" (parse_bits)  set L.read_bits / L.read_reach / L.parse_bits;
     the keys stay in the arm, so they are printed and dumped with it;
  3. after the arm's draws the three flags are None again (later arms read the default);
  4. env R214_LEVELS=<out.json> (no generation): dump every committed word occurrence's concept at each read level, then exit.
An arm without these keys behaves exactly as under t3_story.py.
"""
import json
import math
import os
import sys

sys.path.insert(0, "src")
from trellis2.trellis3_r214 import Trellis3R214, WordTree  # noqa: E402
from trellis2.trellis3 import FILL  # noqa: E402

SRC = os.path.join("experiments", "v3", "diag", "t3_story.py")
GRID = [0, 1, 2, 4, 6, 8, 12, "inf", "basic"]


def _bits(v):
    if v is None or v == "inf":
        return None
    return v if v == "basic" else float(v)


def _r214_start(L, arm):
    L.read_bits = _bits(arm.get("bits"))
    L.read_reach = None if arm.get("wreach") is None else int(arm["wreach"])
    L.parse_bits = _bits(arm.get("pbits"))
    we = arm.get("wemit")
    L.emit_bits = None if we in (None, "leaf", "inf") else float(we)
    L.emit_freq_cut = int(arm.get("wemitk", 0))
    L.emit_uniform = bool(arm.get("wemitu", False))
    wg = arm.get("wgate")
    L.gate_bits = None if wg in (None, "word") else float(wg)
    if L.emit_names is not None:
        print(json.dumps({"r214 emit": we, "classes": len(set(L.emit_names.values()))}), flush=True)
    if L.read_bits is not None or L.read_reach is not None or L.parse_bits is not None:
        print(json.dumps({"r214 read": {"bits": L.read_bits, "reach": L.read_reach, "parse_bits": L.parse_bits}}), flush=True)


def _r214_end(L):
    L.read_bits = L.read_reach = L.parse_bits = None
    L.emit_bits = None
    L.emit_freq_cut = 0
    L.emit_uniform = False
    L.gate_bits = None


def _r214_levels(L, train, EXP, seed, n):
    """No generation: every committed word occurrence (sentence id, position, token, l2, weight) and the concept id it is read
    at for each band of GRID, plus the sentences (for the part-of-speech tagging done by r214_levels.py, env cobint)."""
    L.read_bits = L.read_reach = L.parse_bits = None
    L.emit_bits = None
    L.emit_freq_cut = 0
    L.emit_uniform = False
    L.gate_bits = None
    owner = {}
    for sid, (toks, a0, a1) in enumerate(EXP):
        for i in range(a0, a1):
            owner[i] = sid
    occ, bad = [], 0
    for ri, e in enumerate(L.records):
        if e.leaf is None or e.token is None or e.span[1] - e.span[0] != 1:
            continue
        w = sum(x for which, attr, value, x in e.pas if which == "ctx" and attr == FILL and value[0] == "w")
        if w <= 0:
            continue
        sid, pos = owner.get(ri, -1), e.span[0]
        if sid >= 0 and not (pos < len(EXP[sid][0]) and EXP[sid][0][pos] == e.token):
            bad += 1
            sid = -1
        occ.append({"sid": sid, "pos": pos, "tok": e.token, "l2": e.ctx.get("l2"), "w": w,
                    "c": {str(b): L.lift(e.leaf, math.inf if b == "inf" else b).id for b in GRID}})
    root = L.ctx.root
    lv = {str(b): sorted({(o["c"][str(b)]) for o in occ}) for b in GRID}
    nodes = {}
    for e in L.records:
        if e.leaf is None:
            continue
        for b in GRID:
            c = L.lift(e.leaf, math.inf if b == "inf" else b)
            nodes[c.id] = round(L.level_bits(c, root), 4)
    out = {"seed": seed, "n": n, "grid": [str(b) for b in GRID], "sentences": [t for t, _, _ in EXP], "occ": occ,
           "mismatched owner": bad, "node bits": nodes, "concepts per band": {b: len(v) for b, v in lv.items()},
           "root count": root.count, "records": len(L.records), "window": getattr(L, "window", None)}
    with open(os.environ["R214_LEVELS"], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(json.dumps({"r214 levels": os.environ["R214_LEVELS"], "occurrences": len(occ), "mismatched": bad,
                      "concepts per band": out["concepts per band"]}), flush=True)


def _r214_wordmap(L, train, seed, n):
    """No generation: each word type's learned name truncated to b bits (Trellis3._window_name's rule: the most common concept of the
    bag of the type's reservoir paths, bands up to b; a word with no reservoir keeps its token), for b in 0, 1, 2, 4, 6, 8, 12, None."""
    bs = [0, 1, 2, 4, 6, 8, 12, None]
    toks = sorted({w for s, _ in train for w in s})
    names = {}
    for b in bs:
        d = {}
        for w in toks:
            bag = L._res_bag((w,), b)
            d["%s" % w] = max(bag.items(), key=lambda kv: kv[1])[0] if bag else w
        names["none" if b is None else str(b)] = d
    out = {"seed": seed, "n": n, "sentences": [list(s) for s, _ in train], "names": names,
           "reservoir types": len(L.type_res), "type_names": L.type_names, "window": getattr(L, "window", None)}
    with open(os.environ["R214_WORDMAP"], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(json.dumps({"r214 wordmap": os.environ["R214_WORDMAP"], "types": len(toks), "reservoirs": len(L.type_res)}), flush=True)


def _wtree_b(v):
    return None if v in (None, "leaf") else float(v)


def _r214_make(seed, cfg, train):
    """The learner. cfg key "wtree" (b in bits, or "leaf"): the window names come from a WordTree built over the training stream's
    word types (Trellis3R214.word_names), window_names "all" unless cfg sets it. Env R214_WORDTREE=<out.json>: no learning, dump the
    word classes of the tree at bit levels and depths, then exit."""
    wt = cfg.pop("wtree", None)
    wf = cfg.pop("wtree_feats", "all")
    if os.environ.get("R214_WORDTREE"):
        T = WordTree([s for s, _ in train], seed, wf)
        names = {str(b): T.names(b=math.inf if b == "leaf" else b) for b in [0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.25, 7.5, 7.75, "leaf"]}
        names.update({"d%d" % k: T.names(depth=k) for k in (1, 2, 3, 4, 5, 6, 8)})
        root = T.tree.root
        with open(os.environ["R214_WORDTREE"], "w", encoding="utf-8") as fh:
            json.dump({"seed": seed, "n": len(train), "sentences": [list(s) for s, _ in train], "names": names,
                       "reservoir types": len(T.words), "tree nodes": sum(1 for _ in T.tree.nodes()), "root count": root.count}, fh)
        print(json.dumps({"r214 wordtree": os.environ["R214_WORDTREE"], "types": len(T.words), "root": root.count}), flush=True)
        sys.exit()
    wem = cfg.pop("wemit_tree", False)
    L = Trellis3R214(seed=seed, **cfg)
    T = WordTree([s for s, _ in train], seed, wf) if (wt is not None or wem) else None
    if wt is not None:
        L.word_names = T.names(b=_wtree_b(wt) if wt != "leaf" else None)
        L.window_names = L.window_names or "all"
        print(json.dumps({"r214 wtree": wt, "classes": len(set(L.word_names.values())), "types": len(T.words)}), flush=True)
    if T is not None:
        from collections import Counter
        L.word_tree, L.emit_counts = T, dict(Counter(w for s, _ in train for w in s))
    return L


def patch(src):
    def sub(old, new):
        assert src.count(old) == 1, ("anchor not unique", old)
        return src.replace(old, new)
    src = sub("    L = Trellis3(seed=seed, **cfg)\n", "    L = _r214_make(seed, cfg, train)\n")
    src = sub('        if edge_ == "exact":\n            from trellis2.trellis3_r204 import Trellis3R204\n',
              '        _r214_start(L, arm)\n        if edge_ == "exact":\n            from trellis2.trellis3_r204 import Trellis3R204\n')
    src = sub('    if causal is not None and edge_ == "exact":     # r204: later arms',
              '    if causal is not None:\n        _r214_end(L)\n    if causal is not None and edge_ == "exact":     # r204: later arms')
    src = sub('        def _ok(l1, w):\n            if not pair_ or l1 == T3BOS or (l1, w) in _st_:\n',
              '        def _ok(l1, w):\n            if pair_ and l1 != T3BOS and L.gate_bits is not None:     # r214h: class-level pair gate\n                return L.class_pair_ok(l1, w)\n            if not pair_ or l1 == T3BOS or (l1, w) in _st_:\n')
    src = sub('if os.environ.get("STORY_R204_REPLAY"):',
              'if os.environ.get("R214_WORDMAP"):\n    _r214_wordmap(L, train, seed, n)\n    sys.exit()\n'
              'if os.environ.get("R214_LEVELS"):\n    _r214_levels(L, train, EXP, seed, n)\n    sys.exit()\n'
              'if os.environ.get("STORY_R204_REPLAY"):')
    return src


if __name__ == "__main__":
    src = patch(open(SRC, encoding="utf-8").read())
    sys.argv = [SRC] + sys.argv[1:]
    g = {"__name__": "__main__", "__file__": SRC, "_R214": Trellis3R214, "_r214_start": _r214_start, "_r214_end": _r214_end,
         "_r214_levels": _r214_levels, "_r214_wordmap": _r214_wordmap, "_r214_make": _r214_make}
    exec(compile(src, SRC, "exec"), g)
