"""Review diag: are all parser gains 0 in noconcept / minbits arms? (seed 13, n=500); full arm overlap split."""
import json, os, sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments/v3/diag")
import r230_grammar as G, r233_goldparse as GP
from trellis2.trellis3_r225 import Trellis3R225
from trellis2.trellis3_r233 import Trellis3R233
seed, n = 13, int(sys.argv[1]) if len(sys.argv) > 1 else 500
cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
C = G.load_corpus(seed, 4000); stream = [s for s, _ in C["train"]]; held = C["held"][:500]
golds = [GP.gold(s) for s in held]; tr = set(map(tuple, stream[:n]))
arms = {"noconcept": lambda: Trellis3R225(seed=seed, **cfg, concepts_off=True, concepts_off_parse=True),
        "minbits:12": lambda: Trellis3R233(seed=seed, parse_min_bits=12.0, **cfg),
        "minbits:6": lambda: Trellis3R233(seed=seed, parse_min_bits=6.0, **cfg),
        "full": lambda: Trellis3R225(seed=seed, **cfg)}
for arm in sys.argv[2].split(","):
    L = arms[arm]()
    for s in stream[:n]: L.learn(s)
    gc, gx, npos_c, npos_x, tot = [], [], 0, 0, 0
    trees = []
    for s in held[:500]:
        r = L.parse(s)
        trees.append(GP.tree_spans(r["chosen"], len(s)))
    for s in held[:60]:
        comps = L.parse(s)["comps"]
        for e in comps.values():
            a, b = L._gain(L.cnt, e.cnt), L._gain(L.ctx, e.ctx)
            tot += 1; npos_c += a > 0; npos_x += b > 0; gc.append(a); gx.append(b)
    ov = [tuple(s) in tr for s in held]
    sc_in = GP.score([t for t, o in zip(trees, ov) if o], [g for g, o in zip(golds, ov) if o])
    sc_out = GP.score([t for t, o in zip(trees, ov) if not o], [g for g, o in zip(golds, ov) if not o])
    print(json.dumps({"arm": arm, "n": n, "comps": tot, "share_content_gain_pos": round(npos_c/tot, 4), "share_ctx_gain_pos": round(npos_x/tot, 4),
                      "max_content_gain": round(max(gc), 3), "max_ctx_gain": round(max(gx), 3), "cnt_root": L.cnt.root.count, "ctx_root": L.ctx.root.count,
                      "all": GP.score(trees, golds), "held_in_train": round(sum(ov)/500, 3), "score_in_train": sc_in, "score_not_in_train": sc_out}), flush=True)
