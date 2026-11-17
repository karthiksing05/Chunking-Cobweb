"""Round 187 diagnosis: why does the content-hierarchy gain not matter for story parsing?

Run: t3_content_gain.py <seed> <n1,n2>   (same stream / recipe / held-out set as story_parse_variants.py; read only, learner untouched)
Measures, after each checkpoint: (1) distribution of content / context gain (and raw pre-max best g of the content tree) over
all chart composites of 100 held-out sentences; (2) a term breakdown along the content tree's sorting path for 10 composites;
(3) tree shapes and description attributes; (4) exact_alone vs categorize, stored vs unseen composites.
"""
from __future__ import annotations
import json, math, os, random, re, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
LN2 = math.log(2.0)
S = '"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0'
D0 = '"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,"repair_use":700,"exact_alone":"content"'
CFG = json.loads("{" + S + "," + D0 + ',"story_facet":true}')
WANT = [("the", "bird"), ("to", "play"), ("very", "happy"), ("tim", "was"), ("a", "little"), ("he", "was"), ("in", "the"),
        ("wanted", "to"), ("the", "park"), ("she", "said")]


def q(xs, ps=(0, .1, .25, .5, .75, .9, .99, 1)):
    xs = sorted(xs)
    return {f"q{p}": round(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + .5))], 5) for p in ps} if xs else {}


def stream(n, seed):
    from trellis2.stories import WORD, default_tinystories_path, read_sentences
    raw = open(default_tinystories_path(), encoding="utf-8").read()
    stories = []
    for st in raw.split("<|endoftext|>"):
        st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
        sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
        if sents:
            stories.append(sents)
    vocab = {w for w, _ in Counter(w for s in read_sentences() for w in s).most_common(250)}
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    random.Random(0).shuffle(stories)
    test, train_st = stories[:1500], stories[1500:]
    random.Random(seed).shuffle(train_st)
    train = []
    for st in train_st:
        for i, s in enumerate(st):
            if keep(s):
                train.append((s, any(keep(t) for t in st[:i])))
        if len(train) >= n:
            break
    held = [s for st in test for s in st if keep(s)]
    random.Random(1000).shuffle(held)
    return train[:n], held[:300]


def raw_terms(L, tree, x):
    """per path node (excluding root): (count, log2 share, log2 P(x|C), log2 P(x|root), g)"""
    from trellis2.trellis3 import log_prob
    root = tree.root
    base = log_prob(root, x, root, L.alpha) / LN2
    out = []
    for node in tree.categorize(x)[1:]:
        if node.count <= 0:
            continue
        sh = math.log2(node.count / root.count)
        lp = log_prob(node, x, root, L.alpha) / LN2
        out.append((node.count, sh, lp, base, sh + lp - base, node.id if hasattr(node, "id") else None))
    return out


def shape(tree):
    nodes = list(tree.nodes())
    leaves = tree.leaves()
    depths = [len(tree.ancestors(l)) for l in leaves]
    return {"nodes": len(nodes), "leaves": len(leaves), "max_depth": max(depths), "mean_leaf_depth": round(sum(depths) / len(depths), 2),
            "root_count": tree.root.count, "root_children": len(tree.root.children),
            "root_child_counts": sorted((round(c.count, 1) for c in tree.root.children), reverse=True)[:8]}


def analyse(L, held, cp, rng, out, full=True):
    L._quiet = True
    L._pv = None
    comps_all = []
    for s in held[:100]:
        r = L.parse(s)
        for key, e in r["comps"].items():
            comps_all.append((s, key, e))
    L._quiet = False
    res = {"learned": cp, "n_composites": len(comps_all)}
    gc, gx, raw, nraw_none, ncnt_path = [], [], [], 0, []
    from trellis2.trellis3 import log_prob
    for s, key, e in comps_all:
        gc.append(L._gain(L.cnt, e.cnt))
        gx.append(L._gain(L.ctx, e.ctx))
        t = raw_terms(L, L.cnt, e.cnt)
        ncnt_path.append(len(t))
        if t:
            raw.append(max(z[4] for z in t))
        else:
            nraw_none += 1
    res["content_gain"] = {**q(gc), "share_exact0": round(sum(g == 0.0 for g in gc) / len(gc), 5), "n_pos": sum(g > 0 for g in gc), "max": max(gc), "distinct": len(set(gc))}
    res["context_gain"] = {**q(gx), "share_exact0": round(sum(g == 0.0 for g in gx) / len(gx), 5), "n_pos": sum(g > 0 for g in gx), "max": max(gx), "distinct": len(set(gx))}
    res["content_raw_best_g"] = {**q(raw), "n_no_path_below_root": nraw_none, "share_lt0": round(sum(g < 0 for g in raw) / max(len(raw), 1), 5),
                                 "share_eq0": round(sum(g == 0 for g in raw) / max(len(raw), 1), 5)}
    res["content_path_len"] = {**q(ncnt_path), "mean": round(sum(ncnt_path) / len(ncnt_path), 3)}
    # distinct content descriptions and what they hold
    descs = Counter(json.dumps(e.cnt, sort_keys=True) for _, _, e in comps_all)
    res["distinct_content_descriptions"] = len(descs)
    valtype = Counter()
    blanks = Counter()
    for _, _, e in comps_all:
        for a, v in e.cnt.items():
            valtype[(a, type(v).__name__)] += 1
            if v == "-":
                blanks[a] += 1
    res["cnt_attr_value_types"] = {f"{a}:{t}": c for (a, t), c in sorted(valtype.items())}
    res["cnt_attr_blank_counts"] = dict(blanks)
    res["cnt_values_distinct_per_attr"] = {a: len({e.cnt[a] if not isinstance(e.cnt[a], dict) else json.dumps(e.cnt[a], sort_keys=True) for _, _, e in comps_all}) for a in comps_all[0][2].cnt}
    res["tree_content"] = shape(L.cnt)
    res["tree_context"] = shape(L.ctx)
    res["cnt_attrs"], res["ctx_attrs"] = list(L.cnt_attrs), list(L.ctx_attrs)
    res["root_distinct_values_content"] = {a: len(L.cnt.root.av.get(a, {})) for a in L.cnt_attrs}
    res["root_distinct_values_context"] = {a: len(L.ctx.root.av.get(a, {})) for a in L.ctx_attrs}
    # does content contain the word identity? count tokens held by composites' parts
    res["records"] = len(L.records)
    stored_cnt = [r.cnt for r in L.records if getattr(r, "cnt", None) and r.token is None]
    res["stored_composite_records_with_cnt"] = len(stored_cnt)
    # examples
    exs = []
    byspan = {}
    for s, key, e in comps_all:
        i, k, j = key
        byspan.setdefault(tuple(s[i:j]), []).append((s, key, e))
    chosen = []
    for w in WANT:
        if w in byspan:
            chosen.append((w, byspan[w][0]))
    spans2 = [w for w in byspan if len(w) == 2 and w not in dict(chosen)]
    rng.shuffle(spans2)
    for w in spans2[: max(0, 10 - len(chosen))]:
        chosen.append((w, byspan[w][0]))
    for w, (s, key, e) in chosen[:10]:
        terms = raw_terms(L, L.cnt, e.cnt)
        exs.append({"span": " ".join(w), "sentence": " ".join(s), "key": key, "cnt": e.cnt, "ctx_gain": L._gain(L.ctx, e.ctx), "cnt_gain": L._gain(L.cnt, e.cnt),
                    "path": [{"count": t[0], "log2_share": round(t[1], 4), "log2P_x_given_C": round(t[2], 4), "log2P_x_given_root": round(t[3], 4), "g": round(t[4], 4)} for t in terms]})
    res["examples"] = exs
    # (4) exact_alone and categorize
    ea = L.cnt.exact_alone
    cmp = {"flag_value": bool(ea)}
    pool = [e for _, _, e in comps_all][:: max(1, len(comps_all) // 150)][:150]
    same = diff = 0
    ex_diff = []
    for e in pool:
        L.cnt.exact_alone = True
        p1 = [n.id for n in L.cnt.categorize(e.cnt)]
        L.cnt.exact_alone = False
        p0 = [n.id for n in L.cnt.categorize(e.cnt)]
        L.cnt.exact_alone = ea
        if p1 == p0:
            same += 1
        else:
            diff += 1
            ex_diff.append((p1[-4:], p0[-4:]))
    cmp["unseen_pool"] = {"n": len(pool), "same_path": same, "diff_path": diff, "examples": ex_diff[:3]}
    # stored composites: categorize their own description; gain; path depth; is the final node a leaf holding exactly it
    rr = [r for r in L.records if r.token is None and r.cnt is not None]
    rng.shuffle(rr)
    rr = rr[:150]
    sg, sdepth, sleaf, sc = [], [], 0, []
    for r in rr:
        g = L._gain(L.cnt, r.cnt)
        sg.append(g)
        path = L.cnt.categorize(r.cnt)
        sdepth.append(len(path) - 1)
        sleaf += (not path[-1].children)
    cmp["stored_records"] = {"n": len(rr), "gain": {**q(sg), "share_exact0": round(sum(g == 0.0 for g in sg) / max(len(sg), 1), 4)}, "mean_path_len_below_root": round(sum(sdepth) / max(len(sdepth), 1), 3), "path_ends_at_leaf": sleaf}
    ug = [L._gain(L.cnt, e.cnt) for e in pool]
    cmp["unseen_pool_gain"] = {**q(ug), "share_exact0": round(sum(g == 0.0 for g in ug) / len(ug), 4)}
    # does an unseen description equal a stored one? (exact match by dict)
    storeset = {json.dumps(r.cnt, sort_keys=True) for r in L.records if r.token is None and r.cnt is not None}
    cmp["unseen_pool_desc_stored_exact"] = sum(json.dumps(e.cnt, sort_keys=True) in storeset for e in pool)
    res["exact_alone"] = cmp
    res["alpha_counterfactual"] = alpha_cf(L, comps_all, rr)
    out.append(res)
    print(json.dumps(res, indent=1, default=str), flush=True)


def gain_alpha(L, tree, x, alpha):
    from trellis2.trellis3 import log_prob
    root = tree.root
    base = log_prob(root, x, root, alpha)
    best = 0.0
    for node in tree.categorize(x)[1:]:
        if node.count > 0:
            best = max(best, (math.log(node.count / root.count) + log_prob(node, x, root, alpha) - base) / LN2)
    return best


def alpha_cf(L, comps_all, stored):
    """Measurement only: the same gain with other smoothing (L.alpha is 1.0 in the recipe; alpha<0: total pseudo-count |alpha|)."""
    pool = [e for _, _, e in comps_all][:: max(1, len(comps_all) // 400)][:400]
    out = {}
    for a in (1.0, 0.1, 0.01, 1e-4, -1.0, -10.0):
        row = {}
        for name, tree, attr in (("content", L.cnt, "cnt"), ("context", L.ctx, "ctx")):
            g = [gain_alpha(L, tree, getattr(e, attr), a) for e in pool]
            row[name] = {**q(g, (0.25, 0.5, 0.75, 0.9, 1)), "share_pos": round(sum(x > 0 for x in g) / len(g), 4)}
        g = [gain_alpha(L, L.cnt, r.cnt, a) for r in stored[:100]]
        row["content_stored"] = {**q(g, (0.5, 1)), "share_pos": round(sum(x > 0 for x in g) / len(g), 4)}
        out[str(a)] = row
    return out


def main():
    from trellis2.trellis3 import Trellis3
    seed = int(sys.argv[1])
    cps = [int(x) for x in sys.argv[2].split(",")]
    train, held = stream(cps[-1], seed)
    L = Trellis3(seed=seed, **CFG)
    out, done, t0, rng = [], 0, time.time(), random.Random(7)
    for cp in cps:
        for idx in range(done, cp):
            s, fol = train[idx]
            L.learn(s, follows=fol)
        done = cp
        print(f"=== learned {cp} ({time.time() - t0:.0f}s)", flush=True)
        analyse(L, held, cp, rng, out)
    d = os.path.join(HERE, "..", "results", "content_gain")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"s{seed}.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", d)


if __name__ == "__main__":
    main()
