"""Round 196: why does the same word string get different brackets in different sentences? (diagnosis + parse-time readouts)

Run:  story_parse_consist.py <cfg json> <seed> <checkpoints e.g. 1000,4000> <tag> [variants comma list]
Agg:  story_parse_consist.py agg <tag>

Stream, held-out set, gold and metrics are those of story_parse_eval.py (copied, fixed held-out set). Learning is the cfg
recipe. After each checkpoint:
  * diagnosis (default factor): every occurrence of a repeated held-out string (2..4 words, >= 3 occurrences, as the
    consistency metric) is traced: bracketed or not, its span posterior, and for its best split the factor's parts --
    content gain, context gain, and the context gain at its argmax concept split into the share term and the
    per-attribute log ratios P(x_a|C)/P(x_a|root) grouped as window (l*, r*), own content (cc) and story (pv); plus the
    context gain of the window alone and of the instance without the window, the cc bag (is the string's content concept
    the same in every sentence?) and whether the span touches the sentence edges.
  * parse-time readouts of the same memory (instance attributes set, then restored), scored with the r185 metrics.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "story_parse_consist")
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_VARIANTS = ["V0_default", "Vwin", "Vnowin", "Vpmi", "Vccm", "Vzero"]
LN2 = math.log(2.0)


def prf(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def f1_between(a, b):
    return prf(a, b)[2]


def setup(seed, n):
    import nltk
    from trellis2.stories import WORD, default_tinystories_path, read_sentences
    N_TEST_STORIES, n_test = 1500, 300
    with open(default_tinystories_path(), encoding="utf-8") as f:
        raw = f.read()
    stories = []
    for st in raw.split("<|endoftext|>"):
        st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
        sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
        if sents:
            stories.append(sents)
    all_sents = read_sentences()
    vocab = {w for w, _ in Counter(w for s in all_sents for w in s).most_common(250)}
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    random.Random(0).shuffle(stories)
    test, train_st = stories[:N_TEST_STORIES], stories[N_TEST_STORIES:]
    random.Random(seed).shuffle(train_st)
    train = []
    for st in train_st:
        for i, s in enumerate(st):
            if keep(s):
                train.append((s, any(keep(t) for t in st[:i])))
        if len(train) >= n:
            break
    train = train[:n]
    held = [s for st in test for s in st if keep(s)]
    random.Random(1000).shuffle(held)
    held = held[:n_test]
    del raw, all_sents, stories, test, train_st

    CHUNKER = nltk.RegexpParser(r"""
      NP: {<DT|PRP\$|POS|CD|PDT>*<JJ.*|RBS|RBR>*<NN.*>+}
          {<PRP|EX>}
      AP: {<RB.*>*<JJ.*>+}
      VG: {<MD|TO>?<RB.*>*<VB.*>+(<RB.*>*<VB.*>+)*<RP>?}
      PP: {<IN|TO><NP>}
      VP: {<VG><NP|PP|AP|RB.*>+}
    """)

    def phrases(words):
        tagged = nltk.pos_tag(["I" if w == "i" else w for w in words])
        tree = CHUNKER.parse(tagged)
        spans = defaultdict(set)

        def walk(t, i):
            if isinstance(t, nltk.Tree):
                j = i
                for c in t:
                    j = walk(c, j)
                if t.label() in ("NP", "PP", "VP", "AP", "VG"):
                    spans[(i, j)].add(t.label())
                return j
            return i + 1
        walk(tree, 0)
        for k, (_, t) in enumerate(tagged):
            if t == "CC":
                spans[(k, k + 1)].add("CC")
        return dict(spans)

    def sv(i, j, spans):
        lead = 1 if j - i > 2 and "CC" in spans.get((i, i + 1), ()) else 0
        return any(("NP" in spans.get((i, k), ()) or (lead and "NP" in spans.get((i + 1, k), ()))) and "VG" in spans.get((k, j), ())
                   for k in range(i + 1, j))

    def crosses(i, j, spans):
        if sv(i, j, spans):
            return False
        return any(a < j and i < b and not (a <= i and j <= b) and not (i <= a and b <= j) for a, b in spans)

    GS = [phrases(s) for s in held]
    PH = ("NP", "PP", "VP", "AP", "VG")
    gold_ph = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and ij[1] - ij[0] < len(s) and v & set(PH)} for s, sp in zip(held, GS)]
    gold_sv = [g | {(i, j) for i in range(len(s)) for j in range(i + 2, len(s)) if sv(i, j, sp)} for g, s, sp in zip(gold_ph, held, GS)]
    gold_np = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and "NP" in v and ij[1] - ij[0] < len(s)} for s, sp in zip(held, GS)]
    occ = defaultdict(list)
    for h, s in enumerate(held):
        for i in range(len(s)):
            for j in range(i + 2, min(len(s) - 1, i + 4) + 1):
                occ[tuple(s[i:j])].append((h, i, j))
    strings = {k: v for k, v in occ.items() if len(v) >= 3}

    def metrics(brs):
        P, R, F = prf(brs, gold_ph)
        P2, R2, F2 = prf(brs, gold_sv)
        nb = sum(len(b) for b in brs)
        cr = sum(crosses(i, j, GS[h]) for h, b in enumerate(brs) for i, j in b)
        npm = prf(brs, gold_np)[1]
        tot = hit = dec = 0
        for v in strings.values():
            b = sum((i, j) in brs[h] for h, i, j in v)
            tot += len(v)
            hit += b
            dec += max(b, len(v) - b)
        return {"brackets/sent": round(nb / len(held), 3), "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                "P+sv": round(P2, 4), "R+sv": round(R2, 4), "F1+sv": round(F2, 4), "NP recall": round(npm, 4),
                "crossing": round(cr / nb, 4) if nb else None, "consistency": round(hit / tot, 4),
                "decisive": round(dec / tot, 4), "strings": len(strings)}

    return train, held, gold_ph, strings, metrics


# ---------------------------------------------------------------- factor anatomy
def gain_node(L, tree, x):
    """(gain, argmax concept) as Trellis3._gain (alpha path), the root when no concept beats it."""
    from trellis2.trellis3 import log_prob
    root = tree.root
    if root.count <= 0 or not x:
        return 0.0, root
    base = log_prob(root, x, root, L.alpha)
    best, bn = 0.0, root
    for node in tree.categorize(x)[1:]:
        if node.count <= 0:
            continue
        g = (math.log(node.count / root.count) + log_prob(node, x, root, L.alpha) - base) / LN2
        if g > best:
            best, bn = g, node
    return best, bn


def terms(L, tree, x, node):
    """Per-attribute log2 P(x_a|node)/P(x_a|root) and the share term, at ``node``."""
    from trellis2.trellis3 import log_prob
    root = tree.root
    out = {"share": math.log2(node.count / root.count) if root.count > 0 and node.count > 0 else 0.0}
    for a, v in x.items():
        out[a] = (log_prob(node, {a: v}, root, L.alpha) - log_prob(root, {a: v}, root, L.alpha)) / LN2
    return out


def is_win(a):
    return a[0] in "lr" and (a[1:].isdigit() or a[1:2] == "j")


def anatomy(L, e):
    gc, _ = gain_node(L, L.cnt, e.cnt)
    gx, nx = gain_node(L, L.ctx, e.ctx)
    t = terms(L, L.ctx, e.ctx, nx)
    grp = {"share": t["share"], "win": sum(v for a, v in t.items() if is_win(a)),
           "cc": sum(v for a, v in t.items() if a.startswith("cc")), "pv": t.get("pv", 0.0)}
    win = {a: v for a, v in e.ctx.items() if is_win(a)}
    nowin = {a: v for a, v in e.ctx.items() if not is_win(a)}
    return {"cnt": round(gc, 3), "ctx": round(gx, 3), "at_root": nx is L.ctx.root,
            "coh": round(L._cohesion(e), 3) if hasattr(L, "_cohesion") else 0.0,
            **{f"t_{k}": round(v, 3) for k, v in grp.items()},
            "g_win": round(gain_node(L, L.ctx, win)[0], 3), "g_nowin": round(gain_node(L, L.ctx, nowin)[0], 3),
            "cc": sorted(e.ctx.get("cc", {}).items()) if isinstance(e.ctx.get("cc"), dict) else e.ctx.get("cc")}


# ---------------------------------------------------------------- readouts
def make_learner(cfg, seed):
    """Trellis3, or (cfg key "consist", prototype before porting) a subclass with context-free cohesion:
    a word part's content facet (LC/RC) is the word itself, and the factor adds the content tree's gain on the
    partial instance {LC, RC} (what the two parts are made of, the window marginalised out), each concept's
    value probabilities backed off to its parent's with one pseudo-count (the root's to uniform)."""
    from trellis2.trellis3 import BLANK, Trellis3
    cfg = dict(cfg)
    consist = cfg.pop("consist", None)
    if not consist or "consist" in Trellis3.__init__.__code__.co_varnames:
        if consist:
            cfg["consist"] = consist
        return Trellis3(seed=seed, **cfg)

    class T196(Trellis3):
        def _content(self, e, sides="LR"):
            x = super()._content(e, sides)
            if self.donate_cnt:
                for p, part in (("L", e.left), ("R", e.right)):
                    if p in sides and part is not None and part.token is not None:
                        self._facet(x, f"{p}C", [f"w:{part.token}"])
            return x

        def _cohesion(self, e):
            if e.token is not None or not e.cnt:
                return 0.0
            x = {a: e.cnt[a] for a in ("LC", "RC") if e.cnt.get(a, BLANK) != BLANK}
            if len(x) < 2:
                return 0.0
            tree = self.cnt
            path = tree.categorize(x)
            root = path[0]
            N = root.count
            if N <= 0:
                return 0.0
            bags = {a: (v if isinstance(v, dict) else {v: 1.0}) for a, v in x.items()}
            prev = {}
            for a, bag in bags.items():
                d = root.av.get(a, {})
                S = len(d) + 1
                prev[a] = {u: (d.get(u, 0.0) + 1.0 / S) / (N + 1.0) for u in bag}
            proot = {a: dict(p) for a, p in prev.items()}
            best = 0.0
            for node in path[1:]:
                c = node.count
                if c <= 0:
                    break
                cur, lr = {}, 0.0
                for a, bag in bags.items():
                    d = node.av.get(a, {})
                    cur[a] = {u: (d.get(u, 0.0) + prev[a][u]) / (c + 1.0) for u in bag}
                    lr += sum(p * math.log(cur[a][u] / proot[a][u]) for u, p in bag.items())
                prev = cur
                best = max(best, (math.log(c / N) + lr) / LN2)
            return best

        def _factor(self, e):
            mode = self.__dict__.get("consist")
            if mode == "coh_only":
                return self._cohesion(e)
            base = super()._factor(e)
            return base + self._cohesion(e) if mode == "cohesion" else base

    L = T196(seed=seed, **cfg)
    L.consist = consist
    return L


_MISSING = object()


def set_variant(L, v):
    """Install variant v on L (parse time); returns an undo function."""
    saved = {k: L.__dict__.get(k, _MISSING) for k in ("tau", "readout", "factor", "consist")}
    if v == "Vwin":           # context gain from the window alone
        L._factor = lambda e: L._gain(L.cnt, e.cnt) + L._gain(L.ctx, {a: x for a, x in e.ctx.items() if is_win(a)})
    elif v == "Vnowin":       # context gain without the window (own content + story)
        L._factor = lambda e: L._gain(L.cnt, e.cnt) + L._gain(L.ctx, {a: x for a, x in e.ctx.items() if not is_win(a)})
    elif v == "Vpmi":         # what the own content adds to the window's gain
        L._factor = lambda e: L._gain(L.cnt, e.cnt) + L._gain(L.ctx, e.ctx) - L._gain(L.ctx, {a: x for a, x in e.ctx.items() if a != "cc"})
    elif v == "Vccm":
        L.factor = "ccm"
    elif v == "Vzero":
        L._factor = lambda e: 0.0
    elif v == "Vmarg":
        # the context gain with the observed window replaced by its expectation under the context concept the
        # span's own content (cc) sorts to: E_{w ~ P(.|C)} log2 share(C) P(w|C)/P(w|root) = log2 share(C) +
        # sum_a KL(P(a|C) || P(a|root)) over window attributes, best over the path of {cc}; the same estimator
        # as _gain, context marginalised, so a string's score depends on what it is, not on where it stands
        from trellis2.trellis3 import log_prob
        cache = {}
        wins = [a for a in L.ctx_attrs if is_win(a)]

        def marg(e):
            cc = e.ctx.get("cc")
            if not isinstance(cc, dict) or not cc:
                return L._gain(L.cnt, e.cnt)
            key = tuple(sorted(cc.items()))
            if key not in cache:
                tree = L.ctx
                path = tree.categorize({"cc": cc})
                root = path[0]
                N = root.count
                best = 0.0
                if N > 0:
                    for node in path[1:]:
                        c = node.count
                        if c <= 0:
                            break
                        g = math.log2(c / N)
                        for a in wins:
                            d = node.av.get(a, {})
                            for u, cu in list(d.items()):
                                if cu <= 0:
                                    continue
                                lp = log_prob(node, {a: u}, root, L.alpha)
                                g += math.exp(lp) * (lp - log_prob(root, {a: u}, root, L.alpha)) / LN2
                        best = max(best, g)
                cache[key] = best
            return L._gain(L.cnt, e.cnt) + cache[key]
        L._factor = marg
    elif v in ("Vside", "Vside_coh"):
        # what the context shares ACROSS the span: the context gain minus each side's own gain (the left
        # window's internal coherence, the right window's), so a frame scores by how its two sides and the
        # span's own content go together, not by how familiar each side is on its own
        coh = v == "Vside_coh"

        def side(e):
            x = e.ctx
            lft = {a: u for a, u in x.items() if a[0] == "l" and (a[1:].isdigit() or a[1:2] == "j")}
            rgt = {a: u for a, u in x.items() if a[0] == "r" and (a[1:].isdigit() or a[1:2] == "j")}
            f = L._gain(L.cnt, e.cnt) + L._gain(L.ctx, x) - L._gain(L.ctx, lft) - L._gain(L.ctx, rgt)
            return f + (L._cohesion(e) if coh else 0.0)
        L._factor = side
    elif v.startswith("Vc:"):  # consist readouts ("Vc:none" = the default factor on the same memory)
        L.consist = None if v[3:] == "none" else v[3:]
    elif v != "V0_default":
        raise ValueError(v)

    def undo():
        L.__dict__.pop("_factor", None)
        for k, x in saved.items():
            if x is _MISSING:
                L.__dict__.pop(k, None)
            else:
                setattr(L, k, x)
    return undo


def main():
    from trellis2.trellis3 import Trellis3
    cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
    checkpoints = [int(x) for x in sys.argv[3].split(",")]
    tag = sys.argv[4]
    variants = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else DEFAULT_VARIANTS
    train, held, gold_ph, strings, metrics = setup(seed, checkpoints[-1])
    L = make_learner(cfg, seed)
    out = {"tag": tag, "seed": seed, "cfg": cfg, "held": held, "gold_phrase": [sorted(g) for g in gold_ph],
           "strings": len(strings), "checkpoints": []}
    done, t0 = 0, time.time()
    for cp in checkpoints:
        for idx in range(done, cp):
            s, fol = train[idx]
            L.learn(s, follows=fol)
        done = cp
        row = {"learned": cp, "records": len(L.records), "variants": {}, "diag": {}}
        L._quiet, pv = True, L._pv
        L._pv = None
        for v in variants:
            undo = set_variant(L, v)
            try:
                brs, diag = [], {}
                for h, s in enumerate(held):
                    r = L.parse(s)
                    brs.append({(i, j) for i, k, j in r["chosen"] if 2 <= j - i < len(s)})
                    if v == "V0_default":
                        sp = defaultdict(float)
                        for (a, k, b), p in r["post"].items():
                            sp[(a, b)] += p
                        diag[h] = (r, sp)
            finally:
                undo()
            row["variants"][v] = {"metrics": metrics(brs), "brackets": [sorted(b) for b in brs]}
            print(cp, v, json.dumps(row["variants"][v]["metrics"]), f"{time.time() - t0:.0f}s", flush=True)
            if v == "V0_default":
                # trace every occurrence of every repeated string, and the chosen spans crossing it
                tr = {}
                for key, occs in strings.items():
                    rows = []
                    for h, i, j in occs:
                        r, sp = diag[h]
                        n = len(held[h])
                        k = max(range(i + 1, j), key=lambda k: r["comps"][(i, k, j)].gain)
                        a = anatomy(L, r["comps"][(i, k, j)])
                        cross = [(x, y) for x, y in brs[h] if x < j and i < y and not (x <= i and j <= y) and not (i <= x and y <= j)]
                        cr = []
                        for x, y in cross[:2]:
                            kk = max(range(x + 1, y), key=lambda q: r["comps"][(x, q, y)].gain)
                            cr.append({"span": [x, y], "post": round(sp[(x, y)], 3), **anatomy(L, r["comps"][(x, kk, y)])})
                        rows.append({"h": h, "i": i, "j": j, "in": (i, j) in brs[h], "post": round(sp[(i, j)], 3),
                                     "left_edge": i == 0, "right_edge": j == n, "split": k, **a, "cross": cr})
                    tr[" ".join(key)] = rows
                row["diag"] = tr
        L._quiet, L._pv = False, pv
        out["checkpoints"].append(row)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{tag}_s{seed}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), separators=(",", ":"))
    print("wrote", p)


def agg(tag):
    import glob
    files = sorted(glob.glob(os.path.join(OUT, f"{tag}_s*.json")))
    runs = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        runs[d["seed"]] = d
    seeds = sorted(runs)
    held = runs[seeds[0]]["held"]
    cps = [r["learned"] for r in runs[seeds[0]]["checkpoints"]]
    keys = ["brackets/sent", "P", "R", "F1", "F1+sv", "NP recall", "crossing", "consistency", "decisive", "xseed F1"]
    md = [f"seeds {seeds}; held-out {len(held)}; strings {runs[seeds[0]]['strings']}", ""]
    res = {}
    for ci, cp in enumerate(cps):
        md += [f"### {cp} sentences learned", "", "| variant | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for v in runs[seeds[0]]["checkpoints"][ci]["variants"]:
            per = {s: dict(runs[s]["checkpoints"][ci]["variants"][v]["metrics"]) for s in seeds}
            brs = {s: [set(map(tuple, b)) for b in runs[s]["checkpoints"][ci]["variants"][v]["brackets"]] for s in seeds}
            for s in seeds:
                o = [f1_between(brs[s], brs[t]) for t in seeds if t != s]
                per[s]["xseed F1"] = round(sum(o) / len(o), 4) if o else None
            res[f"{cp}|{v}"] = per
            cells = []
            for k in keys:
                x = [per[s][k] for s in seeds]
                cells.append("n/a" if any(y is None for y in x) else f"{sum(x) / len(x):.3f} [{' / '.join(f'{y:.3f}' for y in x)}]")
            md.append(f"| {v} | " + " | ".join(cells) + " |")
        md.append("")
        # ---- diagnosis, pooled over seeds
        comp = ["post", "ctx", "cnt", "coh", "t_share", "t_win", "t_cc", "t_pv", "g_win", "g_nowin"]
        diffs = defaultdict(list)
        pos = defaultdict(lambda: [0, 0])
        same_cc = [0, 0]
        mixed = allin = allout = 0
        for s in seeds:
            tr = runs[s]["checkpoints"][ci]["diag"]
            for key, rows in tr.items():
                ins = [r for r in rows if r["in"]]
                outs = [r for r in rows if not r["in"]]
                for r in rows:
                    c = ("start" if r["left_edge"] else "") + ("end" if r["right_edge"] else "") or "interior"
                    pos[c][0] += r["in"]
                    pos[c][1] += 1
                for a in range(len(rows)):
                    for b in range(a + 1, len(rows)):
                        same_cc[0] += rows[a]["cc"] == rows[b]["cc"]
                        same_cc[1] += 1
                if ins and outs:
                    mixed += 1
                    for c in comp:
                        diffs[c].append(sum(r[c] for r in ins) / len(ins) - sum(r[c] for r in outs) / len(outs))
                elif ins:
                    allin += 1
                else:
                    allout += 1
        md.append(f"diagnosis (V0, pooled over seeds): strings always bracketed {allin}, never {allout}, mixed {mixed}; "
                  f"pairs of occurrences of one string with the SAME cc (content concept bands) {same_cc[0]}/{same_cc[1]} "
                  f"= {same_cc[0] / max(same_cc[1], 1):.3f}")
        md.append("")
        md.append("bracket rate by position: " + "; ".join(f"{c} {a}/{b} = {a / b:.3f}" for c, (a, b) in sorted(pos.items())))
        md.append("")
        md.append("mixed strings: mean over strings of (mean over bracketed occurrences - mean over unbracketed), with the "
                  "share of strings where the difference is positive")
        md.append("")
        md.append("| quantity | mean diff (bits) | share > 0 |")
        md.append("|---|---|---|")
        for c in comp:
            d = diffs[c]
            if d:
                md.append(f"| {c} | {sum(d) / len(d):+.3f} | {sum(x > 0 for x in d) / len(d):.2f} |")
        md.append("")
        # within-string spread vs between-string spread of the context gain and its parts
        for c in ("ctx", "t_win", "t_cc", "g_win", "g_nowin", "cnt", "coh"):
            within, means = [], []
            for s in seeds:
                for rows in runs[s]["checkpoints"][ci]["diag"].values():
                    xs = [r[c] for r in rows]
                    m = sum(xs) / len(xs)
                    within += [(x - m) ** 2 for x in xs]
                    means.append(m)
            gm = sum(means) / len(means)
            md.append(f"- {c}: within-string sd {math.sqrt(sum(within) / len(within)):.3f}, between-string sd "
                      f"{math.sqrt(sum((m - gm) ** 2 for m in means) / len(means)):.3f}")
        md.append("")
        # case studies (first seed): mixed strings, one bracketed and one unbracketed occurrence, with the crossing winner
        s0 = seeds[0]
        tr = runs[s0]["checkpoints"][ci]["diag"]
        cases = sorted(((k, rows) for k, rows in tr.items() if any(r["in"] for r in rows) and not all(r["in"] for r in rows)),
                       key=lambda kr: -len(kr[1]))[:8]
        md.append(f"case studies (seed {s0}): factor parts in bits; ctx = context gain = share + win + cc + pv at its argmax concept")
        md.append("")
        f = lambda r: (f"post {r['post']:.2f} ctx {r['ctx']:.2f} [share {r['t_share']:.2f} win {r['t_win']:.2f} cc {r['t_cc']:.2f} "
                       f"pv {r['t_pv']:.2f}] cnt {r['cnt']:.2f} g_win {r['g_win']:.2f} g_nowin {r['g_nowin']:.2f}")
        for key, rows in cases:
            md.append(f"- \"{key}\" ({sum(r['in'] for r in rows)}/{len(rows)} bracketed)")
            for want in (True, False):
                r = next(r for r in rows if r["in"] == want)
                w = held[r["h"]]
                md.append(f"  - {'ONE UNIT' if want else 'split   '}: " + " ".join(w[:r['i']]) + f" <{' '.join(w[r['i']:r['j']])}> "
                          + " ".join(w[r['j']:]) + f"  | {f(r)} | cc {r['cc']}")
                for c in r["cross"]:
                    md.append(f"    - crossed by <{' '.join(w[c['span'][0]:c['span'][1]])}>: {f(c)}")
        md.append("")
    open(os.path.join(OUT, f"{tag}_table.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(res, open(os.path.join(OUT, f"{tag}_table.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg(sys.argv[2])
    else:
        main()
