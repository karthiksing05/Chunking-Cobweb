"""Round 204 pre-test (M1, no generation): replay the boundary steps of judged plain-TRELLIS dumps under the default read
and under edge_exact, from the rebuilt learner (called from t3_story.py with STORY_R204_REPLAY=<dump dir>, then exit).

For every generated row (all rows, repeats included) of the plain causal words arm (pair gate, new) of the given dump
dir / seed / n, the start step (history all <s>) and the end step (history = the sentence minus its last word, the
sentence's last word drawn with "ends here") get their log-probability of the drawn (word, done) under the generator's own
step distribution: mass of every filler by Trellis3._gap_fills (go-on mass = the read without r1 minus the end read, as
t3_story.py _fills), pair gate as the arm, then weights ** (1 / temp), normalised. Reported in nats: delta = exact - default.

A boundary is atypical when the word starts (ends) fewer than 2 of the real held-out sentences (the 4 seeds' refs of the
n=1000 reference dir, 1200 sentences: the r204 design's lexicon). Atypical boundary steps of atypical sentences against
the boundary steps of fully typical sentences; medians. Pass: atypical median <= -1.5, typical median >= -0.3.

Env: R204_DUMPFILE (file name inside the dump dir, default story_s{seed}_n{n}_arm1.jsonl), R204_REFS (refs dir).
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter

from trellis2.trellis3 import BOS, EOS
from trellis2.trellis3_r204 import Trellis3R204


def lexicon(refs_dir):
    st, en = Counter(), Counter()
    k = 0
    for f in sorted(glob.glob(os.path.join(refs_dir, "story_s*_n1000_refs.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            s = json.loads(ln)["real"].split()
            if s:
                st[s[0]] += 1
                en[s[-1]] += 1
                k += 1
    return st, en, k


def med(v):
    return None if not v else round(statistics.median(v), 3)


def mean(v):
    return None if not v else round(sum(v) / len(v), 3)


def run(L, train, seed, n, dump_dir, out_path):
    L.__class__ = Trellis3R204
    L.edge_exact = False
    fn = os.environ.get("R204_DUMPFILE", f"story_s{seed}_n{n}_arm1.jsonl")
    lines = [json.loads(l) for l in open(os.path.join(dump_dir, fn), encoding="utf-8")]
    arm, rows = lines[0]["arm"], lines[1:]
    assert arm.get("causal") and arm.get("units") == "words" and arm.get("pair") == "gate" and not arm.get("both") and not arm.get("edge"), arm
    temp = float(arm.get("temp", 1.0))
    W = max(getattr(L, "window", 2), 2)
    st_lex, en_lex, nref = lexicon(os.environ.get("R204_REFS", "experiments/v3/results/story_gen/r200-n1000-refs"))
    _w_, _ix_, _M_ = L.type_licence(near=None)
    _st_ = set(getattr(L, "_pair_counts", {}))
    vals = sorted(float(_M_[_ix_[a], _ix_[b]]) for a, b in _st_ if a in _ix_ and b in _ix_)
    med_ = vals[min(len(vals) - 1, int(len(vals) * 0.5))] if vals else 0.0

    def ok(l1, w):
        if l1 == BOS or (l1, w) in _st_:
            return True
        return l1 in _ix_ and w in _ix_ and float(_M_[_ix_[l1], _ix_[w]]) >= med_ > 0
    cz = {}

    def step(hist, exact):
        t_ = [BOS] * max(5, getattr(L, "window", 2)) + list(hist)
        l2, l1 = t_[-2], t_[-1]
        far = [t_[-3 - i] for i in range(W - 2)]
        k = (exact, l2, l1, tuple(far))
        if k not in cz:
            L.edge_exact = exact
            q = {"l2": l2, "l1": l1, **{f"l{3 + i}": w for i, w in enumerate(far)}}
            a_, e_ = L._gap_fills(q), L._gap_fills({**q, "r1": EOS})
            L.edge_exact = False
            ws = {}
            for v, m_ in a_.items():
                if v[0] != "w" or not ok(l1, v[1]):
                    continue
                me = e_.get(v, 0.0)
                ws[(v[1], True)] = me ** (1.0 / temp) if temp != 1.0 else me
                g = max(m_ - me, 0.0)
                ws[(v[1], False)] = g ** (1.0 / temp) if temp != 1.0 else g
            cz[k] = (ws, sum(ws.values()))
        return cz[k]

    def lp(hist, w, done, exact):
        ws, z = step(hist, exact)
        x = ws.get((w, done), 0.0)
        return math.log(x / z) if x > 0 and z > 0 else None

    recs = []
    for r in rows:
        g = r["gen"].split()
        if not g:
            continue
        s0, s1 = lp([], g[0], len(g) == 1, False), lp([], g[0], len(g) == 1, True)
        e0, e1 = lp(g[:-1], g[-1], True, False), lp(g[:-1], g[-1], True, True)
        recs.append(dict(gen=r["gen"], a_start=st_lex[g[0]] < 2, a_end=en_lex[g[-1]] < 2,
                         start=(s0, s1), end=(e0, e1), judged=r.get("pattern")))
    FLOOR = -30.0       # a step the read gives zero mass (log undefined) is floored, and counted

    def delta(p):
        a, b = p
        zero = ("d" if a is None else "") + ("e" if b is None else "")   # which read gave the drawn step zero mass
        a = FLOOR if a is None else a
        b = FLOOR if b is None else b
        return b - a, zero
    out = dict(seed=seed, n=n, dump=fn, arm=arm, window=getattr(L, "window", None), temp=temp, rows=len(recs),
               refs=nref, floor=FLOOR)
    def detail(xs):
        fin = [d for d, z in xs if not z]
        q = lambda p: None if not fin else round(sorted(fin)[min(len(fin) - 1, int(p * len(fin)))], 3)
        return dict(finite=len(fin), finite_median=med(fin), finite_mean=mean(fin), p10=q(0.1), p25=q(0.25), p75=q(0.75), p90=q(0.9),
                    lose_ge_1_5=round(sum(d <= -1.5 for d, _ in xs) / max(len(xs), 1), 3),
                    lose_gt_0_3=round(sum(d < -0.3 for d, _ in xs) / max(len(xs), 1), 3),
                    gain_gt_0_3=round(sum(d > 0.3 for d, _ in xs) / max(len(xs), 1), 3))
    pools = {"atypical": [], "typical": []}
    for side in ("start", "end"):
        at = [delta(r[side]) for r in recs if r["a_" + side]]
        ty = [delta(r[side]) for r in recs if not r["a_start"] and not r["a_end"]]
        out[side] = dict(atypical=dict(**detail(at), n=len(at), median=med([d for d, _ in at]), mean=mean([d for d, _ in at]),
                                       zero_mass_default=sum("d" in z for _, z in at), zero_mass_exact=sum("e" in z for _, z in at)),
                         typical=dict(**detail(ty), n=len(ty), median=med([d for d, _ in ty]), mean=mean([d for d, _ in ty]),
                                      zero_mass_default=sum("d" in z for _, z in ty), zero_mass_exact=sum("e" in z for _, z in ty)))
        pools["atypical"] += [d for d, _ in at]
        pools["typical"] += [d for d, _ in ty]
    out["pooled"] = {k: dict(n=len(v), median=med(v), mean=mean(v)) for k, v in pools.items()}
    # mean log-probability of the drawn boundary, default vs exact (not the difference)
    for side in ("start", "end"):
        for nm, sel in (("atypical", lambda r: r["a_" + side]), ("typical", lambda r: not r["a_start"] and not r["a_end"])):
            xs = [r[side] for r in recs if sel(r) and r[side][0] is not None and r[side][1] is not None]
            out[side][nm]["median_logp_default"] = med([a for a, _ in xs])
            out[side][nm]["median_logp_exact"] = med([b for _, b in xs])
    sa = [r for r in recs if r["a_start"] or r["a_end"]]
    out["sentence_share_atypical"] = round(len(sa) / len(recs), 3)
    out["pass"] = bool(out["pooled"]["atypical"]["median"] is not None and out["pooled"]["atypical"]["median"] <= -1.5
                       and out["pooled"]["typical"]["median"] is not None and out["pooled"]["typical"]["median"] >= -0.3)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out), flush=True)
