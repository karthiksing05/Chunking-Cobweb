"""Round 199 pre-test (no generation): a ROLE check of the pivot, scored on the judged r192 traces.

The r192 pivot check re-reads the surface pivot word p (the word before the new word w) by per-attribute
back-off, which merges p's roles ("to" infinitive vs preposition, "was" copula vs auxiliary). The role check
asks which role p plays in THIS sentence and whether w continues that role:

  C      the context-tree concept that the pivot's current left context q_p = {l1..lW of p} sorts to
         (Trellis3.ctx.categorize, nothing inserted): level "leaf" = the last node on the path, "parent" = the one above
  S_p    the stored occurrences of p (word fillers; weights as _gap_fills)
  S_pw   those of S_p whose right neighbour is w (ctx r1 == w; the end marker for the end)
  "in C" an occurrence is in C if C is on the path of its context leaf (its own concept at C's level)
  P(C|S) = (mass of S in C + P0(C)) / (mass of S + 1), P0(C) = share of ALL stored word occurrences in C
           (the same single pseudo-count the r192 back-off uses; an empty S reads as the prior)
  ratio  = log P(C | S_pw) - log P(C | S_p); accept w if ratio >= 0

Scored on the steps of the r192-diag trace files (judge labels, classes, and the pivot ratio "both log ratio").
Sentence = its worst (minimum) step, as r192_summarize.py.

Usage (compute node):
  r199_role_auc.py run <cfg json> <seed> <n> <trace.jsonl> <out steps.jsonl>
  r199_role_auc.py agg <out.txt> <steps.jsonl> [...]
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")


def auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = sorted(pos), sorted(neg)
    import bisect
    s = 0.0
    for a in pos:
        lo, hi = bisect.bisect_left(neg, a), bisect.bisect_right(neg, a)
        s += lo + 0.5 * (hi - lo)
    return s / (len(pos) * len(neg))


def run(cfg, seed, n, trace, out):
    from trellis2.stories import WORD, default_tinystories_path, read_sentences
    from trellis2.trellis3 import Trellis3, BOS, EOS, FILL
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
    random.Random(seed).shuffle(stories)
    train = []
    for st in stories[1500:]:
        for i, s in enumerate(st):
            if keep(s):
                train.append((s, any(keep(t) for t in st[:i])))
        if len(train) >= n:
            break
    train = train[:n]
    L = Trellis3(seed=seed, **cfg)
    for s, fol in train:
        L.learn(s, follows=fol)
    L._quiet = True
    W = L.window
    print(json.dumps({"seed": seed, "n": n, "window": W, "records": len(L.records)}), flush=True)

    def ancestors(nd):
        ids = []
        while nd is not None:
            ids.append(nd.id)
            nd = nd.parent
        return ids

    occ = defaultdict(list)           # p -> [(ancestor id set, r1, weight)]
    anc_mass = Counter()              # node id -> mass of all word occurrences below it
    total = 0.0
    n_r1 = n_occ = 0
    for e in L.records:
        if e.leaf is None:
            continue
        for which, attr, value, wt in e.pas:
            if which == "ctx" and attr == FILL and value[0] == "w":
                ids = set(ancestors(e.leaf))
                r1 = e.ctx.get("r1")
                n_occ += 1
                n_r1 += r1 is not None
                occ[value[1]].append((ids, r1, wt))
                total += wt
                for i in ids:
                    anc_mass[i] += wt
    print(json.dumps({"word occurrences": n_occ, "with r1": n_r1, "mass": round(total, 1)}), flush=True)
    sorted_cache = {}

    def path_of(q):
        k = tuple(sorted(q.items()))
        if k not in sorted_cache:
            sorted_cache[k] = [nd.id for nd in L.ctx.categorize(q)]
        return sorted_cache[k]

    def logp(cid, S):
        m = sum(wt for ids, _, wt in S if cid in ids)
        tot = sum(wt for _, _, wt in S)
        p0 = max(anc_mass[cid], 1.0) / total       # a concept holding no stored word occurrence: one occurrence
        return math.log((m + p0) / (tot + 1.0)), m, tot

    rows = [json.loads(l) for l in open(trace, encoding="utf-8")]
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            s = r["gen"].split()
            steps = []
            for st in r["steps"]:
                i, end = st["pos"], st["end"]
                full = [BOS] * 5 + s + [EOS]
                j = 5 + (i - 1 if not end else len(s) - 1)
                piv, right = full[j], full[j + 1]
                ql = {f"l{k}": full[j - k] for k in range(1, W + 1)}
                path = path_of(ql)
                Sp = occ.get(piv, [])
                Spw = [o for o in Sp if o[1] == right]
                d = {"pos": i, "end": end, "gate": st["gate"], "pivot": piv, "w": st["w"],
                     "pivot log ratio": st["both log ratio"], "n Sp": len(Sp), "n Spw": len(Spw),
                     "path len": len(path)}
                for lev, idx in (("leaf", -1), ("parent", -2)):
                    if len(path) < -idx + 0 or len(path) + idx < 0 or path[idx] == path[0]:
                        d[f"role {lev}"] = 0.0       # sorted to the root only: no role information
                        continue
                    a, m_a, t_a = logp(path[idx], Spw)
                    b, m_b, t_b = logp(path[idx], Sp)
                    d[f"role {lev}"] = a - b
                    d[f"share in C {lev} (S_p)"] = round(m_b / t_b, 3) if t_b else None
                steps.append(d)
            fh.write(json.dumps({"gen": r["gen"], "judge": r["judge"], "novel": r["novel trigrams"], "steps": steps}) + "\n")
    print("done", flush=True)


def cls(r):
    st = r["steps"]
    return ("A" if any(s["gate"] == "licensed" and not s["end"] for s in st)
            else "B" if any(s["end"] for s in st) else "C")


def agg(out, files):
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))
    feats = ["pivot log ratio", "role leaf", "role parent", "role min(leaf,parent)"]

    def val(r, f):
        sc = []
        for s in r["steps"]:
            if f == "role min(leaf,parent)":
                sc.append(min(s["role leaf"], s["role parent"]))
            else:
                sc.append(s[f])
        return min(sc)
    sets = {}
    for fn in files:
        key = re.search(r"_w(\d)_", fn).group(1)
        sets[f"w{key}"] = [json.loads(l) for l in open(fn, encoding="utf-8")]
    sets["pooled"] = [r for v in list(sets.values()) for r in v]
    for name, rows in sets.items():
        P(f"\n=== {name}: {len(rows)} sentences with a new trigram, judged grammatical "
          f"{sum(r['judge'] > 0.5 for r in rows) / len(rows):.3f}")
        P(f"{'subset':26s} {'n':>5s} {'gram':>5s} " + " ".join(f"{f:>22s}" for f in feats))
        subs = [("all", rows)] + [(f"class {c}", [r for r in rows if cls(r) == c]) for c in "ABC"] + \
               [("class C, one new step", [r for r in rows if cls(r) == "C" and len(r["steps"]) == 1])]
        for sn, rs in subs:
            g = [r for r in rs if r["judge"] > 0.5]
            b = [r for r in rs if r["judge"] <= 0.5]
            cells = []
            for f in feats:
                a = auc([val(r, f) for r in g], [val(r, f) for r in b])
                cells.append("       -" if a is None else f"{a:22.3f}")
            P(f"{sn:26s} {len(rs):5d} {len(g) / max(len(rs), 1):5.2f} " + " ".join(cells))
        # the accept rule: every step ratio >= 0
        P("accept rule (every new step ratio >= 0): kept share, judged grammatical among kept, among rejected")
        for f in feats:
            for sn, rs in subs[:1] + subs[3:4]:
                k = [r for r in rs if val(r, f) >= 0]
                x = [r for r in rs if val(r, f) < 0]
                P(f"  {f:22s} {sn:22s} kept {len(k) / max(len(rs), 1):.3f} gram|kept "
                  f"{sum(r['judge'] > 0.5 for r in k) / max(len(k), 1):.3f} gram|rejected "
                  f"{sum(r['judge'] > 0.5 for r in x) / max(len(x), 1):.3f}")
    rows = sets["pooled"]
    P("\nstep-level share with an informative role (path deeper than the root), pooled:",
      f"{sum(s['path len'] > 1 for r in rows for s in r['steps']) / sum(len(r['steps']) for r in rows):.3f}",
      "; steps with S_pw empty:", f"{sum(s['n Spw'] == 0 for r in rows for s in r['steps']) / sum(len(r['steps']) for r in rows):.3f}")
    P("\nclass C, lowest role-leaf steps (pivot -> w, judge):")
    cs = sorted([(val(r, "role leaf"), r) for r in rows if cls(r) == "C"], key=lambda x: x[0])
    for v, r in cs[:6] + cs[len(cs) // 2:len(cs) // 2 + 3] + cs[-6:]:
        P(f"  {v:7.2f}  judge {r['judge']:.2f}  {r['gen']}  | {r['novel']}")
    txt = "\n".join(lines)
    print(txt)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        agg(sys.argv[2], sys.argv[3:])
