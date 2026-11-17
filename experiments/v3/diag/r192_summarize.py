"""Round 192: summarize r192_trace.py outputs. Per novel step and per sentence (its worst step), the
features split by the judge (P(yes) > 0.5), the AUC of each candidate statistic against the judge, and
readable case studies of judged-ungrammatical sentences.

Usage: r192_summarize.py <trace.jsonl> [...]
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def stats(st):
    """Candidate statistics for one step (higher = more licensed)."""
    p = st["p"] or 0.0
    if st["end"]:
        N = st["N (l1 w)"]
    elif st["gate"] == "licensed":
        N = st["N l1"]
    else:
        N = st["N l2 l1"]
    pre = N * math.log(max(1 - p, 1e-12))       # log P(never seen | rate p, N chances): pre-emption
    key = [k for k in st if k.endswith("anchor height (mass-wtd)")][0]
    return {"p": p, "log p": math.log(max(p, 1e-12)), "pre-emption log": pre, "N history": -N,
            "A share": st["A share (own l1 = l1)"], "-anchor height": -st[key],
            "stored gate": 1.0 if st["gate"] == "stored" else 0.0,
            "both log ratio": st.get("both log ratio", 0.0), "pivot p2": st.get("pivot p2", 0.0)}


rows = [json.loads(l) for f in sys.argv[1:] for l in open(f, encoding="utf-8")]
rows = [r for r in rows if "steps" in r]
good = [r for r in rows if r["judge"] > 0.5]
bad = [r for r in rows if r["judge"] <= 0.5]
print(f"sentences with an unstored trigram: {len(rows)}; judged grammatical {len(good)} ({len(good) / max(len(rows), 1):.3f})")
# step-level kinds
kind = lambda st: "end" if st["end"] else ("bigram unstored (gate licence)" if st["gate"] == "licensed" else "bigrams stored, trigram not")
for lab, rs in (("grammatical", good), ("ungrammatical", bad)):
    c = Counter(kind(st) for r in rs for st in r["steps"])
    tot = sum(c.values())
    print(f"  {lab}: steps {tot}; " + "; ".join(f"{k} {v / tot:.2f}" for k, v in c.most_common()))
# per-sentence: the worst step's statistics
print("\nAUC of each statistic (sentence = its worst novel step) for judged grammatical vs not:")
names = list(stats(rows[0]["steps"][0]))
for nm in names:
    g = [min(stats(st)[nm] for st in r["steps"]) for r in good]
    b = [min(stats(st)[nm] for st in r["steps"]) for r in bad]
    print(f"  {nm:20s} AUC {auc(g, b):.3f}  mean good {sum(g) / len(g):8.3f}  bad {sum(b) / len(b):8.3f}")
print("\nAUC per kind of step (steps of sentences with ONE novel step):")
for k in ("bigrams stored, trigram not", "bigram unstored (gate licence)", "end"):
    one = [r for r in rows if len(r["steps"]) == 1 and kind(r["steps"][0]) == k]
    gg = [r for r in one if r["judge"] > 0.5]
    bb = [r for r in one if r["judge"] <= 0.5]
    print(f"  {k}: {len(one)} sentences, judged grammatical {len(gg) / max(len(one), 1):.3f}")
    for nm in names:
        a = auc([stats(r["steps"][0])[nm] for r in gg], [stats(r["steps"][0])[nm] for r in bb])
        if a is not None:
            print(f"      {nm:20s} AUC {a:.3f}")
# share kept / gram among kept when filtering by pre-emption (each step must have P(never seen) >= 1/2 etc.)
print("\nfilter on the worst step's pre-emption log P(unseen) >= t: kept share, judged grammatical among kept")
for t in (-0.1, -0.3, -0.69, -1.0, -2.0, -3.0, -5.0):
    kept = [r for r in rows if min(stats(st)["pre-emption log"] for st in r["steps"]) >= t]
    print(f"  t {t:6.2f}: kept {len(kept) / len(rows):.3f}, gram {sum(r['judge'] > 0.5 for r in kept) / max(len(kept), 1):.3f}")
print("\nfilter on the worst step's both-sides log ratio >= t: kept share, judged grammatical among kept")
for t in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0):
    kept = [r for r in rows if min(stats(st)["both log ratio"] for st in r["steps"]) >= t]
    print(f"  t {t:5.2f}: kept {len(kept) / len(rows):.3f}, gram {sum(r['judge'] > 0.5 for r in kept) / max(len(kept), 1):.3f}")
print("\nsentence classes (by the kinds of its novel steps): share of sentences, judged grammatical")
cls = lambda r: ("has licensed bigram" if any(kind(s) == "bigram unstored (gate licence)" for s in r["steps"])
                 else "has novel end (no licensed)" if any(s["end"] for s in r["steps"]) else "middle trigrams only")
cc = defaultdict(list)
for r in rows:
    cc[cls(r)].append(r["judge"] > 0.5)
for k, v in cc.items():
    print(f"  {k:28s} {len(v) / len(rows):.3f}  gram {sum(v) / len(v):.3f}")
for k in ("pre-emption log", "both log ratio"):
    for t in ((-0.69,) if k == "pre-emption log" else (0.0,)):
        kept = [r for r in rows if not any(kind(s) == "bigram unstored (gate licence)" for s in r["steps"])
                and min(stats(st)[k] for st in r["steps"]) >= t]
        print(f"  no licensed bigram AND {k} >= {t}: kept {len(kept) / len(rows):.3f}, gram {sum(r['judge'] > 0.5 for r in kept) / max(len(kept), 1):.3f}")
# cases
print("\nCASES (judged ungrammatical, random 45):")
rng = random.Random(0)
uniq = {}
for r in bad:
    uniq.setdefault(r["gen"], r)
for r in rng.sample(list(uniq.values()), min(45, len(uniq))):
    print(f"\n- \"{r['gen']}\"  (judge {r['judge']:.3f}, {r['file']} temp {r['temp']}); novel: {r['novel trigrams']}")
    for st in r["steps"]:
        t = st["top"][0] if st["top"] else {}
        key = [k for k in st if k.endswith("anchor height (mass-wtd)")][0]
        ns = {k: v for k, v in st.items() if k.startswith("N ")}
        print(f"    step {st['pos']} w={st['w']} after ({st['l2']} {st['l1']}): p {st['p']}, gate {st['gate']} (lic {st['lic']} vs med {st['lic median']}), "
              f"A share {st['A share (own l1 = l1)']}, {key} {st[key]}, {ns}")
        if t:
            print(f"      top instance ({t['mass share']} of w's mass): own {t['own']}; leaf depth {t['leaf depth']} size {t['leaf size']}; "
                  f"anchors {json.dumps(t['anchors'])}; lift {t['lift']}")
