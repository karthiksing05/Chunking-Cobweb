"""Round 192 diagnosis: where do TRELLIS's novel trigrams come from?

Rebuilds the seed's stream and learner exactly as t3_story.py (story follows), then, for every judged
causal words-gated generation in the given dump files, finds each unstored trigram (padded with <s>, </s>)
and traces the filler that produced its last word w after (l2, l1) (with window k, q = l1..lk, BOS padded):

  p        w's share of the gated filler distribution at that step (go-on and end summed)
  gate     "stored" if (l1, w) was stored, else "licensed" (type licence >= median)
  A share  share of w's mass from instances whose own l1 == q's l1 (the novelty is further left: l2 backed off)
  anchors  for the top instances, per attribute l_k: the deepest concept on the instance's context path holding
           q's value (the concept that licensed it), its height above the leaf and its count (size)
  lift     log P(q_a | anchor) / P(q_a | root) per attribute

For the end marker (trigram l1 w </s>) the same, for the end read (r1 = EOS).

Usage (compute node): r192_trace.py <cfg json> <seed> <n> <out.jsonl> <dump.jsonl> [<dump.jsonl> ...]
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3, BOS, EOS, FILL  # noqa: E402

cfg, seed, n, out = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dumps = sys.argv[5:]

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
train_st = stories[1500:]
train = []
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, any(keep(t) for t in st[:i])))
    if len(train) >= n:
        break
train = train[:n]


def grams(s, k):
    t = [BOS] + list(s) + [EOS]
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


tri = {g for s, _ in train for g in grams(s, 3)}
L = Trellis3(seed=seed, **cfg)
for s, fol in train:
    L.learn(s, follows=fol)
L._quiet = True
W = L.window
print(json.dumps({"seed": seed, "n": n, "window": W, "records": len(L.records)}), flush=True)

# the gate (t3_story.py causal "pair": "gate")
_w_, _ix_, _M_ = L.type_licence()
_st_ = set(L._pair_counts)
_vals_ = sorted(float(_M_[_ix_[a], _ix_[b]]) for a, b in _st_ if a in _ix_ and b in _ix_)
_med_ = _vals_[min(len(_vals_) - 1, int(len(_vals_) * 0.5))]


def ok(l1, w):
    if l1 == BOS or (l1, w) in _st_:
        return True
    return l1 in _ix_ and w in _ix_ and float(_M_[_ix_[l1], _ix_[w]]) >= _med_ > 0


def lic(l1, w):
    return float(_M_[_ix_[l1], _ix_[w]]) if l1 in _ix_ and w in _ix_ else 0.0


root = L.ctx.root
alpha = L.alpha
# the stored word occurrences' left contexts: how often each history (l1), (l2 l1) was followed by any word,
# and how often by each word (for pre-emption: a history seen often, never with w)
H1, H2, HW1, HW2 = Counter(), Counter(), Counter(), Counter()
for e in L.records:
    if e.leaf is None:
        continue
    for which, attr, value, _ in e.pas:
        if which == "ctx" and attr == FILL and value[0] == "w":
            l1, l2 = e.ctx.get("l1"), e.ctx.get("l2")
            H1[l1] += 1
            H2[(l2, l1)] += 1
            HW1[(l1, value[1])] += 1
            HW2[(l2, l1, value[1])] += 1


def depth(nd):
    d = 0
    while nd.parent is not None:
        nd, d = nd.parent, d + 1
    return d


def path(nd):
    p = []
    while nd is not None:
        p.append(nd)
        nd = nd.parent
    return p                     # leaf .. root


def probs_on(nd, q):
    """P(q_a | nd) with the same hierarchical back-off as _gap_fills (beta 1)."""
    sizes = {a: len(root.av.get(a, {})) + 1 for a in q}
    p = path(nd)[::-1]
    out = None
    for node in p:
        if node.parent is None:
            out = {a: (node.av.get(a, {}).get(v, 0.0) + alpha) / (node.count + alpha * sizes[a]) for a, v in q.items()}
        else:
            out = {a: (node.av.get(a, {}).get(v, 0.0) + out[a]) / (node.count + 1.0) for a, v in q.items()}
    return out


def anchor(nd, a, v):
    """The deepest concept on nd's path (leaf first) with a count for a = v."""
    for h, node in enumerate(path(nd)):
        c = node.av.get(a, {}).get(v, 0.0)
        if c > 0:
            return {"height": h, "depth": depth(node), "size": round(node.count, 1), "n_v": round(c, 2),
                    "share": round(c / node.count, 3) if node.count else None}
    return {"height": None}


def fills(q, w_filter=None):
    """Per filler: total mass; per instance of w_filter: (mass, instance)."""
    a_ = L._gap_fills(q)
    inst = []
    if w_filter is not None:
        for e in L.records:
            if e.leaf is None:
                continue
            for which, attr, value, wt in e.pas:
                if which == "ctx" and attr == FILL and value == ("w", w_filter):
                    lp = sum(math.log(max(v, 1e-300)) for v in probs_on(e.leaf, q).values())
                    inst.append((wt * math.exp(lp), e))
    return a_, inst


def trace(sent, i):
    """The step that placed sent[i] (i == len(sent): the end after sent[-1])."""
    hist = [BOS] * 5 + list(sent[:i] if i < len(sent) else sent[:-1])
    q = {f"l{k}": hist[-k] for k in range(1, W + 1)}
    w = sent[i] if i < len(sent) else sent[-1]
    end = i == len(sent)
    qq = {**q, "r1": EOS} if end else q
    a_, inst = fills(qq, w)
    if not end:
        tot = sum(m for v, m in a_.items() if v[0] == "w" and ok(q["l1"], v[1]))
        p = a_.get(("w", w), 0.0) / tot if tot else None
    else:   # end: P(end | w placed) among this w's mass, from the end read and the plain read
        a0 = L._gap_fills(q)
        p = a_.get(("w", w), 0.0) / a0.get(("w", w), 1e-300)
    inst.sort(key=lambda x: -x[0])
    z = sum(m for m, _ in inst) or 1e-300
    a_share = sum(m for m, e in inst if e.ctx.get("l1") == q["l1"]) / z
    b2_share = sum(m for m, e in inst if e.ctx.get("l1") == q["l1"] and e.ctx.get("l2") == q["l2"]) / z
    rp = probs_on(root, qq)
    top = []
    for m, e in inst[:3]:
        pr = probs_on(e.leaf, qq)
        top.append({"mass share": round(m / z, 3), "own": {a: e.ctx.get(a) for a in qq},
                    "leaf depth": depth(e.leaf), "leaf size": round(e.leaf.count, 1),
                    "anchors": {a: anchor(e.leaf, a, v) for a, v in qq.items()},
                    "lift": {a: round(math.log(pr[a] / rp[a]), 2) for a in qq}})
    # the deepest-anchor statistic of the mass-weighted instances: for l2 (or r1 at the end)
    key = "r1" if end else "l2"
    hs = [(m / z, anchor(e.leaf, key, qq[key])) for m, e in inst]
    mean_h = sum(s * (a["height"] if a["height"] is not None else 99) for s, a in hs)
    # both sides: the pivot (the word before w; at the end, w itself) re-predicted from its left context
    # with and without its right neighbour (w; at the end, the end marker): the ratio
    # P(pivot | left, right) / P(pivot | left) (> 1: the right neighbour supports the pivot)
    full = [BOS] * 5 + list(sent) + [EOS]
    j = 5 + (i - 1 if not end else len(sent) - 1)       # the pivot's index in full
    piv, right = full[j], full[j + 1]
    ql = {f"l{k}": full[j - k] for k in range(1, W + 1)}
    m1 = {v[1]: x for v, x in L._gap_fills(ql).items() if v[0] == "w"}
    m2 = {v[1]: x for v, x in L._gap_fills({**ql, "r1": right}).items() if v[0] == "w"}
    p1 = m1.get(piv, 0.0) / (sum(m1.values()) or 1e-300)
    p2 = m2.get(piv, 0.0) / (sum(m2.values()) or 1e-300)
    both = math.log(max(p2, 1e-300) / max(p1, 1e-300)) if piv != BOS else 0.0
    if end:
        hist_n ={"N (l1 w)": HW1[(q["l1"], w)], "N (l2 l1 w)": HW2[(q["l2"], q["l1"], w)]}
    else:
        hist_n = {"N l1": H1[q["l1"]], "N l2 l1": H2[(q["l2"], q["l1"])], "N l1 w": HW1[(q["l1"], w)]}
    return {"pos": i, "w": "</s>" if end else w, "l1": q["l1"], "l2": q["l2"], "end": end, "p": None if p is None else round(p, 4),
            **hist_n, "pivot": piv, "both log ratio": round(both, 3), "pivot p1": round(p1, 4), "pivot p2": round(p2, 4),
            "gate": "stored" if (q["l1"], w) in _st_ or q["l1"] == BOS else ("licensed" if ok(q["l1"], w) else "closed"),
            "lic": round(lic(q["l1"], w), 4), "lic median": round(_med_, 4),
            "A share (own l1 = l1)": round(a_share, 3), "l1 l2 both own": round(b2_share, 3),
            "n inst": len(inst), f"{key} anchor height (mass-wtd)": round(mean_h, 2), "top": top}


with open(out, "w", encoding="utf-8") as fh:
    for d in dumps:
        lines = [json.loads(l) for l in open(d, encoding="utf-8")]
        arm, rows = lines[0]["arm"], lines[1:]
        ps = json.load(open(d[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
        for r, pj in zip(rows, ps):
            s = r["gen"].split()
            g = grams(s, 3)
            nov = [k for k, t in enumerate(g) if t not in tri]
            if not nov:
                continue
            # trigram k is (t[k], t[k+1], t[k+2]) over [BOS]+s+[EOS]: its last word is s[k+1] (k+1 == len(s): EOS)
            steps = [trace(s, k + 1) for k in nov]
            fh.write(json.dumps({"file": os.path.basename(d), "temp": arm.get("temp", 1.0), "gen": r["gen"], "judge": pj,
                                 "novel trigrams": [" ".join(g[k]) for k in nov], "steps": steps}) + "\n")
print("done", flush=True)
