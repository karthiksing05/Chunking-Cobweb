"""Round 185: how consistent and phrase-like are TRELLIS's parses of TinyStories sentences, against simple
unsupervised chunking baselines trained on the same stream?

Run:  story_parse_eval.py <cfg json> <seed> <checkpoints e.g. 250,1000,4000> <tag>
Agg:  story_parse_eval.py agg <tag>      (reads results/story_parse_eval/<tag>_s*.json, writes <tag>_table.md/.json)

Stream: as story_chunks.py (same keep filter, story order, `follows`), EXCEPT the held-out set: story_chunks'
held-out stories depend on the seed, so two seeds cannot be compared on them. Here the held-out stories are fixed:
Random(0) shuffles the stories, the first 1500 are the test pool (never trained on by any seed), the rest are
reshuffled with Random(seed) to give the training stream. The 300 held-out sentences are the kept sentences of the
test pool, shuffled with Random(1000). So the training streams differ from story_chunks.py's (same recipe).

Methods (brackets = multi-word spans, whole sentence excluded):
  trellis_chosen  composites of L.parse's chosen tree (not the whole sentence)
  trellis_post    every span whose posterior (summed over splits) >= 0.5, size 2..len-1
  trellis_rec     chosen composites with posterior >= 0.5 (story_chunks' "recognised")
  bpe_x0.5/x1/x2  word-level BPE, merge budget = 0.5/1/2 x the distinct word strings TRELLIS stored as composites
  pmi_chosen/pmi_post  two-level PMI chunker, threshold set so its brackets per held-out sentence match trellis_chosen / trellis_post
  right_branching, random_binary
Gold (scoring only): story_chunks.py's NLTK shallow parse. "phrase" = NP/PP/VP/AP/VG spans of size >= 2;
"+sv" adds subject+verb spans (story_chunks' label).
"""
from __future__ import annotations

import heapq
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "story_parse_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))


# ---------------------------------------------------------------- baselines
def learn_bpe(sents, budget):
    """Word-level BPE: merge the most frequent adjacent pair of units, `budget` times (stops if no pair is left)."""
    seqs = [[(w,) for w in s] for s in sents]
    pc, where = Counter(), defaultdict(set)
    for si, q in enumerate(seqs):
        for p in zip(q, q[1:]):
            pc[p] += 1
            where[p].add(si)
    heap = [(-c, p) for p, c in pc.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) < budget and heap:
        c, p = heapq.heappop(heap)
        if pc.get(p, 0) != -c or -c <= 0:
            continue
        merges.append(p)
        new = p[0] + p[1]
        for si in list(where[p]):
            q = seqs[si]
            for x in zip(q, q[1:]):
                pc[x] -= 1
                if pc[x] <= 0:
                    del pc[x]
            r, k = [], 0
            while k < len(q):
                if k + 1 < len(q) and q[k] == p[0] and q[k + 1] == p[1]:
                    r.append(new)
                    k += 2
                else:
                    r.append(q[k])
                    k += 1
            seqs[si] = r
            for x in zip(r, r[1:]):
                pc[x] += 1
                where[x].add(si)
                heapq.heappush(heap, (-pc[x], x))
        where.pop(p, None)
    return merges


def apply_bpe(words, rank):
    """Greedy: repeatedly merge the adjacent pair with the lowest merge rank (earliest learned); leftmost on ties."""
    units = [((w,), i, i + 1) for i, w in enumerate(words)]
    brackets = set()
    while True:
        best, bk = None, None
        for k in range(len(units) - 1):
            r = rank.get((units[k][0], units[k + 1][0]))
            if r is not None and (best is None or r < best):
                best, bk = r, k
        if bk is None:
            break
        a, b = units[bk], units[bk + 1]
        units[bk:bk + 2] = [(a[0] + b[0], a[1], b[2])]
        brackets.add((a[1], b[2]))
    return {(i, j) for i, j in brackets if j - i < len(words)}


class PMIChunker:
    """Two levels of greedy left-to-right non-overlapping pairing of adjacent units with PMI > thr
    (add-one smoothed from training unigram/bigram counts of that level's units)."""

    def __init__(self, sents, thr):
        self.thr = thr
        self.l1 = self._stats([[(w,) for w in s] for s in sents])
        lvl2 = []
        for s in sents:
            lvl2.append([u for u, _, _ in self._pair([(w,) for w in s], self.l1)])
        self.l2 = self._stats(lvl2)

    @staticmethod
    def _stats(seqs):
        u, b = Counter(), Counter()
        for q in seqs:
            u.update(q)
            b.update(zip(q, q[1:]))
        import math
        return {"u": u, "b": b, "nu": sum(u.values()), "nb": sum(b.values()), "V": len(u), "log": math.log}

    @staticmethod
    def _pmi(a, b, st):
        V = st["V"] + 1
        pab = (st["b"][(a, b)] + 1) / (st["nb"] + V * V)
        pa = (st["u"][a] + 1) / (st["nu"] + V)
        pb = (st["u"][b] + 1) / (st["nu"] + V)
        return st["log"](pab / (pa * pb))

    def _pair(self, units, st):
        """units: list of word tuples -> [(unit, start, end)] pairing greedily left to right."""
        out, pos, k = [], 0, 0
        while k < len(units):
            if k + 1 < len(units) and self._pmi(units[k], units[k + 1], st) > self.thr:
                u = units[k] + units[k + 1]
                out.append((u, pos, pos + len(u)))
                pos += len(u)
                k += 2
            else:
                out.append((units[k], pos, pos + len(units[k])))
                pos += len(units[k])
                k += 1
        return out

    def brackets(self, words):
        n = len(words)
        l1 = self._pair([(w,) for w in words], self.l1)
        br = {(a, b) for _, a, b in l1 if b - a >= 2}
        l2 = self._pair([u for u, _, _ in l1], self.l2)
        br |= {(a, b) for _, a, b in l2 if b - a >= 2}
        return {(a, b) for a, b in br if b - a < n}


def right_branching(n):
    return {(i, n) for i in range(1, n - 1)}


def random_binary(n, rng):
    out = set()

    def rec(i, j, root):
        if j - i < 2:
            return
        if not root:
            out.add((i, j))
        k = rng.randrange(i + 1, j)
        rec(i, k, False)
        rec(k, j, False)
    rec(0, n, True)
    return out


# ---------------------------------------------------------------- metrics
def prf(pred, gold):
    """Micro unlabelled-span P/R/F1 over sentences (lists of sets)."""
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(len(p) for p in pred), sum(len(g) for g in gold)
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def f1_between(a, b):
    return prf(a, b)[2]


# ---------------------------------------------------------------- main
def main():
    import nltk
    from trellis2.stories import WORD, default_tinystories_path, read_sentences
    from trellis2.trellis3_r203 import Trellis3R203 as Trellis3, goodman_nary

    cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
    checkpoints = [int(x) for x in sys.argv[3].split(",")]
    tag = sys.argv[4]
    n = checkpoints[-1]
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

    # gold, with story_chunks.py's definitions (copied: that file runs at import)
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

    def crosses(i, j, spans):
        """story_chunks' 'crosses a boundary' (a subject+verb span never counts)."""
        if sv(i, j, spans):
            return False
        return any(a < j and i < b and not (a <= i and j <= b) and not (i <= a and b <= j) for a, b in spans)

    def sv(i, j, spans):
        lead = 1 if j - i > 2 and "CC" in spans.get((i, i + 1), ()) else 0
        return any(("NP" in spans.get((i, k), ()) or (lead and "NP" in spans.get((i + 1, k), ()))) and "VG" in spans.get((k, j), ())
                   for k in range(i + 1, j))

    GS = [phrases(s) for s in held]
    PH = ("NP", "PP", "VP", "AP", "VG")
    gold_ph = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and ij[1] - ij[0] < len(s) and v & set(PH)} for s, sp in zip(held, GS)]
    gold_sv = [g | {(i, j) for i in range(len(s)) for j in range(i + 2, len(s)) if sv(i, j, sp)} for g, s, sp in zip(gold_ph, held, GS)]
    gold_np = [{ij for ij, v in sp.items() if ij[1] - ij[0] >= 2 and "NP" in v and ij[1] - ij[0] < len(s)} for s, sp in zip(held, GS)]
    # multi-word strings (2..4 words) with >= 3 occurrences as a proper sub-span of held-out sentences
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

    L = Trellis3(seed=seed, **cfg)
    out = {"tag": tag, "seed": seed, "cfg": cfg, "held": held, "gold_phrase": [sorted(g) for g in gold_ph],
           "n_gold_phrase": sum(len(g) for g in gold_ph), "n_gold_sv": sum(len(g) for g in gold_sv),
           "strings": len(strings), "checkpoints": []}
    done, t0 = 0, time.time()
    distinct = set()
    for cp in checkpoints:
        for idx in range(done, cp):
            s, fol = train[idx]
            a0 = len(L.records)
            tree = L.learn(s, follows=fol)
            stored = {(e.span[0], e.split, e.span[1]) for e in L.records[a0:] if e.token is None}
            for (i, j), k in tree.split.items():
                if (i, k, j) in stored:
                    distinct.add(tuple(s[i:j]))
        done = cp
        D = len(distinct)
        sents = [s for s, _ in train[:cp]]
        L._quiet, pv = True, L._pv
        L._pv = None
        chosen, post, rec, SP = [], [], [], []
        for s in held:
            r = L.parse(s)
            sp = defaultdict(float)
            for (a, k, b), p in r["post"].items():
                sp[(a, b)] += p
            SP.append(dict(sp))
            ok = lambda i, j: 2 <= j - i < len(s)
            chosen.append({(i, j) for i, k, j in r["chosen"] if ok(i, j)})
            post.append({ij for ij, p in sp.items() if ok(*ij) and p >= 0.5})
            rec.append({(i, j) for i, k, j in r["chosen"] if ok(i, j) and sp[(i, j)] >= 0.5})
        L._quiet, L._pv = False, pv
        methods = {"trellis_chosen": chosen, "trellis_post": post, "trellis_rec": rec}
        info = {"distinct chunks D": D, "records": len(L.records)}
        # BPE: one run to the largest budget, prefixes for smaller ones
        merges = learn_bpe(sents, 2 * D)
        for mult in (0.5, 1, 2):
            m = merges[:int(round(mult * D))]
            rank = {p: r for r, p in enumerate(m)}
            methods[f"bpe_x{mult}"] = [apply_bpe(s, rank) for s in held]
            info[f"bpe_x{mult} merges"] = len(m)
        # r203: n-ary Goodman readouts and posterior thresholds from the same span posteriors.
        #   gm_p50: lam .5 (spans likelier than not, non-crossing by construction); gm_l0: lam 0 (all spans of the best
        #   binary tree); *_bud: lam / threshold bisected so brackets per held-out sentence equal BPE x1's (matched budget)
        okf = lambda i, j, s: 2 <= j - i < len(s)
        gm = lambda lam: [{ij for ij in goodman_nary(len(s), sp, lam) if okf(*ij, s)} for s, sp in zip(held, SP)]
        methods["gm_p50"] = gm(0.5)
        methods["gm_l0"] = gm(0.0)
        bpe_want = sum(len(b) for b in methods["bpe_x1"]) / len(held)

        def bisect(make, lo, hi):
            best = None
            for _ in range(24):
                mid = (lo + hi) / 2
                got = sum(len(b) for b in make(mid)) / len(held)
                if best is None or abs(got - bpe_want) < best[0]:
                    best = (abs(got - bpe_want), mid, make(mid))
                if got > bpe_want:
                    lo = mid
                else:
                    hi = mid
            return best
        b1 = bisect(gm, 0.0, 1.0)
        methods["gm_bud"] = b1[2]
        info["gm_bud lam"] = round(b1[1], 4)
        b2 = bisect(lambda t: [{ij for ij, p in sp.items() if okf(*ij, s) and p >= t} for s, sp in zip(held, SP)], 0.0, 1.0)
        methods["post_bud"] = b2[2]
        info["post_bud thr"] = round(b2[1], 4)
        info["bpe_x1 budget/sent"] = round(bpe_want, 3)
        # PMI: bisect the threshold so brackets per held-out sentence match the TRELLIS variant
        for name, tgt in (("pmi_chosen", chosen), ("pmi_post", post)):
            want = sum(len(b) for b in tgt) / len(held)
            lo, hi = -8.0, 12.0
            best = None
            for _ in range(22):
                mid = (lo + hi) / 2
                ch = PMIChunker(sents, mid)
                got = sum(len(ch.brackets(s)) for s in held) / len(held)
                if best is None or abs(got - want) < best[0]:
                    best = (abs(got - want), mid, ch)
                if got > want:
                    lo = mid
                else:
                    hi = mid
            methods[name] = [best[2].brackets(s) for s in held]
            info[f"{name} thr"] = round(best[1], 4)
            info[f"{name} target/sent"] = round(want, 3)
        methods["right_branching"] = [right_branching(len(s)) for s in held]
        rng = random.Random(seed * 7 + cp)
        methods["random_binary"] = [random_binary(len(s), rng) for s in held]
        row = {"learned": cp, "seconds": round(time.time() - t0), "info": info, "methods": {}}
        for name, brs in methods.items():
            row["methods"][name] = {"metrics": metrics(brs), "brackets": [sorted(b) for b in brs]}
        out["checkpoints"].append(row)
        print(json.dumps({"learned": cp, "seconds": row["seconds"], **info}), flush=True)
        for name in methods:
            print(name, json.dumps(row["methods"][name]["metrics"]), flush=True)
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
    assert len({json.dumps(runs[s]["held"]) for s in seeds}) == 1, "held-out sets differ across seeds"
    cps = [r["learned"] for r in runs[seeds[0]]["checkpoints"]]
    keys = ["brackets/sent", "P", "R", "F1", "F1+sv", "NP recall", "crossing", "consistency", "decisive", "xseed F1"]
    res = {}
    for ci, cp in enumerate(cps):
        names = list(runs[seeds[0]]["checkpoints"][ci]["methods"])
        for name in names:
            per = {s: dict(runs[s]["checkpoints"][ci]["methods"][name]["metrics"]) for s in seeds}
            brs = {s: [set(map(tuple, b)) for b in runs[s]["checkpoints"][ci]["methods"][name]["brackets"]] for s in seeds}
            pair = {}
            for a in seeds:
                for b in seeds:
                    if a < b:
                        pair[f"{a}-{b}"] = round(f1_between(brs[a], brs[b]), 4)
            for s in seeds:       # per seed: mean F1 against the other seeds
                o = [round(f1_between(brs[s], brs[t]), 4) for t in seeds if t != s]
                per[s]["xseed F1"] = round(sum(o) / len(o), 4) if o else None
            res[f"{cp}|{name}"] = {"per_seed": per, "pair_F1": pair}
    md = [f"seeds {seeds}; held-out sentences {len(runs[seeds[0]]['held'])}; gold phrase spans {runs[seeds[0]]['n_gold_phrase']}; "
          f"strings (>=3 occurrences) {runs[seeds[0]]['strings']}", ""]
    for ci, cp in enumerate(cps):
        info = {s: runs[s]["checkpoints"][ci]["info"] for s in seeds}
        md.append(f"### {cp} sentences learned   info: " + "; ".join(f"s{s}: {json.dumps(info[s])}" for s in seeds))
        md.append("")
        md.append("| method | " + " | ".join(keys) + " |")
        md.append("|---|" + "---|" * len(keys))
        for name in runs[seeds[0]]["checkpoints"][ci]["methods"]:
            r = res[f"{cp}|{name}"]["per_seed"]
            cells = []
            for k in keys:
                v = [r[s][k] for s in seeds]
                if any(x is None for x in v):
                    cells.append("n/a")
                    continue
                cells.append(f"{sum(v) / len(v):.3f} [{' / '.join(f'{x:.3f}' for x in v)}]")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    open(os.path.join(OUT, f"{tag}_table.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(res, open(os.path.join(OUT, f"{tag}_table.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg(sys.argv[2])
    else:
        main()
