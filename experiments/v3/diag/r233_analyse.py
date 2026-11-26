"""Round 233 analysis: exact-judge regimes and novelty of every arm dump.
usage: r233_analyse.py <dump dir> [<dump dir> ...]   (dir name <feats>_s<seed>_n<n>[...])
Per arm: R-sem / R-syn / R-non shares; new = not a training sentence; nb = has a word bigram never in training;
nt = has a trigram (with sentence edges) never in training; NC = new-correct = R-sem AND new AND nt
(the paper's fair count with the exact judge); NCb = R-sem AND has an unseen bigram; distinct = distinct gens / draws;
obj = share of draws violating an object restriction; nh = has a head pair (subject-verb, verb-object, adjective-noun) never in
training; NCh = R-sem AND nh (distinct): correct sentences that combine words never combined in training. NC counts distinct sentences only (a repeat counts against)."""
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402

B, E = "<s>", "</s>"


def grams(s, k):
    t = [B] * (k - 1) + list(s) + [E] * (k - 1) if k > 1 else list(s)
    return {tuple(t[i:i + k]) for i in range(len(t) - k + 1)}


def heads(t):
    """Head pairs of a sentence that parses: (subject noun, verb), (verb, object noun), (adjective, noun); nouns by base form,
    verbs by lemma. None if it does not parse."""
    t = list(t)
    if G.judge(t)[0] == "non":
        return None
    L = G.LEX
    base = lambda w: w[:-1] if L[w][2] == "pl" and w.endswith("s") else w
    out, i, nouns = set(), 0, []
    i = 1 + (L[t[1]][0] == "adj")
    if L[t[1]][0] == "adj":
        out.add(("an", t[1], base(t[2])))
    subj = base(t[i]); i += 1
    v = L[t[i]][1]; i += 1
    out.add(("sv", subj, v))
    if i < len(t) and t[i] in G.PREPS:
        i += 1
    if i < len(t):
        j = i + 1
        if L[t[j]][0] == "adj":
            out.add(("an", t[j], base(t[j + 1])))
            j += 1
        out.add(("vo", v, base(t[j])))
    return out


def train_sets(seed, n):
    C = G.load_corpus(seed, n)
    tr = [s for s, _ in C["train"]]
    bi = set().union(*(grams(s, 2) for s in tr))
    bi_in = {g for g in bi if B not in g and E not in g}
    tri = set().union(*(grams(s, 3) for s in tr))
    hp = set().union(*(heads(s) or set() for s in tr))
    return {tuple(s) for s in tr}, bi_in, tri, hp


def analyse(d):
    m = re.search(r"_s(\d+)_n(\d+)", os.path.basename(d.rstrip("/")))
    seed, n = int(m.group(1)), int(m.group(2))
    seen, bi, tri, hp = train_sets(seed, n)
    rows = []
    ref = [" ".join(x) for x in G.load_corpus(seed, n)["held"][:1600]]
    sources = [({"grammar": 1}, ref)]
    for f in sorted(glob.glob(os.path.join(d, "story_s*_arm*.jsonl")), key=lambda p: int(re.search(r"arm(\d+)", p).group(1))):
        lines = [json.loads(x) for x in open(f, encoding="utf-8")]
        sources.append((lines[0].get("arm", {}), [x["gen"] for x in lines[1:] if x.get("gen")]))
    for arm, txt in sources:
        gens = [x.split() for x in txt]
        N = len(gens)
        if not N:
            continue
        c = Counter()
        done = set()
        for g in gens:
            j, why = G.judge(g)
            c[j] += 1
            c["obj"] += "object" in why
            t = tuple(g)
            new = t not in seen
            nb = bool({x for x in grams(g, 2) if B not in x and E not in x} - bi)
            nt = bool(grams(g, 3) - tri)
            c["new"] += new
            c["nb"] += nb
            c["nt"] += nt
            h = heads(g)
            nh = bool(h - hp) if h is not None else False
            c["nh"] += nh
            first = t not in done
            done.add(t)
            c["NC"] += (j == "sem" and new and nt and first)
            c["NCb"] += (j == "sem" and nb and first)
            c["NCh"] += (j == "sem" and nh and first)
        r = {k: round(c[k] / N, 3) for k in ("sem", "syn", "non", "new", "nb", "nt", "nh", "NC", "NCb", "NCh", "obj")}
        r["distinct"] = round(len(done) / N, 3)
        lab = "grammar samples (reference)" if arm.get("grammar") else (f"W(causal, W'={arm.get('wreach')}, emit {arm.get('wemit')})") if arm.get("causal") else "T " + ",".join(f"{k}={v}" for k, v in arm.items() if k not in ("cond", "fill_name"))
        rows.append((lab, N, r))
    return seed, n, rows


if __name__ == "__main__":
    keys = ("sem", "syn", "non", "obj", "new", "nb", "nt", "nh", "distinct", "NC", "NCb", "NCh")
    for d in sys.argv[1:]:
        seed, n, rows = analyse(d)
        print(f"\n## {os.path.basename(d.rstrip('/'))}")
        print("| arm | N | " + " | ".join(keys) + " |")
        print("|---|---|" + "---|" * len(keys))
        for lab, N, r in rows:
            print(f"| {lab} | {N} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
