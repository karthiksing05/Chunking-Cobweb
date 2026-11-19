"""Round 214 class-emission arms: is class impurity the bottleneck? For every arm with a word-tree emission level b (arm key "wemit"),
each distinct new sentence's words are checked against their class: a word is an IMPURE emission when its majority part of speech
(spaCy UPOS over the training stream, in context) differs from the token-weighted majority POS of its class at b. Reported per arm:
the share of syntax-FAIL and of syntax-PASS sentences holding at least one impure emission, the mean number of impure words, and for
the syntax-fail sentences the most common (word POS <- class POS) pairs. If fails concentrate in impure emissions, impurity is the
bottleneck; if fail and pass sentences carry the same impurity, it is not.
Usage (env cobint): r214_impurity.py <wordtree.json> <scores.tsv> <out.md> <label>=<dump dir> [...]"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc


def main(wt, scores, out_md, specs):
    D = json.load(open(wt, encoding="utf-8"))
    nlp = spacy.load("en_core_web_sm")
    pos = defaultdict(Counter)
    for s, d in zip(D["sentences"], nlp.pipe((Doc(nlp.vocab, words=s) for s in D["sentences"]), batch_size=64)):
        for w, t in zip(s, d):
            pos[w][t.pos_] += 1
    wpos = {w: c.most_common(1)[0][0] for w, c in pos.items()}
    wcnt = {w: sum(c.values()) for w, c in pos.items()}
    stored = {" ".join(s) for s in D["sentences"]}
    rows = [l.rstrip("\n").split("\t") for l in open(scores, encoding="utf-8")]
    ix = {k: i for i, k in enumerate(rows[0])}
    syn = {r[0]: int(r[ix["syn_pass"]]) for r in rows[1:]}
    cmaj = {}
    for b, nm in D["names"].items():
        cc = defaultdict(Counter)
        for w, c in nm.items():
            cc[c][wpos.get(w)] += wcnt.get(w, 0)
        cmaj[b] = {c: v.most_common(1)[0][0] for c, v in cc.items()}
    L = ["| arm | b | classes | distinct new | syn-fail | impure share in syn-fail | impure share in syn-pass | impure words per sentence (fail / pass) | top impure (word POS <- class POS) among fails |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sp in specs:
        lab, d = sp.split("=", 1)
        for f in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
            lines = [json.loads(l) for l in open(f, encoding="utf-8")]
            arm = lines[0]["arm"]
            we = arm.get("wemit")
            if we in (None, "leaf"):
                continue
            key = str(we) if str(we) in D["names"] else str(float(we)).rstrip("0").rstrip(".")
            if key not in D["names"]:
                continue
            nm, cm = D["names"][key], cmaj[key]
            S = {l["gen"] for l in lines[1:]} - stored
            st = {"fail": [0, 0, 0], "pass": [0, 0, 0]}
            pairs = Counter()
            for s in S:
                k = "pass" if syn[s] else "fail"
                imp = [(wpos.get(w), cm[nm[w]]) for w in s.split() if w in nm and wpos.get(w) != cm[nm[w]]]
                st[k][0] += 1
                st[k][1] += bool(imp)
                st[k][2] += len(imp)
                if k == "fail":
                    pairs.update("%s<-%s" % p for p in imp)
            g = lambda k, i: (st[k][i] / st[k][0]) if st[k][0] else float("nan")
            L.append("| %s %s%s | %s | %d | %d | %d | %.3f | %.3f | %.2f / %.2f | %s |" % (
                lab, "gate " if arm.get("pair") else "", "W'=%s" % arm.get("wreach"), key, len(set(nm.values())), len(S), st["fail"][0],
                g("fail", 1), g("pass", 1), g("fail", 2), g("pass", 2), ", ".join("%s %d" % kv for kv in pairs.most_common(4))))
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
