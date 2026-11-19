"""Round 214h part 1: oracle reachability. Take the R-sem sentences (syn_pass and sem_pass, r214_judge2) of a generator arm and, post hoc and
with no context, replace every word outside the K most frequent training types (K=50 by default; the top-K stay) by a uniformly random OTHER member of
its class at level b (class = word-tree class from a wordtree dump; members = non-top-K types of that class; a word with no other member stays).
Variants: <tree>-b<b> for tree in all / func100 and b in 4 / 6; "nocls" = a uniformly random other non-top-K type (the R-non end).
  r214_oracle.py make  <out dir> <scores.tsv> <dump dir> <arm index> <wordtree all.json> <wordtree func100.json> [draws]
  r214_oracle.py report <out dir> <scores.tsv of the oracle sentences> <out.md>
make writes <out>/sentences.txt and <out>/variants.json (variant -> list of [source sentence, oracle sentence]); chunks env, CPU."""
import json
import os
import random
import sys
from collections import Counter, defaultdict

K = 50


def tags(sc):
    return sc[2] == "1", sc[3] == "1"


def load_scores(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")]
    ix = {k: i for i, k in enumerate(rows[0])}
    return {r[0]: (float(r[ix["syn_p"]]), float(r[ix["sem_p"]]), r[ix["syn_pass"]], r[ix["sem_pass"]]) for r in rows[1:]}


def make(out, scores, dump, ai, wt_all, wt_func, draws=3):
    import glob
    import os
    J = load_scores(scores)
    f = sorted(glob.glob(os.path.join(dump, "*_arm%s.jsonl" % ai)))[0]
    lines = [json.loads(l) for l in open(f, encoding="utf-8")]
    arm = lines[0]["arm"]
    D = {"all": json.load(open(wt_all, encoding="utf-8")), "func100": json.load(open(wt_func, encoding="utf-8"))}
    sents = D["all"]["sentences"]
    cnt = Counter(w for s in sents for w in s)
    stored = {" ".join(s) for s in sents}
    top = {w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:K]}
    src = sorted({l["gen"] for l in lines[1:]} - stored)
    sem = [s for s in src if tags(J[s]) == (True, True)]
    rng = random.Random(214)
    members = {}
    for t, d in D.items():
        for b in ("4", "6"):
            cls = defaultdict(list)
            for w, c in d["names"][b].items():
                if w not in top:
                    cls[c].append(w)
            members[(t, b)] = (d["names"][b], cls)
    open_ = sorted(w for w in cnt if w not in top)
    V = {}
    for key, val in list(members.items()) + [(("nocls", "-"), (None, None))]:
        names, cls = val
        name = "nocls" if key[0] == "nocls" else "%s-b%s" % key
        res = []
        for s in sem:
            for _ in range(draws):
                o = []
                for w in s.split():
                    if w in top or w not in cnt:
                        o.append(w)
                        continue
                    pool = [u for u in (open_ if key[0] == "nocls" else cls[names[w]]) if u != w]
                    o.append(rng.choice(pool) if pool else w)
                res.append([s, " ".join(o)])
        V[name] = res
    pm = os.environ.get("R214_POSMAP")
    if pm:                                   # diagnostic variant "pos": swap with a random other non-top-K type of the same majority POS
        P_ = json.load(open(pm, encoding="utf-8"))
        bypos = defaultdict(list)
        for w in open_:
            bypos[P_.get(w)].append(w)
        res = []
        for s in sem:
            for _ in range(draws):
                o = []
                for w in s.split():
                    pool = [u for u in bypos.get(P_.get(w), []) if u != w] if (w not in top and w in cnt) else []
                    o.append(rng.choice(pool) if pool else w)
                res.append([s, " ".join(o)])
        V["pos"] = res
    os.makedirs(out, exist_ok=True)
    json.dump({"arm": arm, "n_sem": len(sem), "n_source": len(src), "variants": V}, open(os.path.join(out, "variants.json"), "w"))
    allv = sorted({o for v in V.values() for _, o in v} | set(sem))
    open(os.path.join(out, "sentences.txt"), "w", encoding="utf-8").write("\n".join(allv) + "\n")
    print("source R-sem", len(sem), "of", len(src), "variants", {k: len(v) for k, v in V.items()}, "sentences", len(allv))


def report(out, scores, out_md):
    import os
    J = load_scores(scores)
    D = json.load(open(os.path.join(out, "variants.json"), encoding="utf-8"))
    L = ["Source: %s; %d of its %d distinct new sentences are R-sem under the judge (they are the oracle's input).\n" % (json.dumps(D["arm"]), D["n_sem"], D["n_source"]),
         "| variant | sentences | changed words per sentence | R-sem | R-syn | R-non | (non with sem_pass) |", "|---|---|---|---|---|---|---|"]
    ex = []
    rng = random.Random(214)
    for name, rows in D["variants"].items():
        cls = Counter()
        ch = 0
        for s, o in rows:
            syn, sem = tags(J[o])
            cls["sem" if syn and sem else "syn" if syn else "non"] += 1
            ch += sum(a != b for a, b in zip(s.split(), o.split()))
        n = len(rows)
        salad = sum(1 for s, o in rows if not tags(J[o])[0] and tags(J[o])[1])
        L.append("| %s | %d | %.2f | %.3f | %.3f | %.3f | %.3f |" % (name, n, ch / n, cls["sem"] / n, cls["syn"] / n, cls["non"] / n, salad / n))
        ex.append("**%s**\n" % name + "\n".join("- %s  ->  %s  (syn_p %.2f, sem_p %.2f)" % (s, o, J[o][0], J[o][1]) for s, o in rng.sample(rows, 5)))
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n\n### Samples (5 per variant)\n\n" + "\n\n".join(ex) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], *(int(x) for x in sys.argv[8:]))
    else:
        report(*sys.argv[2:])
