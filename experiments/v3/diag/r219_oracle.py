"""Round 219 step 1: agreement-aware oracle (diagnostic only; spaCy is never part of the model).
Input: the R-sem sentences of a generator arm (the sources inside an r214 oracle variants.json) and the training stream (a wordtree dump).
Variants (every word outside the K=50 most frequent training types is a candidate; a candidate is replaced by a random OTHER type with the same signature):
  fine    : signature = PTB tag_ (type-level majority over the stream)
  morph   : tag_ + Number/Tense/VerbForm + verb frame (dobj/dative/prt/ccomp/xcomp/acomp/attr/prep children) + adjective role (amod vs predicative)
            + noun determiner habit (mass vs count)
  morph_ar: morph + a/an repair after the swap
  morph1  : morph_ar with exactly one random swappable word swapped (C2one-like)
  chunk   : every spaCy noun chunk is replaced by a stored chunk of the same category (root tag, root role subj/obj/pobj/other, determiner kind,
            pronoun morph), then a/an repair
  chunk1  : one random chunk
  pos     : UPOS control (as in r214)
Signatures of the swapped word come from the source sentence tagged in context; the candidate's signature is its type-level majority over the stream.
  r219_oracle.py make <out> <variants.json (r214 oracle)> <wordtree.json> [draws]
  r219_oracle.py report <out> <scores.tsv> <out.md>
  r219_oracle.py controls <out dir>   (hand-built Chomsky-style syntax-only template sentences + clean controls; no LLM)"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

K = int(os.environ.get("R219_K", "50"))
CPOS = tuple(os.environ.get("R219_CPOS", "NOUN,VERB,ADJ,ADV").split(","))
FRAME = ("dobj", "dative", "prt", "ccomp", "xcomp", "acomp", "attr", "prep", "advcl", "oprd")
VOW = set("aeiou")


def tags(sc):
    return sc[2] == "1", sc[3] == "1"


def load_scores(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")]
    ix = {k: i for i, k in enumerate(rows[0])}
    return {r[0]: (float(r[ix["syn_p"]]), float(r[ix["sem_p"]]), r[ix["syn_pass"]], r[ix["sem_pass"]]) for r in rows[1:]}


def sig(tok, level):
    if level == "pos":
        return tok.pos_
    if level == "fine":
        return tok.tag_
    m = tok.morph
    f = [tok.tag_, "".join(m.get("Number")), "".join(m.get("Tense")), "".join(m.get("VerbForm"))]
    if tok.pos_ in ("VERB", "AUX"):
        f.append("fr:" + ",".join(sorted({c.dep_ for c in tok.children if c.dep_ in FRAME})))
    elif tok.pos_ == "ADJ":
        f.append("amod" if tok.dep_ == "amod" else "pred")
    elif tok.pos_ in ("NOUN",):
        f.append("det" if any(c.dep_ == "det" or c.dep_ == "poss" for c in tok.children) else "nodet")
    return "|".join(f)


def chunk_sig(ch):
    r = ch.root
    dep = r.dep_
    role = "subj" if dep.startswith("nsubj") else "obj" if dep in ("dobj", "dative", "attr", "oprd") else "pobj" if dep == "pobj" else "other"
    first = ch[0]
    det = "a" if first.lower_ in ("a", "an") else "the" if first.lower_ == "the" else "poss" if first.tag_ == "PRP$" else "dem" if first.tag_ == "DT" else "bare"
    if r.tag_ == "PRP":
        return "PRP|" + "".join(r.morph.get("Person")) + "".join(r.morph.get("Number")) + "".join(r.morph.get("Case")) + "|" + role
    return "|".join((r.tag_, role, det, "adj" if any(t.pos_ == "ADJ" for t in ch) else "noadj"))


def repair(words):
    w = list(words)
    for i in range(len(w) - 1):
        if w[i] in ("a", "an"):
            w[i] = "an" if w[i + 1][:1] in VOW else "a"
    return w


def make(out, variants_json, wordtree, draws=3):
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm")
    V0 = json.load(open(variants_json, encoding="utf-8"))
    sem = sorted({s for s, _ in V0["variants"]["pos"]})
    D = json.load(open(wordtree, encoding="utf-8"))
    sents = D["sentences"]
    cnt = Counter(w for s in sents for w in s)
    top = {w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:K]}
    open_ = sorted(w for w in cnt if w not in top)
    mk = lambda ws: Doc(nlp.vocab, words=ws)
    tsig = {lv: defaultdict(Counter) for lv in ("pos", "fine", "morph")}
    chunks = defaultdict(Counter)
    for d in nlp.pipe((mk(s) for s in sents), batch_size=64):
        for t in d:
            for lv in tsig:
                tsig[lv][t.text][sig(t, lv)] += 1
        for ch in d.noun_chunks:
            if len(ch) <= 4:
                chunks[chunk_sig(ch)][ch.text] += 1
    pools = {}
    for lv in tsig:
        p = defaultdict(list)
        for w in open_:
            p[tsig[lv][w].most_common(1)[0][0]].append(w)
        pools[lv] = p
    chunk_list = {k: sorted(c.elements()) for k, c in chunks.items()}
    rng = random.Random(219)
    docs = list(nlp.pipe((mk(s.split()) for s in sem), batch_size=64))
    V = defaultdict(list)

    def swap_tokens(d, lv, one, ar, content=False):
        idx = [t.i for t in d if t.text not in top and t.text in cnt and (not content or t.pos_ in CPOS)]
        if one and idx:
            idx = [rng.choice(idx)]
        ws = [t.text for t in d]
        for i in idx:
            pool = [u for u in pools[lv].get(sig(d[i], lv), []) if u != ws[i]]
            if pool:
                ws[i] = rng.choice(pool)
        return " ".join(repair(ws) if ar else ws)

    def swap_chunks(d, one, content=False):
        chs = [c for c in d.noun_chunks if len(c) <= 4 and chunk_list.get(chunk_sig(c)) and (not content or c.root.tag_ in ("NN", "NNS"))]
        if one and chs:
            chs = [rng.choice(chs)]
        out_ = []
        i = 0
        by = {c.start: c for c in chs}
        while i < len(d):
            if i in by:
                c = by[i]
                alts = [x for x in chunk_list[chunk_sig(c)] if x != c.text]
                out_.extend((rng.choice(alts) if alts else c.text).split())
                i = c.end
            else:
                out_.append(d[i].text)
                i += 1
        return " ".join(repair(out_))

    TR = {}
    for spec in filter(None, os.environ.get("R219_TREES", "").split(",")):
        nm, pth = spec.split("=")
        TR[nm] = json.load(open(pth, encoding="utf-8"))["names"]
    TB = os.environ.get("R219_TB", "4,5,6,7").split(",")
    upos = {w: tsig["pos"][w].most_common(1)[0][0] for w in cnt}
    tcls = {}
    for nm, names in TR.items():
        for b in TB:
            g = defaultdict(list)
            for w in cnt:
                if upos[w] in CPOS:
                    g[names[b][w]].append(w)
            tcls[(nm, b)] = (names[b], g)

    def swap_tree(d, key):
        names, g = tcls[key]
        ws = [t.text for t in d]
        for t in d:
            if t.text in top or t.text not in cnt or t.pos_ not in CPOS:
                continue
            pool = [u for u in g[names[t.text]] if u != t.text]
            if pool:
                ws[t.i] = rng.choice(pool)
        return " ".join(repair(ws))

    for s, d in zip(sem, docs):
        for key in tcls:
            for _ in range(draws):
                V["t_%s_b%s" % key].append([s, swap_tree(d, key)])
        for _ in range(draws):
            V["pos"].append([s, swap_tokens(d, "pos", False, False)])
            V["fine"].append([s, swap_tokens(d, "fine", False, False)])
            V["morph"].append([s, swap_tokens(d, "morph", False, False)])
            V["morph_ar"].append([s, swap_tokens(d, "morph", False, True)])
            V["morph1"].append([s, swap_tokens(d, "morph", True, True)])
            V["chunk"].append([s, swap_chunks(d, False)])
            V["chunk1"].append([s, swap_chunks(d, True)])
            V["fine_c"].append([s, swap_tokens(d, "fine", False, True, True)])
            V["morph_c"].append([s, swap_tokens(d, "morph", False, True, True)])
            V["morph_c1"].append([s, swap_tokens(d, "morph", True, True, True)])
            V["chunk_c"].append([s, swap_chunks(d, False, True)])
            V["chunk_c1"].append([s, swap_chunks(d, True, True)])
    if os.environ.get("R219_ONLY"):
        V = {k: v for k, v in V.items() if any(k.startswith(p) for p in os.environ["R219_ONLY"].split(","))}
    os.makedirs(out, exist_ok=True)
    json.dump({"n_sem": len(sem), "variants": V, "pools": {lv: {k: len(v) for k, v in p.items()} for lv, p in pools.items()},
               "n_chunk_cats": len(chunk_list)}, open(os.path.join(out, "variants.json"), "w"))
    allv = sorted({o for v in V.values() for _, o in v} | set(sem))
    open(os.path.join(out, "sentences.txt"), "w", encoding="utf-8").write("\n".join(allv) + "\n")
    print("sources", len(sem), "variants", {k: len(v) for k, v in V.items()}, "sentences", len(allv))


def report(out, scores, out_md):
    J = load_scores(scores)
    D = json.load(open(os.path.join(out, "variants.json"), encoding="utf-8"))
    L = ["%d R-sem source sentences (input; R-sem = 1 by construction).\n" % D["n_sem"],
         "| variant | n | changed words/sent | share unchanged | R-sem | R-syn | R-non | R-syn (changed only) | R-sem (changed only) | R-non (changed only) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ex = []
    rng = random.Random(219)
    for name, rows in D["variants"].items():
        c_all, c_ch = Counter(), Counter()
        chg = 0
        nun = 0
        for s, o in rows:
            syn, sm = tags(J[o])
            k = "sem" if syn and sm else "syn" if syn else "non"
            c_all[k] += 1
            nch = sum(a != b for a, b in zip(s.split(), o.split())) + abs(len(s.split()) - len(o.split()))
            chg += nch
            if o == s:
                nun += 1
            else:
                c_ch[k] += 1
        n = len(rows)
        m = max(1, sum(c_ch.values()))
        L.append("| %s | %d | %.2f | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            name, n, chg / n, nun / n, c_all["sem"] / n, c_all["syn"] / n, c_all["non"] / n, c_ch["syn"] / m, c_ch["sem"] / m, c_ch["non"] / m))
        ch_rows = [(s, o) for s, o in rows if o != s]
        ex.append("**%s**\n" % name + "\n".join("- %s  ->  %s  (syn_p %.2f, sem_p %.2f)" % (s, o, J[o][0], J[o][1]) for s, o in rng.sample(ch_rows, min(6, len(ch_rows)))))
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n\n### Samples (changed sentences only)\n\n" + "\n\n".join(ex) + "\n")
    print("\n".join(L))


def controls(out):
    """Hand-built template sentences (no LLM). Words from the 250-word story vocabulary (TinyStories). Four classes:
    SYN: syntax-only (Chomsky-style: right frame, number agreement, a/an, nonsensical selection); SEM: ordinary sensible; NON: scrambled/agreement-broken."""
    rng = random.Random(2190)
    cnouns = ["cat", "dog", "bird", "ball", "tree", "house", "door", "cake", "box", "car", "flower", "book", "boy", "girl", "toy", "water", "sun", "moon", "mom", "rock"]
    prop = ["tom", "lily", "sue", "ben", "anna"]
    adj = ["big", "little", "happy", "sad", "red", "new", "old", "small", "soft", "shiny"]
    tv = [("ate", "ate"), ("saw", "saw"), ("found", "found"), ("liked", "liked"), ("took", "took"), ("loved", "loved")]
    iv = ["slept", "smiled", "jumped", "laughed", "ran", "cried"]
    plural = {"cat": "cats", "dog": "dogs", "bird": "birds", "ball": "balls", "tree": "trees", "house": "houses", "door": "doors", "cake": "cakes", "box": "boxes", "car": "cars",
              "flower": "flowers", "book": "books", "boy": "boys", "girl": "girls", "toy": "toys", "rock": "rocks"}
    sens_s = {"cat": ["slept", "ran", "jumped"], "dog": ["ran", "jumped", "slept"], "bird": ["jumped", "slept"], "boy": ["ran", "laughed", "slept", "cried", "smiled"],
              "girl": ["ran", "laughed", "slept", "cried", "smiled"], "mom": ["smiled", "laughed", "slept"]}
    sens_o = {"cat": ["ball", "toy"], "dog": ["ball", "toy", "cake"], "bird": ["flower"], "boy": ["ball", "cake", "book", "toy"], "girl": ["ball", "cake", "book", "toy", "flower"],
              "mom": ["cake", "book", "flower"]}
    art = lambda w: "an" if w[0] in VOW else "a"
    SYN, SEM, NON = set(), set(), set()
    cn = [n for n in cnouns if n in plural and n not in ("water", "sun", "moon", "mom")]
    # SYN templates
    while len(SYN) < 100:
        t = rng.randrange(6)
        n1, n2 = rng.sample(cn, 2)
        a1, a2 = rng.sample(adj, 2)
        v, _ = rng.choice(tv)
        iv1 = rng.choice(iv)
        p = rng.choice(prop)
        if t == 0:
            s = "the %s %s the %s" % (n1, v, n2)
        elif t == 1:
            s = "%s %s %s %s %s" % (p, v, art(a1), a1, n2)
        elif t == 2:
            s = "the %s %s" % (n1, iv1)
        elif t == 3:
            s = "the %s %s %s %s %s" % (a1, n1, v, art(a2), "%s %s" % (a2, n2))
        elif t == 4:
            s = "the %s %s %s the %s" % (plural[n1], v, "", plural[n2])
        else:
            s = "%s and %s %s the %s" % (p, rng.choice(prop), v, n1)
        s = " ".join(s.split())
        SYN.add(s)
    # SEM: sensible and agreeing
    for subj in sens_s:
        for v in sens_s[subj]:
            SEM.add("the %s %s" % (subj, v))
    for subj, objs in sens_o.items():
        for o in objs:
            for v in ("found", "liked", "took", "saw"):
                if (subj in ("cat", "dog", "bird") and v == "took"):
                    continue
                SEM.add("the %s %s the %s" % (subj, v, o))
    for subj in ("boy", "girl"):
        for o in ("cake",):
            SEM.add("the %s ate the %s" % (subj, o))
            SEM.add("the %s ate a %s" % (subj, o))
    for p in prop:
        for o in ("ball", "cake", "book", "toy", "flower", "box"):
            SEM.add("%s found %s %s" % (p, art(o), o))
            SEM.add("%s loved %s %s %s" % (p, art(adj[0]), adj[0], o))
    SEM = sorted(SEM)
    rng.shuffle(SEM)
    SEM = SEM[:100]
    for s in sorted(SYN):
        w = s.split()
        c = rng.randrange(3)
        if c == 0:
            rng.shuffle(w)
            NON.add(" ".join(w))
        elif c == 1:
            NON.add(" ".join(w[::-1]))
        else:
            NON.add(" ".join(x for x in w if x not in ("the", "a", "an")))
    NON = {x for x in NON if x not in SYN}
    os.makedirs(out, exist_ok=True)
    C = {"SYN": sorted(SYN), "SEM": SEM, "NON": sorted(NON)}
    json.dump(C, open(os.path.join(out, "controls.json"), "w"), indent=1)
    open(os.path.join(out, "sentences.txt"), "w", encoding="utf-8").write("\n".join(sorted({s for v in C.values() for s in v})) + "\n")
    print({k: len(v) for k, v in C.items()})


def report_controls(out, scores, out_md):
    J = load_scores(scores)
    C = json.load(open(os.path.join(out, "controls.json"), encoding="utf-8"))
    L = ["| class | n | R-sem | R-syn | R-non | syntax pass | semantics pass |", "|---|---|---|---|---|---|---|"]
    for k, v in C.items():
        c = Counter()
        for s in v:
            syn, sm = tags(J[s])
            c["sem" if syn and sm else "syn" if syn else "non"] += 1
            c["sy"] += syn
            c["sm"] += sm
        n = len(v)
        L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (k, n, c["sem"] / n, c["syn"] / n, c["non"] / n, c["sy"] / n, c["sm"] / n))
    L.append("\nSYN items judged R-syn (5) / failing syntax (5):")
    ok = [s for s in C["SYN"] if tags(J[s]) == (True, False)]
    bad = [s for s in C["SYN"] if not tags(J[s])[0]]
    rng = random.Random(1)
    L += ["- OK  %s (syn_p %.2f sem_p %.2f)" % (s, J[s][0], J[s][1]) for s in rng.sample(ok, min(5, len(ok)))]
    L += ["- FAIL %s (syn_p %.2f sem_p %.2f)" % (s, J[s][0], J[s][1]) for s in rng.sample(bad, min(8, len(bad)))]
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def _main():
    c = sys.argv[1]
    if c == "make":
        make(sys.argv[2], sys.argv[3], sys.argv[4], *(int(x) for x in sys.argv[5:]))
    elif c == "report":
        report(*sys.argv[2:])
    elif c == "controls":
        controls(sys.argv[2])
    elif c == "purity":
        purity(sys.argv[2], *sys.argv[3:])
    elif c == "creport":
        report_controls(*sys.argv[2:])


def purity(out_md, *specs):
    """Morphosyntactic swap purity of word-tree classes (no judge): for each content token (NOUN/VERB/ADJ, in context) the share of the OTHER content
    types of its class at b whose type-level majority morph signature equals the token's in-context signature; token-weighted; also the mean number of
    candidates and the share of tokens with at least one candidate."""
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm")
    trees = {}
    for sp in specs:
        nm, p = sp.split("=")
        trees[nm] = json.load(open(p, encoding="utf-8"))
    S = next(iter(trees.values()))["sentences"]
    docs = list(nlp.pipe((Doc(nlp.vocab, words=s) for s in S), batch_size=64))
    ts = defaultdict(Counter)
    for d in docs:
        for t in d:
            ts[t.text][sig(t, "morph")] += 1
    maj = {w: c.most_common(1)[0][0] for w, c in ts.items()}
    upos = defaultdict(Counter)
    for d in docs:
        for t in d:
            upos[t.text][t.pos_] += 1
    cont = {w for w in upos if upos[w].most_common(1)[0][0] in CPOS}
    L = ["| tree | b | classes (content types) | mean candidates | tokens with a candidate | morph purity of candidates | purity (type-majority sig) |", "|---|---|---|---|---|---|---|"]
    for nm, T in trees.items():
        for b in ("3", "4", "5", "6", "6.5", "7"):
            names = T["names"][b]
            cls = defaultdict(list)
            for w in cont:
                cls[names[w]].append(w)
            n = hit = 0
            pur = cand = 0.0
            pur2 = 0.0
            for d in docs:
                for t in d:
                    if t.text in cont and t.pos_ in CPOS:
                        n += 1
                        pool = [u for u in cls[names[t.text]] if u != t.text]
                        if pool:
                            hit += 1
                            s0 = sig(t, "morph")
                            pur += sum(maj[u] == s0 for u in pool) / len(pool)
                            pur2 += sum(maj[u] == maj[t.text] for u in pool) / len(pool)
                            cand += len(pool)
            L.append("| %s | %s | %d | %.1f | %.2f | %.3f | %.3f |" % (nm, b, len(cls), cand / max(hit, 1), hit / n, pur / max(hit, 1), pur2 / max(hit, 1)))
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    _main()
